use std::sync::Arc;

use llm_provider::{
    CompletionProvider, CompletionRequest, FinishReason, Message,
};
use observability::BudgetTracker;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::repl::{ReplSession, Variable};
use crate::recursion::{
    BudgetPool, LoopDetector, SubAgentConfig, SubAgentResult, SubRlmCall, schedule_sub_agents,
};
use crate::session::{RippleConfig, RippleResult};

/// Tracks variable changes per iteration for convergence detection.
#[derive(Debug, Default)]
struct IterationState {
    /// Variable names that changed in recent iterations (ring buffer of last 3).
    recent_changes: [Vec<String>; 3],
    /// Current ring buffer position.
    ring_pos: usize,
}

impl IterationState {
    fn record_changes(&mut self, changed_vars: Vec<String>) {
        self.recent_changes[self.ring_pos % 3] = changed_vars;
        self.ring_pos += 1;
    }

    /// Returns true if the last 3 iterations produced no variable changes.
    fn has_converged(&self) -> bool {
        if self.ring_pos < 3 {
            return false;
        }
        self.recent_changes.iter().all(|changes| changes.is_empty())
    }
}

/// Tool definition for the `sub_rlm` function, presented to the LLM as a callable tool.
pub fn sub_rlm_tool_definition() -> llm_provider::ToolDefinition {
    llm_provider::ToolDefinition {
        tool_type: "function".to_string(),
        function: llm_provider::FunctionDefinition {
            name: "sub_rlm".to_string(),
            description: "Spawn a recursive sub-agent to process a data variable with a focused instruction. \
                The sub-agent receives the data and returns its result as a variable. \
                Use this for: processing chunks of large data, cross-referencing segments, \
                or any task that benefits from focused sub-processing.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "variable_name": {
                        "type": "string",
                        "description": "Name of the REPL variable containing the data to process"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "The focused instruction for the sub-agent"
                    },
                    "result_variable": {
                        "type": "string",
                        "description": "Name to store the sub-agent's result under in the parent REPL"
                    }
                },
                "required": ["variable_name", "instruction", "result_variable"]
            }),
        },
    }
}

/// The Ripple execution engine. Runs the full REPL loop:
/// READ (present REPL state) → LLM generates actions → EVAL (execute tool calls) →
/// PRINT (append results) → LOOP (repeat until $FINAL or termination).
pub struct RippleExecutor {
    config: RippleConfig,
    sub_agent_config: SubAgentConfig,
}

impl RippleExecutor {
    pub fn new(config: RippleConfig, sub_agent_config: SubAgentConfig) -> Self {
        Self {
            config,
            sub_agent_config,
        }
    }

    pub fn with_defaults() -> Self {
        Self {
            config: RippleConfig::default(),
            sub_agent_config: SubAgentConfig::default(),
        }
    }

    /// Run the Ripple REPL loop.
    ///
    /// This implements the RLM Algorithm 1 from the paper:
    /// 1. Load prompt into REPL as symbolic variable
    /// 2. Loop: present REPL state to LLM, execute generated tool calls, check termination
    /// 3. Return $FINAL variable or best candidate result
    pub async fn run(
        &self,
        provider: &dyn CompletionProvider,
        repl_session: Arc<Mutex<ReplSession>>,
        budget: Arc<BudgetTracker>,
        tool_definitions: Vec<llm_provider::ToolDefinition>,
        tool_dispatch: &dyn ToolDispatch,
        depth: u8,
        system_prompt: &str,
    ) -> Result<RippleResult, anyhow::Error> {
        let mut conversation: Vec<Message> = vec![Message::system(system_prompt)];
        let mut iter_state = IterationState::default();
        let loop_detector = Arc::new(Mutex::new(LoopDetector::new()));
        let budget_pool = Arc::new(BudgetPool::new(budget.remaining_microdollars()));

        // Add sub_rlm tool if we're not at max depth
        let mut all_tools = tool_definitions;
        if depth < self.sub_agent_config.max_depth {
            all_tools.push(sub_rlm_tool_definition());
        }

        let tools_option = if all_tools.is_empty() {
            None
        } else {
            Some(all_tools)
        };

        // Inject complexity hints based on REPL state
        {
            let repl = repl_session.lock().await;
            let hints = self.generate_complexity_hints(&repl, depth);
            if !hints.is_empty() {
                conversation.push(Message::system(hints));
            }
        }

        let mut step = 0u32;
        let mut total_cost = 0u64;

        while step < self.config.max_steps {
            step += 1;
            debug!(step, depth, "Ripple loop iteration");

            // READ: Present current REPL state to LLM
            let repl_summary = {
                let repl = repl_session.lock().await;
                repl.format_state_summary()
            };

            if !repl_summary.starts_with("REPL state: (empty)") {
                conversation.push(Message::system(format!(
                    "Current REPL state (iteration {step}):\n{repl_summary}"
                )));
            }

            // LLM call
            let request = CompletionRequest {
                model: self.config.default_model.clone(),
                messages: conversation.clone(),
                tools: tools_option.clone(),
                temperature: Some(0.0),
                max_tokens: Some(4096),
            };

            let response = provider.complete(request).await?;

            // Record cost
            let caps = provider.capabilities();
            let cost_microdollars = ((response.usage.prompt_tokens as f64
                * caps.cost_per_input_token)
                + (response.usage.completion_tokens as f64 * caps.cost_per_output_token))
                * 1_000_000.0;
            let cost_microdollars = cost_microdollars as u64;
            total_cost += cost_microdollars;

            if let Err(e) = budget.record(cost_microdollars) {
                warn!(depth, step, "Budget exhausted during Ripple loop: {e}");
                break;
            }

            info!(
                depth,
                step,
                finish_reason = ?response.finish_reason,
                cost_usd = format!("{:.6}", cost_microdollars as f64 / 1_000_000.0),
                "Ripple iteration complete"
            );

            // Check for tool calls
            if response.finish_reason == FinishReason::ToolCalls {
                let tool_calls = response
                    .message
                    .tool_calls
                    .as_ref()
                    .cloned()
                    .unwrap_or_default();

                conversation.push(response.message);

                let mut changed_vars = Vec::new();

                // Separate sub_rlm calls from regular tool calls
                let mut regular_calls = Vec::new();
                let mut sub_rlm_raw = Vec::new();

                for (idx, tc) in tool_calls.iter().enumerate() {
                    if tc.function.name == "sub_rlm" {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments)
                                .unwrap_or_default();
                        sub_rlm_raw.push(SubRlmCall {
                            index: idx,
                            call_id: tc.id.clone(),
                            variable_name: args
                                .get("variable_name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            instruction: args
                                .get("instruction")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            result_variable: args
                                .get("result_variable")
                                .and_then(|v| v.as_str())
                                .unwrap_or("result")
                                .to_string(),
                            args,
                        });
                    } else {
                        regular_calls.push(tc);
                    }
                }

                // Process regular tool calls sequentially
                for tc in &regular_calls {
                    let name = &tc.function.name;
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_default();
                    let result = tool_dispatch.dispatch(name, args).await;

                    if name == "repl_set" {
                        if let Ok(parsed) =
                            serde_json::from_str::<serde_json::Value>(
                                &tc.function.arguments,
                            )
                        {
                            if let Some(var_name) =
                                parsed.get("name").and_then(|n| n.as_str())
                            {
                                changed_vars.push(var_name.to_string());
                            }
                        }
                    }

                    conversation.push(Message::tool_result(&tc.id, result));
                }

                // Process sub_rlm calls concurrently using wave scheduling
                if !sub_rlm_raw.is_empty() {
                    let waves = schedule_sub_agents(sub_rlm_raw);
                    debug!(
                        depth,
                        num_waves = waves.len(),
                        "Scheduled sub_rlm calls into execution waves"
                    );

                    // Collect results keyed by original tool_call index
                    let mut sub_results: std::collections::HashMap<
                        usize,
                        (String, String),
                    > = std::collections::HashMap::new();

                    for (wave_idx, wave) in waves.into_iter().enumerate() {
                        let max_conc =
                            self.sub_agent_config.max_concurrent.max(1);

                        // Process the wave in chunks of max_concurrent
                        for chunk in wave.calls.chunks(max_conc) {
                            let futures: Vec<_> = chunk
                                .iter()
                                .map(|call| {
                                    let repl_session =
                                        Arc::clone(&repl_session);
                                    let budget = Arc::clone(&budget);
                                    let budget_pool = Arc::clone(&budget_pool);
                                    let loop_det =
                                        Arc::clone(&loop_detector);
                                    let args = call.args.clone();
                                    let call_id = call.call_id.clone();
                                    let index = call.index;
                                    let result_variable =
                                        call.result_variable.clone();

                                    async move {
                                        let result = self
                                            .handle_sub_rlm(
                                                &args,
                                                provider,
                                                repl_session,
                                                budget,
                                                &budget_pool,
                                                loop_det,
                                                depth,
                                            )
                                            .await;
                                        (
                                            index,
                                            call_id,
                                            result_variable,
                                            result,
                                        )
                                    }
                                })
                                .collect();

                            let results =
                                futures::future::join_all(futures).await;

                            for (index, call_id, result_variable, result) in
                                results
                            {
                                let tool_result = match result {
                                    Ok(sub_result) => {
                                        changed_vars
                                            .push(result_variable.clone());
                                        total_cost +=
                                            sub_result.cost_microdollars;
                                        match &sub_result.output {
                                            Some(_) => format!(
                                                "Sub-agent completed in {} steps. Result stored in ${result_variable}.",
                                                sub_result.steps_taken
                                            ),
                                            None => "Sub-agent completed but produced no output.".to_string(),
                                        }
                                    }
                                    Err(e) => {
                                        format!("Sub-agent error: {e}")
                                    }
                                };
                                sub_results.insert(
                                    index,
                                    (call_id, tool_result),
                                );
                            }
                        }
                        debug!(
                            depth,
                            wave = wave_idx,
                            "Completed execution wave"
                        );
                    }

                    // Add tool results in original tool_call order
                    let mut indices: Vec<usize> =
                        sub_results.keys().cloned().collect();
                    indices.sort();
                    for idx in indices {
                        let (call_id, tool_result) =
                            sub_results.remove(&idx).unwrap();
                        conversation
                            .push(Message::tool_result(&call_id, tool_result));
                    }
                }

                iter_state.record_changes(changed_vars);

                // Check termination: $FINAL set
                {
                    let repl = repl_session.lock().await;
                    if repl.has_final() {
                        let final_output = repl
                            .finals()
                            .map(|(_k, v)| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        info!(depth, step, "Ripple terminated: $FINAL set");
                        return Ok(RippleResult {
                            steps_taken: step,
                            total_cost_microdollars: total_cost,
                            final_output: Some(final_output),
                        });
                    }
                }

                // Check convergence
                if iter_state.has_converged() {
                    info!(depth, step, "Ripple terminated: convergence (no changes in 3 iterations)");
                    let output = self.best_candidate_output(&repl_session).await;
                    return Ok(RippleResult {
                        steps_taken: step,
                        total_cost_microdollars: total_cost,
                        final_output: output,
                    });
                }

                continue;
            }

            // No tool calls — LLM produced a text response
            // This might be the final answer
            let content = response.message.content.clone().unwrap_or_default();
            conversation.push(response.message);

            // If this is the last iteration or looks like a final answer, use it
            if step >= self.config.max_steps - 1 || !content.is_empty() {
                // Store as $FINAL if not already set
                {
                    let mut repl = repl_session.lock().await;
                    if !repl.has_final() && !content.is_empty() {
                        repl.set("final_answer", Variable::Text(content.clone()));
                    }
                }
                return Ok(RippleResult {
                    steps_taken: step,
                    total_cost_microdollars: total_cost,
                    final_output: Some(content),
                });
            }

            iter_state.record_changes(vec![]);
        }

        // Max steps reached
        warn!(depth, "Ripple loop hit max steps ({})", self.config.max_steps);
        let output = self.best_candidate_output(&repl_session).await;
        Ok(RippleResult {
            steps_taken: step,
            total_cost_microdollars: total_cost,
            final_output: output,
        })
    }

    /// Handle a sub_rlm tool call: spawn a recursive sub-agent.
    async fn handle_sub_rlm(
        &self,
        args: &serde_json::Value,
        provider: &dyn CompletionProvider,
        repl_session: Arc<Mutex<ReplSession>>,
        budget: Arc<BudgetTracker>,
        budget_pool: &Arc<BudgetPool>,
        loop_detector: Arc<Mutex<LoopDetector>>,
        parent_depth: u8,
    ) -> Result<SubAgentResult, anyhow::Error> {
        let variable_name = args
            .get("variable_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("sub_rlm: missing variable_name"))?;
        let instruction = args
            .get("instruction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("sub_rlm: missing instruction"))?;
        let result_variable = args
            .get("result_variable")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("sub_rlm: missing result_variable"))?;

        // Get data preview for loop detection
        let data_preview = {
            let repl = repl_session.lock().await;
            match repl.metadata(variable_name) {
                Some(meta) => meta.preview.clone(),
                None => return Err(anyhow::anyhow!("Variable '{}' not found", variable_name)),
            }
        };

        // Loop detection
        {
            let mut detector = loop_detector.lock().await;
            if detector.check_and_record(instruction, &data_preview, parent_depth + 1) {
                return Err(anyhow::anyhow!(
                    "Loop detected: identical sub_rlm call at depth {}",
                    parent_depth + 1
                ));
            }
        }

        // Budget allocation
        let sub_budget = self
            .sub_agent_config
            .allocate_budget(budget_pool.remaining())
            .ok_or_else(|| anyhow::anyhow!("Insufficient budget for sub-agent"))?;

        let reserved = budget_pool
            .try_reserve(sub_budget, self.sub_agent_config.min_viable_budget_microdollars)
            .ok_or_else(|| anyhow::anyhow!("Budget pool exhausted"))?;

        let child_depth = parent_depth + 1;

        // At max depth, fall back to a direct LLM call instead of recursive Ripple
        if child_depth >= self.sub_agent_config.max_depth {
            info!(
                depth = child_depth,
                "At max depth, using direct LLM call instead of recursion"
            );
            let data_content = {
                let repl = repl_session.lock().await;
                repl.get(variable_name).map(|v| format!("{v}")).unwrap_or_default()
            };

            let request = CompletionRequest {
                model: self.config.default_model.clone(),
                messages: vec![
                    Message::system("You are a focused sub-agent. Process the data according to the instruction and provide a clear, structured result."),
                    Message::user(format!("Instruction: {instruction}\n\nData:\n{data_content}")),
                ],
                tools: None,
                temperature: Some(0.0),
                max_tokens: Some(4096),
            };

            let response = provider.complete(request).await?;
            let caps = provider.capabilities();
            let cost = ((response.usage.prompt_tokens as f64 * caps.cost_per_input_token)
                + (response.usage.completion_tokens as f64 * caps.cost_per_output_token))
                * 1_000_000.0;
            let cost = cost as u64;

            let _ = budget.record(cost);
            let unused = reserved.saturating_sub(cost);
            budget_pool.return_unused(unused);

            let output = response.message.content.unwrap_or_default();

            // Store result in parent's REPL
            {
                let mut repl = repl_session.lock().await;
                repl.set(result_variable, Variable::Text(output.clone()));
            }

            return Ok(SubAgentResult {
                output: Some(Variable::Text(output)),
                cost_microdollars: cost,
                steps_taken: 1,
            });
        }

        // Full recursive execution: create a child REPL with the data loaded
        let child_repl = Arc::new(Mutex::new(ReplSession::new()));
        {
            let parent_repl = repl_session.lock().await;
            if let Some(data) = parent_repl.get(variable_name) {
                let mut child = child_repl.lock().await;
                child.set("P", data.clone());
            }
        }

        let child_budget = Arc::new(BudgetTracker::with_microdollar_ceiling(reserved));

        let child_system = format!(
            "You are a focused sub-agent at recursion depth {child_depth}. \
            Your task: {instruction}\n\n\
            The input data is loaded as REPL variable $P. Use REPL tools to inspect and process it.\n\
            When done, use repl_set with name 'final_result' to store your output."
        );

        // Create a minimal RippleConfig for the child
        let child_config = RippleConfig {
            max_steps: (self.config.max_steps / 2).max(10), // Half parent's steps
            max_budget_microdollars: reserved,
            default_model: self.config.default_model.clone(),
        };

        let child_executor = RippleExecutor::new(child_config, self.sub_agent_config.clone());

        // The child needs tool definitions — we pass an empty set since it operates
        // on its own REPL (tools would need to be wired to the child REPL).
        // For now, sub-agents use direct LLM reasoning without tools.
        let result = Box::pin(child_executor.run(
            provider,
            Arc::clone(&child_repl),
            child_budget,
            vec![], // Sub-agents get no tools for now; they use direct LLM calls
            &NoOpDispatch,
            child_depth,
            &child_system,
        ))
        .await?;

        // Return unused budget
        let used = result.total_cost_microdollars;
        let unused = reserved.saturating_sub(used);
        budget_pool.return_unused(unused);

        // Copy result back to parent REPL
        if let Some(ref output) = result.final_output {
            let mut parent_repl = repl_session.lock().await;
            parent_repl.set(result_variable, Variable::Text(output.clone()));
        }

        Ok(SubAgentResult {
            output: result.final_output.map(Variable::Text),
            cost_microdollars: used,
            steps_taken: result.steps_taken,
        })
    }

    /// Generate complexity hints based on REPL state.
    fn generate_complexity_hints(&self, repl: &ReplSession, depth: u8) -> String {
        let mut hints = Vec::new();

        // Check for large variables
        for name in repl.variable_names() {
            if let Some(meta) = repl.metadata(&name) {
                if meta.token_estimate > 4096 {
                    hints.push(format!(
                        "Variable ${name} has ~{} tokens, exceeding single-call capacity. \
                        Consider chunking via repl_chunks and processing via sub_rlm.",
                        meta.token_estimate
                    ));
                }
            }
        }

        if depth > 0 {
            hints.push(format!(
                "You are at recursion depth {depth}. Focus on your specific sub-task \
                and store your result in a final_ variable when done."
            ));
        }

        hints.join("\n")
    }

    /// Get the best candidate output from the REPL when $FINAL wasn't explicitly set.
    async fn best_candidate_output(&self, repl_session: &Arc<Mutex<ReplSession>>) -> Option<String> {
        let repl = repl_session.lock().await;
        // Check for final_ variables first
        if repl.has_final() {
            return Some(
                repl.finals()
                    .map(|(_, v)| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        // Otherwise return the most recently set variable that looks like a result
        // (heuristic: prefer variables named "result", "output", "answer", or the largest one)
        let result_names = ["result", "output", "answer", "summary", "response"];
        for name in &result_names {
            if let Some(var) = repl.get(name) {
                return Some(format!("{var}"));
            }
        }
        None
    }
}

/// Trait for dispatching tool calls (allows the executor to be decoupled from ToolRouter).
#[async_trait::async_trait]
pub trait ToolDispatch: Send + Sync {
    async fn dispatch(&self, tool_name: &str, arguments: serde_json::Value) -> String;
}

/// No-op tool dispatch for sub-agents that don't have tools.
struct NoOpDispatch;

#[async_trait::async_trait]
impl ToolDispatch for NoOpDispatch {
    async fn dispatch(&self, tool_name: &str, _arguments: serde_json::Value) -> String {
        format!("Tool '{tool_name}' not available in this sub-agent context")
    }
}
