use std::sync::Arc;

use llm_provider::{CompletionProvider, CompletionRequest, FinishReason, Message};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::session::SessionStore;
use crate::tool_router::ToolRouter;

pub struct AgentLoop {
    provider: Box<dyn CompletionProvider>,
    tool_router: ToolRouter,
    conversation: Vec<Message>,
    model: String,
    max_steps: u32,
    budget: Arc<BudgetTracker>,
    repl_session: Arc<Mutex<ReplSession>>,
    session_store: Option<SessionStore>,
}

impl AgentLoop {
    pub fn new(
        provider: Box<dyn CompletionProvider>,
        tool_router: ToolRouter,
        system_prompt: impl Into<String>,
        model: impl Into<String>,
        max_steps: u32,
        budget: Arc<BudgetTracker>,
        repl_session: Arc<Mutex<ReplSession>>,
    ) -> Self {
        let mut conversation = Vec::new();
        conversation.push(Message::system(system_prompt));

        Self {
            provider,
            tool_router,
            conversation,
            model: model.into(),
            max_steps,
            budget,
            repl_session,
            session_store: None,
        }
    }

    /// Enable session persistence to a JSONL file.
    pub fn with_session_store(mut self, store: SessionStore) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Compute cost in microdollars from token usage.
    fn compute_cost_microdollars(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        let caps = self.provider.capabilities();
        let cost_usd = (prompt_tokens as f64 * caps.cost_per_input_token)
            + (completion_tokens as f64 * caps.cost_per_output_token);
        (cost_usd * 1_000_000.0) as u64
    }

    /// Get current spend in dollars.
    pub fn spent_dollars(&self) -> f64 {
        self.budget.spent_dollars()
    }

    /// Persist a message to the session store (if enabled).
    async fn persist(&self, message: &Message) {
        if let Some(store) = &self.session_store {
            store.log_message(message).await;
        }
    }

    /// Build the messages list, injecting REPL state summary if variables exist.
    async fn build_messages(&self) -> Vec<Message> {
        let repl = self.repl_session.lock().await;
        let summary = repl.format_state_summary();

        let mut messages = self.conversation.clone();

        if !summary.starts_with("REPL state: (empty)") {
            if let Some(pos) = messages.iter().rposition(|m| m.role == llm_provider::Role::User) {
                messages.insert(
                    pos,
                    Message::system(format!("Current REPL session state:\n{summary}")),
                );
            }
        }

        messages
    }

    /// Process a user message through the agent loop.
    /// Returns the final assistant text response.
    pub async fn process(&mut self, user_input: &str) -> Result<String, anyhow::Error> {
        let user_msg = Message::user(user_input);
        self.persist(&user_msg).await;
        self.conversation.push(user_msg);

        let tools = self.tool_router.tool_definitions();
        let tools_option = if tools.is_empty() { None } else { Some(tools) };

        for step in 0..self.max_steps {
            debug!(step, "Agent loop step");

            let messages = self.build_messages().await;

            let request = CompletionRequest {
                model: self.model.clone(),
                messages,
                tools: tools_option.clone(),
                temperature: Some(0.0),
                max_tokens: Some(4096),
            };

            let response = self.provider.complete(request).await?;

            // Record cost
            let cost_microdollars = self.compute_cost_microdollars(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            );
            if let Err(e) = self.budget.record(cost_microdollars) {
                warn!("Budget exceeded: {e}");
                if let Some(content) = &response.message.content {
                    self.persist(&response.message).await;
                    self.conversation.push(response.message.clone());
                    return Ok(format!(
                        "{content}\n\n[Budget exceeded — session cost: ${:.4}]",
                        self.budget.spent_dollars()
                    ));
                }
                return Err(anyhow::anyhow!("Budget exceeded: {e}"));
            }

            info!(
                finish_reason = ?response.finish_reason,
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                cost_usd = format!("{:.6}", cost_microdollars as f64 / 1_000_000.0),
                total_spent_usd = format!("{:.6}", self.budget.spent_dollars()),
                "LLM response"
            );

            // Check if the assistant wants to call tools
            if response.finish_reason == FinishReason::ToolCalls {
                let tool_calls = response
                    .message
                    .tool_calls
                    .as_ref()
                    .cloned()
                    .unwrap_or_default();

                self.persist(&response.message).await;
                self.conversation.push(response.message);

                for tool_call in &tool_calls {
                    let name = &tool_call.function.name;
                    let arguments: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments).unwrap_or_default();

                    info!(tool = %name, "Executing tool call");

                    let result = self.tool_router.dispatch(name, arguments).await;

                    if result.is_error {
                        warn!(
                            tool = %name,
                            error = %result.content,
                            duration_ms = result.duration_ms,
                            "Tool execution failed"
                        );
                    } else {
                        info!(
                            tool = %name,
                            duration_ms = result.duration_ms,
                            "Tool execution completed"
                        );
                    }

                    let tool_msg = Message::tool_result(&tool_call.id, &result.content);
                    self.persist(&tool_msg).await;
                    self.conversation.push(tool_msg);
                }

                // Check if REPL has final variables — early termination
                {
                    let repl = self.repl_session.lock().await;
                    if repl.has_final() {
                        let finals: Vec<String> = repl
                            .finals()
                            .map(|(k, v)| format!("{k}: {v}"))
                            .collect();
                        info!("REPL has final variables, terminating loop");
                        return Ok(finals.join("\n"));
                    }
                }

                continue;
            }

            // No tool calls — final response
            self.persist(&response.message).await;
            let content = response
                .message
                .content
                .clone()
                .unwrap_or_default();
            self.conversation.push(response.message);
            return Ok(content);
        }

        warn!("Agent loop hit max steps ({})", self.max_steps);
        Err(anyhow::anyhow!(
            "Agent loop exceeded maximum steps ({})",
            self.max_steps
        ))
    }
}
