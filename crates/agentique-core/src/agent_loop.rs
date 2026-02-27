use std::sync::Arc;

use futures::StreamExt;
use llm_provider::{
    CompletionProvider, CompletionRequest, FinishReason, Message, ToolCall, ToolCallFunction,
    Usage,
};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::session::SessionStore;
use crate::tool_router::ToolRouter;

/// Events emitted during streaming agent processing.
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    Token(String),
    ToolCallStart(String),
    ToolCallEnd(String),
    Done(String),
}

/// Accumulator for building tool calls from streaming deltas.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

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

    /// Read-only access to the conversation history.
    pub fn conversation(&self) -> &[Message] {
        &self.conversation
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

    /// Process a user message through the agent loop (non-streaming fallback).
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

    /// Process a user message with streaming, emitting events as tokens arrive.
    pub async fn process_streaming<F>(
        &mut self,
        user_input: &str,
        on_event: F,
    ) -> Result<String, anyhow::Error>
    where
        F: Fn(AgentStreamEvent) + Send + 'static,
    {
        let user_msg = Message::user(user_input);
        self.persist(&user_msg).await;
        self.conversation.push(user_msg);

        let tools = self.tool_router.tool_definitions();
        let tools_option = if tools.is_empty() { None } else { Some(tools) };

        for step in 0..self.max_steps {
            debug!(step, "Agent loop step (streaming)");

            let messages = self.build_messages().await;

            let request = CompletionRequest {
                model: self.model.clone(),
                messages,
                tools: tools_option.clone(),
                temperature: Some(0.0),
                max_tokens: Some(4096),
            };

            // Consume the stream, accumulating content and tool calls
            let mut stream = self.provider.complete_streaming(request).await?;

            let mut full_content = String::new();
            let mut tool_accumulators: Vec<ToolCallAccumulator> = Vec::new();
            let mut finish_reason = FinishReason::Stop;
            let mut usage = Usage::default();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;

                // Emit text tokens as they arrive
                if let Some(delta) = &chunk.delta_content {
                    if !delta.is_empty() {
                        full_content.push_str(delta);
                        on_event(AgentStreamEvent::Token(delta.clone()));
                    }
                }

                // Accumulate tool call deltas
                if let Some(delta_tcs) = &chunk.delta_tool_calls {
                    for tc in delta_tcs {
                        // Find or create accumulator by index (use id presence as signal for new call)
                        if !tc.id.is_empty() {
                            // New tool call
                            tool_accumulators.push(ToolCallAccumulator {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            });
                        } else if let Some(acc) = tool_accumulators.last_mut() {
                            // Continuation of the last tool call
                            if !tc.function.name.is_empty() {
                                acc.name.push_str(&tc.function.name);
                            }
                            acc.arguments.push_str(&tc.function.arguments);
                        }
                    }
                }

                if let Some(fr) = chunk.finish_reason {
                    finish_reason = fr;
                }

                if let Some(u) = chunk.usage {
                    usage = u;
                }
            }

            // Record cost
            let cost_microdollars =
                self.compute_cost_microdollars(usage.prompt_tokens, usage.completion_tokens);
            if let Err(e) = self.budget.record(cost_microdollars) {
                warn!("Budget exceeded: {e}");
                if !full_content.is_empty() {
                    let msg = Message::assistant(&full_content);
                    self.persist(&msg).await;
                    self.conversation.push(msg);
                    let result = format!(
                        "{full_content}\n\n[Budget exceeded — session cost: ${:.4}]",
                        self.budget.spent_dollars()
                    );
                    on_event(AgentStreamEvent::Done(result.clone()));
                    return Ok(result);
                }
                return Err(anyhow::anyhow!("Budget exceeded: {e}"));
            }

            info!(
                finish_reason = ?finish_reason,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                cost_usd = format!("{:.6}", cost_microdollars as f64 / 1_000_000.0),
                total_spent_usd = format!("{:.6}", self.budget.spent_dollars()),
                "LLM response (streaming)"
            );

            // Handle tool calls
            if finish_reason == FinishReason::ToolCalls && !tool_accumulators.is_empty() {
                // Build the assistant message with tool calls
                let tool_calls: Vec<ToolCall> = tool_accumulators
                    .iter()
                    .map(|acc| ToolCall {
                        id: acc.id.clone(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: acc.name.clone(),
                            arguments: acc.arguments.clone(),
                        },
                    })
                    .collect();

                let assistant_msg = Message {
                    role: llm_provider::Role::Assistant,
                    content: if full_content.is_empty() {
                        None
                    } else {
                        Some(full_content.clone())
                    },
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                };

                self.persist(&assistant_msg).await;
                self.conversation.push(assistant_msg);

                for tool_call in &tool_calls {
                    let name = &tool_call.function.name;
                    let arguments: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments).unwrap_or_default();

                    info!(tool = %name, "Executing tool call");
                    on_event(AgentStreamEvent::ToolCallStart(name.clone()));

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

                    on_event(AgentStreamEvent::ToolCallEnd(name.clone()));

                    let tool_msg = Message::tool_result(&tool_call.id, &result.content);
                    self.persist(&tool_msg).await;
                    self.conversation.push(tool_msg);
                }

                // Check for final variables
                {
                    let repl = self.repl_session.lock().await;
                    if repl.has_final() {
                        let finals: Vec<String> = repl
                            .finals()
                            .map(|(k, v)| format!("{k}: {v}"))
                            .collect();
                        info!("REPL has final variables, terminating loop");
                        let result = finals.join("\n");
                        on_event(AgentStreamEvent::Done(result.clone()));
                        return Ok(result);
                    }
                }

                continue;
            }

            // No tool calls — final response
            let msg = Message::assistant(&full_content);
            self.persist(&msg).await;
            self.conversation.push(msg);
            on_event(AgentStreamEvent::Done(full_content.clone()));
            return Ok(full_content);
        }

        warn!("Agent loop hit max steps ({})", self.max_steps);
        Err(anyhow::anyhow!(
            "Agent loop exceeded maximum steps ({})",
            self.max_steps
        ))
    }
}
