use std::sync::Arc;

use futures::StreamExt;
use llm_provider::{
    CompletionProvider, CompletionRequest, FinishReason, Message, ToolCall, ToolCallFunction,
    Usage,
};
use observability::BudgetTracker;
use ripple_engine::{ReplSession, RippleConfig, RippleExecutor, SubAgentConfig, ToolDispatch};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::session::{SessionCompressor, SessionStore};
use crate::tool_router::ToolRouter;

/// Adapter bridging ToolRouter to the ToolDispatch trait used by RippleExecutor.
pub struct ToolRouterDispatch<'a> {
    router: &'a ToolRouter,
}

#[async_trait::async_trait]
impl<'a> ToolDispatch for ToolRouterDispatch<'a> {
    async fn dispatch(&self, tool_name: &str, arguments: serde_json::Value) -> String {
        let result = self.router.dispatch(tool_name, arguments).await;
        result.content
    }
}

// ---------------------------------------------------------------------------
// State machine types
// ---------------------------------------------------------------------------

/// The current state of the agent loop.
#[derive(Debug, Clone)]
pub enum AgentState {
    /// Waiting for user input.
    Idle,
    /// About to call the LLM.
    LlmCall,
    /// LLM responded with tool calls; executing them.
    ToolExecution { calls: Vec<PendingToolCall> },
    /// A tool call requires user approval before execution.
    AwaitingApproval { pending: PendingToolCall, remaining: Vec<PendingToolCall> },
    /// Final response produced.
    Complete { result: String },
    /// An error occurred.
    Error { error: String, recoverable: bool },
}

/// A tool call waiting to be executed (or approved).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// Operations sent *to* the agent loop (from UI / CLI).
#[derive(Debug, Clone)]
pub enum AgentOp {
    /// Submit a new user message.
    UserTurn(String),
    /// Interrupt the agent loop.
    Interrupt,
    /// Respond to a tool approval request.
    ExecApproval {
        call_id: String,
        approved: bool,
        /// If true, auto-approve this tool for the rest of the session.
        remember_session: bool,
    },
}

/// Events emitted *from* the agent loop (to UI / CLI).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Token(String),
    ToolCallStart(String),
    ToolCallEnd(String),
    ToolApprovalRequired {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    AssistantMessage(String),
    StepProgress {
        step: u32,
        state: String,
        model: Option<String>,
        tokens_in: Option<u32>,
        tokens_out: Option<u32>,
        cost_usd: Option<f64>,
    },
    CostUpdate(f64),
    ArtifactCreated {
        file_path: String,
        old_content: Option<String>,
        new_content: String,
    },
    TaskComplete(String),
    Error(String),
}

/// Legacy streaming events (kept for backward-compat `process_streaming`).
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
    /// Tools auto-approved for this session (by name).
    session_approved_tools: std::collections::HashSet<String>,
    /// MCP permission manager for MCP tool approval policies.
    mcp_permissions: Option<Arc<Mutex<mcp_manager::PermissionManager>>>,
    /// Session compressor for long conversations.
    compressor: SessionCompressor,
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
            session_approved_tools: std::collections::HashSet::new(),
            mcp_permissions: None,
            compressor: SessionCompressor::default(),
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

    /// Configure the session compressor.
    pub fn with_compressor(mut self, compressor: SessionCompressor) -> Self {
        self.compressor = compressor;
        self
    }

    /// Attach an MCP permission manager for MCP tool approval policies.
    pub fn with_mcp_permissions(mut self, permissions: Arc<Mutex<mcp_manager::PermissionManager>>) -> Self {
        self.mcp_permissions = Some(permissions);
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

    /// Check whether the conversation needs compaction and perform it if so.
    /// Returns true if compaction occurred.
    async fn maybe_compact(&mut self) -> bool {
        let max_tokens = self.provider.capabilities().max_context_tokens;
        if !self.compressor.needs_compaction(&self.conversation, max_tokens) {
            return false;
        }

        match self.compressor.compact(&self.conversation, self.provider.as_ref()).await {
            Some(result) => {
                // Log the compaction event.
                if let Some(store) = &self.session_store {
                    let event = crate::session::SessionEvent {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        event_type: crate::session::SessionEventType::Compaction,
                        data: serde_json::json!({
                            "original_tokens": result.original_tokens,
                            "compacted_tokens": result.compacted_tokens,
                            "messages_summarized": result.messages_summarized,
                        }),
                    };
                    if let Err(e) = store.append_event(event).await {
                        warn!("Failed to log compaction event: {e}");
                    }
                }

                self.conversation = result.messages;
                true
            }
            None => false,
        }
    }

    /// Check whether a tool should be auto-approved.
    ///
    /// For built-in tools: read-only + non-destructive = auto-approved.
    /// For MCP tools: delegates to the MCP PermissionManager's 3-tier policy.
    /// Session-remembered approvals always auto-approve.
    fn should_auto_approve(&self, tool_name: &str) -> bool {
        if self.session_approved_tools.contains(tool_name) {
            return true;
        }

        // For MCP tools, consult MCP permission manager
        if self.tool_router.is_mcp_tool(tool_name) {
            if let Some(perms) = &self.mcp_permissions {
                // We can't block here, so use try_lock
                if let Ok(pm) = perms.try_lock() {
                    let annotations = self.tool_router.annotations(tool_name)
                        .unwrap_or_default();
                    return pm.is_approved(
                        tool_name,
                        annotations.read_only_hint,
                        annotations.destructive_hint,
                    );
                }
            }
            return false; // MCP tools default to needing approval
        }

        // Built-in tools: auto-approve if read-only and non-destructive
        if let Some(annotations) = self.tool_router.annotations(tool_name) {
            return annotations.read_only_hint && !annotations.destructive_hint;
        }
        false
    }

    // -----------------------------------------------------------------------
    // Channel-based API (new)
    // -----------------------------------------------------------------------

    /// Run the agent loop driven by an op/event channel pair.
    ///
    /// The caller sends `AgentOp` messages and receives `AgentEvent` messages.
    /// The loop runs until the channels close or an unrecoverable error occurs.
    pub async fn run_with_channels(
        &mut self,
        mut op_rx: mpsc::Receiver<AgentOp>,
        event_tx: mpsc::Sender<AgentEvent>,
    ) {
        // Wait for the first UserTurn to kick things off, then loop.
        loop {
            let op = match op_rx.recv().await {
                Some(op) => op,
                None => break, // channel closed
            };

            match op {
                AgentOp::UserTurn(input) => {
                    self.handle_user_turn(&input, &mut op_rx, &event_tx).await;
                }
                AgentOp::Interrupt => {
                    let _ = event_tx.send(AgentEvent::Error("Interrupted".to_string())).await;
                }
                AgentOp::ExecApproval { .. } => {
                    // Spurious approval when not awaiting — ignore.
                }
            }
        }
    }

    /// Core inner loop: process a user turn through LLM calls and tool execution.
    async fn handle_user_turn(
        &mut self,
        user_input: &str,
        op_rx: &mut mpsc::Receiver<AgentOp>,
        event_tx: &mpsc::Sender<AgentEvent>,
    ) {
        let user_msg = Message::user(user_input);
        self.persist(&user_msg).await;
        self.conversation.push(user_msg);

        let tools = self.tool_router.tool_definitions();
        let tools_option = if tools.is_empty() { None } else { Some(tools) };

        for step in 0..self.max_steps {
            // Compact conversation if approaching context limit.
            self.maybe_compact().await;

            debug!(step, "Agent loop step (channel)");
            let _ = event_tx
                .send(AgentEvent::StepProgress {
                    step,
                    state: "llm_call".to_string(),
                    model: Some(self.model.clone()),
                    tokens_in: None,
                    tokens_out: None,
                    cost_usd: None,
                })
                .await;

            // Check for interrupt before LLM call
            if let Ok(op) = op_rx.try_recv() {
                if matches!(op, AgentOp::Interrupt) {
                    let _ = event_tx.send(AgentEvent::Error("Interrupted".to_string())).await;
                    return;
                }
            }

            let messages = self.build_messages().await;
            let request = CompletionRequest {
                model: self.model.clone(),
                messages,
                tools: tools_option.clone(),
                temperature: Some(0.0),
                max_tokens: Some(4096),
            };

            // Stream the LLM response
            let mut stream = match self.provider.complete_streaming(request).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = event_tx.send(AgentEvent::Error(e.to_string())).await;
                    return;
                }
            };

            let mut full_content = String::new();
            let mut tool_accumulators: Vec<ToolCallAccumulator> = Vec::new();
            let mut finish_reason = FinishReason::Stop;
            let mut usage = Usage::default();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = event_tx.send(AgentEvent::Error(e.to_string())).await;
                        return;
                    }
                };

                if let Some(delta) = &chunk.delta_content {
                    if !delta.is_empty() {
                        full_content.push_str(delta);
                        let _ = event_tx.send(AgentEvent::Token(delta.clone())).await;
                    }
                }

                if let Some(delta_tcs) = &chunk.delta_tool_calls {
                    for tc in delta_tcs {
                        if !tc.id.is_empty() {
                            tool_accumulators.push(ToolCallAccumulator {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            });
                        } else if let Some(acc) = tool_accumulators.last_mut() {
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
                    let _ = event_tx.send(AgentEvent::TaskComplete(result)).await;
                } else {
                    let _ = event_tx
                        .send(AgentEvent::Error(format!("Budget exceeded: {e}")))
                        .await;
                }
                return;
            }

            let step_cost_usd = cost_microdollars as f64 / 1_000_000.0;
            let _ = event_tx
                .send(AgentEvent::CostUpdate(self.budget.spent_dollars()))
                .await;
            let _ = event_tx
                .send(AgentEvent::StepProgress {
                    step,
                    state: "llm_done".to_string(),
                    model: Some(self.model.clone()),
                    tokens_in: Some(usage.prompt_tokens),
                    tokens_out: Some(usage.completion_tokens),
                    cost_usd: Some(step_cost_usd),
                })
                .await;

            info!(
                finish_reason = ?finish_reason,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                cost_usd = format!("{:.6}", cost_microdollars as f64 / 1_000_000.0),
                total_spent_usd = format!("{:.6}", self.budget.spent_dollars()),
                "LLM response (channel)"
            );

            // Handle tool calls
            if finish_reason == FinishReason::ToolCalls && !tool_accumulators.is_empty() {
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

                // Process each tool call, potentially requesting approval
                for tool_call in &tool_calls {
                    let name = &tool_call.function.name;
                    let arguments: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments).unwrap_or_default();

                    if self.should_auto_approve(name) {
                        // Auto-approved: execute directly
                        // Capture old content for artifact tracking on file writes
                        let old_content = Self::capture_old_content_if_file_write(name, &arguments).await;
                        let _ = event_tx.send(AgentEvent::ToolCallStart(name.clone())).await;
                        let result = self.tool_router.dispatch(name, arguments.clone()).await;
                        Self::log_tool_result(name, &result);
                        let _ = event_tx.send(AgentEvent::ToolCallEnd(name.clone())).await;

                        // Emit artifact event for successful file writes
                        if !result.is_error {
                            Self::maybe_emit_artifact(name, &arguments, old_content, &event_tx).await;
                        }

                        let tool_msg = Message::tool_result(&tool_call.id, &result.content);
                        self.persist(&tool_msg).await;
                        self.conversation.push(tool_msg);
                    } else {
                        // Needs approval
                        let _ = event_tx
                            .send(AgentEvent::ToolApprovalRequired {
                                call_id: tool_call.id.clone(),
                                tool_name: name.clone(),
                                arguments: arguments.clone(),
                            })
                            .await;

                        // Wait for approval op
                        let approved = loop {
                            match op_rx.recv().await {
                                Some(AgentOp::ExecApproval {
                                    call_id,
                                    approved,
                                    remember_session,
                                }) if call_id == tool_call.id => {
                                    if approved && remember_session {
                                        self.session_approved_tools.insert(name.clone());
                                        // Also record in MCP permission manager
                                        if self.tool_router.is_mcp_tool(name) {
                                            if let Some(perms) = &self.mcp_permissions {
                                                if let Ok(mut pm) = perms.try_lock() {
                                                    pm.approve_for_session(name);
                                                }
                                            }
                                        }
                                    }
                                    break approved;
                                }
                                Some(AgentOp::Interrupt) => {
                                    let _ = event_tx
                                        .send(AgentEvent::Error("Interrupted".to_string()))
                                        .await;
                                    // Add denial result so conversation stays valid
                                    let tool_msg = Message::tool_result(
                                        &tool_call.id,
                                        "Tool execution interrupted by user.",
                                    );
                                    self.persist(&tool_msg).await;
                                    self.conversation.push(tool_msg);
                                    return;
                                }
                                Some(_) => continue, // ignore other ops while awaiting
                                None => return,      // channel closed
                            }
                        };

                        if approved {
                            let old_content = Self::capture_old_content_if_file_write(name, &arguments).await;
                            let _ = event_tx.send(AgentEvent::ToolCallStart(name.clone())).await;
                            let result = self.tool_router.dispatch(name, arguments.clone()).await;
                            Self::log_tool_result(name, &result);
                            let _ = event_tx.send(AgentEvent::ToolCallEnd(name.clone())).await;

                            if !result.is_error {
                                Self::maybe_emit_artifact(name, &arguments, old_content, &event_tx).await;
                            }

                            let tool_msg = Message::tool_result(&tool_call.id, &result.content);
                            self.persist(&tool_msg).await;
                            self.conversation.push(tool_msg);
                        } else {
                            let tool_msg = Message::tool_result(
                                &tool_call.id,
                                "Tool call denied by user.",
                            );
                            self.persist(&tool_msg).await;
                            self.conversation.push(tool_msg);
                        }
                    }
                }

                // Check for final REPL variables
                {
                    let repl = self.repl_session.lock().await;
                    if repl.has_final() {
                        let finals: Vec<String> = repl
                            .finals()
                            .map(|(k, v)| format!("{k}: {v}"))
                            .collect();
                        info!("REPL has final variables, terminating loop");
                        let result = finals.join("\n");
                        let _ = event_tx.send(AgentEvent::TaskComplete(result)).await;
                        return;
                    }
                }

                continue;
            }

            // No tool calls — final response
            let msg = Message::assistant(&full_content);
            self.persist(&msg).await;
            self.conversation.push(msg);
            let _ = event_tx
                .send(AgentEvent::TaskComplete(full_content))
                .await;
            return;
        }

        warn!("Agent loop hit max steps ({})", self.max_steps);
        let _ = event_tx
            .send(AgentEvent::Error(format!(
                "Agent loop exceeded maximum steps ({})",
                self.max_steps
            )))
            .await;
    }

    fn log_tool_result(name: &str, result: &crate::tool_router::ToolResult) {
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
    }

    /// If the tool is `file_write`, read the existing file content before the write.
    async fn capture_old_content_if_file_write(
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Option<String> {
        if tool_name != "file_write" {
            return None;
        }
        let path = arguments.get("path")?.as_str()?;
        tokio::fs::read_to_string(path).await.ok()
    }

    /// Emit an `ArtifactCreated` event if the tool was `file_write`.
    async fn maybe_emit_artifact(
        tool_name: &str,
        arguments: &serde_json::Value,
        old_content: Option<String>,
        event_tx: &mpsc::Sender<AgentEvent>,
    ) {
        if tool_name != "file_write" {
            return;
        }
        let file_path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let new_content = arguments
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !file_path.is_empty() {
            let _ = event_tx
                .send(AgentEvent::ArtifactCreated {
                    file_path,
                    old_content,
                    new_content,
                })
                .await;
        }
    }

    // -----------------------------------------------------------------------
    // Legacy callback-based API (backward compat)
    // -----------------------------------------------------------------------

    /// Process a user message through the agent loop (non-streaming fallback).
    /// Returns the final assistant text response.
    pub async fn process(&mut self, user_input: &str) -> Result<String, anyhow::Error> {
        let user_msg = Message::user(user_input);
        self.persist(&user_msg).await;
        self.conversation.push(user_msg);

        let tools = self.tool_router.tool_definitions();
        let tools_option = if tools.is_empty() { None } else { Some(tools) };

        for step in 0..self.max_steps {
            self.maybe_compact().await;

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
    /// This is a backward-compatible wrapper that auto-approves all tool calls.
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
            self.maybe_compact().await;

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
                        if !tc.id.is_empty() {
                            tool_accumulators.push(ToolCallAccumulator {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            });
                        } else if let Some(acc) = tool_accumulators.last_mut() {
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

    /// Process a user message using the Ripple REPL execution engine.
    /// This enables recursive sub-agent processing for complex tasks.
    pub async fn process_ripple(
        &mut self,
        user_input: &str,
        ripple_config: Option<RippleConfig>,
        sub_agent_config: Option<SubAgentConfig>,
    ) -> Result<String, anyhow::Error> {
        let user_msg = Message::user(user_input);
        self.persist(&user_msg).await;
        self.conversation.push(user_msg);

        // Load user input into REPL as $P if it's large enough
        {
            let mut repl = self.repl_session.lock().await;
            if user_input.len() > 2000 {
                repl.set("P", ripple_engine::Variable::Text(user_input.to_string()));
            }
        }

        let config = ripple_config.unwrap_or_default();
        let sub_config = sub_agent_config.unwrap_or_default();
        let executor = RippleExecutor::new(config, sub_config);

        let tool_defs = self.tool_router.tool_definitions();
        let dispatch = ToolRouterDispatch {
            router: &self.tool_router,
        };

        // Build system prompt from conversation (first message)
        let system_prompt = self
            .conversation
            .first()
            .and_then(|m| m.content.as_deref())
            .unwrap_or("You are a helpful assistant.");

        let result = executor
            .run(
                self.provider.as_ref(),
                Arc::clone(&self.repl_session),
                Arc::clone(&self.budget),
                tool_defs,
                &dispatch,
                0, // Root depth
                system_prompt,
            )
            .await?;

        let output = result
            .final_output
            .unwrap_or_else(|| "No result produced.".to_string());

        let msg = Message::assistant(&output);
        self.persist(&msg).await;
        self.conversation.push(msg);

        info!(
            steps = result.steps_taken,
            cost_usd = format!("{:.6}", result.total_cost_microdollars as f64 / 1_000_000.0),
            "Ripple execution complete"
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_state_debug() {
        let state = AgentState::Idle;
        assert!(format!("{:?}", state).contains("Idle"));

        let state = AgentState::Complete {
            result: "done".to_string(),
        };
        assert!(format!("{:?}", state).contains("Complete"));
    }

    #[test]
    fn agent_event_serialization() {
        let event = AgentEvent::Token("hello".to_string());
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("Token"));
        assert!(json.contains("hello"));

        let event = AgentEvent::ToolApprovalRequired {
            call_id: "tc_1".to_string(),
            tool_name: "file_write".to_string(),
            arguments: serde_json::json!({"path": "/tmp/test"}),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ToolApprovalRequired"));
        assert!(json.contains("file_write"));
    }

    #[test]
    fn pending_tool_call_serde() {
        let ptc = PendingToolCall {
            call_id: "tc_1".to_string(),
            tool_name: "echo".to_string(),
            arguments: serde_json::json!({"text": "hi"}),
        };
        let json = serde_json::to_string(&ptc).unwrap();
        let back: PendingToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back.call_id, "tc_1");
        assert_eq!(back.tool_name, "echo");
    }
}
