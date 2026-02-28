use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use agentique_core::tools::{
    CodeSearchTool, FileReadTool, FileWriteTool, FindRelatedTool, ListFilesTool, ReplChunksTool,
    ReplGetTool, ReplLenTool, ReplLoadFileTool, ReplSearchTool, ReplSetTool, ReplSliceTool,
};
use agentique_core::{AgentEvent, AgentLoop, AgentOp, SessionStore, ToolRouter};
use llm_provider::{AnthropicProvider, CompletionProvider, OpenAiProvider, RetryProvider};
use mcp_manager::McpManager;
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use serde::Serialize;
use tauri::{ipc::Channel, Manager, State};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

const SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.

You have access to:
- File system tools (file_read, file_write, list_files) for working with files.
- Code intelligence tools:
  - code_search: Full-text search across indexed source files (content + symbol names).
  - find_related_files: Find files related to a given file via the dependency graph.
- REPL session tools for managing variables:
  - repl_set / repl_get: Store and retrieve named variables.
  - repl_load_file: Load a file into the REPL as a symbolic variable (you see metadata, not content).
  - repl_slice: Extract a character range from a variable.
  - repl_search: Regex search within a variable, returns matches with line numbers.
  - repl_chunks: Split a large variable into smaller chunks for processing.
  - repl_len: Get size and token info about a variable.

For large files or codebases, use repl_load_file to load them, then use repl_search/repl_slice/repl_chunks
to work with specific parts. This keeps large content out of the conversation context.
Use code_search to find functions, classes, or patterns across the project.
Use find_related_files to understand file dependencies before making changes.
Use 'final_' prefix on variable names to mark terminal outputs (e.g., 'final_answer').

Always explain what you're doing and present results clearly."#;

/// Info about a session, returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub model: String,
    pub budget: f64,
    pub name: String,
    pub created_at: String,
    pub mcp_servers: Vec<String>,
}

/// Events streamed back to the frontend via a Tauri Channel.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEvent {
    TokenDelta(String),
    ToolCallStart(String),
    ToolCallEnd(String),
    ToolApprovalRequired {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    StepProgress {
        step: u32,
        state: String,
        model: Option<String>,
        tokens_in: Option<u32>,
        tokens_out: Option<u32>,
        cost_usd: Option<f64>,
    },
    ArtifactCreated {
        file_path: String,
        old_content: Option<String>,
        new_content: String,
    },
    AssistantMessage(String),
    Error(String),
    CostUpdate(f64),
}

/// Holds a live agent session with its channel handles.
struct AgentSession {
    /// Send ops (UserTurn, Interrupt, ExecApproval) to the agent loop.
    op_tx: mpsc::Sender<AgentOp>,
    /// Receive events from the agent loop. Protected by mutex since only one
    /// send_message call should consume events at a time.
    event_rx: Mutex<mpsc::Receiver<AgentEvent>>,
    model: String,
    budget: f64,
    session_id: String,
    name: String,
    created_at: String,
    mcp_servers: Vec<String>,
}

/// App-level state holding all active sessions.
struct AppState {
    sessions: Mutex<HashMap<String, AgentSession>>,
    data_dir: PathBuf,
}

#[tauri::command]
async fn create_session(
    state: State<'_, AppState>,
    api_key: String,
    model: String,
    budget: f64,
) -> Result<SessionInfo, String> {
    let is_anthropic = model.starts_with("claude") || model.starts_with("anthropic/");
    let provider: Box<dyn CompletionProvider> = if is_anthropic {
        Box::new(RetryProvider::with_defaults(Box::new(
            AnthropicProvider::new(&api_key).with_model(&model),
        )))
    } else {
        Box::new(RetryProvider::with_defaults(Box::new(
            OpenAiProvider::new(&api_key).with_model(&model),
        )))
    };
    let budget_tracker = Arc::new(BudgetTracker::with_dollar_ceiling(budget));
    let repl_session = Arc::new(Mutex::new(ReplSession::new()));

    let mut router = ToolRouter::new();
    router.register(Box::new(FileReadTool));
    router.register(Box::new(FileWriteTool));
    router.register(Box::new(ListFilesTool));
    router.register(Box::new(ReplSetTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplGetTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplLoadFileTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplSliceTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplSearchTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplChunksTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplLenTool::new(Arc::clone(&repl_session))));

    // Data-layer tools: code search index + dependency graph
    let index_dir = state.data_dir.join("index");
    let index_manager = match data_layer::IndexManager::new(&index_dir) {
        Ok(idx) => {
            // Index the current working directory in the background
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            match idx.index_directory(&cwd) {
                Ok(count) => info!(count, dir = %cwd.display(), "Indexed source files"),
                Err(e) => warn!(error = %e, "Failed to index directory"),
            }
            Arc::new(Mutex::new(idx))
        }
        Err(e) => {
            warn!(error = %e, "Failed to create search index, using in-memory fallback");
            Arc::new(Mutex::new(
                data_layer::IndexManager::in_memory().expect("in-memory index"),
            ))
        }
    };

    let dep_graph = Arc::new(Mutex::new(data_layer::DependencyGraph::new()));
    router.register(Box::new(CodeSearchTool::new(Arc::clone(&index_manager))));
    router.register(Box::new(FindRelatedTool::new(Arc::clone(&dep_graph))));

    // Load MCP server configs and connect
    let global_mcp_path = state.data_dir.join("mcp.json");
    // Project-level config would be .agentique/mcp.json in the project root
    // For now we only load the global config
    let mcp_manager = McpManager::load_and_connect(
        None,
        Some(&global_mcp_path),
    )
    .await
    .unwrap_or_else(|e| {
        warn!("Failed to load MCP config: {e}");
        McpManager::new()
    });

    let mcp_servers: Vec<String> = mcp_manager
        .connected_servers()
        .iter()
        .map(|s| s.to_string())
        .collect();

    if !mcp_servers.is_empty() {
        info!(servers = ?mcp_servers, "MCP servers connected");
    }

    // Extract MCP tool definitions and permission manager before wrapping in Arc<Mutex>
    let mcp_tool_defs = mcp_manager.aggregate_tools();
    let mcp_perms = Arc::new(Mutex::new(
        mcp_manager::PermissionManager::new()
    ));

    // Copy trusted servers into the shared permission manager
    {
        let mut perms = mcp_perms.lock().await;
        for server in &mcp_servers {
            // The manager already tracked trusted servers during load_and_connect;
            // we re-use the same trust decisions by checking the original manager.
            if mcp_manager.permissions().is_approved(
                &format!("{server}:__probe__"), true, false
            ) {
                perms.trust_server(server);
            }
        }
    }

    let mcp_arc = Arc::new(Mutex::new(mcp_manager));

    // Register MCP tools into the router
    if !mcp_tool_defs.is_empty() {
        info!(tool_count = mcp_tool_defs.len(), "Registering MCP tools");
        router.register_mcp_tools(Arc::clone(&mcp_arc), mcp_tool_defs);
    }

    let session_store = SessionStore::new(&state.data_dir)
        .await
        .map_err(|e| e.to_string())?;
    let session_id = session_store.session_id().to_string();

    let agent = AgentLoop::new(
        provider,
        router,
        SYSTEM_PROMPT,
        &model,
        30,
        budget_tracker,
        repl_session,
    )
    .with_session_store(session_store)
    .with_mcp_permissions(Arc::clone(&mcp_perms));

    // Create channels for the agent loop
    let (op_tx, op_rx) = mpsc::channel::<AgentOp>(32);
    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>(256);

    let info = SessionInfo {
        session_id: session_id.clone(),
        model: model.clone(),
        budget,
        name: String::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
        mcp_servers: mcp_servers.clone(),
    };

    // Spawn the agent loop — it runs forever, waiting for ops
    tokio::spawn(async move {
        let mut agent = agent;
        agent.run_with_channels(op_rx, event_tx).await;
        // Shutdown MCP on agent loop exit
        let mut mgr = mcp_arc.lock().await;
        mgr.shutdown().await;
    });

    let session = AgentSession {
        op_tx,
        event_rx: Mutex::new(event_rx),
        model,
        budget,
        session_id: session_id.clone(),
        name: String::new(),
        created_at: info.created_at.clone(),
        mcp_servers,
    };

    state.sessions.lock().await.insert(session_id, session);

    Ok(info)
}

#[tauri::command]
async fn send_message(
    state: State<'_, AppState>,
    session_id: String,
    content: String,
    on_event: Channel<StreamEvent>,
) -> Result<(), String> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;

    // Send the user turn op to the agent loop
    session
        .op_tx
        .send(AgentOp::UserTurn(content.clone()))
        .await
        .map_err(|e| format!("Agent loop closed: {e}"))?;

    // Set session name from the first user message (truncated)
    {
        let mut sessions_lock = state.sessions.lock().await;
        if let Some(s) = sessions_lock.get_mut(&session_id) {
            if s.name.is_empty() {
                s.name = content.chars().take(60).collect::<String>();
                if content.len() > 60 {
                    s.name.push_str("...");
                }
            }
        }
    }

    // Lock the event_rx for this message — only one send_message at a time per session
    let mut event_rx = session.event_rx.lock().await;
    drop(sessions); // Release sessions lock while we wait for events

    // Forward events from the agent loop to the Tauri channel until we get
    // a terminal event (TaskComplete or Error)
    loop {
        match event_rx.recv().await {
            Some(event) => {
                let is_terminal = matches!(
                    &event,
                    AgentEvent::TaskComplete(_) | AgentEvent::Error(_)
                );

                let stream_event = match event {
                    AgentEvent::Token(t) => StreamEvent::TokenDelta(t),
                    AgentEvent::ToolCallStart(n) => StreamEvent::ToolCallStart(n),
                    AgentEvent::ToolCallEnd(n) => StreamEvent::ToolCallEnd(n),
                    AgentEvent::ToolApprovalRequired {
                        call_id,
                        tool_name,
                        arguments,
                    } => StreamEvent::ToolApprovalRequired {
                        call_id,
                        tool_name,
                        arguments,
                    },
                    AgentEvent::AssistantMessage(m) => StreamEvent::AssistantMessage(m),
                    AgentEvent::CostUpdate(c) => StreamEvent::CostUpdate(c),
                    AgentEvent::ArtifactCreated {
                        file_path,
                        old_content,
                        new_content,
                    } => StreamEvent::ArtifactCreated {
                        file_path,
                        old_content,
                        new_content,
                    },
                    AgentEvent::TaskComplete(result) => StreamEvent::AssistantMessage(result),
                    AgentEvent::Error(e) => StreamEvent::Error(e),
                    AgentEvent::StepProgress {
                        step,
                        state,
                        model,
                        tokens_in,
                        tokens_out,
                        cost_usd,
                    } => StreamEvent::StepProgress {
                        step,
                        state,
                        model,
                        tokens_in,
                        tokens_out,
                        cost_usd,
                    },
                };

                let _ = on_event.send(stream_event);

                if is_terminal {
                    break;
                }
            }
            None => {
                let _ = on_event.send(StreamEvent::Error("Agent loop ended unexpectedly".into()));
                break;
            }
        }
    }

    Ok(())
}

#[tauri::command]
async fn approve_tool_call(
    state: State<'_, AppState>,
    session_id: String,
    call_id: String,
    approved: bool,
    remember_session: bool,
) -> Result<(), String> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;

    session
        .op_tx
        .send(AgentOp::ExecApproval {
            call_id,
            approved,
            remember_session,
        })
        .await
        .map_err(|e| format!("Agent loop closed: {e}"))?;

    Ok(())
}

#[tauri::command]
async fn list_sessions(state: State<'_, AppState>) -> Result<Vec<SessionInfo>, String> {
    let sessions = state.sessions.lock().await;
    let list = sessions
        .values()
        .map(|s| SessionInfo {
            session_id: s.session_id.clone(),
            model: s.model.clone(),
            budget: s.budget,
            name: s.name.clone(),
            created_at: s.created_at.clone(),
            mcp_servers: s.mcp_servers.clone(),
        })
        .collect();
    Ok(list)
}

#[tauri::command]
async fn delete_session(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().await;
    sessions.remove(&session_id);
    // Also remove persisted JSONL file if it exists
    let session_file = state.data_dir.join("sessions").join(format!("{session_id}.jsonl"));
    if session_file.exists() {
        let _ = tokio::fs::remove_file(&session_file).await;
    }
    Ok(())
}

#[tauri::command]
async fn get_session_cost(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<f64, String> {
    // Cost is now tracked via CostUpdate events; return 0 as a fallback.
    // The frontend should track cost from CostUpdate stream events.
    Ok(0.0)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let data_dir = std::env::var("AGENTIQUE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join(".agentique")
        });

    tauri::Builder::default()
        .manage(AppState {
            sessions: Mutex::new(HashMap::new()),
            data_dir,
        })
        .invoke_handler(tauri::generate_handler![
            create_session,
            send_message,
            approve_tool_call,
            list_sessions,
            delete_session,
            get_session_cost,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Agentique");
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
