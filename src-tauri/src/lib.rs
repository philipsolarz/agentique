use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use agentique_core::tools::{
    FileReadTool, FileWriteTool, ListFilesTool, ReplChunksTool, ReplGetTool, ReplLenTool,
    ReplLoadFileTool, ReplSearchTool, ReplSetTool, ReplSliceTool,
};
use agentique_core::{AgentLoop, AgentStreamEvent, SessionStore, ToolRouter};
use llm_provider::{OpenAiProvider, RetryProvider};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use serde::Serialize;
use tauri::{ipc::Channel, Manager, State};
use tokio::sync::Mutex;

const SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.

You have access to:
- File system tools (file_read, file_write, list_files) for working with files.
- REPL session tools for managing variables:
  - repl_set / repl_get: Store and retrieve named variables.
  - repl_load_file: Load a file into the REPL as a symbolic variable (you see metadata, not content).
  - repl_slice: Extract a character range from a variable.
  - repl_search: Regex search within a variable, returns matches with line numbers.
  - repl_chunks: Split a large variable into smaller chunks for processing.
  - repl_len: Get size and token info about a variable.

For large files or codebases, use repl_load_file to load them, then use repl_search/repl_slice/repl_chunks
to work with specific parts. This keeps large content out of the conversation context.
Use 'final_' prefix on variable names to mark terminal outputs (e.g., 'final_answer').

Always explain what you're doing and present results clearly."#;

/// Info about a session, returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub model: String,
    pub budget: f64,
}

/// Events streamed back to the frontend via a Tauri Channel.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEvent {
    TokenDelta(String),
    ToolCallStart(String),
    ToolCallEnd(String),
    AssistantMessage(String),
    Error(String),
    CostUpdate(f64),
}

/// Holds a live agent session.
struct AgentSession {
    agent: AgentLoop,
    model: String,
    budget: f64,
    session_id: String,
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
    let openai = OpenAiProvider::new(api_key).with_model(&model);
    let provider = RetryProvider::with_defaults(Box::new(openai));
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

    let session_store = SessionStore::new(&state.data_dir)
        .await
        .map_err(|e| e.to_string())?;
    let session_id = session_store.session_id().to_string();

    let agent = AgentLoop::new(
        Box::new(provider),
        router,
        SYSTEM_PROMPT,
        &model,
        30,
        budget_tracker,
        repl_session,
    )
    .with_session_store(session_store);

    let info = SessionInfo {
        session_id: session_id.clone(),
        model: model.clone(),
        budget,
    };

    let session = AgentSession {
        agent,
        model,
        budget,
        session_id: session_id.clone(),
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
    let mut sessions = state.sessions.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;

    let on_event_clone = on_event.clone();
    match session
        .agent
        .process_streaming(&content, move |event| match &event {
            AgentStreamEvent::Token(token) => {
                let _ = on_event_clone.send(StreamEvent::TokenDelta(token.clone()));
            }
            AgentStreamEvent::ToolCallStart(name) => {
                let _ = on_event_clone.send(StreamEvent::ToolCallStart(name.clone()));
            }
            AgentStreamEvent::ToolCallEnd(name) => {
                let _ = on_event_clone.send(StreamEvent::ToolCallEnd(name.clone()));
            }
            AgentStreamEvent::Done(_) => {}
        })
        .await
    {
        Ok(response) => {
            on_event
                .send(StreamEvent::AssistantMessage(response))
                .map_err(|e| e.to_string())?;
            on_event
                .send(StreamEvent::CostUpdate(session.agent.spent_dollars()))
                .map_err(|e| e.to_string())?;
        }
        Err(e) => {
            on_event
                .send(StreamEvent::Error(e.to_string()))
                .map_err(|e| e.to_string())?;
        }
    }

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
        })
        .collect();
    Ok(list)
}

#[tauri::command]
async fn get_session_cost(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<f64, String> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;
    Ok(session.agent.spent_dollars())
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
            list_sessions,
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
