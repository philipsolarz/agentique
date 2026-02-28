use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use agentique_core::{AgentEvent, AgentOp, PersistedSessionInfo, SessionBuilder, SessionStore};
use agentique_core::session::{SessionEvent, SessionEventType};
use serde::{Deserialize, Serialize};
use tauri::{ipc::Channel, State};
use tokio::sync::{mpsc, Mutex};
use tracing::info;

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
    ModelSwitched(String),
    Error(String),
    CostUpdate(f64),
}

/// Holds a live agent session with its channel handles.
struct AgentSession {
    op_tx: mpsc::Sender<AgentOp>,
    event_rx: Arc<Mutex<mpsc::Receiver<AgentEvent>>>,
    model: String,
    budget: f64,
    session_id: String,
    name: String,
    created_at: String,
    mcp_servers: Vec<String>,
    system_prompt: String,
    api_key: String,
    base_url: Option<String>,
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
    base_url: Option<String>,
) -> Result<SessionInfo, String> {
    let mut builder = SessionBuilder::new(&model, &api_key)
        .budget(budget)
        .data_dir(&state.data_dir)
        .mcp_global_config(state.data_dir.join("mcp.json"));
    if let Some(url) = &base_url {
        builder = builder.base_url(url);
    }
    let built = builder.build()
        .await
        .map_err(|e| e.to_string())?;

    let session_id = built.session_id.clone();
    let mcp_servers = built.mcp_servers.clone();
    let mcp_arc = built.mcp_manager;

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

    // Spawn the agent loop
    tokio::spawn(async move {
        let mut agent = built.agent;
        agent.run_with_channels(op_rx, event_tx).await;
        if let Some(mcp) = mcp_arc {
            let mut mgr = mcp.lock().await;
            mgr.shutdown().await;
        }
    });

    let session = AgentSession {
        op_tx,
        event_rx: Arc::new(Mutex::new(event_rx)),
        model,
        budget,
        session_id: session_id.clone(),
        name: String::new(),
        created_at: info.created_at.clone(),
        mcp_servers,
        system_prompt: String::new(),
        api_key: api_key.clone(),
        base_url,
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
    let (op_tx, event_rx_mutex) = {
        let sessions = state.sessions.lock().await;
        let session = sessions
            .get(&session_id)
            .ok_or_else(|| format!("Session not found: {session_id}"))?;
        (session.op_tx.clone(), session.event_rx.clone())
    };

    op_tx
        .send(AgentOp::UserTurn(content.clone()))
        .await
        .map_err(|e| format!("Agent loop closed: {e}"))?;

    // Set session name from the first user message
    {
        let mut sessions = state.sessions.lock().await;
        if let Some(s) = sessions.get_mut(&session_id) {
            if s.name.is_empty() {
                s.name = content.chars().take(60).collect::<String>();
                if content.len() > 60 {
                    s.name.push_str("...");
                }
            }
        }
    }

    let mut event_rx = event_rx_mutex.lock().await;

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
                    AgentEvent::ModelSwitched(m) => StreamEvent::ModelSwitched(m),
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
async fn resume_session(
    state: State<'_, AppState>,
    api_key: String,
    model: String,
    budget: f64,
    session_id: String,
) -> Result<SessionInfo, String> {
    let built = SessionBuilder::new(&model, &api_key)
        .budget(budget)
        .data_dir(&state.data_dir)
        .mcp_global_config(state.data_dir.join("mcp.json"))
        .resume(&session_id)
        .await
        .map_err(|e| e.to_string())?;

    let sid = built.session_id.clone();
    let mcp_servers = built.mcp_servers.clone();
    let mcp_arc = built.mcp_manager;

    let (op_tx, op_rx) = mpsc::channel::<AgentOp>(32);
    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>(256);

    let info = SessionInfo {
        session_id: sid.clone(),
        model: model.clone(),
        budget,
        name: String::from("[resumed]"),
        created_at: chrono::Utc::now().to_rfc3339(),
        mcp_servers: mcp_servers.clone(),
    };

    tokio::spawn(async move {
        let mut agent = built.agent;
        agent.run_with_channels(op_rx, event_tx).await;
        if let Some(mcp) = mcp_arc {
            let mut mgr = mcp.lock().await;
            mgr.shutdown().await;
        }
    });

    let session = AgentSession {
        op_tx,
        event_rx: Arc::new(Mutex::new(event_rx)),
        model,
        budget,
        session_id: sid.clone(),
        name: String::from("[resumed]"),
        created_at: info.created_at.clone(),
        mcp_servers,
        system_prompt: String::new(),
        api_key,
        base_url: None,
    };

    state.sessions.lock().await.insert(sid, session);

    Ok(info)
}

#[tauri::command]
async fn fork_session(
    state: State<'_, AppState>,
    api_key: String,
    model: String,
    budget: f64,
    source_session_id: String,
) -> Result<SessionInfo, String> {
    let built = SessionBuilder::new(&model, &api_key)
        .budget(budget)
        .data_dir(&state.data_dir)
        .mcp_global_config(state.data_dir.join("mcp.json"))
        .fork(&source_session_id)
        .await
        .map_err(|e| e.to_string())?;

    let sid = built.session_id.clone();
    let mcp_servers = built.mcp_servers.clone();
    let mcp_arc = built.mcp_manager;

    let (op_tx, op_rx) = mpsc::channel::<AgentOp>(32);
    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>(256);

    let info = SessionInfo {
        session_id: sid.clone(),
        model: model.clone(),
        budget,
        name: String::from("[forked]"),
        created_at: chrono::Utc::now().to_rfc3339(),
        mcp_servers: mcp_servers.clone(),
    };

    tokio::spawn(async move {
        let mut agent = built.agent;
        agent.run_with_channels(op_rx, event_tx).await;
        if let Some(mcp) = mcp_arc {
            let mut mgr = mcp.lock().await;
            mgr.shutdown().await;
        }
    });

    let session = AgentSession {
        op_tx,
        event_rx: Arc::new(Mutex::new(event_rx)),
        model,
        budget,
        session_id: sid.clone(),
        name: String::from("[forked]"),
        created_at: info.created_at.clone(),
        mcp_servers,
        system_prompt: String::new(),
        api_key,
        base_url: None,
    };

    state.sessions.lock().await.insert(sid, session);

    Ok(info)
}

#[tauri::command]
async fn list_persisted_sessions(
    state: State<'_, AppState>,
) -> Result<Vec<PersistedSessionInfo>, String> {
    SessionStore::list_persisted(&state.data_dir)
        .await
        .map_err(|e| e.to_string())
}

/// Persisted user settings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub budget: Option<f64>,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[tauri::command]
async fn load_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    let path = state.data_dir.join("config.json");
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let data = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_settings(
    state: State<'_, AppState>,
    settings: AppSettings,
) -> Result<(), String> {
    tokio::fs::create_dir_all(&state.data_dir)
        .await
        .map_err(|e| e.to_string())?;
    let path = state.data_dir.join("config.json");
    let data = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    tokio::fs::write(&path, data)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_session(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().await;
    sessions.remove(&session_id);
    let session_file = state.data_dir.join("sessions").join(format!("{session_id}.jsonl"));
    if session_file.exists() {
        let _ = tokio::fs::remove_file(&session_file).await;
    }
    Ok(())
}

#[tauri::command]
async fn update_system_prompt(
    state: State<'_, AppState>,
    session_id: String,
    prompt: String,
) -> Result<(), String> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;

    session
        .op_tx
        .send(AgentOp::UpdateSystemPrompt(prompt))
        .await
        .map_err(|e| format!("Agent loop closed: {e}"))?;

    Ok(())
}

#[tauri::command]
async fn get_system_prompt(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<String, String> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;
    Ok(session.system_prompt.clone())
}

#[derive(Debug, Clone, Serialize)]
struct PromptTemplate {
    name: String,
    content: String,
}

#[tauri::command]
async fn list_prompt_templates() -> Result<Vec<PromptTemplate>, String> {
    Ok(agentique_core::list_prompt_templates()
        .into_iter()
        .map(|(name, content)| PromptTemplate {
            name: name.to_string(),
            content: content.to_string(),
        })
        .collect())
}

#[tauri::command]
async fn switch_model(
    state: State<'_, AppState>,
    session_id: String,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| format!("Session not found: {session_id}"))?;

    let key = api_key.unwrap_or_else(|| session.api_key.clone());
    let url = base_url.or_else(|| session.base_url.clone());

    session
        .op_tx
        .send(AgentOp::SwitchModel {
            model: model.clone(),
            api_key: key,
            base_url: url,
        })
        .await
        .map_err(|e| format!("Agent loop closed: {e}"))?;

    session.model = model;

    Ok(())
}

#[tauri::command]
async fn export_conversation(
    state: State<'_, AppState>,
    session_id: String,
    format: String,
) -> Result<String, String> {
    let sid = uuid::Uuid::parse_str(&session_id).map_err(|e| e.to_string())?;
    let store = SessionStore::resume(&state.data_dir, sid)
        .await
        .map_err(|e| e.to_string())?;
    let events = store.read_events().await.map_err(|e| e.to_string())?;

    match format.as_str() {
        "json" => serde_json::to_string_pretty(&events).map_err(|e| e.to_string()),
        "markdown" | _ => Ok(events_to_markdown(&events)),
    }
}

fn events_to_markdown(events: &[SessionEvent]) -> String {
    let mut md = String::from("# Agentique Conversation\n\n");

    for event in events {
        let time = &event.timestamp;
        match &event.event_type {
            SessionEventType::UserMessage => {
                let content = event.data.get("content").and_then(|v| v.as_str()).unwrap_or("");
                md.push_str(&format!("## User\n*{time}*\n\n{content}\n\n"));
            }
            SessionEventType::AssistantMessage => {
                let content = event.data.get("content").and_then(|v| v.as_str()).unwrap_or("");
                md.push_str(&format!("## Assistant\n*{time}*\n\n{content}\n\n"));
            }
            SessionEventType::ToolCall => {
                if let Some(tool_calls) = event.data.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tool_calls {
                        let name = tc.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("unknown");
                        let args = tc.get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}");
                        md.push_str(&format!(
                            "<details>\n<summary>Tool call: {name}</summary>\n\n```json\n{args}\n```\n\n</details>\n\n"
                        ));
                    }
                }
            }
            SessionEventType::ToolResult => {
                let content = event.data.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let preview: String = content.chars().take(500).collect();
                md.push_str(&format!(
                    "<details>\n<summary>Tool result</summary>\n\n```\n{preview}\n```\n\n</details>\n\n"
                ));
            }
            SessionEventType::SystemMessage => {
                let content = event.data.get("content").and_then(|v| v.as_str()).unwrap_or("");
                md.push_str(&format!("---\n*System: {content}*\n\n"));
            }
            SessionEventType::Compaction => {
                md.push_str("---\n*[Conversation compacted]*\n\n");
            }
        }
    }

    md
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
            resume_session,
            fork_session,
            send_message,
            approve_tool_call,
            list_sessions,
            list_persisted_sessions,
            delete_session,
            load_settings,
            save_settings,
            export_conversation,
            update_system_prompt,
            get_system_prompt,
            list_prompt_templates,
            switch_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Agentique");
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
