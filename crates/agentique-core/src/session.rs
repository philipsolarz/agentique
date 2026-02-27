use std::path::{Path, PathBuf};

use chrono::Utc;
use llm_provider::Message;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tracing::warn;
use uuid::Uuid;

/// A single event in the session transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub timestamp: String,
    pub event_type: SessionEventType,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventType {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    SystemMessage,
}

/// Manages session persistence to a JSONL file.
pub struct SessionStore {
    session_id: Uuid,
    path: PathBuf,
}

impl SessionStore {
    /// Create a new session store. Creates the sessions directory if needed.
    pub async fn new(base_dir: &Path) -> anyhow::Result<Self> {
        let session_id = Uuid::new_v4();
        let sessions_dir = base_dir.join("sessions");
        tokio::fs::create_dir_all(&sessions_dir).await?;

        let path = sessions_dir.join(format!("{session_id}.jsonl"));

        Ok(Self { session_id, path })
    }

    /// Resume an existing session by ID.
    pub async fn resume(base_dir: &Path, session_id: Uuid) -> anyhow::Result<Self> {
        let path = base_dir.join("sessions").join(format!("{session_id}.jsonl"));
        if !path.exists() {
            anyhow::bail!("Session file not found: {}", path.display());
        }
        Ok(Self { session_id, path })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a session event to the JSONL file.
    pub async fn append_event(&self, event: SessionEvent) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(&event)?;
        line.push('\n');

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;

        file.write_all(line.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    /// Log a message to the session transcript.
    pub async fn log_message(&self, message: &Message) {
        let event_type = match message.role {
            llm_provider::Role::User => SessionEventType::UserMessage,
            llm_provider::Role::Assistant => {
                if message.tool_calls.is_some() {
                    SessionEventType::ToolCall
                } else {
                    SessionEventType::AssistantMessage
                }
            }
            llm_provider::Role::Tool => SessionEventType::ToolResult,
            llm_provider::Role::System => SessionEventType::SystemMessage,
        };

        let data = serde_json::to_value(message).unwrap_or_default();

        let event = SessionEvent {
            timestamp: Utc::now().to_rfc3339(),
            event_type,
            data,
        };

        if let Err(e) = self.append_event(event).await {
            warn!("Failed to persist session event: {e}");
        }
    }

    /// Replay a session file, returning the conversation messages.
    pub async fn replay(&self) -> anyhow::Result<Vec<Message>> {
        let content = tokio::fs::read_to_string(&self.path).await?;
        let mut messages = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let event: SessionEvent = serde_json::from_str(line)?;
            if let Ok(msg) = serde_json::from_value::<Message>(event.data) {
                messages.push(msg);
            }
        }

        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn make_store() -> (TempDir, SessionStore) {
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path()).await.unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn creates_session_file() {
        let (_dir, store) = make_store().await;
        // Log a message to create the file
        store.log_message(&Message::user("hello")).await;
        assert!(store.path().exists());
    }

    #[tokio::test]
    async fn persist_and_replay_roundtrip() {
        let (_dir, store) = make_store().await;

        store.log_message(&Message::user("What is 2+2?")).await;
        store.log_message(&Message::assistant("4")).await;
        store
            .log_message(&Message::tool_result("tc_1", "result"))
            .await;

        let replayed = store.replay().await.unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].role, llm_provider::Role::User);
        assert_eq!(replayed[0].content.as_deref(), Some("What is 2+2?"));
        assert_eq!(replayed[1].role, llm_provider::Role::Assistant);
        assert_eq!(replayed[2].role, llm_provider::Role::Tool);
    }

    #[tokio::test]
    async fn session_event_serialization() {
        let event = SessionEvent {
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            event_type: SessionEventType::UserMessage,
            data: serde_json::json!({"role": "user", "content": "test"}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let deser: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.timestamp, "2024-01-01T00:00:00Z");
    }

    #[tokio::test]
    async fn resume_existing_session() {
        let (dir, store) = make_store().await;
        let sid = store.session_id();
        store.log_message(&Message::user("original")).await;

        let resumed = SessionStore::resume(dir.path(), sid).await.unwrap();
        let msgs = resumed.replay().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content.as_deref(), Some("original"));
    }

    #[tokio::test]
    async fn resume_nonexistent_fails() {
        let dir = TempDir::new().unwrap();
        let result = SessionStore::resume(dir.path(), Uuid::new_v4()).await;
        assert!(result.is_err());
    }
}
