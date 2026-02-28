use std::path::{Path, PathBuf};

use chrono::Utc;
use llm_provider::{CompletionProvider, CompletionRequest, Message};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};
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
    Compaction,
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

// ---------------------------------------------------------------------------
// Session compressor
// ---------------------------------------------------------------------------

/// Rough token estimator: ~4 characters per token on average.
fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32 + 3) / 4
}

/// Estimate token count for a single message.
fn estimate_message_tokens(msg: &Message) -> u32 {
    let mut tokens = 0u32;
    if let Some(content) = &msg.content {
        tokens += estimate_tokens(content);
    }
    if let Some(tool_calls) = &msg.tool_calls {
        for tc in tool_calls {
            tokens += estimate_tokens(&tc.function.name);
            tokens += estimate_tokens(&tc.function.arguments);
        }
    }
    // Overhead for role, formatting
    tokens + 4
}

/// Estimate total token count for a conversation.
pub fn estimate_conversation_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(|m| estimate_message_tokens(m)).sum()
}

/// Result of a compaction operation.
pub struct CompactionResult {
    /// The new (compressed) conversation.
    pub messages: Vec<Message>,
    /// Number of tokens before compaction.
    pub original_tokens: u32,
    /// Number of tokens after compaction.
    pub compacted_tokens: u32,
    /// Number of messages that were summarized.
    pub messages_summarized: usize,
}

/// Compresses long conversations by summarizing older messages.
///
/// Strategy:
/// 1. Keep the system message (index 0) verbatim
/// 2. Keep the last `keep_recent_exchanges` user+assistant pairs verbatim
/// 3. Summarize everything in between with a cheap model call
/// 4. Replace summarized messages with a single system message
pub struct SessionCompressor {
    /// Fraction of max_context_tokens at which compaction triggers.
    pub threshold: f32,
    /// Number of recent exchanges (user+assistant pairs) to keep verbatim.
    pub keep_recent_exchanges: usize,
    /// Model to use for the summary call.
    pub summary_model: String,
}

impl Default for SessionCompressor {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            keep_recent_exchanges: 4,
            summary_model: "gpt-4o-mini".to_string(),
        }
    }
}

impl SessionCompressor {
    /// Check whether the conversation needs compaction.
    pub fn needs_compaction(&self, conversation: &[Message], max_context_tokens: u32) -> bool {
        let current = estimate_conversation_tokens(conversation);
        let limit = (max_context_tokens as f32 * self.threshold) as u32;
        current >= limit
    }

    /// Compact the conversation, summarizing older messages via the provider.
    ///
    /// Returns `None` if compaction isn't needed or the conversation is too short.
    pub async fn compact(
        &self,
        conversation: &[Message],
        provider: &dyn CompletionProvider,
    ) -> Option<CompactionResult> {
        if conversation.is_empty() {
            return None;
        }

        let original_tokens = estimate_conversation_tokens(conversation);

        // Find the split point: keep system message + last N exchanges.
        // An "exchange" is a user message + its subsequent non-user messages.
        let system_msg = if conversation[0].role == llm_provider::Role::System {
            Some(&conversation[0])
        } else {
            None
        };

        let non_system_start = if system_msg.is_some() { 1 } else { 0 };
        let non_system = &conversation[non_system_start..];

        // Count exchanges from the end (each starts with a User message).
        let mut keep_from = non_system.len();
        let mut exchanges_found = 0;

        for i in (0..non_system.len()).rev() {
            if non_system[i].role == llm_provider::Role::User {
                exchanges_found += 1;
                if exchanges_found >= self.keep_recent_exchanges {
                    keep_from = i;
                    break;
                }
            }
        }

        // If the entire conversation fits within the recent window, skip compaction.
        if exchanges_found < self.keep_recent_exchanges {
            return None;
        }

        // The messages to summarize (between system msg and the kept tail).
        let to_summarize = &non_system[..keep_from];
        let to_keep = &non_system[keep_from..];

        // Not enough messages to be worth summarizing.
        if to_summarize.len() < 4 {
            return None;
        }

        // Build the summarization prompt.
        let mut context = String::new();
        for msg in to_summarize {
            let role_str = match msg.role {
                llm_provider::Role::User => "User",
                llm_provider::Role::Assistant => "Assistant",
                llm_provider::Role::Tool => "Tool",
                llm_provider::Role::System => "System",
            };
            if let Some(content) = &msg.content {
                context.push_str(&format!("{role_str}: {content}\n\n"));
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    context.push_str(&format!(
                        "{role_str} called tool `{}` with: {}\n\n",
                        tc.function.name, tc.function.arguments
                    ));
                }
            }
        }

        let summary_prompt = format!(
            "Summarize the following conversation history concisely. Preserve:\n\
             - All decisions made and their rationale\n\
             - Tool results and their outcomes\n\
             - REPL variable names (e.g., $P, $result, $FINAL) and their purpose\n\
             - Any file paths or code references mentioned\n\
             \n\
             Conversation:\n{context}\n\
             Provide a concise summary (aim for ~25% of original length):"
        );

        let request = CompletionRequest {
            model: self.summary_model.clone(),
            messages: vec![Message::user(&summary_prompt)],
            tools: None,
            temperature: Some(0.0),
            max_tokens: Some(2048),
        };

        let summary_text = match provider.complete(request).await {
            Ok(response) => response
                .message
                .content
                .unwrap_or_else(|| "[Summary unavailable]".to_string()),
            Err(e) => {
                warn!("Failed to generate compaction summary: {e}");
                // Fall back to a mechanical summary.
                let msg_count = to_summarize.len();
                format!(
                    "[Compacted {msg_count} messages. Key context may be missing. \
                     Last topic before compaction: {}]",
                    to_summarize
                        .last()
                        .and_then(|m| m.content.as_deref())
                        .unwrap_or("unknown")
                )
            }
        };

        // Build the new conversation.
        let mut new_conversation = Vec::new();

        // System message first.
        if let Some(sys) = system_msg {
            new_conversation.push(sys.clone());
        }

        // Summary as a system message.
        new_conversation.push(Message::system(format!(
            "[Conversation summary — {n} earlier messages compacted]\n\n{summary_text}",
            n = to_summarize.len()
        )));

        // Kept recent messages.
        new_conversation.extend_from_slice(to_keep);

        let compacted_tokens = estimate_conversation_tokens(&new_conversation);
        let messages_summarized = to_summarize.len();

        info!(
            original_tokens,
            compacted_tokens,
            messages_summarized,
            "Session compacted"
        );

        Some(CompactionResult {
            messages: new_conversation,
            original_tokens,
            compacted_tokens,
            messages_summarized,
        })
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

    // ---- Compressor tests ----

    #[test]
    fn token_estimation() {
        // ~4 chars per token, rounds down
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars → (11+3)/4 = 3
        assert_eq!(estimate_tokens(""), 0); // empty = 0 tokens
        assert_eq!(estimate_tokens("a"), 1); // (1+3)/4 = 1
        assert_eq!(estimate_tokens("abcdefgh"), 2); // (8+3)/4 = 2
    }

    #[test]
    fn conversation_token_estimation() {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];
        let tokens = estimate_conversation_tokens(&messages);
        // Each message adds content tokens + 4 overhead
        assert!(tokens > 10, "expected >10 tokens, got {tokens}");
    }

    #[test]
    fn needs_compaction_when_over_threshold() {
        let compressor = SessionCompressor {
            threshold: 0.85,
            keep_recent_exchanges: 4,
            summary_model: "test".to_string(),
        };

        // Small conversation — should not need compaction.
        let small = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        assert!(!compressor.needs_compaction(&small, 200_000));

        // Build a large conversation that exceeds 85% of 1000 tokens.
        let mut large = vec![Message::system("sys")];
        for i in 0..50 {
            large.push(Message::user(&format!("User message number {i} with some extra padding text to inflate token count a little more")));
            large.push(Message::assistant(&format!("Assistant response number {i} with some extra padding text to inflate token count")));
        }
        let tokens = estimate_conversation_tokens(&large);
        // Set max_context so that 85% of it equals tokens (so it triggers).
        // tokens >= 0.85 * max → max <= tokens / 0.85
        let max_context = (tokens as f32 / 0.85) as u32;
        assert!(compressor.needs_compaction(&large, max_context));
    }

    #[test]
    fn compaction_skips_short_conversations() {
        let compressor = SessionCompressor::default();
        let rt = tokio::runtime::Runtime::new().unwrap();

        // A conversation with only a system message + 2 exchanges (4 non-system messages).
        // That's fewer than 4 messages to summarize, so compaction should return None.
        let messages = vec![
            Message::system("sys"),
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::assistant("a2"),
        ];

        // We need a mock provider. Since we expect None (too short), it won't be called.
        // Use a simple provider that panics if called.
        struct PanicProvider;
        #[async_trait::async_trait]
        impl CompletionProvider for PanicProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<llm_provider::CompletionResponse, llm_provider::ProviderError> {
                panic!("Should not be called")
            }
            async fn complete_streaming(
                &self,
                _request: CompletionRequest,
            ) -> Result<
                futures::stream::BoxStream<
                    'static,
                    Result<llm_provider::StreamChunk, llm_provider::ProviderError>,
                >,
                llm_provider::ProviderError,
            > {
                panic!("Should not be called")
            }
            fn capabilities(&self) -> llm_provider::ProviderCapabilities {
                llm_provider::ProviderCapabilities {
                    supports_streaming: false,
                    supports_tool_calling: false,
                    max_context_tokens: 100_000,
                    cost_per_input_token: 0.0,
                    cost_per_output_token: 0.0,
                }
            }
            fn name(&self) -> &str {
                "panic"
            }
        }

        let provider = PanicProvider;
        let result = rt.block_on(compressor.compact(&messages, &provider));
        assert!(result.is_none(), "Short conversation should not be compacted");
    }

    #[test]
    fn compaction_preserves_system_and_recent() {
        let compressor = SessionCompressor {
            threshold: 0.85,
            keep_recent_exchanges: 2,
            summary_model: "test".to_string(),
        };
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Build a conversation: system + 10 exchanges + last 2 to keep.
        let mut messages = vec![Message::system("You are helpful.")];
        for i in 0..10 {
            messages.push(Message::user(&format!("Question {i}")));
            messages.push(Message::assistant(&format!("Answer {i}")));
        }

        // Mock provider that returns a short summary.
        struct MockProvider;
        #[async_trait::async_trait]
        impl CompletionProvider for MockProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<llm_provider::CompletionResponse, llm_provider::ProviderError> {
                Ok(llm_provider::CompletionResponse {
                    message: Message::assistant("Summary of previous discussion."),
                    finish_reason: llm_provider::FinishReason::Stop,
                    usage: llm_provider::Usage::default(),
                    model: "mock".to_string(),
                })
            }
            async fn complete_streaming(
                &self,
                _request: CompletionRequest,
            ) -> Result<
                futures::stream::BoxStream<
                    'static,
                    Result<llm_provider::StreamChunk, llm_provider::ProviderError>,
                >,
                llm_provider::ProviderError,
            > {
                panic!("Not used")
            }
            fn capabilities(&self) -> llm_provider::ProviderCapabilities {
                llm_provider::ProviderCapabilities {
                    supports_streaming: false,
                    supports_tool_calling: false,
                    max_context_tokens: 100_000,
                    cost_per_input_token: 0.0,
                    cost_per_output_token: 0.0,
                }
            }
            fn name(&self) -> &str {
                "mock"
            }
        }

        let result = rt
            .block_on(compressor.compact(&messages, &MockProvider))
            .expect("Should compact");

        // Should have: system + summary + last 2 exchanges (4 messages) = 6
        assert_eq!(result.messages.len(), 6);
        assert_eq!(result.messages[0].role, llm_provider::Role::System);
        assert_eq!(
            result.messages[0].content.as_deref(),
            Some("You are helpful.")
        );
        // Second message is the summary.
        assert_eq!(result.messages[1].role, llm_provider::Role::System);
        assert!(result.messages[1]
            .content
            .as_deref()
            .unwrap()
            .contains("compacted"));
        // Last 4 messages should be the last 2 exchanges.
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("Question 8")
        );
        assert_eq!(
            result.messages[3].content.as_deref(),
            Some("Answer 8")
        );
        assert_eq!(
            result.messages[4].content.as_deref(),
            Some("Question 9")
        );
        assert_eq!(
            result.messages[5].content.as_deref(),
            Some("Answer 9")
        );

        assert!(result.compacted_tokens < result.original_tokens);
        assert_eq!(result.messages_summarized, 16); // 8 exchanges = 16 messages
    }
}
