use std::path::PathBuf;
use std::sync::Arc;

use llm_provider::{AnthropicProvider, CompletionProvider, OpenAiProvider, RetryProvider};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::session::{SessionCompressor, SessionStore};
use crate::tool_router::ToolRouter;
use crate::tools::*;
use crate::AgentLoop;

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.

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

/// Result of building a session — contains the agent loop and MCP resources.
pub struct BuiltSession {
    pub agent: AgentLoop,
    pub session_id: String,
    pub mcp_servers: Vec<String>,
    /// MCP manager wrapped in Arc<Mutex> for shutdown on session end.
    pub mcp_manager: Option<Arc<Mutex<mcp_manager::McpManager>>>,
}

/// Builder for creating agent sessions with all tools and providers wired up.
pub struct SessionBuilder {
    model: String,
    api_key: String,
    budget_dollars: f64,
    data_dir: PathBuf,
    system_prompt: String,
    max_steps: u32,
    mcp_global_config: Option<PathBuf>,
    mcp_project_config: Option<PathBuf>,
    base_url: Option<String>,
}

impl SessionBuilder {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            budget_dollars: 5.0,
            data_dir: default_data_dir(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            max_steps: 30,
            mcp_global_config: None,
            mcp_project_config: None,
            base_url: None,
        }
    }

    pub fn budget(mut self, dollars: f64) -> Self {
        self.budget_dollars = dollars;
        self
    }

    pub fn data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_dir = path.into();
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn mcp_global_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.mcp_global_config = Some(path.into());
        self
    }

    pub fn mcp_project_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.mcp_project_config = Some(path.into());
        self
    }

    /// Set a custom base URL for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the session, creating provider, tools, MCP connections, and agent loop.
    pub async fn build(self) -> Result<BuiltSession, anyhow::Error> {
        // Create provider based on model name
        let provider = create_provider(&self.model, &self.api_key, self.base_url.as_deref());
        let budget_tracker = Arc::new(BudgetTracker::with_dollar_ceiling(self.budget_dollars));
        let repl_session = Arc::new(Mutex::new(ReplSession::new()));

        // Register all built-in tools
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
        let index_dir = self.data_dir.join("index");
        let index_manager = match data_layer::IndexManager::new(&index_dir) {
            Ok(idx) => {
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

        // Load MCP servers
        let global_mcp = self
            .mcp_global_config
            .unwrap_or_else(|| self.data_dir.join("mcp.json"));
        let mcp_manager = mcp_manager::McpManager::load_and_connect(
            self.mcp_project_config.as_deref(),
            Some(&global_mcp),
        )
        .await
        .unwrap_or_else(|e| {
            warn!("Failed to load MCP config: {e}");
            mcp_manager::McpManager::new()
        });

        let mcp_servers: Vec<String> = mcp_manager
            .connected_servers()
            .iter()
            .map(|s| s.to_string())
            .collect();

        if !mcp_servers.is_empty() {
            info!(servers = ?mcp_servers, "MCP servers connected");
        }

        let mcp_tool_defs = mcp_manager.aggregate_tools();
        let mcp_perms = Arc::new(Mutex::new(mcp_manager::PermissionManager::new()));

        // Sync trusted servers
        {
            let mut perms = mcp_perms.lock().await;
            for server in &mcp_servers {
                if mcp_manager
                    .permissions()
                    .is_approved(&format!("{server}:__probe__"), true, false)
                {
                    perms.trust_server(server);
                }
            }
        }

        let mcp_arc = Arc::new(Mutex::new(mcp_manager));

        if !mcp_tool_defs.is_empty() {
            info!(tool_count = mcp_tool_defs.len(), "Registering MCP tools");
            router.register_mcp_tools(Arc::clone(&mcp_arc), mcp_tool_defs);
        }

        // Session store
        let session_store = SessionStore::new(&self.data_dir).await?;
        let session_id = session_store.session_id().to_string();

        // Build agent loop
        let agent = AgentLoop::new(
            provider,
            router,
            &self.system_prompt,
            &self.model,
            self.max_steps,
            budget_tracker,
            repl_session,
        )
        .with_session_store(session_store)
        .with_compressor(SessionCompressor::new(&self.model))
        .with_mcp_permissions(Arc::clone(&mcp_perms));

        Ok(BuiltSession {
            agent,
            session_id,
            mcp_servers,
            mcp_manager: Some(mcp_arc),
        })
    }

    /// Resume a persisted session, replaying its conversation history.
    pub async fn resume(self, session_id: &str) -> Result<BuiltSession, anyhow::Error> {
        let sid = uuid::Uuid::parse_str(session_id)?;
        let data_dir = self.data_dir.clone();
        let store = SessionStore::resume(&data_dir, sid).await?;
        let conversation = store.replay().await?;

        info!(
            session_id = %session_id,
            messages = conversation.len(),
            "Resuming session"
        );

        let mut built = self.build().await?;

        // Replace with the resumed session store and inject history
        built.session_id = session_id.to_string();
        let resumed_store = SessionStore::resume(&data_dir, sid).await?;
        built.agent = built.agent
            .with_conversation(conversation)
            .with_session_store(resumed_store);

        Ok(built)
    }

    /// Fork an existing session — create a new session with a copy of the history.
    pub async fn fork(self, source_session_id: &str) -> Result<BuiltSession, anyhow::Error> {
        let sid = uuid::Uuid::parse_str(source_session_id)?;
        let data_dir = self.data_dir.clone();
        let source_store = SessionStore::resume(&data_dir, sid).await?;
        let conversation = source_store.replay().await?;

        // Create new session store with forked content
        let forked_store = source_store.fork(&data_dir).await?;
        let new_session_id = forked_store.session_id().to_string();

        info!(
            source = %source_session_id,
            forked = %new_session_id,
            messages = conversation.len(),
            "Forked session"
        );

        let mut built = self.build().await?;
        built.session_id = new_session_id;
        built.agent = built.agent
            .with_conversation(conversation)
            .with_session_store(forked_store);

        Ok(built)
    }
}

/// Create a provider based on model name prefix.
/// If `base_url` is provided, uses it as a custom OpenAI-compatible endpoint.
pub fn create_provider(model: &str, api_key: &str, base_url: Option<&str>) -> Box<dyn CompletionProvider> {
    let is_anthropic = model.starts_with("claude") || model.starts_with("anthropic/");
    if is_anthropic && base_url.is_none() {
        Box::new(RetryProvider::with_defaults(Box::new(
            AnthropicProvider::new(api_key).with_model(model),
        )))
    } else {
        let mut provider = OpenAiProvider::new(api_key).with_model(model);
        if let Some(url) = base_url {
            provider = provider.with_base_url(url);
        }
        Box::new(RetryProvider::with_defaults(Box::new(provider)))
    }
}

fn default_data_dir() -> PathBuf {
    std::env::var("AGENTIQUE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".agentique"))
                .unwrap_or_else(|_| PathBuf::from(".agentique"))
        })
}
