use std::collections::HashMap;
use std::path::Path;

use rmcp::model::{CallToolRequestParams, Tool as McpTool};
use rmcp::service::RunningService;
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt};
use tokio::process::Command;
use tracing::{error, info, warn};

use crate::config::{McpConfigFile, McpServerConfig, TransportConfig};
use crate::permissions::PermissionManager;

/// A connected MCP server session.
struct McpSession {
    service: RunningService<RoleClient, ()>,
    config: McpServerConfig,
    /// Cached tool list from the server.
    cached_tools: Vec<McpTool>,
}

/// Manages multiple MCP server connections with tool namespacing and permission management.
pub struct McpManager {
    sessions: HashMap<String, McpSession>,
    permissions: PermissionManager,
}

impl McpManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            permissions: PermissionManager::new(),
        }
    }

    /// Load server configs from project and global config files, then connect.
    pub async fn load_and_connect(
        project_config: Option<&Path>,
        global_config: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let mut merged = McpConfigFile::default();

        if let Some(path) = global_config {
            let global = McpConfigFile::load(path).await?;
            merged.merge(global);
        }

        if let Some(path) = project_config {
            let project = McpConfigFile::load(path).await?;
            merged.merge(project);
        }

        let mut manager = Self::new();

        for server_config in merged.servers {
            if server_config.trusted {
                manager.permissions.trust_server(&server_config.name);
            }
            if let Err(e) = manager.connect_server(server_config).await {
                warn!(error = %e, "Failed to connect MCP server, skipping");
            }
        }

        Ok(manager)
    }

    /// Connect to a single MCP server.
    pub async fn connect_server(&mut self, config: McpServerConfig) -> anyhow::Result<()> {
        let name = config.name.clone();
        info!(server = %name, "Connecting to MCP server");

        match &config.transport {
            TransportConfig::Stdio { command, args } => {
                let mut cmd = Command::new(command);
                cmd.args(args);
                for (k, v) in &config.env {
                    cmd.env(k, v);
                }

                let transport = TokioChildProcess::new(cmd)?;
                let service = ().serve(transport).await?;

                // Cache tools
                let tools_response = service.list_tools(Default::default()).await?;
                let cached_tools = tools_response.tools;

                info!(
                    server = %name,
                    tool_count = cached_tools.len(),
                    "MCP server connected"
                );

                self.sessions.insert(
                    name,
                    McpSession {
                        service,
                        config,
                        cached_tools,
                    },
                );
            }
            TransportConfig::Http { url: _ } => {
                warn!(server = %name, "HTTP transport not yet implemented, skipping");
                return Err(anyhow::anyhow!(
                    "HTTP transport not yet implemented for server '{name}'"
                ));
            }
        }

        Ok(())
    }

    /// Get aggregated tool definitions from all connected servers, with namespace prefixes.
    pub fn aggregate_tools(&self) -> Vec<llm_provider::ToolDefinition> {
        let mut tools = Vec::new();

        for (server_name, session) in &self.sessions {
            for tool in &session.cached_tools {
                let namespaced_name = format!("{server_name}:{}", tool.name);
                let description = tool
                    .description
                    .as_deref()
                    .unwrap_or("No description")
                    .to_string();

                // Convert JsonObject (Map<String, Value>) to Value
                let parameters =
                    serde_json::Value::Object(tool.input_schema.as_ref().clone());

                tools.push(llm_provider::ToolDefinition {
                    tool_type: "function".to_string(),
                    function: llm_provider::FunctionDefinition {
                        name: namespaced_name,
                        description,
                        parameters,
                    },
                });
            }
        }

        tools
    }

    /// Dispatch a namespaced tool call (e.g., "github:create_issue") to the correct server.
    pub async fn dispatch(
        &self,
        namespaced_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let (server_name, tool_name) = namespaced_name
            .split_once(':')
            .ok_or_else(|| format!("Invalid namespaced tool name: {namespaced_name}"))?;

        let session = self
            .sessions
            .get(server_name)
            .ok_or_else(|| format!("MCP server not connected: {server_name}"))?;

        let params = CallToolRequestParams {
            meta: None,
            name: tool_name.to_string().into(),
            arguments: if arguments.is_object() {
                Some(arguments.as_object().unwrap().clone())
            } else {
                None
            },
            task: None,
        };

        match session.service.call_tool(params).await {
            Ok(result) => {
                let content: Vec<String> = result
                    .content
                    .iter()
                    .map(|c| {
                        // Extract actual text content instead of debug-printing
                        if let Some(text) = c.as_text() {
                            text.text.clone()
                        } else {
                            // For non-text content (images, resources), use debug as fallback
                            format!("{:?}", c)
                        }
                    })
                    .collect();
                Ok(content.join("\n"))
            }
            Err(e) => {
                error!(
                    server = server_name,
                    tool = tool_name,
                    error = %e,
                    "MCP tool call failed"
                );
                Err(format!("MCP tool call failed: {e}"))
            }
        }
    }

    /// Get a reference to the permission manager.
    pub fn permissions(&self) -> &PermissionManager {
        &self.permissions
    }

    /// Get a mutable reference to the permission manager.
    pub fn permissions_mut(&mut self) -> &mut PermissionManager {
        &mut self.permissions
    }

    /// List connected server names.
    pub fn connected_servers(&self) -> Vec<&str> {
        self.sessions.keys().map(|s| s.as_str()).collect()
    }

    /// Refresh tool list from a specific server (on tools/list_changed notification).
    pub async fn refresh_tools(&mut self, server_name: &str) -> anyhow::Result<()> {
        if let Some(session) = self.sessions.get_mut(server_name) {
            let response = session.service.list_tools(Default::default()).await?;
            session.cached_tools = response.tools;
            info!(
                server = server_name,
                tool_count = session.cached_tools.len(),
                "Refreshed tool list"
            );
        }
        Ok(())
    }

    /// Gracefully disconnect all servers.
    pub async fn shutdown(&mut self) {
        for (name, session) in self.sessions.drain() {
            info!(server = %name, "Shutting down MCP server");
            if let Err(e) = session.service.cancel().await {
                warn!(server = %name, error = %e, "Error shutting down MCP server");
            }
        }
    }
}

impl Default for McpManager {
    fn default() -> Self {
        Self::new()
    }
}
