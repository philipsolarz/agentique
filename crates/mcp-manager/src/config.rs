use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique name for this server (used as namespace prefix).
    pub name: String,
    /// Transport type.
    pub transport: TransportConfig,
    /// Whether this server is trusted (auto-approve read-only tools).
    #[serde(default)]
    pub trusted: bool,
    /// Environment variables to set for the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TransportConfig {
    /// Stdio transport: launch as child process, communicate via stdin/stdout.
    #[serde(rename = "stdio")]
    Stdio {
        /// Command to launch the server.
        command: String,
        /// Arguments to pass to the command.
        #[serde(default)]
        args: Vec<String>,
    },
    /// HTTP transport (Streamable HTTP / SSE).
    #[serde(rename = "http")]
    Http {
        /// URL of the MCP server endpoint.
        url: String,
    },
}

/// Top-level MCP configuration file format.
/// Used for both project-level (.agentique/mcp.json) and global ($AGENTIQUE_HOME/mcp.json).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfigFile {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

impl McpConfigFile {
    /// Load MCP configuration from a JSON file. Returns empty config if file doesn't exist.
    pub async fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = tokio::fs::read_to_string(path).await?;
        let config: McpConfigFile = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Merge another config into this one (later configs take precedence by server name).
    pub fn merge(&mut self, other: McpConfigFile) {
        for server in other.servers {
            // Remove existing server with same name, then add the new one
            self.servers.retain(|s| s.name != server.name);
            self.servers.push(server);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stdio_config() {
        let json = r#"{
            "servers": [
                {
                    "name": "filesystem",
                    "transport": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                    },
                    "trusted": true
                }
            ]
        }"#;
        let config: McpConfigFile = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "filesystem");
        assert!(config.servers[0].trusted);
        match &config.servers[0].transport {
            TransportConfig::Stdio { command, args } => {
                assert_eq!(command, "npx");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected stdio transport"),
        }
    }

    #[test]
    fn parse_http_config() {
        let json = r#"{
            "servers": [
                {
                    "name": "remote",
                    "transport": {
                        "type": "http",
                        "url": "http://localhost:3000/mcp"
                    }
                }
            ]
        }"#;
        let config: McpConfigFile = serde_json::from_str(json).unwrap();
        match &config.servers[0].transport {
            TransportConfig::Http { url } => {
                assert_eq!(url, "http://localhost:3000/mcp");
            }
            _ => panic!("Expected http transport"),
        }
    }

    #[test]
    fn merge_configs() {
        let mut base = McpConfigFile {
            servers: vec![McpServerConfig {
                name: "fs".to_string(),
                transport: TransportConfig::Stdio {
                    command: "old".to_string(),
                    args: vec![],
                },
                trusted: false,
                env: Default::default(),
            }],
        };
        let override_config = McpConfigFile {
            servers: vec![McpServerConfig {
                name: "fs".to_string(),
                transport: TransportConfig::Stdio {
                    command: "new".to_string(),
                    args: vec![],
                },
                trusted: true,
                env: Default::default(),
            }],
        };
        base.merge(override_config);
        assert_eq!(base.servers.len(), 1);
        assert!(base.servers[0].trusted);
    }
}
