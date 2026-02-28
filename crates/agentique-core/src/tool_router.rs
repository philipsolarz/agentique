use async_trait::async_trait;
use llm_provider::ToolDefinition;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only_hint: bool,
    pub destructive_hint: bool,
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only_hint: false,
            destructive_hint: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    pub duration_ms: u64,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
    }
    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String>;

    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: llm_provider::FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters_schema(),
            },
        }
    }
}

/// Adapter that wraps an MCP tool (via McpManager) as a local Tool.
struct McpToolAdapter {
    namespaced_name: String,
    description: String,
    parameters: serde_json::Value,
    annotations_val: ToolAnnotations,
    manager: Arc<Mutex<mcp_manager::McpManager>>,
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.namespaced_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters.clone()
    }

    fn annotations(&self) -> ToolAnnotations {
        self.annotations_val.clone()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let mgr = self.manager.lock().await;
        mgr.dispatch(&self.namespaced_name, arguments).await
    }
}

pub struct ToolRouter {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRouter {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Register all tools from a connected MCP manager.
    ///
    /// Each MCP tool is wrapped in an adapter that routes calls through the
    /// manager. Tool names are namespaced as `{server}:{tool}`.
    pub fn register_mcp_tools(&mut self, manager: Arc<Mutex<mcp_manager::McpManager>>, tool_defs: Vec<ToolDefinition>) {
        for def in tool_defs {
            let namespaced_name = def.function.name.clone();
            let adapter = McpToolAdapter {
                namespaced_name: namespaced_name.clone(),
                description: def.function.description.clone(),
                parameters: def.function.parameters.clone(),
                annotations_val: ToolAnnotations {
                    // MCP tools default to not read-only / not destructive.
                    // The MCP permission manager handles approval policy separately.
                    read_only_hint: false,
                    destructive_hint: false,
                },
                manager: Arc::clone(&manager),
            };
            self.tools.insert(namespaced_name, Box::new(adapter));
        }
    }

    pub async fn dispatch(&self, name: &str, arguments: serde_json::Value) -> ToolResult {
        let start = Instant::now();
        match self.tools.get(name) {
            Some(tool) => {
                let result = tool.execute(arguments).await;
                let duration_ms = start.elapsed().as_millis() as u64;
                match result {
                    Ok(content) => ToolResult {
                        content,
                        is_error: false,
                        duration_ms,
                    },
                    Err(err) => ToolResult {
                        content: format!("Error: {err}"),
                        is_error: true,
                        duration_ms,
                    },
                }
            }
            None => ToolResult {
                content: format!("Unknown tool: {name}"),
                is_error: true,
                duration_ms: 0,
            },
        }
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.tool_definition()).collect()
    }

    /// Get annotations for a tool by name.
    pub fn annotations(&self, name: &str) -> Option<ToolAnnotations> {
        self.tools.get(name).map(|t| t.annotations())
    }

    /// Check if a tool name is an MCP tool (contains ':' namespace separator).
    pub fn is_mcp_tool(&self, name: &str) -> bool {
        name.contains(':') && self.tools.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
            Ok(arguments["text"].as_str().unwrap_or("").to_string())
        }
    }

    struct FailTool;

    #[async_trait]
    impl Tool for FailTool {
        fn name(&self) -> &str {
            "fail"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _arguments: serde_json::Value) -> Result<String, String> {
            Err("intentional failure".to_string())
        }
    }

    #[tokio::test]
    async fn dispatch_registered_tool() {
        let mut router = ToolRouter::new();
        router.register(Box::new(EchoTool));

        let result = router
            .dispatch("echo", serde_json::json!({"text": "hello"}))
            .await;
        assert!(!result.is_error);
        assert_eq!(result.content, "hello");
        assert!(result.duration_ms < 1000);
    }

    #[tokio::test]
    async fn dispatch_unknown_tool() {
        let router = ToolRouter::new();
        let result = router
            .dispatch("nonexistent", serde_json::json!({}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn dispatch_failing_tool() {
        let mut router = ToolRouter::new();
        router.register(Box::new(FailTool));

        let result = router.dispatch("fail", serde_json::json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("intentional failure"));
    }

    #[test]
    fn tool_definitions_list() {
        let mut router = ToolRouter::new();
        router.register(Box::new(EchoTool));
        let defs = router.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "echo");
    }

    #[test]
    fn tool_annotations_default() {
        let annotations = ToolAnnotations::default();
        assert!(!annotations.read_only_hint);
        assert!(!annotations.destructive_hint);
    }

    #[test]
    fn is_mcp_tool_detection() {
        let router = ToolRouter::new();
        assert!(!router.is_mcp_tool("file_read"));
        assert!(!router.is_mcp_tool("github:list_repos")); // not registered
    }
}
