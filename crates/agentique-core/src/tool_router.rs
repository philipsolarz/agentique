use async_trait::async_trait;
use llm_provider::ToolDefinition;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

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
}
