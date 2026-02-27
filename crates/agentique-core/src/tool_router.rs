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
