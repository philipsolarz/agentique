use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::{Tool, ToolAnnotations};

/// Tool that finds files related to a given file using the dependency graph.
pub struct FindRelatedTool {
    graph: Arc<Mutex<data_layer::DependencyGraph>>,
}

impl FindRelatedTool {
    pub fn new(graph: Arc<Mutex<data_layer::DependencyGraph>>) -> Self {
        Self { graph }
    }
}

#[derive(Deserialize, JsonSchema)]
struct FindRelatedParams {
    /// The file path to find related files for.
    file_path: String,
    /// Maximum graph traversal depth (default: 2, max: 5).
    depth: Option<u8>,
}

#[async_trait]
impl Tool for FindRelatedTool {
    fn name(&self) -> &str {
        "find_related_files"
    }

    fn description(&self) -> &str {
        "Find files related to a given file path through import and reference edges \
         in the dependency graph. Returns files ordered by graph distance (closest first). \
         Use this to understand which files are connected and might need changes together."
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only_hint: true,
            destructive_hint: false,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(FindRelatedParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: FindRelatedParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let depth = params.depth.unwrap_or(2).min(5);
        let graph = self.graph.lock().await;

        let related = graph.find_related(&params.file_path, depth);

        if related.is_empty() {
            return Ok(format!(
                "No related files found for '{}' (depth {}). \
                 The file may not be in the dependency graph yet.",
                params.file_path, depth
            ));
        }

        let mut output = format!(
            "Found {} files related to '{}' (depth {}):\n\n",
            related.len(),
            params.file_path,
            depth
        );

        for (i, node) in related.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} ({}, {} symbols)\n",
                i + 1,
                node.path,
                node.language,
                node.symbol_count,
            ));
        }

        Ok(output)
    }
}
