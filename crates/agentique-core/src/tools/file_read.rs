use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool_router::{Tool, ToolAnnotations};

pub struct FileReadTool;

#[derive(Deserialize, JsonSchema)]
struct FileReadParams {
    /// The absolute path to the file to read.
    path: String,
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path. Returns the file contents as a string."
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only_hint: true,
            destructive_hint: false,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(FileReadParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: FileReadParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        tokio::fs::read_to_string(&params.path)
            .await
            .map_err(|e| format!("Failed to read file '{}': {e}", params.path))
    }
}
