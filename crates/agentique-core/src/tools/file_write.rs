use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool_router::{Tool, ToolAnnotations};

pub struct FileWriteTool;

#[derive(Deserialize, JsonSchema)]
struct FileWriteParams {
    /// The absolute path to the file to write.
    path: String,
    /// The content to write to the file.
    content: String,
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only_hint: false,
            destructive_hint: true,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(FileWriteParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: FileWriteParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&params.path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create directory: {e}"))?;
        }

        tokio::fs::write(&params.path, &params.content)
            .await
            .map_err(|e| format!("Failed to write file '{}': {e}", params.path))?;

        Ok(format!("Successfully wrote {} bytes to {}", params.content.len(), params.path))
    }
}
