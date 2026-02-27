use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool_router::{Tool, ToolAnnotations};

pub struct ListFilesTool;

#[derive(Deserialize, JsonSchema)]
struct ListFilesParams {
    /// The directory path to list files from.
    path: String,
}

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files and directories at the given path. Returns a newline-separated list of entries."
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only_hint: true,
            destructive_hint: false,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ListFilesParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ListFilesParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&params.path)
            .await
            .map_err(|e| format!("Failed to read directory '{}': {e}", params.path))?;

        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| format!("Failed to read entry: {e}"))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|e| format!("Failed to get file type: {e}"))?;
            let name = entry.file_name().to_string_lossy().to_string();
            let suffix = if file_type.is_dir() { "/" } else { "" };
            entries.push(format!("{name}{suffix}"));
        }

        entries.sort();
        Ok(entries.join("\n"))
    }
}
