use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::{ReplSession, Variable};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplLoadFileTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplLoadFileTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplLoadFileParams {
    /// The file path to load.
    path: String,
    /// The variable name to store the file content under. Defaults to the file name.
    name: Option<String>,
}

#[async_trait]
impl Tool for ReplLoadFileTool {
    fn name(&self) -> &str {
        "repl_load_file"
    }

    fn description(&self) -> &str {
        "Load a file into the REPL as a symbolic variable. Returns metadata (size, token estimate, preview) instead of the full content, keeping large files out of the conversation context."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplLoadFileParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplLoadFileParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let content = tokio::fs::read_to_string(&params.path)
            .await
            .map_err(|e| format!("Failed to read file '{}': {e}", params.path))?;

        let var_name = params.name.unwrap_or_else(|| {
            std::path::Path::new(&params.path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "file".to_string())
        });

        let mut session = self.session.lock().await;
        session.set(&var_name, Variable::Text(content));

        let meta = session.metadata(&var_name).unwrap();
        Ok(format!(
            "Loaded '{}' into ${}\nType: {}\nSize: {} chars (~{} tokens)\nLines: {}\nPreview: \"{}\"",
            params.path,
            var_name,
            meta.data_type,
            meta.char_count,
            meta.token_estimate,
            meta.line_count,
            meta.preview,
        ))
    }
}
