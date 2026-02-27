use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::ReplSession;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplChunksTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplChunksTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplChunksParams {
    /// The variable name to split into chunks.
    name: String,
    /// Size of each chunk in characters.
    chunk_size: usize,
}

#[async_trait]
impl Tool for ReplChunksTool {
    fn name(&self) -> &str {
        "repl_chunks"
    }

    fn description(&self) -> &str {
        "Split a large REPL variable into smaller chunk variables for processing. Creates variables named {name}_chunk_0, {name}_chunk_1, etc."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplChunksParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplChunksParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let mut session = self.session.lock().await;
        let chunk_names = session.chunk(&params.name, params.chunk_size)?;

        Ok(format!(
            "Split ${} into {} chunks: {}",
            params.name,
            chunk_names.len(),
            chunk_names.join(", ")
        ))
    }
}
