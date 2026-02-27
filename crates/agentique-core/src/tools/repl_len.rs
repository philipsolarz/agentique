use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::ReplSession;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplLenTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplLenTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplLenParams {
    /// The variable name to get the size of.
    name: String,
}

#[async_trait]
impl Tool for ReplLenTool {
    fn name(&self) -> &str {
        "repl_len"
    }

    fn description(&self) -> &str {
        "Get the size and token estimate of a REPL variable."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplLenParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplLenParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let session = self.session.lock().await;
        let meta = session
            .metadata(&params.name)
            .ok_or_else(|| format!("Variable '{}' not found", params.name))?;

        Ok(format!(
            "${}: {} chars, ~{} tokens, {} bytes, {} lines",
            params.name,
            meta.char_count,
            meta.token_estimate,
            meta.size_bytes,
            meta.line_count,
        ))
    }
}
