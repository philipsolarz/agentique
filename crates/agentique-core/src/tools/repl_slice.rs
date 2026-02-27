use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::ReplSession;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplSliceTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplSliceTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplSliceParams {
    /// The variable name to slice.
    name: String,
    /// Start character index (inclusive).
    start: usize,
    /// End character index (exclusive). Clamped to variable length.
    end: usize,
}

#[async_trait]
impl Tool for ReplSliceTool {
    fn name(&self) -> &str {
        "repl_slice"
    }

    fn description(&self) -> &str {
        "Extract a character range from a REPL variable. Returns the sliced text content."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplSliceParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplSliceParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let session = self.session.lock().await;
        session.slice(&params.name, params.start, params.end)
    }
}
