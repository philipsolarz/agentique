use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::ReplSession;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplGetTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplGetTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplGetParams {
    /// The variable name to retrieve from the REPL session.
    name: String,
}

#[async_trait]
impl Tool for ReplGetTool {
    fn name(&self) -> &str {
        "repl_get"
    }

    fn description(&self) -> &str {
        "Get the value of a named variable from the REPL session."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplGetParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplGetParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let session = self.session.lock().await;
        match session.get(&params.name) {
            Some(var) => Ok(format!("{var}")),
            None => Err(format!("Variable '{}' not found", params.name)),
        }
    }
}
