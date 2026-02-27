use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::{ReplSession, Variable};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplSetTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplSetTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplSetParams {
    /// The variable name to set in the REPL session. Prefix with "final_" to mark as terminal output.
    name: String,
    /// The value to store. Will be stored as text.
    value: String,
}

#[async_trait]
impl Tool for ReplSetTool {
    fn name(&self) -> &str {
        "repl_set"
    }

    fn description(&self) -> &str {
        "Set a named variable in the REPL session. Use 'final_' prefix for terminal outputs (e.g., 'final_answer')."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplSetParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplSetParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let mut session = self.session.lock().await;
        session.set(&params.name, Variable::Text(params.value));

        Ok(format!("Variable '{}' set successfully", params.name))
    }
}
