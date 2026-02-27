use std::sync::Arc;

use async_trait::async_trait;
use ripple_engine::ReplSession;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::Tool;

pub struct ReplSearchTool {
    session: Arc<Mutex<ReplSession>>,
}

impl ReplSearchTool {
    pub fn new(session: Arc<Mutex<ReplSession>>) -> Self {
        Self { session }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReplSearchParams {
    /// The variable name to search within.
    name: String,
    /// Regex pattern to search for.
    pattern: String,
    /// Maximum number of matches to return (default: 20).
    max_matches: Option<usize>,
}

#[async_trait]
impl Tool for ReplSearchTool {
    fn name(&self) -> &str {
        "repl_search"
    }

    fn description(&self) -> &str {
        "Regex search within a REPL variable. Returns matching lines with line numbers and match positions."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReplSearchParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: ReplSearchParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let max_matches = params.max_matches.unwrap_or(20);
        let session = self.session.lock().await;
        let matches = session.search(&params.name, &params.pattern, max_matches)?;

        if matches.is_empty() {
            return Ok(format!("No matches for '{}' in ${}", params.pattern, params.name));
        }

        let mut lines = vec![format!("Found {} match(es):", matches.len())];
        for m in &matches {
            lines.push(format!(
                "  L{}: {} (match at {}..{})",
                m.line_number, m.line_content, m.match_start, m.match_end
            ));
        }
        Ok(lines.join("\n"))
    }
}
