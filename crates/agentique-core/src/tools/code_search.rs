use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::tool_router::{Tool, ToolAnnotations};

/// Tool that searches indexed source code using tantivy full-text search.
pub struct CodeSearchTool {
    index: Arc<Mutex<data_layer::IndexManager>>,
}

impl CodeSearchTool {
    pub fn new(index: Arc<Mutex<data_layer::IndexManager>>) -> Self {
        Self { index }
    }
}

#[derive(Deserialize, JsonSchema)]
struct CodeSearchParams {
    /// The search query. Searches across file content and symbol names.
    query: String,
    /// Maximum number of results to return (default: 10).
    limit: Option<usize>,
    /// If true, search only symbol names (function, class, struct names).
    symbols_only: Option<bool>,
}

#[async_trait]
impl Tool for CodeSearchTool {
    fn name(&self) -> &str {
        "code_search"
    }

    fn description(&self) -> &str {
        "Search indexed source code files for content or symbol names. \
         Returns matching file paths with relevance scores and symbol information. \
         Use this to find functions, classes, or code patterns across the codebase."
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only_hint: true,
            destructive_hint: false,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(CodeSearchParams)).unwrap()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        let params: CodeSearchParams =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {e}"))?;

        let limit = params.limit.unwrap_or(10);
        let idx = self.index.lock().await;

        let results = if params.symbols_only.unwrap_or(false) {
            idx.search_symbols(&params.query, limit)
        } else {
            idx.search(&params.query, limit)
        }
        .map_err(|e| format!("Search failed: {e}"))?;

        if results.is_empty() {
            return Ok(format!("No results found for '{}'", params.query));
        }

        let mut output = format!("Found {} results for '{}':\n\n", results.len(), params.query);
        for (i, r) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} ({})\n   Score: {:.2}\n   Symbols: {}\n\n",
                i + 1,
                r.file_path,
                r.language,
                r.score,
                if r.symbols.is_empty() {
                    "(none)".to_string()
                } else {
                    r.symbols.clone()
                },
            ));
        }

        Ok(output)
    }
}
