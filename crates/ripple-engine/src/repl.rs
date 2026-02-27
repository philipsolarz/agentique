use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Variable {
    Text(String),
    Json(serde_json::Value),
    Number(f64),
    Boolean(bool),
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Text(s) => write!(f, "{s}"),
            Variable::Json(v) => write!(f, "{v}"),
            Variable::Number(n) => write!(f, "{n}"),
            Variable::Boolean(b) => write!(f, "{b}"),
        }
    }
}

/// A REPL session holding named variables.
/// Variables prefixed with `final_` are considered terminal outputs.
#[derive(Debug, Default)]
pub struct ReplSession {
    variables: HashMap<String, Variable>,
}

impl ReplSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: impl Into<String>, value: Variable) {
        self.variables.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&Variable> {
        self.variables.get(name)
    }

    pub fn has_final(&self) -> bool {
        self.variables.keys().any(|k| k.starts_with("final_"))
    }

    pub fn finals(&self) -> impl Iterator<Item = (&str, &Variable)> {
        self.variables
            .iter()
            .filter(|(k, _)| k.starts_with("final_"))
            .map(|(k, v)| (k.as_str(), v))
    }

    pub fn format_state_summary(&self) -> String {
        if self.variables.is_empty() {
            return "REPL state: (empty)".to_string();
        }
        let mut lines = vec!["REPL state:".to_string()];
        for (k, v) in &self.variables {
            let display = format!("{v}");
            let truncated = if display.len() > 200 {
                format!("{}...", &display[..200])
            } else {
                display
            };
            lines.push(format!("  {k} = {truncated}"));
        }
        lines.join("\n")
    }
}
