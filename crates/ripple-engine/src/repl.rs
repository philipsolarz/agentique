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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_session() {
        let session = ReplSession::new();
        assert!(!session.has_final());
        assert!(session.get("anything").is_none());
        assert_eq!(session.format_state_summary(), "REPL state: (empty)");
    }

    #[test]
    fn set_and_get_variables() {
        let mut session = ReplSession::new();
        session.set("name", Variable::Text("hello".to_string()));
        session.set("count", Variable::Number(42.0));
        session.set("flag", Variable::Boolean(true));

        assert!(matches!(session.get("name"), Some(Variable::Text(s)) if s == "hello"));
        assert!(matches!(session.get("count"), Some(Variable::Number(n)) if (*n - 42.0).abs() < f64::EPSILON));
        assert!(matches!(session.get("flag"), Some(Variable::Boolean(true))));
        assert!(session.get("missing").is_none());
    }

    #[test]
    fn overwrite_variable() {
        let mut session = ReplSession::new();
        session.set("x", Variable::Text("old".to_string()));
        session.set("x", Variable::Text("new".to_string()));
        assert!(matches!(session.get("x"), Some(Variable::Text(s)) if s == "new"));
    }

    #[test]
    fn final_detection() {
        let mut session = ReplSession::new();
        session.set("temp", Variable::Text("not final".to_string()));
        assert!(!session.has_final());

        session.set("final_answer", Variable::Text("42".to_string()));
        assert!(session.has_final());

        let finals: Vec<_> = session.finals().collect();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].0, "final_answer");
    }

    #[test]
    fn state_summary_contains_variables() {
        let mut session = ReplSession::new();
        session.set("x", Variable::Number(3.14));
        let summary = session.format_state_summary();
        assert!(summary.contains("REPL state:"));
        assert!(summary.contains("x = 3.14"));
    }

    #[test]
    fn state_summary_truncates_long_values() {
        let mut session = ReplSession::new();
        let long_value = "a".repeat(300);
        session.set("big", Variable::Text(long_value));
        let summary = session.format_state_summary();
        assert!(summary.contains("..."));
    }

    #[test]
    fn variable_display() {
        assert_eq!(format!("{}", Variable::Text("hello".to_string())), "hello");
        assert_eq!(format!("{}", Variable::Number(3.14)), "3.14");
        assert_eq!(format!("{}", Variable::Boolean(false)), "false");
    }

    #[test]
    fn json_variable() {
        let val = serde_json::json!({"key": "value"});
        let mut session = ReplSession::new();
        session.set("data", Variable::Json(val));
        assert!(session.get("data").is_some());
    }
}
