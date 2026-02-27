use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Threshold in characters: variables larger than this show metadata only in state summaries.
pub const SYMBOLIC_THRESHOLD_CHARS: usize = 500;

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

/// Metadata about a stored variable, computed on insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMetadata {
    pub data_type: String,
    pub char_count: usize,
    pub token_estimate: usize,
    pub size_bytes: usize,
    pub line_count: usize,
    pub preview: String,
}

impl VariableMetadata {
    fn from_variable(var: &Variable) -> Self {
        let text = format!("{var}");
        let char_count = text.len();
        Self {
            data_type: match var {
                Variable::Text(_) => "Text".to_string(),
                Variable::Json(_) => "Json".to_string(),
                Variable::Number(_) => "Number".to_string(),
                Variable::Boolean(_) => "Boolean".to_string(),
            },
            char_count,
            token_estimate: char_count / 4,
            size_bytes: text.as_bytes().len(),
            line_count: text.lines().count().max(1),
            preview: if text.len() > 120 {
                format!("{}...", &text[..120])
            } else {
                text
            },
        }
    }
}

/// A match result from searching within a variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// Internal wrapper holding a variable and its metadata.
#[derive(Debug, Clone)]
struct StoredVariable {
    value: Variable,
    metadata: VariableMetadata,
}

/// A REPL session holding named variables.
/// Variables prefixed with `final_` are considered terminal outputs.
#[derive(Debug, Default)]
pub struct ReplSession {
    variables: HashMap<String, StoredVariable>,
}

impl ReplSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: impl Into<String>, value: Variable) {
        let metadata = VariableMetadata::from_variable(&value);
        self.variables.insert(
            name.into(),
            StoredVariable { value, metadata },
        );
    }

    pub fn get(&self, name: &str) -> Option<&Variable> {
        self.variables.get(name).map(|sv| &sv.value)
    }

    pub fn metadata(&self, name: &str) -> Option<&VariableMetadata> {
        self.variables.get(name).map(|sv| &sv.metadata)
    }

    pub fn has_final(&self) -> bool {
        self.variables.keys().any(|k| k.starts_with("final_"))
    }

    pub fn finals(&self) -> impl Iterator<Item = (&str, &Variable)> {
        self.variables
            .iter()
            .filter(|(k, _)| k.starts_with("final_"))
            .map(|(k, sv)| (k.as_str(), &sv.value))
    }

    /// Extract a character range from a text variable.
    pub fn slice(&self, name: &str, start: usize, end: usize) -> Result<String, String> {
        let var = self.get(name).ok_or_else(|| format!("Variable '{name}' not found"))?;
        let text = format!("{var}");
        if start >= text.len() {
            return Err(format!("Start index {start} out of range (len={})", text.len()));
        }
        let end = end.min(text.len());
        if start > end {
            return Err(format!("Start ({start}) > end ({end})"));
        }
        Ok(text[start..end].to_string())
    }

    /// Regex search within a variable, returning matches with line numbers.
    pub fn search(
        &self,
        name: &str,
        pattern: &str,
        max_matches: usize,
    ) -> Result<Vec<SearchMatch>, String> {
        let var = self.get(name).ok_or_else(|| format!("Variable '{name}' not found"))?;
        let text = format!("{var}");
        let re = Regex::new(pattern).map_err(|e| format!("Invalid regex: {e}"))?;

        let mut matches = Vec::new();
        for (line_idx, line) in text.lines().enumerate() {
            for mat in re.find_iter(line) {
                matches.push(SearchMatch {
                    line_number: line_idx + 1,
                    line_content: line.to_string(),
                    match_start: mat.start(),
                    match_end: mat.end(),
                });
                if matches.len() >= max_matches {
                    return Ok(matches);
                }
            }
        }
        Ok(matches)
    }

    /// Split a variable into chunks stored as separate variables.
    /// Returns the names of the created chunk variables.
    pub fn chunk(&mut self, name: &str, chunk_size: usize) -> Result<Vec<String>, String> {
        if chunk_size == 0 {
            return Err("Chunk size must be > 0".to_string());
        }
        let var = self.get(name).ok_or_else(|| format!("Variable '{name}' not found"))?;
        let text = format!("{var}");

        let mut chunk_names = Vec::new();
        let mut start = 0;
        let mut idx = 0;

        while start < text.len() {
            let end = (start + chunk_size).min(text.len());
            let chunk_name = format!("{name}_chunk_{idx}");
            let chunk_text = text[start..end].to_string();
            self.set(&chunk_name, Variable::Text(chunk_text));
            chunk_names.push(chunk_name);
            start = end;
            idx += 1;
        }

        Ok(chunk_names)
    }

    /// Get the character count of a variable.
    pub fn len(&self, name: &str) -> Result<usize, String> {
        let meta = self.metadata(name).ok_or_else(|| format!("Variable '{name}' not found"))?;
        Ok(meta.char_count)
    }

    pub fn format_state_summary(&self) -> String {
        if self.variables.is_empty() {
            return "REPL state: (empty)".to_string();
        }
        let mut lines = vec!["REPL state:".to_string()];
        for (k, sv) in &self.variables {
            if sv.metadata.char_count > SYMBOLIC_THRESHOLD_CHARS {
                lines.push(format!(
                    "  ${}: {} ({} chars, ~{} tokens) \u{2014} preview: \"{}\"",
                    k,
                    sv.metadata.data_type,
                    sv.metadata.char_count,
                    sv.metadata.token_estimate,
                    sv.metadata.preview,
                ));
            } else {
                let display = format!("{}", sv.value);
                lines.push(format!("  {k} = {display}"));
            }
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
    fn state_summary_symbolic_for_large_values() {
        let mut session = ReplSession::new();
        let long_value = "a".repeat(600);
        session.set("big", Variable::Text(long_value));
        let summary = session.format_state_summary();
        assert!(summary.contains("$big: Text"));
        assert!(summary.contains("600 chars"));
        assert!(summary.contains("~150 tokens"));
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

    #[test]
    fn metadata_computed_on_set() {
        let mut session = ReplSession::new();
        session.set("txt", Variable::Text("hello world".to_string()));
        let meta = session.metadata("txt").unwrap();
        assert_eq!(meta.data_type, "Text");
        assert_eq!(meta.char_count, 11);
        assert_eq!(meta.token_estimate, 2); // 11/4 = 2
        assert_eq!(meta.line_count, 1);
        assert_eq!(meta.preview, "hello world");
    }

    #[test]
    fn slice_basic() {
        let mut session = ReplSession::new();
        session.set("s", Variable::Text("hello world".to_string()));
        assert_eq!(session.slice("s", 0, 5).unwrap(), "hello");
        assert_eq!(session.slice("s", 6, 11).unwrap(), "world");
    }

    #[test]
    fn slice_clamps_end() {
        let mut session = ReplSession::new();
        session.set("s", Variable::Text("abc".to_string()));
        assert_eq!(session.slice("s", 0, 100).unwrap(), "abc");
    }

    #[test]
    fn slice_errors() {
        let mut session = ReplSession::new();
        session.set("s", Variable::Text("abc".to_string()));
        assert!(session.slice("missing", 0, 1).is_err());
        assert!(session.slice("s", 100, 200).is_err());
    }

    #[test]
    fn search_basic() {
        let mut session = ReplSession::new();
        session.set("code", Variable::Text("fn main() {\n    println!(\"hello\");\n}\n".to_string()));
        let matches = session.search("code", "println", 10).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line_number, 2);
    }

    #[test]
    fn search_max_matches() {
        let mut session = ReplSession::new();
        session.set("data", Variable::Text("aaa\naaa\naaa\naaa\n".to_string()));
        let matches = session.search("data", "aaa", 2).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn search_errors() {
        let mut session = ReplSession::new();
        assert!(session.search("missing", "pat", 10).is_err());
        session.set("x", Variable::Text("hello".to_string()));
        assert!(session.search("x", "[invalid", 10).is_err());
    }

    #[test]
    fn chunk_basic() {
        let mut session = ReplSession::new();
        session.set("data", Variable::Text("abcdefghij".to_string()));
        let names = session.chunk("data", 3).unwrap();
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "data_chunk_0");
        assert_eq!(session.get("data_chunk_0").unwrap().to_string(), "abc");
        assert_eq!(session.get("data_chunk_1").unwrap().to_string(), "def");
        assert_eq!(session.get("data_chunk_2").unwrap().to_string(), "ghi");
        assert_eq!(session.get("data_chunk_3").unwrap().to_string(), "j");
    }

    #[test]
    fn chunk_errors() {
        let mut session = ReplSession::new();
        assert!(session.chunk("missing", 10).is_err());
        session.set("x", Variable::Text("abc".to_string()));
        assert!(session.chunk("x", 0).is_err());
    }

    #[test]
    fn len_basic() {
        let mut session = ReplSession::new();
        session.set("s", Variable::Text("hello".to_string()));
        assert_eq!(session.len("s").unwrap(), 5);
        assert!(session.len("missing").is_err());
    }
}
