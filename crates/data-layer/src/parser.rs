use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Parser, Tree};

/// A symbol extracted from source code via tree-sitter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: usize,
    pub end_line: usize,
    pub file_path: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Class,
    Struct,
    Enum,
    Import,
    Const,
    Interface,
    Module,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolKind::Function => write!(f, "function"),
            SymbolKind::Class => write!(f, "class"),
            SymbolKind::Struct => write!(f, "struct"),
            SymbolKind::Enum => write!(f, "enum"),
            SymbolKind::Import => write!(f, "import"),
            SymbolKind::Const => write!(f, "const"),
            SymbolKind::Interface => write!(f, "interface"),
            SymbolKind::Module => write!(f, "module"),
        }
    }
}

/// Supported programming languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Rust,
    Python,
    TypeScript,
    JavaScript,
}

impl SupportedLanguage {
    /// Detect language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "ts" | "tsx" => Some(Self::TypeScript),
            "js" | "jsx" => Some(Self::JavaScript),
            _ => None,
        }
    }

    fn tree_sitter_language(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        }
    }

    #[allow(dead_code)]
    fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
        }
    }
}

/// Extracts symbols from source code using tree-sitter.
pub struct SymbolExtractor;

impl SymbolExtractor {
    /// Parse source code and extract top-level symbols.
    pub fn extract(
        source: &str,
        language: SupportedLanguage,
        file_path: &str,
    ) -> Vec<Symbol> {
        let mut parser = Parser::new();
        parser
            .set_language(&language.tree_sitter_language())
            .expect("Failed to set tree-sitter language");

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut symbols = Vec::new();
        Self::walk_tree(&tree, source, language, file_path, &mut symbols);
        symbols
    }

    /// Extract import paths from source code.
    pub fn extract_imports(
        source: &str,
        language: SupportedLanguage,
        _file_path: &str,
    ) -> Vec<String> {
        let mut parser = Parser::new();
        parser
            .set_language(&language.tree_sitter_language())
            .expect("Failed to set tree-sitter language");

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut imports = Vec::new();
        Self::walk_imports(&tree, source, language, &mut imports);
        imports
    }

    fn walk_tree(
        tree: &Tree,
        source: &str,
        language: SupportedLanguage,
        file_path: &str,
        symbols: &mut Vec<Symbol>,
    ) {
        let root = tree.root_node();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let kind_str = child.kind();
            let (symbol_kind, name_field) = match language {
                SupportedLanguage::Rust => match kind_str {
                    "function_item" => (Some(SymbolKind::Function), "name"),
                    "struct_item" => (Some(SymbolKind::Struct), "name"),
                    "enum_item" => (Some(SymbolKind::Enum), "name"),
                    "const_item" => (Some(SymbolKind::Const), "name"),
                    "mod_item" => (Some(SymbolKind::Module), "name"),
                    "impl_item" => {
                        // Extract methods from impl blocks
                        Self::walk_impl_block(&child, source, file_path, symbols);
                        continue;
                    }
                    _ => (None, ""),
                },
                SupportedLanguage::Python => match kind_str {
                    "function_definition" => (Some(SymbolKind::Function), "name"),
                    "class_definition" => (Some(SymbolKind::Class), "name"),
                    _ => (None, ""),
                },
                SupportedLanguage::TypeScript | SupportedLanguage::JavaScript => match kind_str {
                    "function_declaration" => (Some(SymbolKind::Function), "name"),
                    "class_declaration" => (Some(SymbolKind::Class), "name"),
                    "interface_declaration" => (Some(SymbolKind::Interface), "name"),
                    "enum_declaration" => (Some(SymbolKind::Enum), "name"),
                    "lexical_declaration" | "export_statement" => {
                        // Walk children to find variable_declarator or nested declarations
                        let mut inner_cursor = child.walk();
                        for inner in child.children(&mut inner_cursor) {
                            match inner.kind() {
                                "variable_declarator" => {
                                    if let Some(name_node) = inner.child_by_field_name("name") {
                                        let name = &source[name_node.byte_range()];
                                        symbols.push(Symbol {
                                            name: name.to_string(),
                                            kind: SymbolKind::Const,
                                            start_line: child.start_position().row + 1,
                                            end_line: child.end_position().row + 1,
                                            file_path: file_path.to_string(),
                                        });
                                    }
                                }
                                "function_declaration" => {
                                    if let Some(name_node) = inner.child_by_field_name("name") {
                                        let name = &source[name_node.byte_range()];
                                        symbols.push(Symbol {
                                            name: name.to_string(),
                                            kind: SymbolKind::Function,
                                            start_line: inner.start_position().row + 1,
                                            end_line: inner.end_position().row + 1,
                                            file_path: file_path.to_string(),
                                        });
                                    }
                                }
                                "class_declaration" => {
                                    if let Some(name_node) = inner.child_by_field_name("name") {
                                        let name = &source[name_node.byte_range()];
                                        symbols.push(Symbol {
                                            name: name.to_string(),
                                            kind: SymbolKind::Class,
                                            start_line: inner.start_position().row + 1,
                                            end_line: inner.end_position().row + 1,
                                            file_path: file_path.to_string(),
                                        });
                                    }
                                }
                                "interface_declaration" => {
                                    if let Some(name_node) = inner.child_by_field_name("name") {
                                        let name = &source[name_node.byte_range()];
                                        symbols.push(Symbol {
                                            name: name.to_string(),
                                            kind: SymbolKind::Interface,
                                            start_line: inner.start_position().row + 1,
                                            end_line: inner.end_position().row + 1,
                                            file_path: file_path.to_string(),
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                        continue;
                    }
                    _ => (None, ""),
                },
            };

            if let Some(kind) = symbol_kind {
                if let Some(name_node) = child.child_by_field_name(name_field) {
                    let name = &source[name_node.byte_range()];
                    symbols.push(Symbol {
                        name: name.to_string(),
                        kind,
                        start_line: child.start_position().row + 1,
                        end_line: child.end_position().row + 1,
                        file_path: file_path.to_string(),
                    });
                }
            }
        }
    }

    fn walk_impl_block(
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        symbols: &mut Vec<Symbol>,
    ) {
        // Get the type name being implemented
        let type_name = node
            .child_by_field_name("type")
            .map(|n| source[n.byte_range()].to_string())
            .unwrap_or_default();

        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "function_item" {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let method_name = &source[name_node.byte_range()];
                        let qualified = if type_name.is_empty() {
                            method_name.to_string()
                        } else {
                            format!("{type_name}::{method_name}")
                        };
                        symbols.push(Symbol {
                            name: qualified,
                            kind: SymbolKind::Function,
                            start_line: child.start_position().row + 1,
                            end_line: child.end_position().row + 1,
                            file_path: file_path.to_string(),
                        });
                    }
                }
            }
        }
    }

    fn walk_imports(
        tree: &Tree,
        source: &str,
        language: SupportedLanguage,
        imports: &mut Vec<String>,
    ) {
        let root = tree.root_node();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match language {
                SupportedLanguage::Rust => {
                    if child.kind() == "use_declaration" {
                        let text = &source[child.byte_range()];
                        // Extract the path portion: `use foo::bar;` → `foo::bar`
                        if let Some(stripped) = text.strip_prefix("use ") {
                            let path = stripped.trim_end_matches(';').trim();
                            imports.push(path.to_string());
                        }
                    }
                }
                SupportedLanguage::Python => {
                    if child.kind() == "import_statement"
                        || child.kind() == "import_from_statement"
                    {
                        let text = &source[child.byte_range()];
                        imports.push(text.to_string());
                    }
                }
                SupportedLanguage::TypeScript | SupportedLanguage::JavaScript => {
                    if child.kind() == "import_statement" {
                        // Extract source string: import { x } from './foo' → ./foo
                        if let Some(source_node) = child.child_by_field_name("source") {
                            let raw = &source[source_node.byte_range()];
                            let cleaned = raw.trim_matches(|c| c == '\'' || c == '"');
                            imports.push(cleaned.to_string());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_symbols() {
        let code = r#"
use std::collections::HashMap;

const MAX_SIZE: usize = 100;

struct Config {
    name: String,
}

enum Mode {
    Fast,
    Slow,
}

fn main() {
    println!("hello");
}

impl Config {
    fn new() -> Self {
        Config { name: String::new() }
    }
}
"#;
        let symbols = SymbolExtractor::extract(code, SupportedLanguage::Rust, "test.rs");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MAX_SIZE"), "got: {names:?}");
        assert!(names.contains(&"Config"), "got: {names:?}");
        assert!(names.contains(&"Mode"), "got: {names:?}");
        assert!(names.contains(&"main"), "got: {names:?}");
        assert!(names.contains(&"Config::new"), "got: {names:?}");
    }

    #[test]
    fn rust_imports() {
        let code = r#"
use std::collections::HashMap;
use crate::tool_router::ToolRouter;

fn main() {}
"#;
        let imports = SymbolExtractor::extract_imports(code, SupportedLanguage::Rust, "test.rs");
        assert_eq!(imports.len(), 2);
        assert!(imports[0].contains("HashMap"));
        assert!(imports[1].contains("ToolRouter"));
    }

    #[test]
    fn python_symbols() {
        let code = r#"
import os

class Config:
    def __init__(self):
        pass

def main():
    print("hello")
"#;
        let symbols = SymbolExtractor::extract(code, SupportedLanguage::Python, "test.py");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Config"), "got: {names:?}");
        assert!(names.contains(&"main"), "got: {names:?}");
    }

    #[test]
    fn typescript_symbols() {
        let code = r#"
import { useState } from 'react';

interface Props {
    name: string;
}

const MAX_SIZE = 100;

function App() {
    return null;
}

class Component {
    render() {}
}
"#;
        let symbols =
            SymbolExtractor::extract(code, SupportedLanguage::TypeScript, "test.ts");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Props"), "got: {names:?}");
        assert!(names.contains(&"App"), "got: {names:?}");
        assert!(names.contains(&"Component"), "got: {names:?}");
        assert!(names.contains(&"MAX_SIZE"), "got: {names:?}");
    }

    #[test]
    fn language_from_extension() {
        assert_eq!(
            SupportedLanguage::from_extension("rs"),
            Some(SupportedLanguage::Rust)
        );
        assert_eq!(
            SupportedLanguage::from_extension("py"),
            Some(SupportedLanguage::Python)
        );
        assert_eq!(
            SupportedLanguage::from_extension("ts"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(SupportedLanguage::from_extension("md"), None);
    }
}
