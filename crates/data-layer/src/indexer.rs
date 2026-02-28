use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::parser::{SupportedLanguage, SymbolExtractor};

/// A search result from the index.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub language: String,
    pub symbols: String,
    pub snippet: String,
    pub score: f32,
}

/// Manages a Tantivy full-text search index for code and documents.
pub struct IndexManager {
    index: Index,
    reader: IndexReader,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    f_file_path: Field,
    f_language: Field,
    f_symbol_names: Field,
    f_symbol_kinds: Field,
    f_content: Field,
}

impl IndexManager {
    /// Open or create an index at the given directory.
    pub fn new(index_dir: &Path) -> anyhow::Result<Self> {
        let mut schema_builder = Schema::builder();
        let f_file_path = schema_builder.add_text_field("file_path", STRING | STORED);
        let f_language = schema_builder.add_text_field("language", STRING | STORED);
        let f_symbol_names = schema_builder.add_text_field("symbol_names", TEXT | STORED);
        let f_symbol_kinds = schema_builder.add_text_field("symbol_kinds", TEXT | STORED);
        let f_content = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();

        std::fs::create_dir_all(index_dir)?;

        let index = Index::open_or_create(
            tantivy::directory::MmapDirectory::open(index_dir)?,
            schema.clone(),
        )?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            schema,
            f_file_path,
            f_language,
            f_symbol_names,
            f_symbol_kinds,
            f_content,
        })
    }

    /// Create an in-memory index (for testing).
    pub fn in_memory() -> anyhow::Result<Self> {
        let mut schema_builder = Schema::builder();
        let f_file_path = schema_builder.add_text_field("file_path", STRING | STORED);
        let f_language = schema_builder.add_text_field("language", STRING | STORED);
        let f_symbol_names = schema_builder.add_text_field("symbol_names", TEXT | STORED);
        let f_symbol_kinds = schema_builder.add_text_field("symbol_kinds", TEXT | STORED);
        let f_content = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            schema,
            f_file_path,
            f_language,
            f_symbol_names,
            f_symbol_kinds,
            f_content,
        })
    }

    /// Index all supported source files in a directory tree.
    pub fn index_directory(&self, root: &Path) -> anyhow::Result<usize> {
        let mut writer = self.index.writer(50_000_000)?;
        let mut count = 0;

        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !is_hidden(e) && !is_build_dir(e))
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let language = match SupportedLanguage::from_extension(ext) {
                Some(l) => l,
                None => continue,
            };

            if let Err(e) = self.index_file(&mut writer, path, language) {
                warn!(path = %path.display(), error = %e, "Failed to index file");
            } else {
                count += 1;
            }
        }

        writer.commit()?;
        self.reader.reload()?;
        info!(count, root = %root.display(), "Indexed files");
        Ok(count)
    }

    /// Index a single file.
    pub fn index_file(
        &self,
        writer: &mut IndexWriter,
        path: &Path,
        language: SupportedLanguage,
    ) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let path_str = path.to_string_lossy().to_string();

        // Delete existing document for this path
        writer.delete_term(Term::from_field_text(self.f_file_path, &path_str));

        // Extract symbols
        let symbols = SymbolExtractor::extract(&content, language, &path_str);
        let symbol_names: Vec<String> = symbols.iter().map(|s| s.name.clone()).collect();
        let symbol_kinds: Vec<String> = symbols.iter().map(|s| s.kind.to_string()).collect();

        let lang_name = match language {
            SupportedLanguage::Rust => "rust",
            SupportedLanguage::Python => "python",
            SupportedLanguage::TypeScript => "typescript",
            SupportedLanguage::JavaScript => "javascript",
        };

        writer.add_document(doc!(
            self.f_file_path => path_str,
            self.f_language => lang_name,
            self.f_symbol_names => symbol_names.join(" "),
            self.f_symbol_kinds => symbol_kinds.join(" "),
            self.f_content => content,
        ))?;

        debug!(path = %path.display(), symbols = symbols.len(), "Indexed file");
        Ok(())
    }

    /// Re-index a single file (for incremental updates).
    pub fn reindex_file(&self, path: &Path) -> anyhow::Result<()> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let language = SupportedLanguage::from_extension(ext)
            .ok_or_else(|| anyhow::anyhow!("Unsupported file type: {ext}"))?;

        let mut writer = self.index.writer(50_000_000)?;
        self.index_file(&mut writer, path, language)?;
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Full-text search across all indexed content and symbols.
    pub fn search(&self, query_str: &str, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.f_content, self.f_symbol_names]);

        let query = query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let file_path = doc
                .get_first(self.f_file_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let language = doc
                .get_first(self.f_language)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let symbols = doc
                .get_first(self.f_symbol_names)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(SearchResult {
                file_path,
                language,
                symbols,
                snippet: String::new(), // Could extract snippet with tantivy highlight
                score,
            });
        }

        Ok(results)
    }

    /// Search specifically for symbol names.
    pub fn search_symbols(
        &self,
        symbol_query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_symbol_names]);

        let query = query_parser.parse_query(symbol_query)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let file_path = doc
                .get_first(self.f_file_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let language = doc
                .get_first(self.f_language)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let symbols = doc
                .get_first(self.f_symbol_names)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(SearchResult {
                file_path,
                language,
                symbols,
                snippet: String::new(),
                score,
            });
        }

        Ok(results)
    }
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    // Don't filter the root directory (depth 0) — the user explicitly chose to index it.
    entry.depth() > 0
        && entry
            .file_name()
            .to_str()
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
}

fn is_build_dir(entry: &walkdir::DirEntry) -> bool {
    if entry.depth() == 0 {
        return false;
    }
    let name = entry.file_name().to_str().unwrap_or("");
    matches!(
        name,
        "target" | "node_modules" | "__pycache__" | "dist" | "build" | ".git"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn index_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();

        fs::write(
            src_dir.join("main.rs"),
            r#"
fn calculate_total(items: &[f64]) -> f64 {
    items.iter().sum()
}

struct Invoice {
    amount: f64,
}
"#,
        )
        .unwrap();

        fs::write(
            src_dir.join("utils.py"),
            r#"
def format_currency(amount):
    return f"${amount:.2f}"

class Calculator:
    pass
"#,
        )
        .unwrap();

        let idx = IndexManager::in_memory().unwrap();
        let count = idx.index_directory(dir.path()).unwrap();
        assert_eq!(count, 2);

        // Search for content
        let results = idx.search("calculate", 10).unwrap();
        assert!(!results.is_empty(), "expected results for 'calculate'");
        assert!(results[0].file_path.contains("main.rs"));

        // Search for symbols
        let results = idx.search_symbols("Calculator", 10).unwrap();
        assert!(!results.is_empty(), "expected results for 'Calculator'");
        assert!(results[0].file_path.contains("utils.py"));
    }

    #[test]
    fn reindex_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.rs");
        fs::write(&file, "fn old_function() {}").unwrap();

        let idx = IndexManager::in_memory().unwrap();
        {
            let mut writer = idx.index.writer(50_000_000).unwrap();
            idx.index_file(&mut writer, &file, SupportedLanguage::Rust)
                .unwrap();
            writer.commit().unwrap();
        }
        idx.reader.reload().unwrap();

        let results = idx.search_symbols("old_function", 10).unwrap();
        assert!(!results.is_empty());

        // Update the file and reindex
        fs::write(&file, "fn new_function() {}").unwrap();
        idx.reindex_file(&file).unwrap();

        let results = idx.search_symbols("new_function", 10).unwrap();
        assert!(!results.is_empty());
    }
}
