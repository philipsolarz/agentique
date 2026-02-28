use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

/// A file node in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNode {
    pub path: String,
    pub language: String,
    pub symbol_count: usize,
}

/// Edge types in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// File A imports symbols from file B.
    Imports,
    /// Symbol A references/calls symbol B.
    References,
}

/// A code dependency graph built on petgraph.
pub struct DependencyGraph {
    graph: StableGraph<FileNode, EdgeKind>,
    /// Map from file path to node index for O(1) lookup.
    path_index: HashMap<String, NodeIndex>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            path_index: HashMap::new(),
        }
    }

    /// Add a file node to the graph. Returns the node index.
    /// If the file already exists, updates its metadata.
    pub fn add_file(&mut self, node: FileNode) -> NodeIndex {
        if let Some(&idx) = self.path_index.get(&node.path) {
            // Update existing node
            self.graph[idx] = node;
            idx
        } else {
            let path = node.path.clone();
            let idx = self.graph.add_node(node);
            self.path_index.insert(path, idx);
            idx
        }
    }

    /// Add an import edge from one file to another.
    pub fn add_import(&mut self, from_path: &str, to_path: &str) {
        let from_idx = match self.path_index.get(from_path) {
            Some(&idx) => idx,
            None => return,
        };
        let to_idx = match self.path_index.get(to_path) {
            Some(&idx) => idx,
            None => return,
        };

        // Avoid duplicate edges
        if !self
            .graph
            .edges_connecting(from_idx, to_idx)
            .any(|e| *e.weight() == EdgeKind::Imports)
        {
            self.graph.add_edge(from_idx, to_idx, EdgeKind::Imports);
        }
    }

    /// Add a reference edge between files.
    pub fn add_reference(&mut self, from_path: &str, to_path: &str) {
        let from_idx = match self.path_index.get(from_path) {
            Some(&idx) => idx,
            None => return,
        };
        let to_idx = match self.path_index.get(to_path) {
            Some(&idx) => idx,
            None => return,
        };

        if !self
            .graph
            .edges_connecting(from_idx, to_idx)
            .any(|e| *e.weight() == EdgeKind::References)
        {
            self.graph.add_edge(from_idx, to_idx, EdgeKind::References);
        }
    }

    /// Remove a file and all its edges from the graph.
    pub fn remove_file(&mut self, path: &str) {
        if let Some(idx) = self.path_index.remove(path) {
            self.graph.remove_node(idx);
        }
    }

    /// Find files related to a given file path using BFS up to `max_depth` hops.
    /// Returns files in order of graph distance (closest first).
    pub fn find_related(&self, file_path: &str, max_depth: u8) -> Vec<&FileNode> {
        let start_idx = match self.path_index.get(file_path) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let mut visited = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_idx, 0u8));
        visited.insert(start_idx, 0u8);

        while let Some((node_idx, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Traverse both incoming and outgoing edges
            for direction in &[Direction::Outgoing, Direction::Incoming] {
                for neighbor in self.graph.neighbors_directed(node_idx, *direction) {
                    if !visited.contains_key(&neighbor) {
                        visited.insert(neighbor, depth + 1);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        // Remove the starting node and sort by depth
        visited.remove(&start_idx);
        let mut results: Vec<(NodeIndex, u8)> = visited.into_iter().collect();
        results.sort_by_key(|(_, depth)| *depth);

        results
            .iter()
            .filter_map(|(idx, _)| self.graph.node_weight(*idx))
            .collect()
    }

    /// Compute a simple PageRank-like score for each file based on incoming edges.
    /// Returns files sorted by score (highest first).
    pub fn rank_files(&self) -> Vec<(&FileNode, f32)> {
        let node_count = self.graph.node_count();
        if node_count == 0 {
            return Vec::new();
        }

        let damping = 0.85f32;
        let iterations = 20;
        let initial = 1.0 / node_count as f32;

        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let mut scores: HashMap<NodeIndex, f32> =
            node_indices.iter().map(|&idx| (idx, initial)).collect();

        for _ in 0..iterations {
            let mut new_scores: HashMap<NodeIndex, f32> = HashMap::new();

            for &idx in &node_indices {
                let incoming_sum: f32 = self
                    .graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .map(|src| {
                        let src_out = self
                            .graph
                            .neighbors_directed(src, Direction::Outgoing)
                            .count();
                        if src_out > 0 {
                            scores[&src] / src_out as f32
                        } else {
                            0.0
                        }
                    })
                    .sum();

                new_scores.insert(idx, (1.0 - damping) / node_count as f32 + damping * incoming_sum);
            }

            scores = new_scores;
        }

        let mut ranked: Vec<(NodeIndex, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
            .iter()
            .filter_map(|(idx, score)| self.graph.node_weight(*idx).map(|node| (node, *score)))
            .collect()
    }

    /// Get the number of files in the graph.
    pub fn file_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a file node by path.
    pub fn get_file(&self, path: &str) -> Option<&FileNode> {
        self.path_index
            .get(path)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// List all file paths in the graph.
    pub fn file_paths(&self) -> Vec<&str> {
        self.path_index.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(path: &str) -> FileNode {
        FileNode {
            path: path.to_string(),
            language: "rust".to_string(),
            symbol_count: 5,
        }
    }

    #[test]
    fn add_and_find_files() {
        let mut graph = DependencyGraph::new();
        graph.add_file(make_node("src/main.rs"));
        graph.add_file(make_node("src/lib.rs"));
        graph.add_file(make_node("src/utils.rs"));

        assert_eq!(graph.file_count(), 3);
        assert!(graph.get_file("src/main.rs").is_some());
        assert!(graph.get_file("nonexistent.rs").is_none());
    }

    #[test]
    fn import_edges_and_related() {
        let mut graph = DependencyGraph::new();
        graph.add_file(make_node("src/main.rs"));
        graph.add_file(make_node("src/lib.rs"));
        graph.add_file(make_node("src/utils.rs"));
        graph.add_file(make_node("src/deep.rs"));

        graph.add_import("src/main.rs", "src/lib.rs");
        graph.add_import("src/lib.rs", "src/utils.rs");
        graph.add_import("src/utils.rs", "src/deep.rs");

        assert_eq!(graph.edge_count(), 3);

        // Depth 1: only direct neighbors
        let related = graph.find_related("src/main.rs", 1);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].path, "src/lib.rs");

        // Depth 2: two hops
        let related = graph.find_related("src/main.rs", 2);
        assert_eq!(related.len(), 2);

        // Depth 3: full chain
        let related = graph.find_related("src/main.rs", 3);
        assert_eq!(related.len(), 3);
    }

    #[test]
    fn no_duplicate_edges() {
        let mut graph = DependencyGraph::new();
        graph.add_file(make_node("a.rs"));
        graph.add_file(make_node("b.rs"));

        graph.add_import("a.rs", "b.rs");
        graph.add_import("a.rs", "b.rs"); // duplicate
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn remove_file() {
        let mut graph = DependencyGraph::new();
        graph.add_file(make_node("a.rs"));
        graph.add_file(make_node("b.rs"));
        graph.add_import("a.rs", "b.rs");

        graph.remove_file("b.rs");
        assert_eq!(graph.file_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn pagerank_scores() {
        let mut graph = DependencyGraph::new();
        graph.add_file(make_node("hub.rs"));
        graph.add_file(make_node("a.rs"));
        graph.add_file(make_node("b.rs"));
        graph.add_file(make_node("c.rs"));

        // a, b, c all import hub → hub should have highest rank
        graph.add_import("a.rs", "hub.rs");
        graph.add_import("b.rs", "hub.rs");
        graph.add_import("c.rs", "hub.rs");

        let ranked = graph.rank_files();
        assert!(!ranked.is_empty());
        assert_eq!(ranked[0].0.path, "hub.rs");
    }
}
