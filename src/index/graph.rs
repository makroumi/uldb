// src/index/graph.rs
//
// Compressed Sparse Row (CSR) relation graph for code structure traversal.
//
// Proven: Cell 7 of reference notebook
//   5000 nodes, 15000 edges, BFS P50 0.23ms, P99 0.64ms
//
// Relation types: imports, calls, inherits, tests, defines
//
// Architecture:
//   Dynamic construction phase: dict-of-sets adjacency
//   Query phase: CSR arrays for cache-friendly traversal
//   Forward and backward adjacency for bidirectional queries
//
// Complexity:
//   add_edge:   O(1) amortized (dynamic phase)
//   build_csr:  O(V + E)
//   neighbors:  O(degree)
//   bfs:        O(V + E)
//   dfs:        O(V + E)
//   space:      O(V + E)

use std::collections::{HashMap, HashSet, VecDeque};

/// Supported relation types between code entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    Imports,
    Calls,
    Inherits,
    Tests,
    Defines,
}

impl Relation {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "imports" => Some(Relation::Imports),
            "calls" => Some(Relation::Calls),
            "inherits" => Some(Relation::Inherits),
            "tests" => Some(Relation::Tests),
            "defines" => Some(Relation::Defines),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Relation::Imports => "imports",
            Relation::Calls => "calls",
            Relation::Inherits => "inherits",
            Relation::Tests => "tests",
            Relation::Defines => "defines",
        }
    }
}

/// Edge in the relation graph.
#[derive(Debug, Clone)]
struct Edge {
    dst: usize,
    relation: Relation,
}

/// CSR graph for code relation traversal.
///
/// Construction: call add_node/add_edge, then build().
/// Query: call neighbours, bfs, dfs after build().
pub struct RelationGraph {
    // Node management
    key_to_idx: HashMap<String, usize>,
    idx_to_key: Vec<String>,

    // Dynamic adjacency (construction phase)
    fwd_adj: Vec<Vec<Edge>>,
    bwd_adj: Vec<Vec<Edge>>,

    // CSR arrays (query phase)
    fwd_ptr: Vec<usize>,
    fwd_dst: Vec<usize>,
    fwd_rel: Vec<Relation>,
    bwd_ptr: Vec<usize>,
    bwd_dst: Vec<usize>,
    bwd_rel: Vec<Relation>,
    built: bool,

    edge_count: usize,
}

/// One result from a BFS/DFS traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub key: String,
    pub depth: usize,
    pub relation: Option<Relation>,
}

impl RelationGraph {
    pub fn new() -> Self {
        Self {
            key_to_idx: HashMap::new(),
            idx_to_key: Vec::new(),
            fwd_adj: Vec::new(),
            bwd_adj: Vec::new(),
            fwd_ptr: Vec::new(),
            fwd_dst: Vec::new(),
            fwd_rel: Vec::new(),
            bwd_ptr: Vec::new(),
            bwd_dst: Vec::new(),
            bwd_rel: Vec::new(),
            built: false,
            edge_count: 0,
        }
    }

    /// Register a node. Returns its internal index. O(1) amortized.
    pub fn add_node(&mut self, key: &str) -> usize {
        if let Some(&idx) = self.key_to_idx.get(key) {
            return idx;
        }
        let idx = self.idx_to_key.len();
        self.key_to_idx.insert(key.to_string(), idx);
        self.idx_to_key.push(key.to_string());
        self.fwd_adj.push(Vec::new());
        self.bwd_adj.push(Vec::new());
        self.built = false;
        idx
    }

    /// Add a directed edge. Nodes are created if they do not exist. O(1).
    pub fn add_edge(&mut self, src: &str, dst: &str, relation: Relation) {
        let s = self.add_node(src);
        let d = self.add_node(dst);
        self.fwd_adj[s].push(Edge { dst: d, relation });
        self.bwd_adj[d].push(Edge { dst: s, relation });
        self.edge_count += 1;
        self.built = false;
    }

    /// Build CSR arrays from dynamic adjacency. O(V + E).
    /// Must be called before querying.
    pub fn build(&mut self) {
        let n = self.idx_to_key.len();

        // Forward CSR.
        self.fwd_ptr = Vec::with_capacity(n + 1);
        self.fwd_dst = Vec::with_capacity(self.edge_count);
        self.fwd_rel = Vec::with_capacity(self.edge_count);

        let mut offset = 0;
        for i in 0..n {
            self.fwd_ptr.push(offset);
            for edge in &self.fwd_adj[i] {
                self.fwd_dst.push(edge.dst);
                self.fwd_rel.push(edge.relation);
            }
            offset += self.fwd_adj[i].len();
        }
        self.fwd_ptr.push(offset);

        // Backward CSR.
        self.bwd_ptr = Vec::with_capacity(n + 1);
        self.bwd_dst = Vec::with_capacity(self.edge_count);
        self.bwd_rel = Vec::with_capacity(self.edge_count);

        offset = 0;
        for i in 0..n {
            self.bwd_ptr.push(offset);
            for edge in &self.bwd_adj[i] {
                self.bwd_dst.push(edge.dst);
                self.bwd_rel.push(edge.relation);
            }
            offset += self.bwd_adj[i].len();
        }
        self.bwd_ptr.push(offset);

        self.built = true;
    }

    fn ensure_built(&mut self) {
        if !self.built {
            self.build();
        }
    }

    /// Get forward or backward neighbours of a node.
    ///
    /// direction: "forward" or "backward"
    /// relation_filter: if Some, only return edges of that relation type
    pub fn neighbours(
        &mut self,
        key: &str,
        direction: &str,
        relation_filter: Option<Relation>,
    ) -> Vec<(String, Relation)> {
        self.ensure_built();
        let idx = match self.key_to_idx.get(key) {
            Some(&i) => i,
            None => return Vec::new(),
        };

        let (ptr, dst, rel) = if direction == "backward" {
            (&self.bwd_ptr, &self.bwd_dst, &self.bwd_rel)
        } else {
            (&self.fwd_ptr, &self.fwd_dst, &self.fwd_rel)
        };

        let start = ptr[idx];
        let end = ptr[idx + 1];
        let mut results = Vec::new();

        for i in start..end {
            if let Some(filter) = relation_filter {
                if rel[i] != filter {
                    continue;
                }
            }
            results.push((self.idx_to_key[dst[i]].clone(), rel[i]));
        }

        results
    }

    /// Breadth-first search from a start node.
    ///
    /// direction: "forward" or "backward"
    /// max_depth: maximum traversal depth
    /// relation_filter: if Some, only follow edges of that type
    ///
    /// Returns all reachable nodes with their depth and relation.
    /// Complexity: O(V + E) for the reachable subgraph.
    pub fn bfs(
        &mut self,
        start: &str,
        direction: &str,
        max_depth: usize,
        relation_filter: Option<Relation>,
    ) -> Vec<TraversalResult> {
        self.ensure_built();
        let start_idx = match self.key_to_idx.get(start) {
            Some(&i) => i,
            None => return Vec::new(),
        };

        let (ptr, dst, rel) = if direction == "backward" {
            (&self.bwd_ptr, &self.bwd_dst, &self.bwd_rel)
        } else {
            (&self.fwd_ptr, &self.fwd_dst, &self.fwd_rel)
        };

        let mut visited: HashMap<usize, usize> = HashMap::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        let mut results = Vec::new();

        visited.insert(start_idx, 0);
        queue.push_back((start_idx, 0));
        results.push(TraversalResult {
            key: self.idx_to_key[start_idx].clone(),
            depth: 0,
            relation: None,
        });

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let start_edge = ptr[node];
            let end_edge = ptr[node + 1];

            for i in start_edge..end_edge {
                if let Some(filter) = relation_filter {
                    if rel[i] != filter {
                        continue;
                    }
                }

                let nb = dst[i];
                if !visited.contains_key(&nb) {
                    visited.insert(nb, depth + 1);
                    queue.push_back((nb, depth + 1));
                    results.push(TraversalResult {
                        key: self.idx_to_key[nb].clone(),
                        depth: depth + 1,
                        relation: Some(rel[i]),
                    });
                }
            }
        }

        results
    }

    /// Depth-first search from a start node.
    pub fn dfs(
        &mut self,
        start: &str,
        direction: &str,
        max_depth: usize,
        relation_filter: Option<Relation>,
    ) -> Vec<TraversalResult> {
        self.ensure_built();
        let start_idx = match self.key_to_idx.get(start) {
            Some(&i) => i,
            None => return Vec::new(),
        };

        let (ptr, dst, rel) = if direction == "backward" {
            (&self.bwd_ptr, &self.bwd_dst, &self.bwd_rel)
        } else {
            (&self.fwd_ptr, &self.fwd_dst, &self.fwd_rel)
        };

        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(usize, usize)> = vec![(start_idx, 0)];
        let mut results = Vec::new();

        while let Some((node, depth)) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }

            results.push(TraversalResult {
                key: self.idx_to_key[node].clone(),
                depth,
                relation: None,
            });

            if depth >= max_depth {
                continue;
            }

            let start_edge = ptr[node];
            let end_edge = ptr[node + 1];

            for i in start_edge..end_edge {
                if let Some(filter) = relation_filter {
                    if rel[i] != filter {
                        continue;
                    }
                }
                if !visited.contains(&dst[i]) {
                    stack.push((dst[i], depth + 1));
                }
            }
        }

        results
    }

    pub fn node_count(&self) -> usize { self.idx_to_key.len() }
    pub fn edge_count(&self) -> usize { self.edge_count }
    pub fn is_built(&self) -> bool { self.built }

    /// Check if a node exists.
    pub fn has_node(&self, key: &str) -> bool {
        self.key_to_idx.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> RelationGraph {
        let mut g = RelationGraph::new();
        g.add_edge("api.py::ViewSet", "auth.py::AuthService", Relation::Imports);
        g.add_edge("api.py::ViewSet", "models.py::User", Relation::Imports);
        g.add_edge("auth.py::AuthService", "auth.py::validate_token", Relation::Calls);
        g.add_edge("auth.py::AuthService", "auth.py::hash_password", Relation::Calls);
        g.add_edge("models.py::User", "BaseModel", Relation::Inherits);
        g.add_edge("tests::test_auth", "auth.py::AuthService", Relation::Tests);
        g.build();
        g
    }

    #[test]
    fn node_and_edge_counts() {
        let g = sample_graph();
        assert_eq!(g.node_count(), 7);
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn forward_neighbours() {
        let mut g = sample_graph();
        let nbs = g.neighbours("auth.py::AuthService", "forward", None);
        assert_eq!(nbs.len(), 2);
        let keys: Vec<&str> = nbs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"auth.py::validate_token"));
        assert!(keys.contains(&"auth.py::hash_password"));
    }

    #[test]
    fn backward_neighbours() {
        let mut g = sample_graph();
        let nbs = g.neighbours("auth.py::AuthService", "backward", None);
        assert_eq!(nbs.len(), 2);
        let keys: Vec<&str> = nbs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"api.py::ViewSet"));
        assert!(keys.contains(&"tests::test_auth"));
    }

    #[test]
    fn filtered_neighbours() {
        let mut g = sample_graph();
        let nbs = g.neighbours(
            "auth.py::AuthService",
            "backward",
            Some(Relation::Tests),
        );
        assert_eq!(nbs.len(), 1);
        assert_eq!(nbs[0].0, "tests::test_auth");
    }

    #[test]
    fn bfs_forward() {
        let mut g = sample_graph();
        let results = g.bfs("api.py::ViewSet", "forward", 3, None);
        let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
        assert!(keys.contains(&"api.py::ViewSet"));
        assert!(keys.contains(&"auth.py::AuthService"));
        assert!(keys.contains(&"auth.py::validate_token"));
    }

    #[test]
    fn bfs_depth_limited() {
        let mut g = sample_graph();
        let results = g.bfs("api.py::ViewSet", "forward", 1, None);
        // Depth 0: ViewSet, Depth 1: AuthService + User
        assert!(results.len() <= 4);
        // validate_token is depth 2, should not appear
        let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
        assert!(!keys.contains(&"auth.py::validate_token"));
    }

    #[test]
    fn bfs_relation_filtered() {
        let mut g = sample_graph();
        let results = g.bfs(
            "api.py::ViewSet",
            "forward",
            5,
            Some(Relation::Imports),
        );
        // Only follows Imports edges
        let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
        assert!(keys.contains(&"auth.py::AuthService"));
        assert!(keys.contains(&"models.py::User"));
        // Should NOT traverse Calls edges from AuthService
        assert!(!keys.contains(&"auth.py::validate_token"));
    }

    #[test]
    fn bfs_backward() {
        let mut g = sample_graph();
        let results = g.bfs("auth.py::validate_token", "backward", 5, None);
        let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
        assert!(keys.contains(&"auth.py::AuthService"));
        assert!(keys.contains(&"api.py::ViewSet"));
    }

    #[test]
    fn dfs_forward() {
        let mut g = sample_graph();
        let results = g.dfs("api.py::ViewSet", "forward", 5, None);
        assert!(results.len() >= 3);
        assert_eq!(results[0].key, "api.py::ViewSet");
    }

    #[test]
    fn missing_node_returns_empty() {
        let mut g = sample_graph();
        assert!(g.neighbours("nonexistent", "forward", None).is_empty());
        assert!(g.bfs("nonexistent", "forward", 5, None).is_empty());
        assert!(g.dfs("nonexistent", "forward", 5, None).is_empty());
    }

    #[test]
    fn has_node() {
        let g = sample_graph();
        assert!(g.has_node("auth.py::AuthService"));
        assert!(!g.has_node("nonexistent"));
    }

    #[test]
    fn relation_from_str() {
        assert_eq!(Relation::from_str("imports"), Some(Relation::Imports));
        assert_eq!(Relation::from_str("calls"), Some(Relation::Calls));
        assert_eq!(Relation::from_str("inherits"), Some(Relation::Inherits));
        assert_eq!(Relation::from_str("tests"), Some(Relation::Tests));
        assert_eq!(Relation::from_str("defines"), Some(Relation::Defines));
        assert_eq!(Relation::from_str("unknown"), None);
    }

    #[test]
    fn auto_build_on_query() {
        let mut g = RelationGraph::new();
        g.add_edge("a", "b", Relation::Calls);
        // Do NOT call build() explicitly.
        let nbs = g.neighbours("a", "forward", None);
        assert_eq!(nbs.len(), 1);
        assert!(g.is_built());
    }

    #[test]
    fn scale_1000_nodes() {
        let mut g = RelationGraph::new();
        for i in 0..1000u32 {
            g.add_node(&format!("node_{i}"));
        }
        for i in 0..999u32 {
            g.add_edge(
                &format!("node_{i}"),
                &format!("node_{}", i + 1),
                Relation::Calls,
            );
        }
        g.build();
        assert_eq!(g.node_count(), 1000);
        assert_eq!(g.edge_count(), 999);

        let results = g.bfs("node_0", "forward", 1000, None);
        assert_eq!(results.len(), 1000);
    }
}
