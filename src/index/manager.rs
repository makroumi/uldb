// src/index/manager.rs
//
// IndexManager: owns and coordinates all query indices.
//
// When a record is written to the engine, the IndexManager updates
// all relevant indices. When a query arrives, the planner decides
// which indices to consult, and the manager dispatches to each.
//
// Indices owned:
//   bm25:   keyword search over record keys and values
//   fuzzy:  trigram + Levenshtein symbol lookup over keys
//   hnsw:   vector nearest neighbor (when embeddings are provided)
//   graph:  relation traversal (when edges are added)
//
// Thread safety: NOT thread-safe internally. Engine wraps in Mutex.

use crate::index::bm25::{Bm25Index, Bm25Result};
use crate::index::fuzzy::{FuzzyMatcher, FuzzyMatch};
use crate::index::hnsw::{HnswIndex, SearchResult as HnswResult};
use crate::index::graph::{RelationGraph, Relation, TraversalResult};
use crate::query::planner::{QuerySpec, QueryStep, QueryPlanner, RankedHit, merge_rrf};

/// Default HNSW vector dimension. Set to 0 to disable until first vector arrives.
const DEFAULT_DIM: usize = 0;

/// Manages all query indices for one engine instance.
pub struct IndexManager {
    pub bm25: Bm25Index,
    pub fuzzy: FuzzyMatcher,
    pub hnsw: Option<HnswIndex>,
    pub graph: RelationGraph,
    planner: QueryPlanner,
    hnsw_dim: usize,
}

impl IndexManager {
    pub fn new() -> Self {
        Self {
            bm25: Bm25Index::new(),
            fuzzy: FuzzyMatcher::new(4),
            hnsw: None,
            graph: RelationGraph::new(),
            planner: QueryPlanner::new(),
            hnsw_dim: DEFAULT_DIM,
        }
    }

    /// Called when a record is written.
    /// Updates BM25 and fuzzy indices with the key and value.
    pub fn on_put(&mut self, key: &[u8], value: &[u8]) {
        let key_str = String::from_utf8_lossy(key);

        // BM25: index key + value text
        let content = format!(
            "{} {}",
            key_str,
            String::from_utf8_lossy(value)
        );
        self.bm25.add_document(key.to_vec(), &content);

        // Fuzzy: index the key as a symbol name
        self.fuzzy.add(&key_str);
    }

    /// Called when a vector embedding is associated with a key.
    pub fn on_put_vector(&mut self, key: &[u8], vector: Vec<f32>) {
        if self.hnsw.is_none() {
            self.hnsw_dim = vector.len();
            self.hnsw = Some(HnswIndex::with_defaults(self.hnsw_dim));
        }
        if let Some(ref mut hnsw) = self.hnsw {
            let _ = hnsw.add(key.to_vec(), vector);
        }
    }

    /// Called when a relation edge is added.
    pub fn on_add_edge(&mut self, src: &str, dst: &str, relation: &str) {
        if let Some(rel) = Relation::from_str(relation) {
            self.graph.add_edge(src, dst, rel);
        }
    }

    /// Execute a full multi-index query.
    ///
    /// Uses the planner to decide which indices to hit,
    /// runs each, then merges results with RRF.
    pub fn query(&mut self, spec: &QuerySpec) -> Vec<RankedHit> {
        let plan = self.planner.plan(spec);
        let mut all_lists: Vec<Vec<RankedHit>> = Vec::new();

        for step in &plan.steps {
            match step {
                QueryStep::Bm25 => {
                    let results = self.bm25.search(&spec.text, spec.top_k * 2);
                    let hits: Vec<RankedHit> = results
                        .into_iter()
                        .enumerate()
                        .map(|(i, r)| RankedHit {
                            key: r.key,
                            score: r.score,
                            rank: i,
                        })
                        .collect();
                    if !hits.is_empty() {
                        all_lists.push(hits);
                    }
                }

                QueryStep::Fuzzy => {
                    let results = self.fuzzy.query(&spec.text, spec.top_k * 2);
                    let hits: Vec<RankedHit> = results
                        .into_iter()
                        .enumerate()
                        .map(|(i, m)| RankedHit {
                            key: m.symbol.into_bytes(),
                            score: m.jaccard,
                            rank: i,
                        })
                        .collect();
                    if !hits.is_empty() {
                        all_lists.push(hits);
                    }
                }

                QueryStep::Hnsw => {
                    if let Some(ref hnsw) = self.hnsw {
                        if !spec.vector.is_empty() {
                            let results = hnsw.search(&spec.vector, spec.top_k * 2);
                            let hits: Vec<RankedHit> = results
                                .into_iter()
                                .enumerate()
                                .map(|(i, r)| RankedHit {
                                    key: r.key,
                                    score: 1.0 - r.distance as f64,
                                    rank: i,
                                })
                                .collect();
                            if !hits.is_empty() {
                                all_lists.push(hits);
                            }
                        }
                    }
                }

                QueryStep::Graph => {
                    if !spec.text.is_empty() {
                        let relation_filter = spec.relations.first()
                            .and_then(|r| Relation::from_str(r));

                        let results = self.graph.bfs(
                            &spec.text,
                            "forward",
                            spec.max_depth,
                            relation_filter,
                        );

                        let hits: Vec<RankedHit> = results
                            .into_iter()
                            .filter(|r| r.depth > 0)
                            .enumerate()
                            .map(|(i, r)| RankedHit {
                                key: r.key.into_bytes(),
                                score: 1.0 / (1.0 + r.depth as f64),
                                rank: i,
                            })
                            .collect();
                        if !hits.is_empty() {
                            all_lists.push(hits);
                        }
                    }
                }
            }
        }

        if all_lists.is_empty() {
            return Vec::new();
        }

        let mut merged = merge_rrf(&all_lists, 60.0);
        merged.truncate(spec.top_k);
        merged
    }

    /// Stats about index sizes.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            bm25_docs: self.bm25.len(),
            bm25_vocab: self.bm25.vocab_size(),
            fuzzy_symbols: self.fuzzy.len(),
            hnsw_vectors: self.hnsw.as_ref().map(|h| h.len()).unwrap_or(0),
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
        }
    }
}

/// Summary statistics for all indices.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub bm25_docs: usize,
    pub bm25_vocab: usize,
    pub fuzzy_symbols: usize,
    pub hnsw_vectors: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_put_indexes_bm25_and_fuzzy() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"auth.py::validate_token", b"def validate_token(t): ...");
        mgr.on_put(b"auth.py::hash_password", b"def hash_password(p): ...");
        mgr.on_put(b"models.py::User", b"class User(BaseModel): ...");

        assert_eq!(mgr.bm25.len(), 3);
        assert_eq!(mgr.fuzzy.len(), 3);
    }

    #[test]
    fn bm25_query_finds_matching_docs() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"auth.py::validate_token", b"validate jwt token signature");
        mgr.on_put(b"auth.py::hash_password", b"hash password with bcrypt");
        mgr.on_put(b"models.py::User", b"user model with email field");

        let spec = QuerySpec {
            text: "validate token".into(),
            top_k: 5,
            ..Default::default()
        };
        let results = mgr.query(&spec);
        assert!(!results.is_empty());
        assert_eq!(results[0].key, b"auth.py::validate_token");
    }

    #[test]
    fn fuzzy_query_finds_typo() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"getUserById", b"function");
        mgr.on_put(b"validateEmail", b"function");
        mgr.on_put(b"hashPassword", b"function");

        let spec = QuerySpec {
            text: "getUsrById".into(),
            top_k: 3,
            ..Default::default()
        };
        let results = mgr.query(&spec);
        assert!(!results.is_empty());
        // fuzzy should find getUserById despite typo
        let keys: Vec<String> = results.iter()
            .map(|r| String::from_utf8_lossy(&r.key).to_string())
            .collect();
        assert!(keys.contains(&"getUserById".to_string()));
    }

    #[test]
    fn graph_query_follows_edges() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"AuthService", b"class");
        mgr.on_put(b"validate_token", b"function");
        mgr.on_put(b"hash_password", b"function");

        mgr.on_add_edge("AuthService", "validate_token", "calls");
        mgr.on_add_edge("AuthService", "hash_password", "calls");

        let spec = QuerySpec {
            text: "AuthService".into(),
            relations: vec!["calls".into()],
            top_k: 5,
            max_depth: 3,
            ..Default::default()
        };
        let results = mgr.query(&spec);
        let keys: Vec<String> = results.iter()
            .map(|r| String::from_utf8_lossy(&r.key).to_string())
            .collect();
        assert!(keys.contains(&"validate_token".to_string()));
        assert!(keys.contains(&"hash_password".to_string()));
    }

    #[test]
    fn multi_index_merge() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"auth.py::validate_token", b"validate jwt token auth");
        mgr.on_put(b"auth.py::hash_password", b"hash password bcrypt");
        mgr.on_put(b"models.py::User", b"user model email");

        // Text query should use both BM25 and potentially fuzzy
        let spec = QuerySpec {
            text: "validate_token".into(),
            top_k: 5,
            ..Default::default()
        };
        let results = mgr.query(&spec);
        assert!(!results.is_empty());
        // validate_token should be top result from both BM25 and fuzzy
        assert_eq!(
            String::from_utf8_lossy(&results[0].key),
            "auth.py::validate_token"
        );
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"k", b"v");
        let spec = QuerySpec::default();
        let results = mgr.query(&spec);
        assert!(results.is_empty());
    }

    #[test]
    fn stats_correct() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"a", b"x");
        mgr.on_put(b"b", b"y");
        mgr.on_add_edge("a", "b", "calls");

        let stats = mgr.stats();
        assert_eq!(stats.bm25_docs, 2);
        assert_eq!(stats.fuzzy_symbols, 2);
        assert_eq!(stats.graph_nodes, 2);
        assert_eq!(stats.graph_edges, 1);
        assert_eq!(stats.hnsw_vectors, 0);
    }

    #[test]
    fn hnsw_query_works_when_vectors_present() {
        let mut mgr = IndexManager::new();
        mgr.on_put(b"doc1", b"hello");
        mgr.on_put_vector(b"doc1", vec![1.0, 0.0, 0.0, 0.0]);
        mgr.on_put(b"doc2", b"world");
        mgr.on_put_vector(b"doc2", vec![0.0, 1.0, 0.0, 0.0]);

        let spec = QuerySpec {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            top_k: 2,
            ..Default::default()
        };
        let results = mgr.query(&spec);
        assert!(!results.is_empty());
        assert_eq!(results[0].key, b"doc1");
    }
}
