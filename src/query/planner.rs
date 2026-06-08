// src/query/planner.rs
//
// Deterministic multi-index query planner for uldb.
//
// Purpose:
//   Route one user query to the right combination of indices:
//     - BM25 keyword search
//     - HNSW vector search
//     - fuzzy symbol search
//     - CSR relation graph traversal
//
// This planner is intentionally rule-based, not learned.
// Predictability matters more than "smartness" in a database.
//
// Planning rules:
//   1. If relations are requested, include graph traversal.
//   2. If a vector is provided, include HNSW.
//   3. If text is present, include BM25.
//   4. If text looks symbol-like, include fuzzy.
//   5. If text is short and symbol-like, prioritize fuzzy first.
//   6. If text is long natural language, prioritize BM25 first.
//
// Future:
//   Metadata filters (lang/type/file) can be pushed down into index scans.
//
// Complexity:
//   plan(): O(len(text))
//   merge_rrf(): O(total_result_count)

use std::collections::HashMap;

/// One step in a query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStep {
    /// BM25 keyword search over indexed documents.
    Bm25,
    /// HNSW vector nearest-neighbor search.
    Hnsw,
    /// Trigram + Levenshtein fuzzy symbol lookup.
    Fuzzy,
    /// CSR graph traversal.
    Graph,
}

/// Internal query spec used by the planner.
///
/// This mirrors the wire-level `ulmp::messages::query::Query` shape
/// but is decoupled so the storage engine can be used independently.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySpec {
    pub text: String,
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub max_depth: usize,
    pub relations: Vec<String>,
    pub lang_filter: Vec<String>,
    pub type_filter: Vec<String>,
    pub file_filter: Vec<String>,
    pub merge_strategy: u8,
    pub timeout_ms: u32,
}

impl Default for QuerySpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            vector: Vec::new(),
            top_k: 10,
            max_depth: 3,
            relations: Vec::new(),
            lang_filter: Vec::new(),
            type_filter: Vec::new(),
            file_filter: Vec::new(),
            merge_strategy: 0x01, // RRF
            timeout_ms: 5000,
        }
    }
}

/// Planned execution order.
///
/// `steps` is ordered by execution priority.
/// For example:
///   symbol query: [Fuzzy, Bm25]
///   NL query:     [Bm25, Hnsw]
///   graph query:  [Graph, Bm25]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlan {
    pub steps: Vec<QueryStep>,
}

/// One ranked hit from an index.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    pub key: Vec<u8>,
    pub score: f64,
    pub rank: usize,
}

/// Heuristic planner.
pub struct QueryPlanner;

impl QueryPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Produce a deterministic plan from a query spec.
    ///
    /// Rules:
    ///   - Graph included if relations requested
    ///   - HNSW included if vector present
    ///   - BM25 included if text present
    ///   - Fuzzy included if text looks like a symbol
    ///
    /// Ordering:
    ///   - short symbol-like text: Fuzzy before BM25
    ///   - long natural language: BM25 before HNSW
    ///   - graph prepended if relations present
    pub fn plan(&self, q: &QuerySpec) -> QueryPlan {
        let mut steps = Vec::new();

        let has_text = !q.text.trim().is_empty();
        let has_vector = !q.vector.is_empty();
        let has_graph = !q.relations.is_empty();
        let is_symbol = looks_symbol_like(&q.text);
        let is_short = q.text.len() <= 32;

        if has_graph {
            steps.push(QueryStep::Graph);
        }

        if has_text && is_symbol && is_short {
            steps.push(QueryStep::Fuzzy);
            steps.push(QueryStep::Bm25);
        } else {
            if has_text {
                steps.push(QueryStep::Bm25);
            }
            if has_vector {
                steps.push(QueryStep::Hnsw);
            }
            if has_text && is_symbol {
                steps.push(QueryStep::Fuzzy);
            }
        }

        // If only vector exists and no text, add HNSW.
        if has_vector && !steps.contains(&QueryStep::Hnsw) {
            steps.push(QueryStep::Hnsw);
        }

        // Deduplicate while preserving order.
        let mut seen = Vec::new();
        steps.retain(|s| {
            if seen.contains(s) {
                false
            } else {
                seen.push(s.clone());
                true
            }
        });

        QueryPlan { steps }
    }
}

/// Reciprocal Rank Fusion merge.
///
/// Each input list is assumed sorted by descending quality.
/// The merged score for one key is:
///
///   RRF(key) = sum( 1 / (k + rank_i(key)) )
///
/// with rank_i starting at 1, and k usually 60.
/// This is robust to incomparable score scales between indices.
///
/// Complexity: O(total number of hits across all lists)
pub fn merge_rrf(lists: &[Vec<RankedHit>], k: f64) -> Vec<RankedHit> {
    let mut scores: HashMap<Vec<u8>, f64> = HashMap::new();

    for list in lists {
        for (i, hit) in list.iter().enumerate() {
            let rank = (i + 1) as f64;
            *scores.entry(hit.key.clone()).or_insert(0.0) += 1.0 / (k + rank);
        }
    }

    let mut merged: Vec<RankedHit> = scores
        .into_iter()
        .map(|(key, score)| RankedHit {
            key,
            score,
            rank: 0,
        })
        .collect();

    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, hit) in merged.iter_mut().enumerate() {
        hit.rank = i;
    }

    merged
}

/// Heuristic: does this text look like a code symbol?
///
/// Strong signal if:
///   - contains "::", ".", "_"
///   - has camelCase or PascalCase shape
///   - no spaces and mostly alnum/_
///
/// Complexity: O(len(text))
pub fn looks_symbol_like(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    if text.contains("::") || text.contains('.') || text.contains('_') {
        return true;
    }

    if !text.contains(' ') && text.chars().all(|c| c.is_ascii_alphanumeric()) {
        // detect camelCase / PascalCase by internal uppercase letters
        let chars: Vec<char> = text.chars().collect();
        for i in 1..chars.len() {
            if chars[i].is_ascii_uppercase() && chars[i - 1].is_ascii_lowercase() {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_plans_nothing() {
        let qp = QueryPlanner::new();
        let plan = qp.plan(&QuerySpec::default());
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn text_query_uses_bm25() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "jwt authentication validation".into(),
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(plan.steps, vec![QueryStep::Bm25]);
    }

    #[test]
    fn vector_query_uses_hnsw() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            vector: vec![0.1, 0.2, 0.3],
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(plan.steps, vec![QueryStep::Hnsw]);
    }

    #[test]
    fn symbol_query_prefers_fuzzy_then_bm25() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "getUserById".into(),
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(plan.steps, vec![QueryStep::Fuzzy, QueryStep::Bm25]);
    }

    #[test]
    fn snake_case_symbol_query_prefers_fuzzy() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "validate_token".into(),
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(plan.steps, vec![QueryStep::Fuzzy, QueryStep::Bm25]);
    }

    #[test]
    fn graph_query_prepends_graph_step() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "AuthService".into(),
            relations: vec!["calls".into()],
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(
            plan.steps,
            vec![QueryStep::Graph, QueryStep::Fuzzy, QueryStep::Bm25]
        );
    }

    #[test]
    fn graph_plus_vector_and_text() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "jwt auth".into(),
            vector: vec![0.1, 0.2],
            relations: vec!["imports".into()],
            ..Default::default()
        };
        let plan = qp.plan(&q);
        assert_eq!(
            plan.steps,
            vec![QueryStep::Graph, QueryStep::Bm25, QueryStep::Hnsw]
        );
    }

    #[test]
    fn symbol_detection() {
        assert!(looks_symbol_like("getUserById"));
        assert!(looks_symbol_like("validate_token"));
        assert!(looks_symbol_like("auth::AuthService"));
        assert!(looks_symbol_like("os.path.join"));
        assert!(!looks_symbol_like("how do i validate a token"));
        assert!(!looks_symbol_like(""));
    }

    #[test]
    fn rrf_merge_combines_lists() {
        let list1 = vec![
            RankedHit { key: b"a".to_vec(), score: 0.9, rank: 0 },
            RankedHit { key: b"b".to_vec(), score: 0.8, rank: 1 },
            RankedHit { key: b"c".to_vec(), score: 0.7, rank: 2 },
        ];
        let list2 = vec![
            RankedHit { key: b"b".to_vec(), score: 0.95, rank: 0 },
            RankedHit { key: b"a".to_vec(), score: 0.85, rank: 1 },
            RankedHit { key: b"d".to_vec(), score: 0.75, rank: 2 },
        ];

        let merged = merge_rrf(&[list1, list2], 60.0);
        // a and b should be at the top because both appear in both lists
        let top_keys: Vec<&[u8]> = merged.iter().take(2).map(|h| h.key.as_slice()).collect();
        assert!(top_keys.contains(&b"a".as_ref()));
        assert!(top_keys.contains(&b"b".as_ref()));
    }

    #[test]
    fn rrf_rank_assigned() {
        let list = vec![
            RankedHit { key: b"x".to_vec(), score: 1.0, rank: 0 },
            RankedHit { key: b"y".to_vec(), score: 0.9, rank: 1 },
        ];
        let merged = merge_rrf(&[list], 60.0);
        for (i, hit) in merged.iter().enumerate() {
            assert_eq!(hit.rank, i);
        }
    }

    #[test]
    fn no_duplicate_steps() {
        let qp = QueryPlanner::new();
        let q = QuerySpec {
            text: "getUserById".into(),
            vector: vec![0.1, 0.2],
            relations: vec!["calls".into()],
            ..Default::default()
        };
        let plan = qp.plan(&q);

        let mut seen = std::collections::HashSet::new();
        for step in plan.steps {
            assert!(seen.insert(format!("{step:?}")));
        }
    }
}
