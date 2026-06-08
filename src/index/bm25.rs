// src/index/bm25.rs
//
// BM25 inverted index for keyword search.
//
// Proven: reference notebook Cell 2
//   BM25 lexical scoring with tokenization
//   hybrid use alongside HNSW and fuzzy matching
//
// Formula:
//   score(D,Q) = sum(idf(qi) * f(qi,D) * (k1 + 1) /
//                    (f(qi,D) + k1 * (1 - b + b * |D| / avgdl)))
//
// Defaults:
//   k1 = 1.5
//   b  = 0.75
//
// Complexity:
//   add_document: O(tokens)
//   search:       O(sum postings(qi)) for query terms qi
//   space:        O(total_term_occurrences + vocabulary)

use std::collections::HashMap;

/// One search result from the BM25 index.
#[derive(Debug, Clone)]
pub struct Bm25Result {
    /// External document key.
    pub key: Vec<u8>,
    /// BM25 relevance score. Higher = better.
    pub score: f64,
    /// 0-based rank in the result list.
    pub rank: usize,
}

/// BM25 inverted index.
///
/// Stores:
///   docs:    external key by internal doc_id
///   df:      document frequency per term
///   postings: term -> Vec<(doc_id, term_frequency_in_doc)>
///   doc_len: token count per document
pub struct Bm25Index {
    docs: Vec<Vec<u8>>,
    df: HashMap<String, u32>,
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<u32>,
    avgdl: f64,
    k1: f64,
    b: f64,
}

impl Bm25Index {
    /// Create a new BM25 index with default parameters.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            df: HashMap::new(),
            postings: HashMap::new(),
            doc_len: Vec::new(),
            avgdl: 0.0,
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Create with custom BM25 parameters.
    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            docs: Vec::new(),
            df: HashMap::new(),
            postings: HashMap::new(),
            doc_len: Vec::new(),
            avgdl: 0.0,
            k1,
            b,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.df.len()
    }

    /// Average document length.
    pub fn avgdl(&self) -> f64 {
        self.avgdl
    }

    /// Add one document to the index.
    ///
    /// key:     opaque external identifier
    /// content: raw text, tokenized internally
    ///
    /// Complexity: O(tokens)
    pub fn add_document(&mut self, key: Vec<u8>, content: &str) {
        let doc_id = self.docs.len();
        self.docs.push(key);

        let tokens = tokenize(content);
        let len = tokens.len() as u32;
        self.doc_len.push(len);

        // term frequencies within this document
        let mut tf: HashMap<String, u32> = HashMap::new();
        for tok in tokens {
            *tf.entry(tok).or_insert(0) += 1;
        }

        // update document frequency and postings
        for (term, freq) in tf {
            self.postings.entry(term.clone())
                .or_insert_with(Vec::new)
                .push((doc_id, freq));
            *self.df.entry(term).or_insert(0) += 1;
        }

        // update average document length
        let total_len: u64 = self.doc_len.iter().map(|&l| l as u64).sum();
        self.avgdl = total_len as f64 / self.doc_len.len() as f64;
    }

    /// Remove a document from the index by key.
    ///
    /// Removes the document from postings, decrements df, and
    /// recalculates avgdl. Used when a key is deleted or overwritten.
    pub fn remove_document(&mut self, key: &[u8]) {
        // Find the doc_id for this key
        let doc_id = match self.docs.iter().position(|k| k == key) {
            Some(id) => id,
            None => return, // not indexed
        };


        // Remove from postings and update df
        let mut empty_terms = Vec::new();
        for (term, entries) in self.postings.iter_mut() {
            let before = entries.len();
            entries.retain(|(id, _)| *id != doc_id);
            if entries.len() < before {
                if let Some(df) = self.df.get_mut(term) {
                    *df = df.saturating_sub(1);
                }
            }
            if entries.is_empty() {
                empty_terms.push(term.clone());
            }
        }
        for term in empty_terms {
            self.postings.remove(&term);
            self.df.remove(&term);
        }

        // Mark doc as removed (empty key so it won't match future lookups)
        self.docs[doc_id] = Vec::new();
        self.doc_len[doc_id] = 0;

        // Recalculate avgdl
        let total_len: u64 = self.doc_len.iter().map(|&l| l as u64).sum();
        let active_docs = self.docs.iter().filter(|k| !k.is_empty()).count();
        self.avgdl = if active_docs > 0 {
            total_len as f64 / active_docs as f64
        } else {
            0.0
        };
    }

        /// Search with a natural-language or keyword query.
    ///
    /// Returns top_k results sorted by score descending.
    ///
    /// Complexity: O(sum postings(qi))
    pub fn search(&self, query: &str, top_k: usize) -> Vec<Bm25Result> {
        if self.docs.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let q_terms = tokenize(query);
        if q_terms.is_empty() {
            return Vec::new();
        }

        let n_docs = self.docs.len() as f64;
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for term in q_terms {
            let df = match self.df.get(&term) {
                Some(&df) => df as f64,
                None => continue, // unseen term
            };

            // BM25 IDF
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

            let postings = &self.postings[&term];
            for &(doc_id, tf) in postings {
                let tf = tf as f64;
                let dl = self.doc_len[doc_id] as f64;

                let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl.max(1e-9));
                let score = idf * (tf * (self.k1 + 1.0)) / denom;

                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        let mut results: Vec<Bm25Result> = scores
            .into_iter()
            .map(|(doc_id, score)| Bm25Result {
                key: self.docs[doc_id].clone(),
                score,
                rank: 0,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);

        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i;
        }

        results
    }
}

/// Tokenize text for BM25.
///
/// Rules:
///   - lowercase
///   - split on non-alphanumeric boundaries
///   - split snake_case and kebab-case naturally via non-alnum rule
///   - split camelCase boundaries by inserting separators first
///
/// Example:
///   "getUserById and validate_token" -> ["get", "user", "by", "id", "and", "validate", "token"]
fn tokenize(text: &str) -> Vec<String> {
    // Insert spaces before CamelCase boundaries.
    let mut expanded = String::with_capacity(text.len() * 2);
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if i > 0 && c.is_ascii_uppercase() && chars[i - 1].is_ascii_lowercase() {
            expanded.push(' ');
        }
        expanded.push(c.to_ascii_lowercase());
    }

    expanded
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = Bm25Index::new();
        assert!(idx.search("test", 10).is_empty());
    }

    #[test]
    fn add_and_search_basic() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"doc1".to_vec(), "jwt authentication validate token");
        idx.add_document(b"doc2".to_vec(), "hash password with salt");
        idx.add_document(b"doc3".to_vec(), "validate email and phone");

        let results = idx.search("validate token", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].key, b"doc1");
    }

    #[test]
    fn more_matching_terms_scores_higher() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "validate token jwt auth");
        idx.add_document(b"b".to_vec(), "validate token");
        idx.add_document(b"c".to_vec(), "validate");

        let results = idx.search("validate token jwt", 10);
        assert_eq!(results[0].key, b"a");
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn rank_matches_position() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "foo bar baz");
        idx.add_document(b"b".to_vec(), "foo bar");
        idx.add_document(b"c".to_vec(), "foo");

        let results = idx.search("foo bar baz", 3);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i);
        }
    }

    #[test]
    fn top_k_limit() {
        let mut idx = Bm25Index::new();
        for i in 0..20u32 {
            idx.add_document(format!("doc{i}").into_bytes(), "same term");
        }
        let results = idx.search("same", 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn unseen_terms_ignored() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "hello world");
        let results = idx.search("nonexistent", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn tokenize_splits_camel_case() {
        let toks = tokenize("getUserById");
        assert_eq!(toks, vec!["get", "user", "by", "id"]);
    }

    #[test]
    fn tokenize_splits_snake_case() {
        let toks = tokenize("validate_token");
        assert_eq!(toks, vec!["validate", "token"]);
    }

    #[test]
    fn tokenize_splits_kebab_case() {
        let toks = tokenize("hash-password");
        assert_eq!(toks, vec!["hash", "password"]);
    }

    #[test]
    fn tokenize_mixed() {
        let toks = tokenize("getUserById and validate_token");
        assert_eq!(toks, vec!["get", "user", "by", "id", "and", "validate", "token"]);
    }

    #[test]
    fn avgdl_updates() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "one two");
        assert_eq!(idx.avgdl(), 2.0);
        idx.add_document(b"b".to_vec(), "one two three four");
        assert_eq!(idx.avgdl(), 3.0);
    }

    #[test]
    fn vocab_size() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "alpha beta gamma");
        idx.add_document(b"b".to_vec(), "beta gamma delta");
        assert_eq!(idx.vocab_size(), 4);
    }

    #[test]
    fn repeated_terms_increase_score() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"a".to_vec(), "jwt jwt jwt authentication");
        idx.add_document(b"b".to_vec(), "jwt authentication");

        let results = idx.search("jwt", 2);
        assert_eq!(results[0].key, b"a");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn exact_keyword_match() {
        let mut idx = Bm25Index::new();
        idx.add_document(b"doc1".to_vec(), "validate token");
        idx.add_document(b"doc2".to_vec(), "hash password");

        let results = idx.search("hash", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, b"doc2");
    }
}
