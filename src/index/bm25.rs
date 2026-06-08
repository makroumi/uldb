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
    key_to_doc: HashMap<Vec<u8>, usize>,
    doc_terms: Vec<Vec<String>>,
    df: HashMap<String, u32>,
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<u32>,
    total_doc_len: u64,
    active_docs: usize,
    avgdl: f64,
    k1: f64,
    b: f64,
}

impl Bm25Index {
    /// Create a new BM25 index with default parameters.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            key_to_doc: HashMap::new(),
            doc_terms: Vec::new(),
            df: HashMap::new(),
            postings: HashMap::new(),
            doc_len: Vec::new(),
            total_doc_len: 0,
            active_docs: 0,
            avgdl: 0.0,
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Create with custom BM25 parameters.
    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            docs: Vec::new(),
            key_to_doc: HashMap::new(),
            doc_terms: Vec::new(),
            df: HashMap::new(),
            postings: HashMap::new(),
            doc_len: Vec::new(),
            total_doc_len: 0,
            active_docs: 0,
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
        self.key_to_doc.insert(key.clone(), doc_id);
        self.docs.push(key);

        let tokens = tokenize(content);
        let len = tokens.len() as u32;
        self.doc_len.push(len);

        // term frequencies: reuse tokens Vec, count in-place
        let mut tf: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for tok in tokens {
            *tf.entry(tok).or_insert(0) += 1;
        }

        // Store terms and update postings in a single pass (no extra clone)
        let mut terms = Vec::with_capacity(tf.len());
        for (term, freq) in tf {
            self.postings.entry(term.clone())
                .or_default()
                .push((doc_id, freq));
            *self.df.entry(term.clone()).or_insert(0) += 1;
            terms.push(term);
        }
        self.doc_terms.push(terms);

        // update average document length incrementally O(1)
        self.total_doc_len += len as u64;
        self.active_docs += 1;
        self.avgdl = self.total_doc_len as f64 / self.active_docs as f64;
    }

    /// Remove a document from the index by key.
    ///
    /// Removes the document from postings, decrements df, and
    /// recalculates avgdl. Used when a key is deleted or overwritten.
    /// Check if a key is currently indexed. O(1).
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.key_to_doc.contains_key(key)
    }

        pub fn remove_document(&mut self, key: &[u8]) {
        // O(1) lookup via key_to_doc HashMap
        let doc_id = match self.key_to_doc.remove(key) {
            Some(id) => id,
            None => return, // not indexed
        };

        // Only touch posting lists for terms this document contains.
        // This is O(terms_in_doc) instead of O(all_terms_in_index).
        if doc_id < self.doc_terms.len() {
            let terms = std::mem::take(&mut self.doc_terms[doc_id]);
            for term in &terms {
                if let Some(entries) = self.postings.get_mut(term.as_str()) {
                    entries.retain(|(id, _)| *id != doc_id);
                    if let Some(df) = self.df.get_mut(term.as_str()) {
                        *df = df.saturating_sub(1);
                    }
                    if entries.is_empty() {
                        self.postings.remove(term.as_str());
                        self.df.remove(term.as_str());
                    }
                }
            }
        }

        // Mark doc as removed; update counters incrementally
        let removed_len = self.doc_len[doc_id] as u64;
        self.docs[doc_id] = Vec::new();
        self.doc_len[doc_id] = 0;
        self.total_doc_len = self.total_doc_len.saturating_sub(removed_len);
        self.active_docs = self.active_docs.saturating_sub(1);
        self.avgdl = if self.active_docs > 0 {
            self.total_doc_len as f64 / self.active_docs as f64
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
    // Fast tokenizer: processes bytes directly, avoids intermediate
    // String allocation for camelCase expansion.
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut in_token = false;

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        // Check for camelCase boundary: lowercase followed by uppercase
        if i > 0 && b.is_ascii_uppercase() && bytes[i - 1].is_ascii_lowercase() && in_token {
            // Emit the token before the uppercase letter
            if start < i {
                let tok = unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) };
                tokens.push(tok.to_ascii_lowercase());
            }
            start = i;
        }

        if b.is_ascii_alphanumeric() {
            if !in_token {
                start = i;
                in_token = true;
            }
        } else {
            if in_token {
                let tok = unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) };
                tokens.push(tok.to_ascii_lowercase());
                in_token = false;
            }
        }
        i += 1;
    }

    // Emit last token
    if in_token && start < bytes.len() {
        let tok = unsafe { std::str::from_utf8_unchecked(&bytes[start..]) };
        tokens.push(tok.to_ascii_lowercase());
    }

    tokens
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
