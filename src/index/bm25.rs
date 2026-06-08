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
    term_pool: Vec<String>,
    term_ids: HashMap<String, u32>,
    doc_terms: Vec<Vec<u32>>,
    df: HashMap<u32, u32>,
    postings: HashMap<u32, Vec<(usize, u32)>>,
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
            term_pool: Vec::new(),
            term_ids: HashMap::new(),
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
            term_pool: Vec::new(),
            term_ids: HashMap::new(),
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
        self.term_ids.len()
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

        let mut tokens = tokenize(content);
        let len = tokens.len() as u32;
        self.doc_len.push(len);

        // Sort tokens so identical terms are adjacent.
        tokens.sort_unstable();

        let mut term_ids = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let mut freq = 1u32;
            while i + (freq as usize) < tokens.len() && tokens[i + freq as usize] == tokens[i] {
                freq += 1;
            }

            // Intern the term: reuse existing ID or assign new one.
            let tid = match self.term_ids.get(&tokens[i]) {
                Some(&id) => id,
                None => {
                    let id = self.term_pool.len() as u32;
                    self.term_ids.insert(std::mem::take(&mut tokens[i]), id);
                    self.term_pool.push(String::new()); // placeholder
                    id
                }
            };

            self.postings.entry(tid)
                .or_default()
                .push((doc_id, freq));
            *self.df.entry(tid).or_insert(0) += 1;
            term_ids.push(tid);

            i += freq as usize;
        }
        self.doc_terms.push(term_ids);

        self.total_doc_len += len as u64;
        self.active_docs += 1;
        self.avgdl = self.total_doc_len as f64 / self.active_docs as f64;
    }

    /// Remove a document from the index by key.
    ///
    /// Removes the document from postings, decrements df, and
    /// recalculates avgdl. Used when a key is deleted or overwritten.
    /// Index a document by tokenizing directly into term IDs.
    /// Zero intermediate String allocation per token.
    /// Uses a reusable byte buffer for lowercasing.
    pub fn add_document_direct(&mut self, key: Vec<u8>, key_text: &str, value_text: &str) {
        let doc_id = self.docs.len();
        self.key_to_doc.insert(key.clone(), doc_id);
        self.docs.push(key);

        // Tokenize directly into term IDs. No Vec<String> intermediate.
        let mut term_freq: Vec<(u32, u32)> = Vec::new(); // (term_id, freq)
        let mut buf = Vec::with_capacity(32);
        let mut total_tokens = 0u32;

        // Process both text parts with the same inline tokenizer.
        for text in &[key_text, value_text] {
            let bytes = text.as_bytes();
            let _start = 0;
            let mut in_token = false;
            let mut i = 0;

            while i < bytes.len() {
                let b = bytes[i];

                // camelCase boundary
                if b.is_ascii_uppercase() && in_token && i > 0 && bytes[i - 1].is_ascii_lowercase() {
                    if !buf.is_empty() {
                        total_tokens += 1;
                        self.intern_and_count(&buf, &mut term_freq);
                        buf.clear();
                    }
                }

                if b.is_ascii_alphanumeric() {
                    buf.push(b.to_ascii_lowercase());
                    in_token = true;
                } else if in_token {
                    if !buf.is_empty() {
                        total_tokens += 1;
                        self.intern_and_count(&buf, &mut term_freq);
                        buf.clear();
                    }
                    in_token = false;
                }
                i += 1;
            }

            if !buf.is_empty() {
                total_tokens += 1;
                self.intern_and_count(&buf, &mut term_freq);
                buf.clear();
            }
        }

        self.doc_len.push(total_tokens);

        // Update postings and df from collected term frequencies.
        let mut term_ids = Vec::with_capacity(term_freq.len());
        for &(tid, freq) in &term_freq {
            self.postings.entry(tid).or_default().push((doc_id, freq));
            *self.df.entry(tid).or_insert(0) += 1;
            term_ids.push(tid);
        }
        self.doc_terms.push(term_ids);

        self.total_doc_len += total_tokens as u64;
        self.active_docs += 1;
        self.avgdl = self.total_doc_len as f64 / self.active_docs as f64;
    }

    /// Intern a lowercased token buffer and update term frequency counts.
    /// If the term exists, increments its count. Otherwise creates a new term ID.
    fn intern_and_count(&mut self, buf: &[u8], term_freq: &mut Vec<(u32, u32)>) {
        // Look up the term by bytes without allocating a String first.
        // SAFETY: buf contains only ASCII lowercase bytes from to_ascii_lowercase().
        let token_str = unsafe { std::str::from_utf8_unchecked(buf) };

        let tid = match self.term_ids.get(token_str) {
            Some(&id) => id,
            None => {
                let id = self.term_pool.len() as u32;
                let owned = token_str.to_string(); // single alloc for new term only
                self.term_ids.insert(owned, id);
                self.term_pool.push(String::new());
                id
            }
        };

        // Update frequency: check if this term was already seen in this doc
        if let Some(entry) = term_freq.iter_mut().find(|(t, _)| *t == tid) {
            entry.1 += 1;
        } else {
            term_freq.push((tid, 1));
        }
    }

        /// Index a document from key and value text parts directly.
    /// Avoids allocating a combined content String.
    /// Tokenizes both parts separately and merges token counts.
    pub fn add_document_parts(&mut self, key: Vec<u8>, key_text: &str, value_text: &str) {
        let doc_id = self.docs.len();
        self.key_to_doc.insert(key.clone(), doc_id);
        self.docs.push(key);

        // Tokenize both parts separately, merge into one sorted list.
        let mut tokens = tokenize(key_text);
        tokens.extend(tokenize(value_text));
        let len = tokens.len() as u32;
        self.doc_len.push(len);

        // Sort for run-based tf counting.
        tokens.sort_unstable();

        let mut term_ids = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let mut freq = 1u32;
            while i + (freq as usize) < tokens.len() && tokens[i + freq as usize] == tokens[i] {
                freq += 1;
            }

            let tid = match self.term_ids.get(&tokens[i]) {
                Some(&id) => id,
                None => {
                    let id = self.term_pool.len() as u32;
                    self.term_ids.insert(std::mem::take(&mut tokens[i]), id);
                    self.term_pool.push(String::new());
                    id
                }
            };

            self.postings.entry(tid)
                .or_default()
                .push((doc_id, freq));
            *self.df.entry(tid).or_insert(0) += 1;
            term_ids.push(tid);

            i += freq as usize;
        }
        self.doc_terms.push(term_ids);

        self.total_doc_len += len as u64;
        self.active_docs += 1;
        self.avgdl = self.total_doc_len as f64 / self.active_docs as f64;
    }

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

        // Mark doc as deleted. Do NOT scan posting lists.
        // Deleted docs are filtered at query time during search().
        // This makes delete O(1) instead of O(terms * posting_length).
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
        // doc_terms and df are left as-is. Stale postings are filtered
        // during search by checking if docs[doc_id] is empty.
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
            let tid = match self.term_ids.get(&term) {
                Some(&id) => id,
                None => continue, // unseen term
            };
            let df = match self.df.get(&tid) {
                Some(&df) => df as f64,
                None => continue,
            };

            // BM25 IDF
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

            let postings = match self.postings.get(&tid) {
                Some(p) => p,
                None => continue,
            };
            for &(doc_id, tf) in postings {
                // Skip deleted documents
                if doc_id >= self.docs.len() || self.docs[doc_id].is_empty() {
                    continue;
                }
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
    // Fast tokenizer: lowercases in-place into a reusable buffer,
    // then splits on non-alnum boundaries with camelCase detection.
    let bytes = text.as_bytes();
    let mut tokens = Vec::with_capacity(16);
    let mut buf = Vec::with_capacity(32);
    let mut in_token = false;

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        // camelCase boundary: emit current token, start new one
        if b.is_ascii_uppercase() && in_token && i > 0 && bytes[i - 1].is_ascii_lowercase() {
            if !buf.is_empty() {
                // SAFETY: buf contains only ASCII lowercase bytes
                tokens.push(unsafe { String::from_utf8_unchecked(buf.clone()) });
                buf.clear();
            }
        }

        if b.is_ascii_alphanumeric() {
            buf.push(b.to_ascii_lowercase());
            in_token = true;
        } else if in_token {
            if !buf.is_empty() {
                tokens.push(unsafe { String::from_utf8_unchecked(buf.clone()) });
                buf.clear();
            }
            in_token = false;
        }
        i += 1;
    }

    if !buf.is_empty() {
        tokens.push(unsafe { String::from_utf8_unchecked(buf) });
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
