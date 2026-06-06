// src/index/fuzzy.rs
//
// Fuzzy symbol matcher using trigram index and Levenshtein edit distance.
//
// Validated: Cell 8 (100% accuracy on 10 typo/variant queries)
//            Cell 18 (100% accuracy in final benchmark)
//
// Two-phase approach:
//   Phase 1: Trigram Jaccard similarity for candidate filtering
//   Phase 2: Levenshtein edit distance for precise ranking
//
// Normalization: case-insensitive, separator-insensitive
//   getUserById    -> getuserbyid
//   get_user_by_id -> getuserbyid
//   GET-USER-BY-ID -> getuserbyid
//
// Complexity:
//   add:   O(|symbol| * trigram_count)
//   query: O(|query_trigrams| * bucket_size + candidates * |s| * |t|)

use std::collections::HashMap;

/// Normalize a symbol name: lowercase, strip separators.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '_' | '-' | ' ' => {} // skip separators
            c => out.push(c.to_ascii_lowercase()),
        }
    }
    out
}

/// Extract character trigrams from a string, with edge padding.
/// "hello" -> {"##h", "#he", "hel", "ell", "llo", "lo#", "o##"}
fn trigrams(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let padded = format!("##{s}##");
    let chars: Vec<char> = padded.chars().collect();
    (0..chars.len().saturating_sub(2))
        .map(|i| chars[i..i + 3].iter().collect())
        .collect()
}

/// Bounded Levenshtein edit distance. Returns max_dist+1 if exceeded.
fn levenshtein(a: &str, b: &str, max_dist: usize) -> usize {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    if b.len() - a.len() > max_dist {
        return max_dist + 1;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let la = a_chars.len();
    let lb = b_chars.len();

    let mut prev: Vec<usize> = (0..=lb).collect();

    for i in 0..la {
        let mut curr = vec![i + 1; lb + 1];
        let mut row_min = curr[0];

        for j in 0..lb {
            let cost = if a_chars[i] == b_chars[j] { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
            row_min = row_min.min(curr[j + 1]);
        }

        if row_min > max_dist {
            return max_dist + 1;
        }
        prev = curr;
    }

    prev[lb]
}

/// A match result from a fuzzy query.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub symbol: String,
    pub distance: usize,
    pub jaccard: f64,
}

/// Trigram-indexed fuzzy symbol matcher.
pub struct FuzzyMatcher {
    symbols: Vec<String>,           // original symbol names
    normalized: Vec<String>,        // normalized forms
    tri_counts: Vec<usize>,         // trigram count per symbol
    index: HashMap<String, Vec<usize>>,  // trigram -> [symbol indices]
    max_distance: usize,
}

impl FuzzyMatcher {
    pub fn new(max_distance: usize) -> Self {
        Self {
            symbols: Vec::new(),
            normalized: Vec::new(),
            tri_counts: Vec::new(),
            index: HashMap::new(),
            max_distance,
        }
    }

    /// Add a symbol to the index.
    pub fn add(&mut self, symbol: &str) {
        let idx = self.symbols.len();
        let norm = normalize(symbol);
        let trigs = trigrams(&norm);
        let tri_count = trigs.len();

        self.symbols.push(symbol.to_string());
        self.normalized.push(norm);
        self.tri_counts.push(tri_count);

        // Deduplicate trigrams for this symbol to avoid inflating hit counts
        let mut seen = std::collections::HashSet::new();
        for tg in trigs {
            if seen.insert(tg.clone()) {
                self.index.entry(tg).or_default().push(idx);
            }
        }
    }

    /// Query for the top-k closest symbols to the input.
    pub fn query(&self, q: &str, top_k: usize) -> Vec<FuzzyMatch> {
        self.query_with_max(q, top_k, self.max_distance)
    }

    /// Query with a custom max edit distance.
    pub fn query_with_max(
        &self,
        q: &str,
        top_k: usize,
        max_distance: usize,
    ) -> Vec<FuzzyMatch> {
        let q_norm = normalize(q);
        let q_trigs = trigrams(&q_norm);
        if q_trigs.is_empty() {
            return Vec::new();
        }

        // Deduplicate query trigrams
        let mut q_trig_set = std::collections::HashSet::new();
        for tg in &q_trigs {
            q_trig_set.insert(tg.as_str());
        }
        let q_tri_n = q_trig_set.len();

        // Phase 1: count trigram hits per candidate
        let mut hit_count: HashMap<usize, usize> = HashMap::new();
        for tg in &q_trig_set {
            if let Some(indices) = self.index.get(*tg) {
                for &idx in indices {
                    *hit_count.entry(idx).or_default() += 1;
                }
            }
        }

        if hit_count.is_empty() {
            return Vec::new();
        }

        // Score by Jaccard similarity, sort descending
        let mut candidates: Vec<(f64, usize)> = hit_count
            .iter()
            .map(|(&idx, &hits)| {
                let union = q_tri_n + self.tri_counts[idx] - hits;
                let jaccard = if union > 0 {
                    hits as f64 / union as f64
                } else {
                    0.0
                };
                (jaccard, idx)
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        let candidate_limit = top_k.max(50) * 4;
        candidates.truncate(candidate_limit);

        // Phase 2: Levenshtein ranking
        let mut results: Vec<(usize, bool, f64, usize)> = Vec::new();

        for (jaccard, idx) in candidates {
            let dist = levenshtein(&q_norm, &self.normalized[idx], max_distance);
            if dist <= max_distance {
                let raw_exact = self.symbols[idx] == q;
                results.push((dist, !raw_exact, jaccard, idx));
            }
        }

        // Sort: distance ASC, raw_exact first, jaccard DESC, index ASC
        results.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(
                    b.2.partial_cmp(&a.2)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.3.cmp(&b.3))
        });

        results.truncate(top_k);

        results
            .into_iter()
            .map(|(dist, _, jaccard, idx)| FuzzyMatch {
                symbol: self.symbols[idx].clone(),
                distance: dist,
                jaccard,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_matcher() -> FuzzyMatcher {
        let mut fm = FuzzyMatcher::new(4);
        let symbols = [
            "AuthService",
            "validate_token",
            "hash_password",
            "UserViewSet",
            "ArticleViewSet",
            "validateEmail",
            "sendEmail",
            "formatDate",
            "hashPassword",
            "getUserById",
            "createUserProfile",
            "deleteUserProfile",
            "connectDB",
            "disconnectDB",
            "queryDB",
            "logInfo",
            "logWarning",
            "logError",
            "cacheGet",
            "cacheSet",
            "cacheDel",
            "encryptAES",
            "decryptAES",
            "generateToken",
            "verifyToken",
        ];
        for sym in &symbols {
            fm.add(sym);
        }
        fm
    }

    #[test]
    fn exact_match() {
        let fm = build_matcher();
        let results = fm.query("getUserById", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].symbol, "getUserById");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn case_insensitive() {
        let fm = build_matcher();
        let results = fm.query("GETUSERBYID", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn separator_insensitive() {
        let fm = build_matcher();
        let results = fm.query("get_user_by_id", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn typo_tolerance() {
        let fm = build_matcher();

        let cases = [
            ("AuthServce", "AuthService"),
            ("vallidate_token", "validate_token"),
            ("sendEmal", "sendEmail"),
            ("connetDB", "connectDB"),
            ("logWarnig", "logWarning"),
            ("generateTokn", "generateToken"),
        ];

        for (query, expected) in &cases {
            let results = fm.query(query, 3);
            assert!(
                results.iter().any(|r| r.symbol == *expected),
                "query '{query}' should find '{expected}', got {:?}",
                results.iter().map(|r| &r.symbol).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn garbage_returns_nothing() {
        let fm = build_matcher();
        let results = fm.query_with_max(
            "xyzzy_not_a_symbol_at_all_xyz",
            3,
            2,
        );
        assert!(
            results.is_empty(),
            "garbage query should return empty at max_dist=2"
        );
    }
}
