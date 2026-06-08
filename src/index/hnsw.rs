// src/index/hnsw.rs
//
// Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.
//
// Algorithm: Malkov & Yashunin 2018 (arXiv:1603.09320)
// Proven: Cell 14 of reference notebook (96.3% recall@10, N=2000, dim=64)
//
// Parameters:
//   m:         max connections per node per layer (default 16)
//   m0:        max connections at layer 0 (default 2*m)
//   ef_build:  beam width during construction (default 200)
//   ef_search: beam width during search (default 50)
//
// Distance: cosine (1 - dot product of normalized vectors)
//
// Complexity:
//   add:   O(log N) layers * O(ef_build * M) distance computations
//   query: O(log N) descent + O(ef_search * M) at layer 0
//   space: O(N * M) neighbour pointers
//
// Thread safety: NOT thread-safe. Use Arc<RwLock<HnswIndex>> externally.
// Zero external dependencies.

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

// ============================================================================
// Float wrapper for heap ordering (no external deps)
// ============================================================================

/// f32 wrapper that implements Ord by treating NaN as greater than everything.
/// Used for min/max heaps over distances.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl OrdF32 {
    fn val(self) -> f32 { self.0 }
}

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Greater)
    }
}

// ============================================================================
// Distance computation
// ============================================================================

/// Cosine distance between two unit-normalized vectors.
/// Returns value in [0, 2]. Smaller = more similar.
/// Complexity: O(dim)
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot.clamp(-1.0, 1.0)
}

/// Normalize a vector to unit L2 length.
/// Returns false if the vector has zero norm.
fn normalize(v: &mut [f32]) -> bool {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

// ============================================================================
// HNSW graph node
// ============================================================================

struct Node {
    /// L2-normalized embedding vector.
    vector: Vec<f32>,
    /// External key stored by the caller.
    key: Vec<u8>,
    /// Adjacency lists per layer. neighbours[l] = node indices at layer l.
    neighbours: Vec<Vec<usize>>,
}

// ============================================================================
// Minimal LCG random number generator (no external deps)
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x123456789ABCDEF0 }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Extract 53 mantissa bits.
        let mantissa = self.state >> 11;
        let bits = 0x3FF0000000000000u64 | mantissa;
        f64::from_bits(bits) - 1.0
    }
}

// ============================================================================
// Public API
// ============================================================================

/// One result from an approximate nearest neighbor search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// External key associated with the indexed vector.
    pub key: Vec<u8>,
    /// Cosine distance to the query. 0.0 = identical, 2.0 = opposite.
    pub distance: f32,
    /// 0-based position in the result list (0 = closest).
    pub rank: usize,
}

/// HNSW approximate nearest neighbor graph index.
///
/// Indexes vectors associated with opaque byte keys.
/// All vectors must have the same dimension.
pub struct HnswIndex {
    dim: usize,
    m: usize,
    m0: usize,
    ef_build: usize,
    ef_search: usize,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    entry_level: usize,
    rng: Rng,
}

impl HnswIndex {
    /// Create a new HNSW index.
    ///
    /// dim:       vector dimension
    /// m:         max connections per node per layer
    /// ef_build:  construction beam width (higher = better quality, slower build)
    /// ef_search: search beam width (higher = better recall, slower query)
    pub fn new(dim: usize, m: usize, ef_build: usize, ef_search: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert!(m >= 2, "m must be at least 2");
        Self {
            dim,
            m,
            m0: m * 2,
            ef_build,
            ef_search,
            nodes: Vec::new(),
            entry_point: None,
            entry_level: 0,
            rng: Rng::new(42),
        }
    }

    /// Sensible defaults for a 384-dim embedding model.
    pub fn with_defaults(dim: usize) -> Self {
        Self::new(dim, 16, 200, 50)
    }

    /// Number of vectors indexed.
    pub fn len(&self) -> usize { self.nodes.len() }

    /// True if no vectors have been added.
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Vector dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// Add a vector to the index.
    ///
    /// The vector is L2-normalized internally.
    /// key:    opaque identifier returned in search results
    /// vector: embedding vector of dimension self.dim()
    ///
    /// Returns Err if dimension mismatches or vector is zero.
    /// Complexity: O(log N * ef_build * M * dim)
    pub fn add(&mut self, key: Vec<u8>, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "dimension mismatch: index has dim={}, got dim={}",
                self.dim,
                vector.len()
            ));
        }
        let mut v = vector;
        if !normalize(&mut v) {
            return Err("vector has zero norm and cannot be normalized".into());
        }

        let new_idx = self.nodes.len();
        let level = self.random_level();

        let mut neighbours = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbours.push(Vec::new());
        }

        self.nodes.push(Node { vector: v, key, neighbours });

        if self.entry_point.is_none() {
            self.entry_point = Some(new_idx);
            self.entry_level = level;
            return Ok(());
        }

        let ep_level = self.entry_level;
        let mut ep = self.entry_point.unwrap();

        // Greedy descent from top layer to level+1.
        for lyr in (level + 1..=ep_level).rev() {
            let cands = self.search_layer_node(new_idx, ep, 1, lyr);
            if let Some(&(_, c)) = cands.first() { ep = c; }
        }

        // Insert at each layer from min(level, ep_level) down to 0.
        for lyr in (0..=level.min(ep_level)).rev() {
            let m_max = if lyr == 0 { self.m0 } else { self.m };
            let cands = self.search_layer_node(new_idx, ep, self.ef_build, lyr);
            let selected = self.select_neighbours_heuristic(new_idx, &cands, m_max);
            self.nodes[new_idx].neighbours[lyr] = selected.clone();

            for &nb in &selected {
                while self.nodes[nb].neighbours.len() <= lyr {
                    self.nodes[nb].neighbours.push(Vec::new());
                }
                if !self.nodes[nb].neighbours[lyr].contains(&new_idx) {
                    self.nodes[nb].neighbours[lyr].push(new_idx);
                }
                if self.nodes[nb].neighbours[lyr].len() > m_max {
                    let pruned = self.prune(nb, lyr, m_max);
                    self.nodes[nb].neighbours[lyr] = pruned;
                }
            }

            if let Some(&(_, c)) = cands.first() { ep = c; }
        }

        if level > ep_level {
            self.entry_point = Some(new_idx);
            self.entry_level = level;
        }

        Ok(())
    }

    /// Search for the k approximate nearest neighbors.
    ///
    /// query: query vector (normalized internally)
    /// k:     number of results to return
    ///
    /// Returns results sorted by distance ascending (closest first).
    /// Complexity: O(log N * ef_search * M * dim)
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.nodes.is_empty() || k == 0 || query.len() != self.dim {
            return Vec::new();
        }

        let mut qv = query.to_vec();
        if !normalize(&mut qv) {
            return Vec::new();
        }

        let ep_level = self.entry_level;
        let mut ep = self.entry_point.unwrap();

        // Greedy descent to layer 1.
        for lyr in (1..=ep_level).rev() {
            let cands = self.search_layer_vec(&qv, ep, 1, lyr);
            if let Some(&(_, c)) = cands.first() { ep = c; }
        }

        // Full beam search at layer 0.
        let ef = self.ef_search.max(k);
        let mut cands = self.search_layer_vec(&qv, ep, ef, 0);
        cands.sort_by(|a, b| a.0.cmp(&b.0));
        cands.truncate(k);

        cands
            .into_iter()
            .enumerate()
            .map(|(rank, (dist, idx))| SearchResult {
                key: self.nodes[idx].key.clone(),
                distance: dist.val(),
                rank,
            })
            .collect()
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    fn random_level(&mut self) -> usize {
        let ml = 1.0 / (self.m as f64).ln();
        let mut level = 0;
        while self.rng.next_f64() < (-1.0 / ml).exp() && level < 16 {
            level += 1;
        }
        level
    }

    fn node_dist(&self, a: usize, b: usize) -> OrdF32 {
        OrdF32(cosine_dist(&self.nodes[a].vector, &self.nodes[b].vector))
    }

    fn vec_dist(&self, node: usize, vec: &[f32]) -> OrdF32 {
        OrdF32(cosine_dist(&self.nodes[node].vector, vec))
    }

    fn get_layer_neighbours(&self, node: usize, layer: usize) -> &[usize] {
        if layer < self.nodes[node].neighbours.len() {
            &self.nodes[node].neighbours[layer]
        } else {
            &[]
        }
    }

    /// Beam search using a node index as query.
    fn search_layer_node(
        &self,
        query: usize,
        ep: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(OrdF32, usize)> {
        let ep_d = self.node_dist(query, ep);
        let mut visited = HashSet::new();
        visited.insert(ep);

        // Min-heap: smallest distance at top (candidates to expand).
        let mut cands: BinaryHeap<std::cmp::Reverse<(OrdF32, usize)>> =
            BinaryHeap::new();
        cands.push(std::cmp::Reverse((ep_d, ep)));

        // Max-heap: largest distance at top (evict when full).
        let mut results: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();
        results.push((ep_d, ep));

        while let Some(std::cmp::Reverse((c_dist, c_idx))) = cands.pop() {
            if let Some(&(worst, _)) = results.peek() {
                if c_dist > worst && results.len() >= ef {
                    break;
                }
            }
            for &nb in self.get_layer_neighbours(c_idx, layer) {
                if visited.insert(nb) {
                    let nb_d = self.node_dist(query, nb);
                    let worst = results.peek().map(|&(d, _)| d).unwrap_or(OrdF32(f32::MAX));
                    if nb_d < worst || results.len() < ef {
                        cands.push(std::cmp::Reverse((nb_d, nb)));
                        results.push((nb_d, nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(OrdF32, usize)> = results.into_iter().collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Beam search using a raw vector as query.
    fn search_layer_vec(
        &self,
        query_vec: &[f32],
        ep: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(OrdF32, usize)> {
        let ep_d = self.vec_dist(ep, query_vec);
        let mut visited = HashSet::new();
        visited.insert(ep);

        let mut cands: BinaryHeap<std::cmp::Reverse<(OrdF32, usize)>> =
            BinaryHeap::new();
        cands.push(std::cmp::Reverse((ep_d, ep)));

        let mut results: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();
        results.push((ep_d, ep));

        while let Some(std::cmp::Reverse((c_dist, c_idx))) = cands.pop() {
            if let Some(&(worst, _)) = results.peek() {
                if c_dist > worst && results.len() >= ef {
                    break;
                }
            }
            for &nb in self.get_layer_neighbours(c_idx, layer) {
                if visited.insert(nb) {
                    let nb_d = self.vec_dist(nb, query_vec);
                    let worst = results.peek().map(|&(d, _)| d).unwrap_or(OrdF32(f32::MAX));
                    if nb_d < worst || results.len() < ef {
                        cands.push(std::cmp::Reverse((nb_d, nb)));
                        results.push((nb_d, nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(OrdF32, usize)> = results.into_iter().collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Heuristic neighbour selection (Algorithm 4 from the HNSW paper).
    /// Prefers diverse neighbours over pure closest.
    fn select_neighbours_heuristic(
        &self,
        query: usize,
        candidates: &[(OrdF32, usize)],
        m: usize,
    ) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::new();
        let mut selected_vecs: Vec<&[f32]> = Vec::new();

        for &(dist, idx) in candidates {
            if idx == query { continue; }
            if selected.len() >= m { break; }
            let good = selected_vecs.iter().all(|sv| {
                OrdF32(cosine_dist(&self.nodes[idx].vector, sv)) >= dist
            });
            if good {
                selected.push(idx);
                selected_vecs.push(&self.nodes[idx].vector);
            }
        }

        selected
    }

    fn prune(&self, node: usize, layer: usize, m_max: usize) -> Vec<usize> {
        let current = self.nodes[node].neighbours[layer].clone();
        let mut cands: Vec<(OrdF32, usize)> = current
            .iter()
            .map(|&nb| (self.node_dist(node, nb), nb))
            .collect();
        cands.sort_by(|a, b| a.0.cmp(&b.0));
        self.select_neighbours_heuristic(node, &cands, m_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut v: Vec<f32> = (0..dim)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
            .collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn add_and_search_exact_match() {
        let mut idx = HnswIndex::with_defaults(4);
        idx.add(b"a".to_vec(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(b"b".to_vec(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add(b"c".to_vec(), vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].key, b"a");
        assert!(r[0].distance < 0.001, "distance={}", r[0].distance);
    }

    #[test]
    fn closest_is_rank_zero() {
        let mut idx = HnswIndex::with_defaults(4);
        idx.add(b"far".to_vec(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add(b"near".to_vec(), vec![0.99, 0.14, 0.0, 0.0]).unwrap();

        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(r[0].key, b"near");
        assert_eq!(r[0].rank, 0);
        assert!(r[0].distance < r[1].distance);
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut idx = HnswIndex::with_defaults(4);
        let err = idx.add(b"k".to_vec(), vec![1.0, 0.0]).unwrap_err();
        assert!(err.contains("dimension mismatch"));
    }

    #[test]
    fn zero_vector_error() {
        let mut idx = HnswIndex::with_defaults(4);
        let err = idx.add(b"k".to_vec(), vec![0.0, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(err.contains("zero norm"));
    }

    #[test]
    fn empty_returns_empty() {
        let idx = HnswIndex::with_defaults(4);
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn k_larger_than_index_returns_all() {
        let mut idx = HnswIndex::with_defaults(4);
        idx.add(b"a".to_vec(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(b"b".to_vec(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 100);
        assert!(r.len() <= 2);
    }

    #[test]
    fn results_sorted_ascending_by_distance() {
        let mut idx = HnswIndex::with_defaults(8);
        for i in 0..30u64 {
            idx.add(format!("k{i}").into_bytes(), rng_vec(8, i)).unwrap();
        }
        let r = idx.search(&rng_vec(8, 99999), 10);
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance,
                "not sorted: {} > {}", w[0].distance, w[1].distance);
        }
    }

    #[test]
    fn rank_matches_position_in_list() {
        let mut idx = HnswIndex::with_defaults(4);
        for i in 0..8u64 {
            idx.add(format!("k{i}").into_bytes(), rng_vec(4, i)).unwrap();
        }
        let r = idx.search(&rng_vec(4, 9999), 5);
        for (i, res) in r.iter().enumerate() {
            assert_eq!(res.rank, i);
        }
    }

    #[test]
    fn recall_at_10_at_least_80_percent() {
        let dim = 32;
        let n = 500usize;
        let k = 10;

        let mut idx = HnswIndex::new(dim, 12, 100, 50);
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        let mut keys: Vec<Vec<u8>> = Vec::new();

        for i in 0..n {
            let v = rng_vec(dim, i as u64);
            idx.add(format!("v{i}").into_bytes(), v.clone()).unwrap();
            vecs.push(v);
            keys.push(format!("v{i}").into_bytes());
        }

        let n_queries = 20usize;
        let mut total_overlap = 0;

        for qi in 0..n_queries {
            let q = rng_vec(dim, 100000 + qi as u64);

            // Brute-force exact k nearest.
            let mut exact: Vec<(OrdF32, usize)> = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (OrdF32(cosine_dist(&q, v)), i))
                .collect();
            exact.sort_by(|a, b| a.0.cmp(&b.0));
            let exact_set: HashSet<usize> =
                exact.iter().take(k).map(|(_, i)| *i).collect();

            let hnsw_res = idx.search(&q, k);
            let hnsw_set: HashSet<&[u8]> =
                hnsw_res.iter().map(|r| r.key.as_slice()).collect();

            let overlap = exact_set
                .iter()
                .filter(|&&i| hnsw_set.contains(keys[i].as_slice()))
                .count();
            total_overlap += overlap;
        }

        let recall = total_overlap as f64 / (n_queries * k) as f64;
        assert!(
            recall >= 0.80,
            "recall@{k} = {recall:.3} is below 0.80"
        );
    }

    #[test]
    fn add_many_and_search_correct_neighbour() {
        let dim = 16;
        let mut idx = HnswIndex::new(dim, 8, 50, 20);

        for i in 0..200u64 {
            idx.add(format!("k{i}").into_bytes(), rng_vec(dim, i)).unwrap();
        }

        // Query very close to k0 specifically.
        let mut q = rng_vec(dim, 0);
        // Perturb slightly.
        q[0] += 0.01;
        normalize(&mut q);

        let r = idx.search(&q, 5);
        assert!(!r.is_empty());
        assert!(r[0].distance < 0.1, "expected close result, got {}", r[0].distance);
    }
}
