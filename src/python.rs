use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::storage::fnv;
use crate::storage::wal;
use crate::index::bloom::BloomFilter;
use crate::index::fuzzy::FuzzyMatcher;
use crate::tx::mvcc::{MvccStore, TxManager, Isolation};

use std::sync::Arc;

#[pyfunction]
fn fnv1a(data: &[u8]) -> u64 {
    fnv::fnv1a(data)
}

#[pyfunction]
fn fnv1a_parts(parts: Vec<Vec<u8>>) -> u64 {
    let refs: Vec<&[u8]> = parts.iter().map(|v| v.as_slice()).collect();
    fnv::fnv1a_parts(&refs)
}

#[pyfunction]
fn cosine_dist(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vectors must have same dimension",
        ));
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        Ok(1.0)
    } else {
        Ok(1.0 - dot / denom)
    }
}

#[pyfunction]
fn levenshtein(a: &str, b: &str, max_dist: usize) -> usize {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
    if b.len() - a.len() > max_dist {
        return max_dist + 1;
    }
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let la = ac.len();
    let lb = bc.len();
    let mut prev: Vec<usize> = (0..=lb).collect();
    for i in 0..la {
        let mut curr = vec![i + 1; lb + 1];
        let mut row_min = curr[0];
        for j in 0..lb {
            let cost = if ac[i] == bc[j] { 0 } else { 1 };
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

#[pyfunction]
fn wal_serialize<'py>(py: Python<'py>, key: &[u8], value: &[u8]) -> Bound<'py, PyBytes> {
    let data = wal::serialize(key, value);
    PyBytes::new(py, &data)
}

#[pyclass]
struct PyBloomFilter {
    inner: BloomFilter,
}

#[pymethods]
impl PyBloomFilter {
    #[new]
    fn new(capacity: usize, fpr: f64) -> Self {
        Self { inner: BloomFilter::new(capacity, fpr) }
    }
    fn add(&mut self, key: &[u8]) { self.inner.add(key); }
    fn may_contain(&self, key: &[u8]) -> bool { self.inner.may_contain(key) }
    fn count(&self) -> usize { self.inner.count() }
    fn size_bytes(&self) -> usize { self.inner.size_bytes() }
}

#[pyclass]
struct PyFuzzyMatcher {
    inner: FuzzyMatcher,
}

#[pymethods]
impl PyFuzzyMatcher {
    #[new]
    fn new(max_distance: usize) -> Self {
        Self { inner: FuzzyMatcher::new(max_distance) }
    }
    fn add(&mut self, symbol: &str) { self.inner.add(symbol); }
    fn query(&self, q: &str, top_k: usize) -> Vec<(String, usize, f64)> {
        self.inner.query(q, top_k)
            .into_iter()
            .map(|m| (m.symbol, m.distance, m.jaccard))
            .collect()
    }
    fn len(&self) -> usize { self.inner.len() }
}

#[pyclass]
struct PyMvccStore {
    mgr: TxManager,
}

#[pymethods]
impl PyMvccStore {
    #[new]
    fn new() -> Self {
        Self { mgr: TxManager::new(Arc::new(MvccStore::new())) }
    }
    fn put(&self, key: String, value: Vec<u8>) -> PyResult<u64> {
        let mut txn = self.mgr.begin(Isolation::Snapshot);
        txn.put(key, value);
        txn.commit().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        let store = self.mgr.store();
        store.read(key, store.current_ts())
    }
    fn version_count(&self) -> usize {
        self.mgr.store().version_count()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fnv1a, m)?)?;
    m.add_function(wrap_pyfunction!(fnv1a_parts, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_dist, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(wal_serialize, m)?)?;
    m.add_class::<PyBloomFilter>()?;
    m.add_class::<PyFuzzyMatcher>()?;
    m.add_class::<PyMvccStore>()?;
    Ok(())
}
