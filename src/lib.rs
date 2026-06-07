// src/lib.rs
//
// uldb: Agentic AI database storage engine.
//
// Modules:
//   storage/  WAL, memtable, page store, compaction, FNV-1a
//   index/    bloom filter, fuzzy symbol matcher
//   tx/       MVCC transactions
//   query/    schema validation

pub mod storage;
pub mod index;
pub mod tx;
pub mod query;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
