// src/index/mod.rs
//
// Index subsystem: HNSW vectors, CSR graph, fuzzy matching, bloom filter.
//
// Each index is independent and can be queried in parallel.
// The query planner decides which indices to consult.

pub mod bloom;
pub mod fuzzy;
