// src/index/mod.rs
//
// Index subsystem.
//
// bloom:  probabilistic membership filter
// fuzzy:  trigram + Levenshtein symbol search
// hnsw:   approximate nearest neighbor vector search

pub mod bloom;
pub mod fuzzy;
pub mod hnsw;
