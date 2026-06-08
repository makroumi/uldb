// src/tx/mod.rs
//
// Transaction subsystem.
//
// mvcc:  Multi-Version Concurrency Control (snapshot/serializable isolation)
// hamt:  Persistent Hash Array Mapped Trie (O(1) snapshots, structural sharing)

pub mod mvcc;
pub mod hamt;
