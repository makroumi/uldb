// src/storage/mod.rs
//
// Storage subsystem: hashing, WAL, memtable, page store, compaction.
//
// Data flow:
//
//   write --> WAL (durability) --> Memtable (hot reads)
//                                      |
//                                      v  (flush at threshold)
//                                 Page Store (compressed, sorted)
//                                      |
//                                      v  (background)
//                                 Compaction (merge, tombstone GC)

pub mod fnv;
pub mod wal;
pub mod memtable;
pub mod page;
pub mod compaction;
