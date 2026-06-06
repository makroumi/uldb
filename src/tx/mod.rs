// src/tx/mod.rs
//
// Transaction subsystem: MVCC, isolation levels, conflict detection.
//
// Validated: Cell 12 (full ACID), Cell 13 (multi-agent concurrency)
// Results: 0 isolation violations under 8 threads, money conservation exact

pub mod mvcc;
