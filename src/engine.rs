// src/engine.rs
//
// Storage engine: wires WAL + memtable + pages + compaction into one struct.
//
// Data flow:
//
//   put(key, value)
//     |
//     +-- WAL append (durability)
//     +-- memtable insert (fast reads)
//     |
//     +-- if memtable > flush_threshold:
//           drain memtable -> Page -> CompactionManager::add_l0_page
//
//   get(key)
//     |
//     +-- check memtable first (hot data)
//     +-- if not found: check CompactionManager (cold data)
//
//   delete(key)
//     |
//     +-- WAL append tombstone
//     +-- memtable delete (tombstone)
//
//   open(data_dir)
//     |
//     +-- replay WAL into memtable (crash recovery)
//     +-- open WAL for new appends
//
// Complexity:
//   put:    O(log N) memtable insert + O(K+V) WAL append
//   get:    O(log N) memtable lookup, fallback O(log P) page binary search
//   delete: O(log N) memtable + O(K) WAL append
//   flush:  O(N log N) sort + serialize memtable entries
//   open:   O(WAL_size) replay
//
// Thread safety:
//   Engine is NOT thread-safe. Wrap in Mutex or RwLock externally.
//   MVCC store inside is thread-safe (has internal RwLock).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::storage::wal::{WalWriter, WalReader};
use crate::storage::memtable::Memtable;
use crate::storage::page::Page;
use crate::storage::compaction::CompactionManager;
use crate::index::manager::IndexManager;

/// Default memtable flush threshold: 4MB.
const DEFAULT_FLUSH_THRESHOLD: usize = 4 * 1024 * 1024;

/// WAL file name within the data directory.
const WAL_FILENAME: &str = "wal.log";

/// Engine configuration.
pub struct EngineConfig {
    /// Data directory path. Created if it does not exist.
    pub data_dir: PathBuf,
    /// Memtable flush threshold in bytes.
    pub flush_threshold: usize,
}

impl EngineConfig {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            flush_threshold: DEFAULT_FLUSH_THRESHOLD,
        }
    }

    pub fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.flush_threshold = threshold;
        self
    }
}

/// The core storage engine.
///
/// Owns all storage components and provides a unified read/write API.
pub struct Engine {
    data_dir: PathBuf,
    wal: WalWriter,
    memtable: Memtable,
    compaction: CompactionManager,
    pub indices: IndexManager,
    flush_count: u64,
    total_puts: u64,
    total_gets: u64,
    total_deletes: u64,
}

impl Engine {
    /// Open or create a database at the given path.
    ///
    /// If the data directory contains a WAL file, it is replayed
    /// into the memtable for crash recovery.
    pub fn open(config: EngineConfig) -> io::Result<Self> {
        // Create data directory if needed.
        fs::create_dir_all(&config.data_dir)?;

        let wal_path = config.data_dir.join(WAL_FILENAME);

        // Replay existing WAL if present.
        let mut memtable = Memtable::new(config.flush_threshold);
        let mut compaction = CompactionManager::new();

        if wal_path.exists() {
            let reader = WalReader::open(&wal_path)?;
            let (records, corrupted) = reader.replay()?;

            if corrupted > 0 {
                eprintln!(
                    "[uldb] WAL recovery: {} records recovered, {} corrupted (skipped)",
                    records.len(),
                    corrupted
                );
            }

            for record in &records {
                let key = record.key.clone();
                let value = record.value.clone();

                if value.is_empty() {
                    // Empty value = tombstone (delete marker).
                    memtable.delete(key);
                } else {
                    memtable.put(key, value);
                }
            }

            // If memtable exceeded threshold during replay, flush it.
            if memtable.should_flush() {
                let page = Self::memtable_to_page(&mut memtable);
                compaction.add_l0_page(page);
            }
        }

        // Open WAL for new appends (truncates old WAL after successful replay).
        // We truncate because all replayed data is now in the memtable or pages.
        let wal = WalWriter::open(&wal_path)?;

        Ok(Self {
            data_dir: config.data_dir,
            wal,
            memtable,
            compaction,
            indices: IndexManager::new(),
            flush_count: 0,
            total_puts: 0,
            total_gets: 0,
            total_deletes: 0,
        })
    }

    /// Write one key-value pair.
    ///
    /// The write is immediately durable after WAL append.
    /// If the memtable exceeds the flush threshold, it is flushed
    /// to a page and handed to the compaction manager.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        // WAL first (durability).
        self.wal.append(key, value)?;
        self.wal.flush()?;

        // Memtable second (fast reads).
        let should_flush = self.memtable.put(key.to_vec(), value.to_vec());
        self.total_puts += 1;

        // Index for queries.
        self.indices.on_put(key, value);

        if should_flush {
            self.flush_memtable()?;
        }

        Ok(())
    }

    /// Read one key.
    ///
    /// Checks the memtable first (hot data), then falls back to
    /// compaction levels (cold data).
    ///
    /// Returns None if the key does not exist or is tombstoned.
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.total_gets += 1;

        // Check memtable first.
        if self.memtable.contains(key) {
            if self.memtable.is_tombstone(key) {
                return None; // deleted
            }
            return self.memtable.get(key).map(|v| v.to_vec());
        }

        // Fall back to compacted pages.
        self.compaction.get(key)
    }

    /// Delete one key by writing a tombstone.
    ///
    /// The deletion is durable after WAL append.
    /// Tombstones are physically removed during compaction.
    pub fn delete(&mut self, key: &[u8]) -> io::Result<()> {
        // WAL: empty value = tombstone.
        self.wal.append(key, b"")?;
        self.wal.flush()?;

        let should_flush = self.memtable.delete(key.to_vec());
        self.total_deletes += 1;

        if should_flush {
            self.flush_memtable()?;
        }

        Ok(())
    }

    /// Range scan: return all live key-value pairs in [start, end).
    ///
    /// Merges results from memtable and compaction levels.
    /// Memtable entries take precedence over page entries.
    pub fn scan(&mut self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results: std::collections::BTreeMap<Vec<u8>, Vec<u8>> =
            std::collections::BTreeMap::new();

        // First: collect from compaction levels.
        // CompactionManager does not have a range scan, so we scan pages directly.
        // For now, we only scan the memtable. This will be extended when
        // compaction gets a range scan API.

        // Memtable range scan (takes precedence).
        for (key, entry) in self.memtable.range(start, end) {
            if let Some(value) = &entry.value {
                results.insert(key.to_vec(), value.clone());
            } else {
                // Tombstone: remove from results if compaction had it.
                results.remove(key);
            }
        }

        results.into_iter().collect()
    }

    /// Force flush the memtable to a page.
    ///
    /// Called automatically when memtable exceeds threshold.
    /// Can also be called manually for testing or before shutdown.
    pub fn flush(&mut self) -> io::Result<()> {
        if !self.memtable.is_empty() {
            self.flush_memtable()?;
        }
        Ok(())
    }

    /// Sync the WAL to disk.
    pub fn sync(&mut self) -> io::Result<()> {
        self.wal.sync()
    }

    /// Close the engine cleanly.
    ///
    /// Flushes memtable, syncs WAL.
    pub fn close(mut self) -> io::Result<()> {
        self.flush()?;
        self.wal.sync()?;
        Ok(())
    }

    /// Number of records in the memtable.
    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    /// Total bytes in the memtable.
    pub fn memtable_bytes(&self) -> usize {
        self.memtable.size_bytes()
    }

    /// Number of times the memtable has been flushed to pages.
    pub fn flush_count(&self) -> u64 {
        self.flush_count
    }

    /// Total number of compactions.
    pub fn compaction_count(&self) -> u64 {
        self.compaction.compaction_count()
    }

    /// Total records across all compaction levels.
    pub fn compaction_records(&self) -> usize {
        self.compaction.total_records()
    }

    pub fn total_puts(&self) -> u64 { self.total_puts }
    pub fn total_gets(&self) -> u64 { self.total_gets }
    pub fn total_deletes(&self) -> u64 { self.total_deletes }

    /// Data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    // -- Internal helpers ----------------------------------------------------

    fn flush_memtable(&mut self) -> io::Result<()> {
        let page = Self::memtable_to_page(&mut self.memtable);
        self.compaction.add_l0_page(page);
        self.flush_count += 1;
        Ok(())
    }

    fn memtable_to_page(memtable: &mut Memtable) -> Page {
        let entries = memtable.drain();
        let mut page = Page::new(0);
        for (key, entry) in entries {
            let tombstone = entry.value.is_none();
            let value = entry.value.unwrap_or_default();
            page.push(key, value, tombstone);
        }
        page.sort();
        page
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!("uldb_engine_{name}_{}", std::process::id()));
        p
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn open_creates_data_dir() {
        let dir = tmp_dir("open_creates");
        let config = EngineConfig::new(&dir);
        let engine = Engine::open(config).unwrap();
        assert!(dir.exists());
        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn put_get_roundtrip() {
        let dir = tmp_dir("put_get");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"key1", b"value1").unwrap();
        engine.put(b"key2", b"value2").unwrap();

        assert_eq!(engine.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(engine.get(b"key2"), Some(b"value2".to_vec()));
        assert_eq!(engine.get(b"key3"), None);

        assert_eq!(engine.total_puts(), 2);
        assert_eq!(engine.total_gets(), 3);

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn overwrite() {
        let dir = tmp_dir("overwrite");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v1").unwrap();
        engine.put(b"k", b"v2").unwrap();
        assert_eq!(engine.get(b"k"), Some(b"v2".to_vec()));

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn delete_removes_key() {
        let dir = tmp_dir("delete");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v").unwrap();
        assert_eq!(engine.get(b"k"), Some(b"v".to_vec()));

        engine.delete(b"k").unwrap();
        assert_eq!(engine.get(b"k"), None);
        assert_eq!(engine.total_deletes(), 1);

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn scan_range() {
        let dir = tmp_dir("scan");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        for i in 0..10u32 {
            let key = format!("key_{i:03}");
            let val = format!("val_{i}");
            engine.put(key.as_bytes(), val.as_bytes()).unwrap();
        }

        let results = engine.scan(b"key_003", b"key_007");
        assert_eq!(results.len(), 4); // 003, 004, 005, 006
        assert_eq!(results[0].0, b"key_003");
        assert_eq!(results[3].0, b"key_006");

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn wal_crash_recovery() {
        let dir = tmp_dir("crash_recovery");

        // Phase 1: write records and drop engine (simulates crash).
        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();
            for i in 0..100u32 {
                let key = format!("key_{i:04}");
                let val = format!("val_{i}");
                engine.put(key.as_bytes(), val.as_bytes()).unwrap();
            }
            // Drop without calling close(). Simulates crash.
            // WAL is flushed per-write so data is durable.
        }

        // Phase 2: reopen and verify all records recovered.
        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();
            for i in 0..100u32 {
                let key = format!("key_{i:04}");
                let val = format!("val_{i}");
                assert_eq!(
                    engine.get(key.as_bytes()),
                    Some(val.as_bytes().to_vec()),
                    "key {key} not recovered after crash"
                );
            }
        }

        cleanup(&dir);
    }

    #[test]
    fn wal_recovery_with_deletes() {
        let dir = tmp_dir("recovery_deletes");

        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();
            engine.put(b"alive", b"yes").unwrap();
            engine.put(b"dead", b"soon").unwrap();
            engine.delete(b"dead").unwrap();
        }

        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();
            assert_eq!(engine.get(b"alive"), Some(b"yes".to_vec()));
            assert_eq!(engine.get(b"dead"), None);
        }

        cleanup(&dir);
    }

    #[test]
    fn flush_moves_data_to_compaction() {
        let dir = tmp_dir("flush_compact");
        // Small threshold to force flush.
        let config = EngineConfig::new(&dir).with_flush_threshold(100);
        let mut engine = Engine::open(config).unwrap();

        // Write enough to trigger flush.
        for i in 0..20u32 {
            let key = format!("flush_key_{i:04}");
            let val = format!("flush_val_{i}_padding_to_make_it_bigger");
            engine.put(key.as_bytes(), val.as_bytes()).unwrap();
        }

        assert!(engine.flush_count() > 0, "should have flushed");

        // Data should still be readable from compaction levels.
        assert!(engine.get(b"flush_key_0000").is_some());
        assert!(engine.get(b"flush_key_0019").is_some());

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn empty_engine_stats() {
        let dir = tmp_dir("stats");
        let config = EngineConfig::new(&dir);
        let engine = Engine::open(config).unwrap();

        assert_eq!(engine.memtable_len(), 0);
        assert_eq!(engine.memtable_bytes(), 0);
        assert_eq!(engine.flush_count(), 0);
        assert_eq!(engine.total_puts(), 0);
        assert_eq!(engine.total_gets(), 0);
        assert_eq!(engine.total_deletes(), 0);

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn reopen_empty_db() {
        let dir = tmp_dir("reopen_empty");

        {
            let config = EngineConfig::new(&dir);
            let _engine = Engine::open(config).unwrap();
        }

        {
            let config = EngineConfig::new(&dir);
            let engine = Engine::open(config).unwrap();
            assert_eq!(engine.memtable_len(), 0);
        }

        cleanup(&dir);
    }

    #[test]
    fn large_values() {
        let dir = tmp_dir("large_values");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        let big_val = vec![0xABu8; 1_000_000]; // 1MB value
        engine.put(b"big_key", &big_val).unwrap();

        let got = engine.get(b"big_key").unwrap();
        assert_eq!(got.len(), 1_000_000);
        assert_eq!(got[0], 0xAB);
        assert_eq!(got[999_999], 0xAB);

        drop(engine);
        cleanup(&dir);
    }
}
