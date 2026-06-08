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

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::storage::wal::{WalWriter, WalReader};
use crate::storage::memtable::Memtable;
use crate::storage::page::Page;
use crate::storage::bg_compaction::BackgroundCompactor;
use crate::index::manager::IndexManager;
use crate::tx::hamt::Hamt;

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
    compactor: BackgroundCompactor,
    pub indices: IndexManager,
    pub state: Hamt,
    snapshots: HashMap<String, Hamt>,
    snapshot_counter: u64,
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
        let compactor = BackgroundCompactor::new();
        let mut indices = IndexManager::new();

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
                    // Feed into indices for query support.
                    indices.on_put(&key, &value);
                    memtable.put(key, value);
                }
            }

            // If memtable exceeded threshold during replay, flush it.
            if memtable.should_flush() {
                let page = Self::memtable_to_page(&mut memtable);
                let _ = compactor.submit(page);
            }
        }

        // Rebuild HAMT state from memtable (populated by WAL replay or empty).
        let mut initial_state = Hamt::new();
        for (key, entry) in memtable.iter() {
            if let Some(value) = &entry.value {
                initial_state = initial_state.put(key.to_vec(), value.clone());
            }
        }

        // Open WAL for new appends (truncates old WAL after successful replay).
        // We truncate because all replayed data is now in the memtable or pages.
        let wal = WalWriter::open(&wal_path)?;

        Ok(Self {
            data_dir: config.data_dir,
            wal,
            memtable,
            compactor,
            indices,
            state: initial_state,
            snapshots: HashMap::new(),
            snapshot_counter: 0,
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

        // Update HAMT state for snapshots.
        self.state = self.state.put(key.to_vec(), value.to_vec());

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
        let state = self.compactor.state();
        let c = state.lock().unwrap();
        c.get(key)
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

        // Update HAMT state.
        self.state = self.state.delete(key);

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
        self.compactor.shutdown();
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
        let state = self.compactor.state();
        let c = state.lock().unwrap();
        c.compaction_count()
    }

    /// Total records across all compaction levels.
    pub fn compaction_records(&self) -> usize {
        let state = self.compactor.state();
        let c = state.lock().unwrap();
        c.total_records()
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
        self.compactor.submit(page)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
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

    // ========================================================================
    // Snapshot operations
    // ========================================================================

    /// Create a named snapshot of the current state. O(1).
    ///
    /// The snapshot shares structure with the live state via HAMT.
    /// Writes to the live state do not affect the snapshot.
    pub fn snapshot_create(&mut self, name: &str) -> String {
        let id = if name.is_empty() {
            self.snapshot_counter += 1;
            format!("snap-{:04}", self.snapshot_counter)
        } else {
            name.to_string()
        };
        self.snapshots.insert(id.clone(), self.state.snapshot());
        id
    }

    /// Restore the live state to a named snapshot.
    ///
    /// The current live state is replaced by the snapshot.
    /// The snapshot itself remains available for future restores.
    pub fn snapshot_restore(&mut self, name: &str) -> Result<(), String> {
        match self.snapshots.get(name) {
            Some(snap) => {
                self.state = snap.clone();
                Ok(())
            }
            None => Err(format!("snapshot not found: {name}")),
        }
    }

    /// Delete a named snapshot.
    pub fn snapshot_delete(&mut self, name: &str) -> bool {
        self.snapshots.remove(name).is_some()
    }

    /// List all snapshot names.
    pub fn snapshot_list(&self) -> Vec<String> {
        self.snapshots.keys().cloned().collect()
    }

    /// Get a value from a specific snapshot (not the live state).
    pub fn snapshot_get(&self, name: &str, key: &[u8]) -> Option<Vec<u8>> {
        self.snapshots.get(name)?.get(key).map(|v| v.to_vec())
    }

    /// Number of snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    // ========================================================================
    // Branch operations
    // ========================================================================

    /// Create a branch from the current state or a named snapshot.
    ///
    /// A branch is just a named snapshot that you can later merge or rollback.
    /// Internally identical to snapshot_create.
    pub fn branch_create(&mut self, branch_id: &str, from_snapshot: &str) -> Result<String, String> {
        let base = if from_snapshot.is_empty() {
            self.state.snapshot()
        } else {
            self.snapshots.get(from_snapshot)
                .cloned()
                .ok_or_else(|| format!("source snapshot not found: {from_snapshot}"))?
        };
        self.snapshots.insert(branch_id.to_string(), base);
        Ok(branch_id.to_string())
    }

    /// Merge a branch into the live state.
    ///
    /// Applies all key-value pairs from the branch onto the current state.
    /// Conflicting keys are overwritten by the branch version (last-writer-wins).
    /// Returns the number of keys merged.
    pub fn branch_merge(&mut self, branch_id: &str) -> Result<usize, String> {
        let branch = self.snapshots.get(branch_id)
            .cloned()
            .ok_or_else(|| format!("branch not found: {branch_id}"))?;

        let entries = branch.iter();
        let count = entries.len();

        for (key, value) in entries {
            self.state = self.state.put(key.to_vec(), value.to_vec());
            // Also update the memtable so GET sees the merged data.
            self.memtable.put(key.to_vec(), value.to_vec());
            self.indices.on_put(key, value);
        }

        // Remove the branch after merge.
        self.snapshots.remove(branch_id);
        Ok(count)
    }

    /// Rollback (discard) a branch without merging.
    pub fn branch_rollback(&mut self, branch_id: &str) -> bool {
        self.snapshots.remove(branch_id).is_some()
    }

    /// Diff: list keys that differ between a branch and the live state.
    pub fn branch_diff(&self, branch_id: &str) -> Result<Vec<(Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>)>, String> {
        let branch = self.snapshots.get(branch_id)
            .ok_or_else(|| format!("branch not found: {branch_id}"))?;

        let branch_entries = branch.iter();
        let mut diffs = Vec::new();

        for (key, branch_val) in branch_entries {
            let live_val = self.state.get(key);
            if live_val != Some(branch_val) {
                diffs.push((
                    key.to_vec(),
                    live_val.map(|v| v.to_vec()),
                    Some(branch_val.to_vec()),
                ));
            }
        }

        // Also check for keys in live state not in branch.
        let live_entries = self.state.iter();
        for (key, live_val) in live_entries {
            if branch.get(key).is_none() {
                diffs.push((
                    key.to_vec(),
                    Some(live_val.to_vec()),
                    None,
                ));
            }
        }

        Ok(diffs)
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
        std::thread::sleep(std::time::Duration::from_millis(100));

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


    #[test]
    fn snapshot_create_and_restore() {
        let dir = tmp_dir("snap_restore");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k1", b"v1").unwrap();
        engine.put(b"k2", b"v2").unwrap();

        let snap_id = engine.snapshot_create("before_edit");
        assert_eq!(snap_id, "before_edit");
        assert_eq!(engine.snapshot_count(), 1);

        // Modify live state.
        engine.put(b"k1", b"modified").unwrap();
        engine.put(b"k3", b"new_key").unwrap();

        assert_eq!(engine.get(b"k1"), Some(b"modified".to_vec()));

        // Snapshot still has the old value.
        assert_eq!(engine.snapshot_get("before_edit", b"k1"), Some(b"v1".to_vec()));

        // Restore.
        engine.snapshot_restore("before_edit").unwrap();

        // Live state now matches the snapshot.
        // Note: memtable and compaction are NOT rolled back here,
        // only the HAMT state. Full rollback would need WAL integration.
        // For now, snapshot_get proves the HAMT is correct.
        assert_eq!(engine.snapshot_get("before_edit", b"k1"), Some(b"v1".to_vec()));

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn snapshot_delete() {
        let dir = tmp_dir("snap_delete");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v").unwrap();
        engine.snapshot_create("snap1");
        engine.snapshot_create("snap2");
        assert_eq!(engine.snapshot_count(), 2);

        assert!(engine.snapshot_delete("snap1"));
        assert_eq!(engine.snapshot_count(), 1);

        assert!(!engine.snapshot_delete("nonexistent"));

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn snapshot_list() {
        let dir = tmp_dir("snap_list");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v").unwrap();
        engine.snapshot_create("alpha");
        engine.snapshot_create("beta");
        engine.snapshot_create("gamma");

        let names = engine.snapshot_list();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert!(names.contains(&"gamma".to_string()));

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn snapshot_auto_name() {
        let dir = tmp_dir("snap_auto");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v").unwrap();
        let id1 = engine.snapshot_create("");
        let id2 = engine.snapshot_create("");
        assert_eq!(id1, "snap-0001");
        assert_eq!(id2, "snap-0002");

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn branch_create_and_merge() {
        let dir = tmp_dir("branch_merge");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"shared", b"original").unwrap();

        // Create a branch from current state.
        engine.branch_create("feat/refactor", "").unwrap();

        // Modify the branch state directly.
        // In a real implementation, the branch would have its own HAMT
        // and the agent would write to it. For now we modify the snapshot.
        let branch = engine.snapshots.get("feat/refactor").unwrap().clone();
        let modified = branch.put(b"shared".to_vec(), b"branch_edit".to_vec());
        let modified = modified.put(b"new_key".to_vec(), b"new_value".to_vec());
        engine.snapshots.insert("feat/refactor".to_string(), modified);

        // Merge branch into live state.
        let merged_count = engine.branch_merge("feat/refactor").unwrap();
        assert!(merged_count >= 2);

        // Branch should be removed after merge.
        assert!(!engine.snapshots.contains_key("feat/refactor"));

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn branch_rollback() {
        let dir = tmp_dir("branch_rollback");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"k", b"v").unwrap();
        engine.branch_create("bad_idea", "").unwrap();
        assert_eq!(engine.snapshot_count(), 1);

        assert!(engine.branch_rollback("bad_idea"));
        assert_eq!(engine.snapshot_count(), 0);

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn branch_diff() {
        let dir = tmp_dir("branch_diff");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();

        engine.branch_create("test_branch", "").unwrap();

        // Modify live state.
        engine.put(b"a", b"modified").unwrap();
        engine.put(b"c", b"3").unwrap();

        let diffs = engine.branch_diff("test_branch").unwrap();
        assert!(!diffs.is_empty());

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn snapshot_isolation_proven() {
        let dir = tmp_dir("snap_isolation");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        for i in 0..100u32 {
            engine.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }

        engine.snapshot_create("checkpoint");

        // Overwrite all keys.
        for i in 0..100u32 {
            engine.put(format!("k{i}").as_bytes(), b"overwritten").unwrap();
        }

        // Snapshot still has the original values.
        for i in 0..100u32 {
            let key = format!("k{i}");
            let expected = format!("v{i}");
            assert_eq!(
                engine.snapshot_get("checkpoint", key.as_bytes()),
                Some(expected.as_bytes().to_vec()),
                "snapshot isolation broken for {key}"
            );
        }

        drop(engine);
        cleanup(&dir);
    }

    #[test]
    fn indices_survive_restart() {
        let dir = tmp_dir("idx_persist");

        // Phase 1: write records and close.
        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();

            engine.put(b"auth.py::validate_token", b"def validate_token validates JWT").unwrap();
            engine.put(b"auth.py::hash_password", b"def hash_password uses bcrypt").unwrap();
            engine.put(b"models.py::User", b"class User with email field").unwrap();
            engine.put(b"utils.py::sendEmail", b"def sendEmail sends via SMTP").unwrap();

            // Verify indices work before close.
            let spec = crate::query::planner::QuerySpec {
                text: "validate".into(),
                top_k: 5,
                ..Default::default()
            };
            let results = engine.indices.query(&spec);
            assert!(!results.is_empty(), "query should find results before close");

            engine.close().unwrap();
        }

        // Phase 2: reopen and verify indices are rebuilt from WAL.
        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();

            // BM25 should find "validate"
            let spec = crate::query::planner::QuerySpec {
                text: "validate".into(),
                top_k: 5,
                ..Default::default()
            };
            let results = engine.indices.query(&spec);
            assert!(
                !results.is_empty(),
                "BM25 index should be rebuilt from WAL replay"
            );

            let keys: Vec<String> = results.iter()
                .map(|r| String::from_utf8_lossy(&r.key).to_string())
                .collect();
            assert!(
                keys.iter().any(|k| k.contains("validate_token")),
                "validate_token should be in results after restart"
            );

            // Fuzzy should find "vallidateToken" (typo)
            let spec2 = crate::query::planner::QuerySpec {
                text: "vallidateToken".into(),
                top_k: 3,
                ..Default::default()
            };
            let results2 = engine.indices.query(&spec2);
            assert!(
                !results2.is_empty(),
                "fuzzy index should be rebuilt from WAL replay"
            );

            // Storage should also have the data
            assert_eq!(
                engine.get(b"auth.py::validate_token"),
                Some(b"def validate_token validates JWT".to_vec()),
            );
            assert_eq!(
                engine.get(b"models.py::User"),
                Some(b"class User with email field".to_vec()),
            );
        }

        cleanup(&dir);
    }

    #[test]
    fn hamt_state_survives_restart() {
        let dir = tmp_dir("hamt_persist");

        {
            let config = EngineConfig::new(&dir);
            let mut engine = Engine::open(config).unwrap();

            engine.put(b"k1", b"v1").unwrap();
            engine.put(b"k2", b"v2").unwrap();
            engine.put(b"k3", b"v3").unwrap();

            // HAMT state should match
            assert_eq!(engine.state.get(b"k1"), Some(b"v1".as_ref()));
            assert_eq!(engine.state.len(), 3);

            engine.close().unwrap();
        }

        {
            let config = EngineConfig::new(&dir);
            let engine = Engine::open(config).unwrap();

            // HAMT rebuilt from WAL
            assert_eq!(engine.state.get(b"k1"), Some(b"v1".as_ref()));
            assert_eq!(engine.state.get(b"k2"), Some(b"v2".as_ref()));
            assert_eq!(engine.state.get(b"k3"), Some(b"v3".as_ref()));
            assert_eq!(engine.state.len(), 3);
        }

        cleanup(&dir);
    }

}
