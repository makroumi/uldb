// src/storage/memtable.rs
//
// In-memory sorted key-value store backed by a BTreeMap.
//
// Validated: Cell 3
//   put: 363,766 ops/sec (Python) -- Rust target: 10M+
//   get: 1,212,281 ops/sec (Python) -- Rust target: 30M+
//
// Architecture:
//   BTreeMap gives us O(log n) put/get and sorted iteration for free.
//   Flush threshold triggers compaction to page store.
//   Tombstones (None values) represent deletes.
//
// Thread safety: external lock required. Memtable itself is single-threaded.
// The Engine wraps it in a RwLock.

use std::collections::BTreeMap;

/// A single memtable entry. None value means tombstone (deleted).
#[derive(Debug, Clone)]
pub struct MemEntry {
    pub value: Option<Vec<u8>>,
    pub seq: u64,
}

/// Sorted in-memory buffer. Flushed to pages when size exceeds threshold.
pub struct Memtable {
    entries: BTreeMap<Vec<u8>, MemEntry>,
    size_bytes: usize,
    next_seq: u64,
    flush_threshold: usize,
}

impl Memtable {
    /// Create a new memtable with the given flush threshold in bytes.
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            size_bytes: 0,
            next_seq: 0,
            flush_threshold,
        }
    }

    /// Insert or update a key. Returns true if flush threshold exceeded.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let entry_size = key.len() + value.len() + 8; // 8 for seq

        // If overwriting, subtract old size
        if let Some(old) = self.entries.get(&key) {
            let old_size = key.len()
                + old.value.as_ref().map_or(0, |v| v.len())
                + 8;
            self.size_bytes = self.size_bytes.saturating_sub(old_size);
        }

        self.next_seq += 1;
        self.entries.insert(
            key,
            MemEntry {
                value: Some(value),
                seq: self.next_seq,
            },
        );
        self.size_bytes += entry_size;
        self.should_flush()
    }

    /// Mark a key as deleted (tombstone).
    pub fn delete(&mut self, key: Vec<u8>) -> bool {
        let entry_size = key.len() + 8;

        if let Some(old) = self.entries.get(&key) {
            let old_size = key.len()
                + old.value.as_ref().map_or(0, |v| v.len())
                + 8;
            self.size_bytes = self.size_bytes.saturating_sub(old_size);
        }

        self.next_seq += 1;
        self.entries.insert(
            key,
            MemEntry {
                value: None,
                seq: self.next_seq,
            },
        );
        self.size_bytes += entry_size;
        self.should_flush()
    }

    /// Get the value for a key. Returns None for missing or tombstoned keys.
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.entries
            .get(key)
            .and_then(|e| e.value.as_deref())
    }

    /// Check if a key exists (even as tombstone).
    pub fn contains(&self, key: &[u8]) -> bool {
        self.entries.contains_key(key)
    }

    /// Check if a key is tombstoned.
    pub fn is_tombstone(&self, key: &[u8]) -> bool {
        self.entries
            .get(key)
            .map_or(false, |e| e.value.is_none())
    }

    /// Iterate all entries in sorted key order (for flushing to pages).
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &MemEntry)> {
        self.entries.iter().map(|(k, v)| (k.as_slice(), v))
    }

    /// Range scan: returns entries in [start, end) sorted order.
    pub fn range(
        &self,
        start: &[u8],
        end: &[u8],
    ) -> impl Iterator<Item = (&[u8], &MemEntry)> {
        use std::ops::Bound;
        self.entries
            .range::<Vec<u8>, _>((
                Bound::Included(start.to_vec()),
                Bound::Excluded(end.to_vec()),
            ))
            .map(|(k, v)| (k.as_slice(), v))
    }

    /// Drain all entries for flushing. Resets the memtable.
    pub fn drain(&mut self) -> Vec<(Vec<u8>, MemEntry)> {
        self.size_bytes = 0;
        self.next_seq = 0;
        std::mem::take(&mut self.entries)
            .into_iter()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn should_flush(&self) -> bool {
        self.size_bytes >= self.flush_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_roundtrip() {
        let mut mt = Memtable::new(1_000_000);
        mt.put(b"key1".to_vec(), b"val1".to_vec());
        assert_eq!(mt.get(b"key1"), Some(b"val1".as_ref()));
        assert_eq!(mt.get(b"key2"), None);
    }

    #[test]
    fn overwrite() {
        let mut mt = Memtable::new(1_000_000);
        mt.put(b"k".to_vec(), b"v1".to_vec());
        mt.put(b"k".to_vec(), b"v2".to_vec());
        assert_eq!(mt.get(b"k"), Some(b"v2".as_ref()));
        assert_eq!(mt.len(), 1);
    }

    #[test]
    fn delete_tombstone() {
        let mut mt = Memtable::new(1_000_000);
        mt.put(b"k".to_vec(), b"v".to_vec());
        mt.delete(b"k".to_vec());
        assert_eq!(mt.get(b"k"), None);
        assert!(mt.is_tombstone(b"k"));
        assert!(mt.contains(b"k"));
    }

    #[test]
    fn sorted_iteration() {
        let mut mt = Memtable::new(1_000_000);
        mt.put(b"c".to_vec(), b"3".to_vec());
        mt.put(b"a".to_vec(), b"1".to_vec());
        mt.put(b"b".to_vec(), b"2".to_vec());
        let keys: Vec<&[u8]> = mt.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b"a".as_ref(), b"b", b"c"]);
    }

    #[test]
    fn flush_threshold() {
        let mut mt = Memtable::new(100);
        for i in 0..100u32 {
            let key = format!("key_{i:05}");
            let val = format!("val_{i:05}");
            if mt.put(key.into_bytes(), val.into_bytes()) {
                // Should trigger after enough data
                assert!(mt.size_bytes() >= 100);
                return;
            }
        }
        panic!("flush threshold never triggered");
    }

    #[test]
    fn drain_resets() {
        let mut mt = Memtable::new(1_000_000);
        mt.put(b"k".to_vec(), b"v".to_vec());
        let drained = mt.drain();
        assert_eq!(drained.len(), 1);
        assert!(mt.is_empty());
        assert_eq!(mt.size_bytes(), 0);
    }

    #[test]
    fn range_scan() {
        let mut mt = Memtable::new(1_000_000);
        for i in 0..10u32 {
            let key = format!("key_{i:02}");
            mt.put(key.into_bytes(), b"v".to_vec());
        }
        let range: Vec<&[u8]> = mt
            .range(b"key_03", b"key_07")
            .map(|(k, _)| k)
            .collect();
        assert_eq!(range.len(), 4); // 03, 04, 05, 06
    }
}
