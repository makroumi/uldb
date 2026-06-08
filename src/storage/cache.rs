// src/storage/cache.rs
//
// HashMap-backed LRU cache for hot key acceleration.
//
// Uses HashMap for O(1) lookup and a generation counter for LRU eviction.
// On get: returns value, updates generation (marks as recently used).
// On insert when full: evicts entry with lowest generation.
//
// Complexity:
//   get:    O(1) HashMap lookup
//   insert: O(1) amortized, O(n) worst case on eviction scan
//   clear:  O(1)

use std::collections::HashMap;

/// Default cache capacity: 1024 entries.
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;

struct CacheEntry {
    value: Vec<u8>,
    generation: u64,
}

pub struct LruCache {
    entries: HashMap<Vec<u8>, CacheEntry>,
    capacity: usize,
    generation: u64,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            generation: 0,
        }
    }

    /// O(1) lookup. On hit, updates generation (marks as MRU).
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        if let Some(entry) = self.entries.get_mut(key) {
            self.generation += 1;
            entry.generation = self.generation;
            return Some(entry.value.clone());
        }
        None
    }

    /// Insert a key-value pair. Evicts LRU entry if at capacity.
    pub fn insert(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.generation += 1;

        // Update existing
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.value = value;
            entry.generation = self.generation;
            return;
        }

        // Evict LRU if at capacity
        if self.entries.len() >= self.capacity {
            // Find the entry with the lowest generation
            let mut min_gen = u64::MAX;
            let mut min_key: Option<Vec<u8>> = None;
            for (k, e) in self.entries.iter() {
                if e.generation < min_gen {
                    min_gen = e.generation;
                    min_key = Some(k.clone());
                }
            }
            if let Some(k) = min_key {
                self.entries.remove(&k);
            }
        }

        self.entries.insert(key, CacheEntry {
            value,
            generation: self.generation,
        });
    }

    /// Invalidate a specific key. O(1).
    pub fn invalidate(&mut self, key: &[u8]) {
        self.entries.remove(key);
    }

    /// Clear the entire cache. O(1).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_and_insert() {
        let mut c = LruCache::new(3);
        c.insert(b"k1".to_vec(), b"v1".to_vec());
        c.insert(b"k2".to_vec(), b"v2".to_vec());
        assert_eq!(c.get(b"k1"), Some(b"v1".to_vec()));
        assert_eq!(c.get(b"k3"), None);
    }

    #[test]
    fn evicts_lru() {
        let mut c = LruCache::new(2);
        c.insert(b"k1".to_vec(), b"v1".to_vec());
        c.insert(b"k2".to_vec(), b"v2".to_vec());
        c.insert(b"k3".to_vec(), b"v3".to_vec()); // evicts k1
        assert_eq!(c.get(b"k1"), None);
        assert_eq!(c.get(b"k2"), Some(b"v2".to_vec()));
        assert_eq!(c.get(b"k3"), Some(b"v3".to_vec()));
    }

    #[test]
    fn access_promotes_to_mru() {
        let mut c = LruCache::new(2);
        c.insert(b"k1".to_vec(), b"v1".to_vec());
        c.insert(b"k2".to_vec(), b"v2".to_vec());
        c.get(b"k1"); // promote k1 to MRU
        c.insert(b"k3".to_vec(), b"v3".to_vec()); // evicts k2 (LRU)
        assert_eq!(c.get(b"k1"), Some(b"v1".to_vec()));
        assert_eq!(c.get(b"k2"), None);
    }

    #[test]
    fn invalidate() {
        let mut c = LruCache::new(10);
        c.insert(b"k1".to_vec(), b"v1".to_vec());
        c.invalidate(b"k1");
        assert_eq!(c.get(b"k1"), None);
    }

    #[test]
    fn overwrite_existing_key() {
        let mut c = LruCache::new(10);
        c.insert(b"k1".to_vec(), b"v1".to_vec());
        c.insert(b"k1".to_vec(), b"v2".to_vec());
        assert_eq!(c.get(b"k1"), Some(b"v2".to_vec()));
        assert_eq!(c.len(), 1);
    }
}
