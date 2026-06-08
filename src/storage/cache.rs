// src/storage/cache.rs
//
// Simple bounded LRU cache for hot key acceleration.
//
// Uses a Vec of (key, value) entries with LRU eviction.
// On hit: entry is moved to the back (most recently used).
// On miss: caller fetches from storage and calls insert().
// On insert when full: front entry (least recently used) is evicted.
//
// This is intentionally simple. For production scale (>10K entries),
// replace with a proper hash-indexed LRU. For typical agentic workloads
// where the working set is small (hundreds of symbols), Vec-based LRU
// is cache-line friendly and fast.
//
// Complexity:
//   get:    O(n) scan, n <= capacity (typically 1024)
//   insert: O(n) for eviction shift
//   clear:  O(1)
//
// Thread safety: NOT thread-safe. Engine wraps in RwLock externally.

/// Default cache capacity: 1024 entries.
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;

pub struct LruCache {
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    capacity: usize,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(4096)),
            capacity,
        }
    }

    /// Look up a key. On hit, moves the entry to the back (MRU position).
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            let entry = self.entries.remove(pos);
            let value = entry.1.clone();
            self.entries.push(entry);
            Some(value)
        } else {
            None
        }
    }

    /// Insert a key-value pair. Evicts the LRU entry if at capacity.
    pub fn insert(&mut self, key: Vec<u8>, value: Vec<u8>) {
        // Remove existing entry for this key if present
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(pos);
        }

        // Evict LRU (front) if at capacity
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }

        self.entries.push((key, value));
    }

    /// Invalidate a specific key.
    pub fn invalidate(&mut self, key: &[u8]) {
        self.entries.retain(|(k, _)| k != key);
    }

    /// Clear the entire cache.
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
