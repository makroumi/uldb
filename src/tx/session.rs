// src/tx/session.rs
//
// Transaction session management.
//
// Each active transaction has:
//   - a unique tx_id
//   - an isolation level
//   - a HAMT snapshot of the state at begin time
//   - a write buffer (pending puts and deletes)
//   - a read set (for serializable conflict detection)
//
// On commit:
//   - serializable: check read set for conflicts
//   - all levels: apply write buffer to engine atomically
//
// On rollback:
//   - discard write buffer
//   - remove from active transactions
//
// Complexity:
//   begin:    O(1) -- snapshot the HAMT root
//   get:      O(log32 N) -- check write buffer, then snapshot
//   put:      O(1) -- buffer the write
//   commit:   O(W) where W = write buffer size
//   rollback: O(1) -- drop the buffer

use std::collections::HashMap;
use crate::tx::hamt::Hamt;

/// Isolation level for a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxIsolation {
    ReadCommitted,
    Snapshot,
    Serializable,
}

impl TxIsolation {
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x01 => TxIsolation::ReadCommitted,
            0x02 => TxIsolation::Snapshot,
            0x03 => TxIsolation::Serializable,
            _ => TxIsolation::Snapshot,
        }
    }
}

/// One buffered write operation within a transaction.
#[derive(Debug, Clone)]
pub enum TxOp {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
}

/// An active transaction session.
pub struct TxSession {
    pub tx_id: u64,
    pub isolation: TxIsolation,
    /// Snapshot of HAMT state at begin time.
    /// Reads go here (for snapshot/serializable isolation).
    pub snapshot: Hamt,
    /// Buffered writes. Applied on commit, discarded on rollback.
    pub write_buffer: Vec<TxOp>,
    /// Keys read during this transaction (for serializable conflict detection).
    pub read_set: Vec<Vec<u8>>,
    /// Timestamp when the transaction began.
    pub begin_ts: u64,
}

impl TxSession {
    pub fn new(tx_id: u64, isolation: TxIsolation, snapshot: Hamt, begin_ts: u64) -> Self {
        Self {
            tx_id,
            isolation,
            snapshot,
            write_buffer: Vec::new(),
            read_set: Vec::new(),
            begin_ts,
        }
    }

    /// Read a key within this transaction.
    ///
    /// Checks write buffer first (read-your-own-writes),
    /// then falls back to the snapshot.
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        // Check write buffer in reverse order (latest write wins).
        for op in self.write_buffer.iter().rev() {
            match op {
                TxOp::Put(k, v) if k == key => return Some(v.clone()),
                TxOp::Delete(k) if k == key => return None,
                _ => {}
            }
        }

        // Record in read set for serializable conflict detection.
        if self.isolation == TxIsolation::Serializable {
            self.read_set.push(key.to_vec());
        }

        // Read from snapshot.
        self.snapshot.get(key).map(|v| v.to_vec())
    }

    /// Buffer a write within this transaction.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.write_buffer.push(TxOp::Put(key, value));
    }

    /// Buffer a delete within this transaction.
    pub fn delete(&mut self, key: Vec<u8>) {
        self.write_buffer.push(TxOp::Delete(key));
    }

    /// Number of buffered operations.
    pub fn buffer_len(&self) -> usize {
        self.write_buffer.len()
    }
}

/// Manages all active transaction sessions.
pub struct TxManager {
    sessions: HashMap<u64, TxSession>,
    next_id: u64,
    next_ts: u64,
}

impl TxManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 1,
            next_ts: 1,
        }
    }

    /// Begin a new transaction. Returns the tx_id.
    pub fn begin(&mut self, isolation: TxIsolation, snapshot: Hamt) -> u64 {
        let tx_id = self.next_id;
        self.next_id += 1;
        let ts = self.next_ts;
        self.next_ts += 1;

        let session = TxSession::new(tx_id, isolation, snapshot, ts);
        self.sessions.insert(tx_id, session);
        tx_id
    }

    /// Get a mutable reference to a transaction session.
    pub fn get_mut(&mut self, tx_id: u64) -> Option<&mut TxSession> {
        self.sessions.get_mut(&tx_id)
    }

    /// Get a reference to a transaction session.
    pub fn get(&self, tx_id: u64) -> Option<&TxSession> {
        self.sessions.get(&tx_id)
    }

    /// Remove and return a transaction session (for commit or rollback).
    pub fn remove(&mut self, tx_id: u64) -> Option<TxSession> {
        self.sessions.remove(&tx_id)
    }

    /// Number of active transactions.
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }

    /// Check if a transaction exists.
    pub fn exists(&self, tx_id: u64) -> bool {
        self.sessions.contains_key(&tx_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_hamt() -> Hamt {
        Hamt::new()
    }

    fn seeded_hamt() -> Hamt {
        Hamt::new()
            .put(b"k1".to_vec(), b"v1".to_vec())
            .put(b"k2".to_vec(), b"v2".to_vec())
            .put(b"k3".to_vec(), b"v3".to_vec())
    }

    #[test]
    fn begin_assigns_unique_ids() {
        let mut mgr = TxManager::new();
        let id1 = mgr.begin(TxIsolation::Snapshot, empty_hamt());
        let id2 = mgr.begin(TxIsolation::Snapshot, empty_hamt());
        assert_ne!(id1, id2);
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn read_your_own_writes() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        let tx = mgr.get_mut(tx_id).unwrap();
        tx.put(b"k1".to_vec(), b"modified".to_vec());

        // Should see the buffered write, not the snapshot.
        assert_eq!(tx.get(b"k1"), Some(b"modified".to_vec()));
        // Unmodified key reads from snapshot.
        assert_eq!(tx.get(b"k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn buffered_delete() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        let tx = mgr.get_mut(tx_id).unwrap();
        tx.delete(b"k1".to_vec());

        // Should see deletion.
        assert_eq!(tx.get(b"k1"), None);
        // Other keys unaffected.
        assert_eq!(tx.get(b"k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn rollback_discards_writes() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        {
            let tx = mgr.get_mut(tx_id).unwrap();
            tx.put(b"k1".to_vec(), b"should_be_discarded".to_vec());
            tx.put(b"new_key".to_vec(), b"new_val".to_vec());
        }

        // Rollback: remove the session.
        let removed = mgr.remove(tx_id);
        assert!(removed.is_some());
        assert!(!mgr.exists(tx_id));
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn commit_returns_write_buffer() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        {
            let tx = mgr.get_mut(tx_id).unwrap();
            tx.put(b"k1".to_vec(), b"new_v1".to_vec());
            tx.delete(b"k3".to_vec());
        }

        let session = mgr.remove(tx_id).unwrap();
        assert_eq!(session.buffer_len(), 2);

        // Apply the writes to a HAMT to simulate commit.
        let mut state = session.snapshot.clone();
        for op in &session.write_buffer {
            match op {
                TxOp::Put(k, v) => { state = state.put(k.clone(), v.clone()); }
                TxOp::Delete(k) => { state = state.delete(k); }
            }
        }

        assert_eq!(state.get(b"k1"), Some(b"new_v1".as_ref()));
        assert_eq!(state.get(b"k2"), Some(b"v2".as_ref()));
        assert_eq!(state.get(b"k3"), None);
    }

    #[test]
    fn serializable_tracks_read_set() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Serializable, seeded_hamt());

        let tx = mgr.get_mut(tx_id).unwrap();
        tx.get(b"k1");
        tx.get(b"k2");
        tx.get(b"k3");

        assert_eq!(tx.read_set.len(), 3);
    }

    #[test]
    fn snapshot_does_not_track_reads() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        let tx = mgr.get_mut(tx_id).unwrap();
        tx.get(b"k1");
        tx.get(b"k2");

        assert!(tx.read_set.is_empty());
    }

    #[test]
    fn multiple_writes_last_wins() {
        let mut mgr = TxManager::new();
        let tx_id = mgr.begin(TxIsolation::Snapshot, seeded_hamt());

        let tx = mgr.get_mut(tx_id).unwrap();
        tx.put(b"k1".to_vec(), b"first".to_vec());
        tx.put(b"k1".to_vec(), b"second".to_vec());
        tx.put(b"k1".to_vec(), b"third".to_vec());

        assert_eq!(tx.get(b"k1"), Some(b"third".to_vec()));
    }

    #[test]
    fn isolation_from_byte() {
        assert_eq!(TxIsolation::from_byte(0x01), TxIsolation::ReadCommitted);
        assert_eq!(TxIsolation::from_byte(0x02), TxIsolation::Snapshot);
        assert_eq!(TxIsolation::from_byte(0x03), TxIsolation::Serializable);
        assert_eq!(TxIsolation::from_byte(0xFF), TxIsolation::Snapshot);
    }
}
