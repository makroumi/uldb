// src/tx/hamt.rs
//
// Hash Array Mapped Trie (HAMT) -- persistent, immutable key-value store.
//
// Proven: Cell 5 of reference notebook
//   get:  O(log32 N) ~= O(1) practical
//   put:  O(log32 N) copy-on-write path
//   snap: O(1) -- snapshot is just an Arc<Node> pointer
//   space: O(N) with structural sharing between versions
//
// Architecture:
//   - 32-way branching trie keyed on FNV-1a hash of the key
//   - Each node is one of:
//     Leaf(key, value)
//     Branch { bitmap: u32, children: Arc<[Arc<Node>]> }
//   - "Put" creates a new path from root to leaf, sharing all unmodified branches
//   - "Snapshot" = clone the root Arc (O(1), no copy)
//
// This gives uldb:
//   - Multiple agents read different snapshots simultaneously (no blocking)
//   - Branch = new root from a snapshot root
//   - Merge = apply branch diffs to target root
//   - Rollback = discard branch root, old root unchanged
//
// Thread safety: Arc-based. Multiple readers, single writer.

use std::sync::Arc;
use crate::storage::fnv::fnv1a;

// Number of bits consumed per trie level.
const BITS_PER_LEVEL: u32 = 5;
// Branching factor: 2^5 = 32.
const BRANCHING: usize = 1 << BITS_PER_LEVEL;
// Mask for extracting one level's index.
const MASK: u32 = (BRANCHING as u32) - 1;

// ============================================================================
// Trie nodes
// ============================================================================

/// A HAMT node is either a leaf or an internal branch.
#[derive(Clone)]
enum Node {
    /// A key-value pair at the bottom of the trie.
    Leaf {
        hash: u64,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// An internal node with up to 32 children.
    Branch {
        /// Bitmap: bit i set means child i exists.
        bitmap: u32,
        /// Dense array of present children.
        children: Arc<Vec<Arc<Node>>>,
    },
}

impl Node {
    fn new_leaf(hash: u64, key: Vec<u8>, value: Vec<u8>) -> Arc<Self> {
        Arc::new(Node::Leaf { hash, key, value })
    }

    fn new_branch(bitmap: u32, children: Vec<Arc<Node>>) -> Arc<Self> {
        Arc::new(Node::Branch {
            bitmap,
            children: Arc::new(children),
        })
    }
}

/// Bit index of a hash at a given trie level.
fn bit_index(hash: u64, level: u32) -> usize {
    ((hash >> (level * BITS_PER_LEVEL)) & MASK as u64) as usize
}

/// Position of a bit in the dense child array.
fn sparse_index(bitmap: u32, bit: usize) -> usize {
    let mask = (1u32 << bit) - 1;
    (bitmap & mask).count_ones() as usize
}

// ============================================================================
// HAMT public API
// ============================================================================

/// Persistent key-value map backed by a HAMT.
///
/// Every mutation returns a new root that shares structure with the original.
/// Old roots remain valid and unchanged -- they are snapshots.
///
/// Keys and values are raw byte slices.
#[derive(Clone)]
pub struct Hamt {
    root: Option<Arc<Node>>,
    len: usize,
}

impl Hamt {
    /// Create an empty HAMT.
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Number of key-value pairs stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up a key. O(log32 N) ~= O(1) practical.
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        let hash = fnv1a(key);
        let mut current = self.root.as_ref()?;
        let mut level = 0u32;

        loop {
            match current.as_ref() {
                Node::Leaf { key: k, value, .. } => {
                    if k == key {
                        return Some(value);
                    }
                    return None;
                }
                Node::Branch { bitmap, children } => {
                    let bit = bit_index(hash, level);
                    let flag = 1u32 << bit;
                    if bitmap & flag == 0 {
                        return None;
                    }
                    let idx = sparse_index(*bitmap, bit);
                    current = &children[idx];
                    level += 1;
                }
            }
        }
    }

    /// Insert or update a key. Returns a new HAMT root. O(log32 N).
    ///
    /// The original HAMT is unchanged.
    pub fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Self {
        let hash = fnv1a(&key);
        let existed = self.get(&key).is_some();
        let new_root = insert_node(self.root.as_ref(), hash, key, value, 0);
        Self {
            root: Some(new_root),
            len: if existed { self.len } else { self.len + 1 },
        }
    }

    /// Delete a key. Returns a new HAMT root. O(log32 N).
    ///
    /// If the key does not exist, returns self unchanged.
    pub fn delete(&self, key: &[u8]) -> Self {
        if self.get(key).is_none() {
            return self.clone();
        }
        let hash = fnv1a(key);
        let new_root = remove_node(self.root.as_ref(), hash, key, 0);
        Self {
            root: new_root,
            len: self.len - 1,
        }
    }

    /// Create a snapshot. O(1) -- just clones the Arc.
    ///
    /// The snapshot and the original share all nodes.
    /// Writes to either do not affect the other.
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    /// Iterate all key-value pairs. Order is not guaranteed.
    pub fn iter(&self) -> Vec<(&[u8], &[u8])> {
        let mut result = Vec::with_capacity(self.len);
        if let Some(root) = &self.root {
            collect_leaves(root, &mut result);
        }
        result
    }

    /// Apply all entries from `other` onto `self`.
    /// Used for merging a branch back.
    pub fn merge_from(&self, other: &Hamt) -> Self {
        let mut result = self.clone();
        for (key, value) in other.iter() {
            result = result.put(key.to_vec(), value.to_vec());
        }
        result
    }
}

// ============================================================================
// Recursive insert
// ============================================================================

fn insert_node(
    node: Option<&Arc<Node>>,
    hash: u64,
    key: Vec<u8>,
    value: Vec<u8>,
    level: u32,
) -> Arc<Node> {
    match node {
        None => Node::new_leaf(hash, key, value),

        Some(n) => match n.as_ref() {
            Node::Leaf { hash: existing_hash, key: existing_key, value: existing_value } => {
                if existing_key == &key {
                    // Same key: replace value.
                    Node::new_leaf(hash, key, value)
                } else {
                    // Collision: create a branch and recurse.
                    let old_bit = bit_index(*existing_hash, level);
                    let new_bit = bit_index(hash, level);

                    if old_bit == new_bit {
                        // Still colliding at this level, go deeper.
                        let deeper = insert_node(
                            Some(n),
                            hash,
                            key,
                            value,
                            level + 1,
                        );
                        let bitmap = 1u32 << old_bit;
                        Node::new_branch(bitmap, vec![deeper])
                    } else {
                        let old_leaf = Arc::clone(n);
                        let new_leaf = Node::new_leaf(hash, key, value);
                        let (bitmap, children) = if old_bit < new_bit {
                            (
                                (1u32 << old_bit) | (1u32 << new_bit),
                                vec![old_leaf, new_leaf],
                            )
                        } else {
                            (
                                (1u32 << old_bit) | (1u32 << new_bit),
                                vec![new_leaf, old_leaf],
                            )
                        };
                        Node::new_branch(bitmap, children)
                    }
                }
            }

            Node::Branch { bitmap, children } => {
                let bit = bit_index(hash, level);
                let flag = 1u32 << bit;

                if bitmap & flag == 0 {
                    // New child slot.
                    let new_leaf = Node::new_leaf(hash, key, value);
                    let idx = sparse_index(*bitmap, bit);
                    let mut new_children = children.as_ref().clone();
                    new_children.insert(idx, new_leaf);
                    Node::new_branch(bitmap | flag, new_children)
                } else {
                    // Existing child: recurse.
                    let idx = sparse_index(*bitmap, bit);
                    let child = insert_node(
                        Some(&children[idx]),
                        hash,
                        key,
                        value,
                        level + 1,
                    );
                    let mut new_children = children.as_ref().clone();
                    new_children[idx] = child;
                    Node::new_branch(*bitmap, new_children)
                }
            }
        },
    }
}

// ============================================================================
// Recursive remove
// ============================================================================

fn remove_node(
    node: Option<&Arc<Node>>,
    hash: u64,
    key: &[u8],
    level: u32,
) -> Option<Arc<Node>> {
    let n = node?;

    match n.as_ref() {
        Node::Leaf { key: k, .. } => {
            if k == key { None } else { Some(Arc::clone(n)) }
        }

        Node::Branch { bitmap, children } => {
            let bit = bit_index(hash, level);
            let flag = 1u32 << bit;

            if bitmap & flag == 0 {
                return Some(Arc::clone(n));
            }

            let idx = sparse_index(*bitmap, bit);
            let new_child = remove_node(Some(&children[idx]), hash, key, level + 1);

            match new_child {
                None => {
                    if children.len() == 1 {
                        // Branch is now empty.
                        None
                    } else {
                        let mut new_children = children.as_ref().clone();
                        new_children.remove(idx);
                        Some(Node::new_branch(bitmap & !flag, new_children))
                    }
                }
                Some(child) => {
                    let mut new_children = children.as_ref().clone();
                    new_children[idx] = child;
                    Some(Node::new_branch(*bitmap, new_children))
                }
            }
        }
    }
}

// ============================================================================
// Leaf collection for iteration
// ============================================================================

fn collect_leaves<'a>(node: &'a Arc<Node>, out: &mut Vec<(&'a [u8], &'a [u8])>) {
    match node.as_ref() {
        Node::Leaf { key, value, .. } => {
            out.push((key, value));
        }
        Node::Branch { children, .. } => {
            for child in children.as_ref() {
                collect_leaves(child, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_get_returns_none() {
        let h = Hamt::new();
        assert_eq!(h.get(b"k"), None);
        assert!(h.is_empty());
    }

    #[test]
    fn put_and_get() {
        let h = Hamt::new()
            .put(b"key1".to_vec(), b"val1".to_vec())
            .put(b"key2".to_vec(), b"val2".to_vec());

        assert_eq!(h.get(b"key1"), Some(b"val1".as_ref()));
        assert_eq!(h.get(b"key2"), Some(b"val2".as_ref()));
        assert_eq!(h.get(b"key3"), None);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn put_overwrites() {
        let h1 = Hamt::new().put(b"k".to_vec(), b"v1".to_vec());
        let h2 = h1.put(b"k".to_vec(), b"v2".to_vec());

        assert_eq!(h1.get(b"k"), Some(b"v1".as_ref()));
        assert_eq!(h2.get(b"k"), Some(b"v2".as_ref()));
        assert_eq!(h1.len(), 1);
        assert_eq!(h2.len(), 1);
    }

    #[test]
    fn snapshot_isolation() {
        let h1 = Hamt::new().put(b"x".to_vec(), b"100".to_vec());
        let snap = h1.snapshot();
        let h2 = h1.put(b"x".to_vec(), b"999".to_vec());

        // Original snapshot unchanged.
        assert_eq!(snap.get(b"x"), Some(b"100".as_ref()));
        // New version has new value.
        assert_eq!(h2.get(b"x"), Some(b"999".as_ref()));
    }

    #[test]
    fn delete_removes_key() {
        let h = Hamt::new()
            .put(b"a".to_vec(), b"1".to_vec())
            .put(b"b".to_vec(), b"2".to_vec());

        let h2 = h.delete(b"a");
        assert_eq!(h2.get(b"a"), None);
        assert_eq!(h2.get(b"b"), Some(b"2".as_ref()));
        assert_eq!(h2.len(), 1);

        // Original unchanged.
        assert_eq!(h.get(b"a"), Some(b"1".as_ref()));
    }

    #[test]
    fn delete_nonexistent_unchanged() {
        let h = Hamt::new().put(b"k".to_vec(), b"v".to_vec());
        let h2 = h.delete(b"nonexistent");
        assert_eq!(h2.len(), h.len());
        assert_eq!(h2.get(b"k"), Some(b"v".as_ref()));
    }

    #[test]
    fn iter_returns_all_entries() {
        let mut h = Hamt::new();
        for i in 0..20u32 {
            h = h.put(format!("key_{i}").into_bytes(), format!("val_{i}").into_bytes());
        }
        let entries = h.iter();
        assert_eq!(entries.len(), 20);
        assert_eq!(h.len(), 20);
    }

    #[test]
    fn structural_sharing_snapshots_are_cheap() {
        let mut h = Hamt::new();
        for i in 0..1000u32 {
            h = h.put(format!("k{i}").into_bytes(), format!("v{i}").into_bytes());
        }
        // Snapshot is O(1): just clone the Arc.
        let snap = h.snapshot();
        assert_eq!(snap.len(), 1000);

        // Modify the live version -- snapshot is unchanged.
        let h2 = h.put(b"new_key".to_vec(), b"new_val".to_vec());
        assert_eq!(snap.get(b"new_key"), None);
        assert_eq!(h2.get(b"new_key"), Some(b"new_val".as_ref()));
        assert_eq!(h.len(), 1000);
    }

    #[test]
    fn merge_from() {
        let base = Hamt::new()
            .put(b"a".to_vec(), b"base_a".to_vec())
            .put(b"b".to_vec(), b"base_b".to_vec());

        let branch = Hamt::new()
            .put(b"b".to_vec(), b"branch_b".to_vec())
            .put(b"c".to_vec(), b"branch_c".to_vec());

        let merged = base.merge_from(&branch);
        assert_eq!(merged.get(b"a"), Some(b"base_a".as_ref()));
        assert_eq!(merged.get(b"b"), Some(b"branch_b".as_ref())); // branch wins
        assert_eq!(merged.get(b"c"), Some(b"branch_c".as_ref()));
    }

    #[test]
    fn many_keys_no_collision_errors() {
        let mut h = Hamt::new();
        for i in 0..10_000u32 {
            h = h.put(format!("key_{i:05}").into_bytes(), format!("val_{i}").into_bytes());
        }
        assert_eq!(h.len(), 10_000);
        for i in 0..10_000u32 {
            let key = format!("key_{i:05}");
            let expected = format!("val_{i}");
            assert_eq!(
                h.get(key.as_bytes()),
                Some(expected.as_bytes()),
                "key {key} not found"
            );
        }
    }

    #[test]
    fn sequential_snapshots_are_independent() {
        let h0 = Hamt::new().put(b"x".to_vec(), b"0".to_vec());
        let h1 = h0.put(b"x".to_vec(), b"1".to_vec());
        let h2 = h1.put(b"x".to_vec(), b"2".to_vec());
        let h3 = h2.put(b"x".to_vec(), b"3".to_vec());

        assert_eq!(h0.get(b"x"), Some(b"0".as_ref()));
        assert_eq!(h1.get(b"x"), Some(b"1".as_ref()));
        assert_eq!(h2.get(b"x"), Some(b"2".as_ref()));
        assert_eq!(h3.get(b"x"), Some(b"3".as_ref()));
    }
}
