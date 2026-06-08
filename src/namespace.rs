// src/namespace.rs
//
// Namespace key scoping for uldb.
//
// Every record key is prefixed with its namespace before being stored
// in the engine. This provides complete isolation between namespaces
// without any engine-level changes.
//
// Key format:
//   [8B namespace_id as big-endian u64][key_bytes...]
//
// Examples:
//   namespace_id=42, key="auth.py::AuthService"
//   stored as: \x00\x00\x00\x00\x00\x00\x00\x2a auth.py::AuthService
//
// Namespace ID = fnv1a(repo_url || "::" || commit_sha)
// This is deterministic: the same repo+commit always maps to the same ID.
//
// Scan range for one namespace:
//   start = [namespace_id_bytes][0x00...]
//   end   = [namespace_id_bytes][0xFF...]
//
// Complexity:
//   scope_key:    O(key_len)
//   unscope_key:  O(1)
//   ns_scan_range: O(1)

use crate::storage::fnv::fnv1a;

/// Number of prefix bytes (8 bytes = u64 big-endian namespace_id).
pub const NS_PREFIX_LEN: usize = 8;

/// Derive a numeric namespace ID from repo URL and commit SHA.
///
/// This is deterministic: same inputs always produce the same ID.
pub fn derive_namespace_id(repo_url: &str, commit_sha: &str) -> u64 {
    let combined = format!("{repo_url}::{commit_sha}");
    fnv1a(combined.as_bytes())
}

/// Prefix a key with the namespace ID.
///
/// [8B namespace_id_be][key_bytes]
pub fn scope_key(namespace_id: u64, key: &[u8]) -> Vec<u8> {
    let mut scoped = Vec::with_capacity(NS_PREFIX_LEN + key.len());
    scoped.extend_from_slice(&namespace_id.to_be_bytes());
    scoped.extend_from_slice(key);
    scoped
}

/// Remove the namespace prefix from a stored key.
///
/// Returns None if the key is too short to have a valid prefix.
pub fn unscope_key(scoped_key: &[u8]) -> Option<&[u8]> {
    if scoped_key.len() < NS_PREFIX_LEN {
        return None;
    }
    Some(&scoped_key[NS_PREFIX_LEN..])
}

/// Extract the namespace ID from a stored key.
pub fn namespace_id_of(scoped_key: &[u8]) -> Option<u64> {
    if scoped_key.len() < NS_PREFIX_LEN {
        return None;
    }
    Some(u64::from_be_bytes(scoped_key[..NS_PREFIX_LEN].try_into().unwrap()))
}

/// Scan range bounds for one namespace.
///
/// Returns (start, end) for an engine.scan() call that returns
/// only keys belonging to namespace_id.
pub fn ns_scan_range(namespace_id: u64) -> (Vec<u8>, Vec<u8>) {
    let prefix = namespace_id.to_be_bytes();
    let mut start = prefix.to_vec();
    let mut end = prefix.to_vec();
    start.push(0x00);
    end.push(0xFF);
    (start, end)
}

/// True if a scoped key belongs to the given namespace.
pub fn key_in_namespace(scoped_key: &[u8], namespace_id: u64) -> bool {
    namespace_id_of(scoped_key) == Some(namespace_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_and_unscope_roundtrip() {
        let ns_id = 42u64;
        let key = b"auth.py::AuthService";
        let scoped = scope_key(ns_id, key);
        assert_eq!(scoped.len(), NS_PREFIX_LEN + key.len());
        assert_eq!(unscope_key(&scoped), Some(key.as_ref()));
    }

    #[test]
    fn different_namespaces_different_prefixes() {
        let a = scope_key(1, b"key");
        let b = scope_key(2, b"key");
        assert_ne!(a, b);
        assert_eq!(unscope_key(&a), Some(b"key".as_ref()));
        assert_eq!(unscope_key(&b), Some(b"key".as_ref()));
    }

    #[test]
    fn unscope_too_short_returns_none() {
        assert_eq!(unscope_key(&[0u8; 3]), None);
        assert_eq!(unscope_key(&[]), None);
    }

    #[test]
    fn ns_scan_range_correct() {
        let ns_id = 99u64;
        let (start, end) = ns_scan_range(ns_id);
        assert_eq!(&start[..8], &ns_id.to_be_bytes());
        assert_eq!(&end[..8], &ns_id.to_be_bytes());
        assert_eq!(start[8], 0x00);
        assert_eq!(end[8], 0xFF);
    }

    #[test]
    fn key_in_namespace_correct() {
        let ns_id = 7u64;
        let key = scope_key(ns_id, b"my_key");
        assert!(key_in_namespace(&key, ns_id));
        assert!(!key_in_namespace(&key, ns_id + 1));
    }

    #[test]
    fn derive_namespace_id_deterministic() {
        let id1 = derive_namespace_id("github.com/org/repo", "abc123");
        let id2 = derive_namespace_id("github.com/org/repo", "abc123");
        assert_eq!(id1, id2);
    }

    #[test]
    fn derive_namespace_id_differs_on_commit() {
        let id1 = derive_namespace_id("github.com/org/repo", "abc123");
        let id2 = derive_namespace_id("github.com/org/repo", "def456");
        assert_ne!(id1, id2);
    }

    #[test]
    fn derive_namespace_id_differs_on_repo() {
        let id1 = derive_namespace_id("github.com/org/repo-a", "sha");
        let id2 = derive_namespace_id("github.com/org/repo-b", "sha");
        assert_ne!(id1, id2);
    }

    #[test]
    fn namespace_id_of_correct() {
        let ns_id = 12345u64;
        let scoped = scope_key(ns_id, b"key");
        assert_eq!(namespace_id_of(&scoped), Some(ns_id));
    }

    #[test]
    fn namespace_isolation_in_scan_range() {
        let ns_a = 100u64;
        let ns_b = 200u64;

        let key_a = scope_key(ns_a, b"shared_key");
        let key_b = scope_key(ns_b, b"shared_key");

        let (start_a, end_a) = ns_scan_range(ns_a);
        let (start_b, end_b) = ns_scan_range(ns_b);

        assert!(key_a >= start_a && key_a <= end_a);
        assert!(key_b >= start_b && key_b <= end_b);
        assert!(key_a < start_b); // ns_a keys come before ns_b keys
    }
}
