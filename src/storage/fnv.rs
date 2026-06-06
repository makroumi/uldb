// src/storage/fnv.rs
//
// FNV-1a 64-bit hash function.
//
// Validated: Cell 1 (0 collisions at 10K keys)
// Projected: Cell 15 (3,152x speedup over Python)
//
// Usage:
//   Namespace key:  fnv1a(repo_url + commit_sha)
//   Record key:     fnv1a(namespace + node_type + qualified_name)
//
// Complexity: O(n) where n = byte length of input
// Collision:  follows birthday paradox at 2^32 keys for 64-bit output

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Compute FNV-1a hash of a byte slice.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Compute FNV-1a hash of multiple byte slices concatenated logically.
/// Avoids allocation for multi-part keys (namespace + type + name).
pub fn fnv1a_parts(parts: &[&[u8]]) -> u64 {
    let mut h = FNV_OFFSET;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Derive a namespace key from repo URL and commit SHA.
pub fn namespace_key(repo_url: &str, commit_sha: &str) -> u64 {
    fnv1a_parts(&[repo_url.as_bytes(), b"::", commit_sha.as_bytes()])
}

/// Derive a record key from namespace, node type, and qualified name.
pub fn record_key(namespace: u64, node_type: &str, qualified_name: &str) -> u64 {
    let ns_bytes = namespace.to_be_bytes();
    fnv1a_parts(&[
        &ns_bytes,
        b"::",
        node_type.as_bytes(),
        b"::",
        qualified_name.as_bytes(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        assert_eq!(fnv1a(b"hello"), fnv1a(b"hello"));
    }

    #[test]
    fn different_inputs_differ() {
        assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
    }

    #[test]
    fn parts_matches_concatenated() {
        let full = fnv1a(b"abc::def::ghi");
        let parts = fnv1a_parts(&[b"abc", b"::", b"def", b"::", b"ghi"]);
        assert_eq!(full, parts);
    }

    #[test]
    fn namespace_key_deterministic() {
        let k1 = namespace_key("github.com/repo", "abc123");
        let k2 = namespace_key("github.com/repo", "abc123");
        assert_eq!(k1, k2);
    }

    #[test]
    fn namespace_key_differs_on_commit() {
        let k1 = namespace_key("github.com/repo", "abc123");
        let k2 = namespace_key("github.com/repo", "def456");
        assert_ne!(k1, k2);
    }

    #[test]
    fn record_key_deterministic() {
        let ns = namespace_key("repo", "sha");
        let r1 = record_key(ns, "function", "auth.validate_token");
        let r2 = record_key(ns, "function", "auth.validate_token");
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_collisions_10k() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for i in 0..10_000u64 {
            let key = format!("ns::python::module_{i}::function::fn_{i}");
            let h = fnv1a(key.as_bytes());
            assert!(seen.insert(h), "collision at i={i}");
        }
    }
}
