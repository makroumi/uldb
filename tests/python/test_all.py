"""
UlmenDB integration tests.

Validates every component works identically regardless of backend
(Rust or Python fallback). Mirrors the notebook Cell 18 scorecard.
"""

import time
import random
import threading

# Import ulmen -- will use Rust if available, else Python fallback
import ulmen


def test_backend():
    print(f"Backend: {ulmen._BACKEND}")
    assert ulmen._BACKEND in ("rust", "python")


def test_fnv1a():
    h1 = ulmen.fnv1a(b"hello")
    h2 = ulmen.fnv1a(b"hello")
    h3 = ulmen.fnv1a(b"world")
    assert h1 == h2, "deterministic"
    assert h1 != h3, "different inputs differ"
    print(f"  fnv1a: {hex(h1)}")


def test_cosine_dist():
    d_same = ulmen.cosine_dist([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    d_diff = ulmen.cosine_dist([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert abs(d_same) < 0.001, f"same vector: {d_same}"
    assert abs(d_diff - 1.0) < 0.001, f"orthogonal: {d_diff}"
    print(f"  cosine_dist: same={d_same:.4f} ortho={d_diff:.4f}")


def test_levenshtein():
    assert ulmen.levenshtein("kitten", "sitting", 10) == 3
    assert ulmen.levenshtein("test", "test", 10) == 0
    assert ulmen.levenshtein("a", "b", 0) == 1  # exceeds max
    print("  levenshtein: OK")


def test_bloom_filter():
    bf = ulmen.BloomFilter(capacity=10000, fpr=0.01)
    for i in range(10000):
        bf.add(f"key_{i}".encode())

    # No false negatives
    for i in range(10000):
        assert bf.may_contain(f"key_{i}".encode()), f"false neg at {i}"

    # Measure FPR
    fp = sum(
        1
        for i in range(10000)
        if bf.may_contain(f"absent_{i}".encode())
    )
    fpr = fp / 10000
    assert fpr < 0.03, f"FPR too high: {fpr}"
    print(f"  bloom: 0 false neg, FPR={fpr:.4f}, size={bf.size_bytes()}B")


def test_fuzzy_matcher():
    fm = ulmen.FuzzyMatcher(max_distance=4)
    symbols = [
        "AuthService", "validate_token", "hash_password",
        "getUserById", "validateEmail", "sendEmail",
        "connectDB", "logWarning", "encryptAES", "generateToken",
    ]
    for s in symbols:
        fm.add(s)

    cases = [
        ("AuthServce", "AuthService"),
        ("GETUSERBYID", "getUserById"),
        ("sendEmal", "sendEmail"),
        ("connetDB", "connectDB"),
        ("logWarnig", "logWarning"),
    ]

    correct = 0
    for query, expected in cases:
        results = fm.query(query, 3)
        if results and results[0][0] == expected:
            correct += 1
        elif results and any(r[0] == expected for r in results):
            correct += 1

    accuracy = correct / len(cases)
    assert accuracy >= 0.80, f"accuracy too low: {accuracy}"
    print(f"  fuzzy: accuracy={accuracy:.0%} ({correct}/{len(cases)})")


def test_mvcc_store():
    store = ulmen.MvccStore()
    store.put("key1", b"value1")
    store.put("key2", b"value2")
    assert store.get("key1") == b"value1"
    assert store.get("key2") == b"value2"
    assert store.get("absent") is None

    # Overwrite
    store.put("key1", b"updated")
    assert store.get("key1") == b"updated"
    print(f"  mvcc: put/get OK, versions={store.version_count()}")


def test_throughput():
    # Ingestion
    store = ulmen.MvccStore()
    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        store.put(f"k{i}", f"v{i}".encode())
    elapsed = time.perf_counter() - t0
    rps = n / elapsed
    print(f"  throughput: {rps:,.0f} put/sec ({n} records in {elapsed*1000:.1f}ms)")

    # FNV-1a
    t0 = time.perf_counter()
    for i in range(100000):
        ulmen.fnv1a(f"key_{i}".encode())
    elapsed = time.perf_counter() - t0
    hps = 100000 / elapsed
    print(f"  fnv1a throughput: {hps:,.0f} hash/sec")


if __name__ == "__main__":
    tests = [
        test_backend,
        test_fnv1a,
        test_cosine_dist,
        test_levenshtein,
        test_bloom_filter,
        test_fuzzy_matcher,
        test_mvcc_store,
        test_throughput,
    ]

    print("=" * 60)
    print("UlmenDB Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} tests failed"
    print("ALL TESTS PASS")
