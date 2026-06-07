"""uldb integration tests."""

import time

import uldb


def test_backend():
    print(f"Backend: {uldb._BACKEND}")
    assert uldb._BACKEND in ("rust", "python-fallback")


def test_fnv1a():
    h1 = uldb.fnv1a(b"hello")
    h2 = uldb.fnv1a(b"hello")
    h3 = uldb.fnv1a(b"world")
    assert h1 == h2
    assert h1 != h3


def test_cosine_dist():
    assert abs(uldb.cosine_dist([1.0, 0.0], [1.0, 0.0])) < 0.001
    assert abs(uldb.cosine_dist([1.0, 0.0], [0.0, 1.0]) - 1.0) < 0.001


def test_levenshtein():
    assert uldb.levenshtein("kitten", "sitting", 10) == 3
    assert uldb.levenshtein("test", "test", 10) == 0


def test_bloom_filter():
    bf = uldb.BloomFilter(capacity=1000, fpr=0.01)
    for i in range(100):
        bf.add(f"key_{i}".encode())
    for i in range(100):
        assert bf.may_contain(f"key_{i}".encode())


def test_fuzzy_matcher():
    fm = uldb.FuzzyMatcher(max_distance=3)
    for s in ["AuthService", "validateToken", "getUserById"]:
        fm.add(s)
    results = fm.query("getUserById", 3)
    assert results[0][0] == "getUserById"


def test_mvcc_store():
    store = uldb.MvccStore()
    store.put("k1", b"v1")
    store.put("k2", b"v2")
    assert store.get("k1") == b"v1"
    assert store.get("k2") == b"v2"
    store.put("k1", b"updated")
    assert store.get("k1") == b"updated"


if __name__ == "__main__":
    import inspect
    tests = [
        (n, f) for n, f in inspect.getmembers(
            __import__(__name__), inspect.isfunction
        ) if n.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    assert failed == 0, f"{failed} tests failed"
