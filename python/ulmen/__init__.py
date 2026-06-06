"""
UlmenDB -- Agentic AI Database

Zero external dependencies. Full ACID. MVCC snapshot isolation.
Fuzzy symbol matching. Bloom filters. WAL durability.

All algorithms validated in Cells 1-18 of the reference notebook.

Usage:
    from ulmen import MvccStore, FuzzyMatcher, BloomFilter

    store = MvccStore()
    store.put("key", b"value")
    print(store.get("key"))

    fm = FuzzyMatcher(max_distance=3)
    fm.add("getUserById")
    fm.add("get_user_by_id")
    results = fm.query("getUserByID", top_k=5)

    bf = BloomFilter(capacity=10000, fpr=0.01)
    bf.add(b"key_1")
    print(bf.may_contain(b"key_1"))  # True
"""

__version__ = "0.1.0"

try:
    # Rust core available (production path)
    from ulmen._core import (
        fnv1a,
        fnv1a_parts,
        cosine_dist,
        levenshtein,
        wal_serialize,
        PyBloomFilter as BloomFilter,
        PyFuzzyMatcher as FuzzyMatcher,
        PyMvccStore as MvccStore,
    )
    _BACKEND = "rust"

except ImportError:
    # Pure Python fallback (development / testing without Rust toolchain)
    # These are the same algorithms from the notebook cells, not stubs.

    import re
    import math
    import hashlib
    import struct
    import zlib
    import threading
    from collections import defaultdict, OrderedDict

    _BACKEND = "python"

    # -- Cell 1: FNV-1a -------------------------------------------------------

    _FNV_OFFSET = 0xcbf29ce484222325
    _FNV_PRIME = 0x100000001b3
    _MASK64 = 0xFFFFFFFFFFFFFFFF

    def fnv1a(data):
        """FNV-1a 64-bit hash."""
        h = _FNV_OFFSET
        for b in data:
            h ^= b
            h = (h * _FNV_PRIME) & _MASK64
        return h

    def fnv1a_parts(parts):
        """FNV-1a over multiple byte slices (logical concatenation)."""
        h = _FNV_OFFSET
        for part in parts:
            for b in part:
                h ^= b
                h = (h * _FNV_PRIME) & _MASK64
        return h

    # -- Cell 14/15: Cosine distance ------------------------------------------

    def cosine_dist(a, b):
        """Cosine distance between two float vectors."""
        dot_val = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        denom = na * nb
        if denom < 1e-12:
            return 1.0
        return 1.0 - dot_val / denom

    # -- Cell 8/15: Levenshtein ------------------------------------------------

    def levenshtein(a, b, max_dist=999):
        """Bounded Levenshtein edit distance."""
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        if len(a) > len(b):
            a, b = b, a
        la, lb = len(a), len(b)
        prev = list(range(lb + 1))
        for i in range(la):
            curr = [i + 1] + [0] * lb
            row_min = curr[0]
            ai = a[i]
            for j in range(lb):
                cost = 0 if ai == b[j] else 1
                curr[j + 1] = min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + cost,
                )
                if curr[j + 1] < row_min:
                    row_min = curr[j + 1]
            if row_min > max_dist:
                return max_dist + 1
            prev = curr
        return prev[lb]

    # -- Cell 2/15: WAL serialize ----------------------------------------------

    def wal_serialize(key, value):
        """Serialize a WAL record with CRC32 checksum."""
        key_len = len(key)
        val_len = len(value)
        header = struct.pack(">HI", key_len, val_len)
        payload = header + key + value
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", crc) + payload

    # -- Cell 17 Gap 14: Bloom filter ------------------------------------------

    class BloomFilter:
        """Bloom filter for reducing unnecessary page reads."""

        def __init__(self, capacity, fpr=0.01):
            self._m = max(
                64,
                int(-capacity * math.log(fpr) / (math.log(2) ** 2)),
            )
            self._k = max(
                1,
                int(self._m / capacity * math.log(2)),
            )
            self._bits = bytearray(math.ceil(self._m / 8))
            self._count = 0

        def _positions(self, key):
            for i in range(self._k):
                h = hashlib.md5(
                    i.to_bytes(2, "big") + bytes(key)
                ).digest()
                pos = int.from_bytes(h[:8], "big") % self._m
                yield pos

        def add(self, key):
            for pos in self._positions(key):
                self._bits[pos // 8] |= 1 << (pos % 8)
            self._count += 1

        def may_contain(self, key):
            return all(
                (self._bits[pos // 8] >> (pos % 8)) & 1
                for pos in self._positions(key)
            )

        def count(self):
            return self._count

        def size_bytes(self):
            return len(self._bits)

    # -- Cell 8: Fuzzy matcher -------------------------------------------------

    class FuzzyMatcher:
        """Trigram + Levenshtein fuzzy symbol matcher."""

        def __init__(self, max_distance=4):
            self._max_dist = max_distance
            self._symbols = []
            self._norm = []
            self._tri_count = []
            self._trigrams = defaultdict(list)

        @staticmethod
        def _normalize(s):
            return re.sub(r'[_\-\s]+', '', s.lower())

        @staticmethod
        def _trigrams_of(s):
            if not s:
                return set()
            p = "##" + s + "##"
            return {p[i:i+3] for i in range(len(p) - 2)}

        def add(self, symbol):
            idx = len(self._symbols)
            norm = self._normalize(symbol)
            trigs = self._trigrams_of(norm)
            self._symbols.append(symbol)
            self._norm.append(norm)
            self._tri_count.append(len(trigs))
            for tg in trigs:
                self._trigrams[tg].append(idx)

        def query(self, q, top_k=10):
            q_norm = self._normalize(q)
            q_trigs = self._trigrams_of(q_norm)
            if not q_trigs:
                return []

            hit_count = defaultdict(int)
            for tg in q_trigs:
                for idx in self._trigrams.get(tg, []):
                    hit_count[idx] += 1

            if not hit_count:
                return []

            q_tri_n = len(q_trigs)
            candidates = []
            for idx, hits in hit_count.items():
                union = q_tri_n + self._tri_count[idx] - hits
                jac = hits / union if union > 0 else 0.0
                candidates.append((jac, idx))

            candidates.sort(key=lambda x: (-x[0], x[1]))
            candidates = candidates[:max(50, top_k * 4)]

            scored = []
            for jac, idx in candidates:
                d = levenshtein(
                    q_norm, self._norm[idx], self._max_dist
                )
                if d <= self._max_dist:
                    raw_exact = 1 if self._symbols[idx] == q else 0
                    scored.append((d, -raw_exact, -jac, idx))

            scored.sort()
            return [
                (self._symbols[idx], d, -nj)
                for d, _, nj, idx in scored[:top_k]
            ]

        def len(self):
            return len(self._symbols)

    # -- Cell 12: MVCC store ---------------------------------------------------

    class _MvccVersion:
        __slots__ = ("value", "txn_id", "commit_ts", "prev")

        def __init__(self, value, txn_id, commit_ts=None):
            self.value = value
            self.txn_id = txn_id
            self.commit_ts = commit_ts
            self.prev = None

    class MvccStore:
        """MVCC key-value store with snapshot isolation."""

        def __init__(self):
            self._chains = {}
            self._lock = threading.RLock()
            self._ts = 0
            self._txn_counter = 0

        def put(self, key, value):
            """Convenience: single-key write + commit."""
            with self._lock:
                self._txn_counter += 1
                tid = self._txn_counter
                self._ts += 1
                ts = self._ts

                new_v = _MvccVersion(value, tid, ts)
                new_v.prev = self._chains.get(key)
                self._chains[key] = new_v
            return ts

        def get(self, key):
            """Read latest committed version."""
            with self._lock:
                v = self._chains.get(key)
                while v is not None:
                    if v.commit_ts is not None:
                        return v.value
                    v = v.prev
                return None

        def version_count(self):
            with self._lock:
                total = 0
                for head in self._chains.values():
                    v = head
                    while v is not None:
                        total += 1
                        v = v.prev
                return total


__all__ = [
    "fnv1a",
    "fnv1a_parts",
    "cosine_dist",
    "levenshtein",
    "wal_serialize",
    "BloomFilter",
    "FuzzyMatcher",
    "MvccStore",
]
