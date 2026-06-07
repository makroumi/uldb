# uldb

Agentic AI database. Storage engine with MVCC, WAL, LSM compaction, bloom filters, fuzzy symbol matching.

## Install

```bash
pip install uldb
```

## Quick start

```python
from uldb import MvccStore, FuzzyMatcher, BloomFilter

store = MvccStore()
store.put("auth.py::AuthService", b"class AuthService: pass")
print(store.get("auth.py::AuthService"))

fm = FuzzyMatcher(max_distance=3)
fm.add("getUserById")
results = fm.query("getUserByID", top_k=5)

bf = BloomFilter(capacity=10000, fpr=0.01)
bf.add(b"key_1")
print(bf.may_contain(b"key_1"))
```

## License
BSL-1.1

## Author
El Mehdi Makroumi