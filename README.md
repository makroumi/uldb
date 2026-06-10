<div align="center">

# uldb

**The database for AI agents.**

[![Rust Tests](https://img.shields.io/badge/rust_tests-260-brightgreen)]()
[![Python Tests](https://img.shields.io/badge/python_tests-86-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSL--1.1-blue)]()

[Benchmarks](#benchmarks) | [Quick Start](#quick-start) | [Architecture](#architecture) | [API Reference](#api-reference)

</div>

---

uldb is a storage engine purpose-built for agentic AI workloads. MVCC transactions, WAL crash recovery, LSM compaction, and five search indices in a single binary with zero required external services.

Agents get isolated branch workspaces with snapshot-level reads and atomic merge-or-rollback semantics. Every write is crash-safe. Every search returns ranked results across all indices via reciprocal rank fusion.

Native [ulmen-core](https://github.com/makroumi/ulmen) integration for typed agent payload storage.

```python
import uldb

app = uldb.open("./my_project")

app.put("auth.py::validate", b"def validate(token): ...")
results = app.search("validate token")

with app.agent("refactor-auth") as agent:
    docs = agent.search("validate token")
    agent.put("auth.py::validate", b"def validate_v2(token): ...")
    # auto-merge on success, auto-rollback on failure

app.close()
```

``` Rust
use uldb::engine::{Engine, EngineConfig};
use uldb::agent_store;
use ulmen_core::*;

let mut engine = Engine::open(EngineConfig::new("./data")).unwrap();

// Store typed agent payloads
let payload = AgentPayload { /* ... */ };
agent_store::store_payload(&mut engine, "session:001", &payload).unwrap();

// Search across agent records
let results = agent_store::search_records(&mut engine, "authentication", 10);
```

---
## Benchmarks
AMD Ryzen 7 4700U, 8 cores, NVMe SSD. Release build.


### Throughput
| Operation | Speed | Latency |
| --- | --- | --- |
| PUT sequential | 141K ops/sec | 7.1 us |
| PUT batch | 171K ops/sec | 5.9 us |
| GET (hit) | 4.2M ops/sec | 238 ns |
| GET (cached) | 10.0M ops/sec | 100 ns |
| DELETE | 349K ops/sec | 2.9 us |
| SCAN 100 keys | 230K ops/sec | 4.4 us |
| BM25 query | 83K queries/sec | 12.1 us |
| Fuzzy query | 3.9K queries/sec | 257 us |
| Snapshot create | 10.3M ops/sec | 97 ns |


### Agentic Workloads
| Workload | Throughput |
| --- | --- |
| Codebase ingest (10K symbols) | 143K symbols/sec |
| Codebase ingest (50K symbols) | 126K symbols/sec |
| Agent session (1K mixed ops) | 25.3K ops/sec |
| 4 concurrent agents | 113K ops/sec |


### Storage Efficiency
| Metric | Value |
| --- | --- |
| Disk overhead | 1.30x raw data |
| WAL recovery | 207K records/sec |


### PUT Cost Breakdown
| Component | Latency | Share |
| --- | --- | --- |
| WAL | 618 ns | 13% |
| Memtable | 460 ns | 10% |
| Index | 2,543 ns | 52% |
| HAMT | 1,180 ns | 25% |

---

## Installation

``` Rust
[dependencies]
uldb = { git = "https://github.com/makroumi/uldb" }
```

``` Python
pip install uldb
```

---

## Quick Start

### Python
``` python
import uldb

app = uldb.open("./my_project")

# CRUD
app.put("auth.py::validate", b"def validate(token): ...")
doc = app.get("auth.py::validate")        # returns bytes
app.delete("auth.py::validate")

# Search (BM25 + fuzzy, RRF merged)
results = app.search("validate token", limit=10)
for doc in results:
    print(f"{doc.id}: score={doc.score:.3f}")

# Fuzzy symbol search (typo-tolerant)
results = app.search_fuzzy("getUserByID", limit=5)

# Bulk ingest
app.load({
    "auth.py::validate": b"def validate(token): ...",
    "auth.py::login": b"def login(user, pwd): ...",
})

# Snapshots
snap_id = app.snapshot("before_refactor")
app.restore("before_refactor")

# Stats
print(app.stats())

app.close()
```

### Agent Workflow

``` python
import uldb

app = uldb.open("./codebase")
app.put("auth.py::validate", b"def validate(token): return True")

# Agent: isolated workspace, auto-merge/rollback
with app.agent("security-review") as agent:
    # Search sees committed state
    docs = agent.search("validate token")

    # Writes are isolated
    agent.put("auth.py::validate", b"def validate_v2(token): return check_jwt(token)")
    agent.put("auth.py::refresh", b"def refresh(token): return renew_jwt(token)")

    # On clean exit: all writes merge to main atomically
    # On exception: all writes are discarded

# Main now has the agent's changes
assert b"validate_v2" in app.get("auth.py::validate")
```

### Agent Rollback
``` python
try:
    with app.agent("risky-change") as agent:
        agent.put("config.py", b"SECRET_KEY = 'HACKED'")
        raise RuntimeError("detected unsafe change")
except RuntimeError:
    pass

# Config is untouched
assert b"HACKED" not in app.get("config.py")
```

### Multiple Agents
``` python
agent_a = app.agent("agent_alpha")
agent_b = app.agent("agent_beta")

agent_a.put("shared_key", b"from alpha")
agent_b.put("shared_key", b"from beta")

# Each sees only its own writes
agent_a.discard()
agent_b.commit()

# Beta wins
assert app.get("shared_key") == b"from beta"
```

### Vector Search
``` python
with app.agent("embedding-agent") as agent:
    results = agent.search_vector([0.1, 0.2, 0.3, ...], limit=10)
```

### Graph Traversal
``` python
with app.agent("graph-agent") as agent:
    results = agent.search_graph("AuthService", relation="calls", depth=3)
```

### Rust

``` rust
use uldb::engine::{Engine, EngineConfig};

let mut engine = Engine::open(EngineConfig::new("./data")).unwrap();

engine.put(b"auth.py::validate", b"def validate(): ...").unwrap();

// Read (238ns)
let value = engine.get(b"auth.py::validate");

// Batch
engine.put_batch(&[
    (b"a.py::fn1", b"code1"),
    (b"a.py::fn2", b"code2"),
]).unwrap();

// Search
use uldb::query::planner::QuerySpec;
let spec = QuerySpec { text: "validate token".into(), top_k: 10, ..Default::default() };
let hits = engine.indices.query(&spec);

// Agent payloads (via ulmen-core)
use uldb::agent_store;
use ulmen_core::*;

let payload = AgentPayload { /* ... */ };
agent_store::store_payload(&mut engine, "session:001", &payload).unwrap();
let loaded = agent_store::load_payload(&engine, "session:001").unwrap();
let records = agent_store::search_records(&mut engine, "jwt token", 10);
```

### Server
``` Bash
uldb serve --port 7771 --data ./data --token mytoken
```
connect with [ulmp](https://github.com/makroumi/ulmp) Python client:
``` Python
from ulmp import AsyncClient

async with AsyncClient.connect("localhost", 7771, token=b"mytoken") as db:
    ns = db.namespace("my_project")
    await ns.put("auth.py::validate", b"code")
    results = await ns.search("validate")
```
---
## Architecture
```text
Engine
  WAL            crash-safe write-ahead log
  Memtable       sorted in-memory buffer
  Compaction     tiered LSM (3 levels)
  PageStore      compressed pages on disk
  Cache          LRU read cache
  HAMT           persistent hash trie (snapshots, branches)

  IndexManager
    BM25           keyword search
    HNSW           vector nearest-neighbor
    FuzzyMatcher   typo-tolerant symbol lookup
    RelationGraph  CSR graph traversal
    BloomFilter    per-page membership test
    QueryPlanner   RRF multi-index fusion

  Transactions   MVCC (snapshot + serializable isolation)
  Namespaces     repo+commit scoped isolation
  Branches       copy-on-write via HAMT
  Snapshots      zero-cost via structural sharing

  AgentStore     ulmen-core typed payload storage
  UmpHandler     ulmp wire protocol server (46 handlers)
```

### Agent Isolation
Each agent gets a branch (copy-on-write HAMT fork):
- Reads: snapshot state at creation + own writes
- Writes: invisible to main and other agents until commit
- Commit: atomic merge to main
- Rollback: discard everything
- Search: reflects committed main state

### Sever Handlers (46 implemented)
- **Records**: `put`, `get`, `delete`, `scan`, `put_batch`, `get_batch`, `range_delete`
- **Query**: `text` (BM25), `fuzzy`, `vector` (HNSW), `graph`
- **Transactions**: `begin`, `commit`, `rollback`, `status`
- **Snapshots**: `create`, `restore`, `delete`, `list`
- **Branches**: `create`, `merge`, `rollback`, `diff`, `list`
- **Namespaces**: `create`, `open`, `delete`, `list`, `stat`, `grant`
- **Admin**: `stats`, `compact`, `config_get`, `config_set`, `backup`, `restore`
- **Watch**: `register`, `unwatch`, `window` (credit-based)
- **Auth**: `rotate_request`, `rotate`
- **Streaming**: `checkpoint`, `stream_resume`

---
## API Reference
### Python (DB)
``` Python
app = uldb.open(path)
app.put(key, value_bytes)
app.get(key) -> bytes | None
app.delete(key)
app.search(query, limit=10) -> list[Document]
app.search_fuzzy(symbol, limit=5) -> list[Document]
app.scan(prefix, limit=100) -> list[Document]
app.load(records_dict) -> int
app.delete_range(start, end) -> int
app.keys(prefix, limit) -> list[str]
app.snapshot(name) -> str
app.restore(name)
app.snapshots() -> list[str]
app.stats() -> dict
app.agent(name) -> Agent
app.close()
```

### Python (Agent)
``` Python
with app.agent(name) as agent:
    agent.put(key, value)
    agent.get(key) -> Document | None
    agent.delete(key)
    agent.search(query) -> list[Document]
    agent.search_fuzzy(symbol) -> list[Document]
    agent.search_vector(embedding) -> list[Document]
    agent.search_graph(start, relation, depth) -> list[Document]
    agent.scan(prefix) -> list[Document]
    agent.load(records_dict) -> int
    agent.commit()
    agent.discard()
    agent.checkpoint(name) -> str
```

### Rust (Engine)
``` Rust
Engine::open(config) -> Engine
engine.put(key, value) -> io::Result<()>
engine.get(key) -> Option<Vec<u8>>
engine.delete(key) -> io::Result<()>
engine.scan(start, end) -> Vec<(Vec<u8>, Vec<u8>)>
engine.put_batch(entries) -> io::Result<()>
engine.flush() -> io::Result<()>
engine.close() -> io::Result<()>
```

### Rust (Agent Store)
``` Rust
agent_store::store_payload(engine, key, payload) -> io::Result<()>
agent_store::load_payload(engine, key) -> Result<AgentPayload>
agent_store::search_records(engine, query, limit) -> Vec<(AgentRecord, f64)>
agent_store::append_records(engine, key, records) -> io::Result<()>
agent_store::get_records_by_type(engine, key, type) -> Vec<AgentRecord>
agent_store::get_recent_records(engine, key, limit) -> Vec<AgentRecord>
agent_store::list_sessions(engine) -> Vec<String>
agent_store::delete_payload(engine, key) -> io::Result<()>
```

---
## Testing
``` Bash
# Rust (260 tests)
cargo test --features server

# Python (86 tests)
source .venv/bin/activate
maturin develop --release --features python
pytest tests/python/ -v

# Benchmarks
cargo bench --bench engine_bench
```
---
## Ecosystem
- ulmen - Serialization + agent protocol
- uldb - Storage, indexing, caching
- ulmp - Wire protocol, networking

---
## License
Business Source License 1.1. See [LICENSE](https://github.com/makroumi/ulmen/blob/main/LICENSE).

Copyright (c) 2026 El Mehdi Makroumi.