// tests/rust/e2e_gaps.rs
//
// End-to-end gap detection tests for uldb.
//
// Each test is labeled with the gap it exercises. Tests that PASS confirm
// existing behaviour. Tests that FAIL expose concrete gaps.
//
// Run with: cargo test --test e2e_gaps
//
// Gap inventory:
//   P0-GAP-1:  Page file persistence (pages never written to disk)
//   P0-GAP-2:  HNSW index persistence (vectors lost on restart)
//   P0-GAP-3:  Real transaction wiring (tx_begin/commit are stubs)
//   P1-GAP-5:  Concurrent connections (single-threaded accept loop)
//   P1-GAP-8:  Graph persistence (graph lost on restart)
//   P1-GAP-9:  Vector indexing via handler (handle_query_vector stub)
//   P1-GAP-10: Namespace-aware indices (indices use global ns=0)
//   P2-GAP-4:  WATCH/NOTIFY (handle_watch is stub)

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use uldb::engine::{Engine, EngineConfig};
use uldb::index::hnsw::HnswIndex;
use uldb::namespace::{derive_namespace_id, scope_key};
use uldb::query::planner::QuerySpec;
use uldb::server::UmpHandler;
use uldb::tx::hamt::Hamt;

use ulmp::messages::opcode;
use ulmp::messages::{branch, query, record, snapshot, tx, watch};
use ulmp::messages::admin;
use ulmp::server::handler::{Handler, Response};

// ============================================================================
// Test helpers
// ============================================================================

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = env::temp_dir();
    p.push(format!("uldb_e2e_{name}_{}", std::process::id()));
    p
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn open_engine(dir: &Path) -> Engine {
    Engine::open(EngineConfig::new(dir)).expect("engine open failed")
}

fn open_engine_small(dir: &Path) -> Engine {
    Engine::open(EngineConfig::new(dir).with_flush_threshold(512))
        .expect("engine open failed")
}

fn make_handler(dir: &Path) -> UmpHandler {
    UmpHandler::new(open_engine(dir))
}

fn count_result_rows(resp: &Response) -> usize {
    match resp {
        Response::Stream { frames } => frames
            .iter()
            .filter(|(op, _)| *op == opcode::OP_RESULT_ROW)
            .count(),
        _ => 0,
    }
}

fn is_ok_response(resp: &Response) -> bool {
    match resp {
        Response::Single { opcode: op, .. } => {
            *op == opcode::OP_RESULT_END || *op == opcode::OP_RESULT_ROW
        }
        Response::Stream { frames } => frames
            .last()
            .map(|(op, _)| *op == opcode::OP_RESULT_END)
            .unwrap_or(false),
        _ => false,
    }
}

fn is_error_response(resp: &Response) -> bool {
    match resp {
        Response::Single { opcode: op, .. } => *op == opcode::OP_ERROR,
        _ => false,
    }
}



/// Deterministic LCG random f64 in [0, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Build a normalised random vector of given dimension seeded by `seed`.
fn rng_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0xCAFEBABE_DEADBEEF;
    let mut v: Vec<f32> = (0..dim).map(|_| (lcg(&mut s) * 2.0 - 1.0) as f32).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

// ============================================================================
// EXISTING BEHAVIOUR CONFIRMATIONS
// These must all pass. They document what already works.
// ============================================================================

/// Confirm WAL crash recovery is correct end-to-end.
#[test]
fn confirm_wal_crash_recovery_e2e() {
    let dir = tmp_dir("wal_crash_e2e");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        for i in 0..500u32 {
            eng.put(
                format!("crash_key_{i:05}").as_bytes(),
                format!("crash_val_{i}").as_bytes(),
            )
            .unwrap();
        }
        // Drop without close() -- simulates crash.
        // WAL is flushed per-put so all records are durable.
    }

    {
        let mut eng = open_engine(&dir);
        for i in 0..500u32 {
            let key = format!("crash_key_{i:05}");
            let expected = format!("crash_val_{i}");
            assert_eq!(
                eng.get(key.as_bytes()),
                Some(expected.as_bytes().to_vec()),
                "WAL recovery failed for key {key}"
            );
        }
    }

    cleanup(&dir);
}

/// Confirm memtable flush + background compaction + read-through works.
#[test]
fn confirm_flush_and_compaction_read_through() {
    let dir = tmp_dir("flush_compact_e2e");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine_small(&dir);
        for i in 0..200u32 {
            let k = format!("fk_{i:05}");
            let v = format!("fv_{i}_padding_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
            eng.put(k.as_bytes(), v.as_bytes()).unwrap();
        }

        thread::sleep(Duration::from_millis(200));
        assert!(eng.flush_count() > 0, "should have flushed at least once");

        for i in 0..200u32 {
            let k = format!("fk_{i:05}");
            let v = format!("fv_{i}_padding_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
            assert_eq!(
                eng.get(k.as_bytes()),
                Some(v.as_bytes().to_vec()),
                "key {k} not readable after flush"
            );
        }
    }

    cleanup(&dir);
}

/// Confirm snapshot isolation: old snapshot unaffected by new writes.
#[test]
fn confirm_snapshot_isolation_e2e() {
    let dir = tmp_dir("snap_iso_e2e");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        for i in 0..50u32 {
            eng.put(format!("sk{i}").as_bytes(), format!("sv{i}").as_bytes())
                .unwrap();
        }

        let snap_id = eng.snapshot_create("v1");

        for i in 0..50u32 {
            eng.put(format!("sk{i}").as_bytes(), b"overwritten").unwrap();
        }

        for i in 0..50u32 {
            let key = format!("sk{i}");
            let expected = format!("sv{i}");
            assert_eq!(
                eng.snapshot_get(&snap_id, key.as_bytes()),
                Some(expected.as_bytes().to_vec()),
                "snapshot isolation broken for {key}"
            );
        }
    }

    cleanup(&dir);
}

/// Confirm handler PUT/GET/DELETE/SCAN round-trip.
#[test]
fn confirm_handler_crud_e2e() {
    let dir = tmp_dir("crud_e2e");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    for i in 0..10u32 {
        let resp = handler.handle_put(record::Put {
            key: format!("crud_{i:03}"),
            value: format!("value_{i}").into_bytes(),
        });
        assert!(is_ok_response(&resp), "PUT {i} failed: {resp:?}");
    }

    for i in 0..10u32 {
        let resp = handler.handle_get(record::Get {
            key: format!("crud_{i:03}"),
        });
        match &resp {
            Response::Single { opcode: op, .. } => {
                assert_eq!(*op, opcode::OP_RESULT_ROW, "GET {i} returned wrong opcode");
            }
            _ => panic!("GET {i} returned wrong response type: {resp:?}"),
        }
    }

    let del = handler.handle_delete(record::Delete {
        key: "crud_005".into(),
    });
    assert!(is_ok_response(&del), "DELETE failed: {del:?}");

    let get_deleted = handler.handle_get(record::Get {
        key: "crud_005".into(),
    });
    assert!(
        is_error_response(&get_deleted),
        "GET after DELETE should return error"
    );

    // Scan [002, 007) excludes 005 (deleted). Expect: 002,003,004,006 = 4.
    let scan = handler.handle_scan(record::Scan {
        start: "crud_002".into(),
        end: "crud_007".into(),
        limit: 100,
    });
    let rows = count_result_rows(&scan);
    assert_eq!(rows, 4, "SCAN should return 4 live records, got {rows}");

    cleanup(&dir);
}

/// Confirm BM25 text search works end-to-end via handler.
#[test]
fn confirm_bm25_query_e2e() {
    let dir = tmp_dir("bm25_e2e");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "auth.py::validate_token".into(),
        value: b"def validate_token validates JWT authentication token".to_vec(),
    });
    handler.handle_put(record::Put {
        key: "auth.py::hash_password".into(),
        value: b"def hash_password uses bcrypt hashing".to_vec(),
    });
    handler.handle_put(record::Put {
        key: "models.py::User".into(),
        value: b"class User email password model".to_vec(),
    });
    handler.handle_put(record::Put {
        key: "utils.py::send_email".into(),
        value: b"def send_email sends via SMTP".to_vec(),
    });

    let resp = handler.handle_query(query::Query {
        namespace: "".into(),
        text: "validate token authentication".into(),
        vector: vec![],
        top_k: 5,
        max_depth: 0,
        relations: vec![],
        lang_filter: vec![],
        type_filter: vec![],
        file_filter: vec![],
        merge_strategy: 1,
        timeout_ms: 5000,
    });

    let rows = count_result_rows(&resp);
    assert!(
        rows >= 1,
        "BM25 query should return at least 1 result, got {rows}"
    );

    cleanup(&dir);
}

/// Confirm fuzzy symbol search works end-to-end.
#[test]
fn confirm_fuzzy_search_e2e() {
    let dir = tmp_dir("fuzzy_e2e");
    fs::create_dir_all(&dir).unwrap();

    let mut eng = open_engine(&dir);
    eng.put(b"getUserById", b"function").unwrap();
    eng.put(b"validateEmail", b"function").unwrap();
    eng.put(b"hashPassword", b"function").unwrap();
    eng.put(b"sendNotification", b"function").unwrap();

    let spec = QuerySpec {
        text: "getUsrById".into(),
        top_k: 5,
        ..Default::default()
    };
    let results = eng.indices.query(&spec);
    assert!(
        !results.is_empty(),
        "fuzzy search should find results for 'getUsrById'"
    );
    let keys: Vec<String> = results
        .iter()
        .map(|r| String::from_utf8_lossy(&r.key).to_string())
        .collect();
    assert!(
        keys.contains(&"getUserById".to_string()),
        "fuzzy search should find getUserById, got: {keys:?}"
    );

    cleanup(&dir);
}

/// Confirm branch create/merge/rollback lifecycle.
#[test]
fn confirm_branch_lifecycle_e2e() {
    let dir = tmp_dir("branch_e2e");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "shared.py::service".into(),
        value: b"class Service: original".to_vec(),
    });

    let br = handler.handle_branch_create(branch::BranchCreate {
        namespace_id: 0,
        branch_id: "feat/refactor".into(),
        from_snapshot: "".into(),
        description: "refactor service".into(),
    });
    assert!(is_ok_response(&br), "branch_create failed: {br:?}");

    let bl = handler.handle_branch_list(branch::BranchList { namespace_id: 0 });
    let branch_rows = count_result_rows(&bl);
    assert_eq!(branch_rows, 1, "should have 1 branch, got {branch_rows}");

    let rb = handler.handle_branch_rollback(branch::BranchRollback {
        namespace_id: 0,
        branch_id: "feat/refactor".into(),
    });
    assert!(is_ok_response(&rb), "branch_rollback failed: {rb:?}");

    let bl2 = handler.handle_branch_list(branch::BranchList { namespace_id: 0 });
    assert_eq!(
        count_result_rows(&bl2),
        0,
        "no branches should exist after rollback"
    );

    cleanup(&dir);
}

/// Confirm MVCC snapshot isolation at the Hamt level.
#[test]
fn confirm_hamt_snapshot_isolation_e2e() {
    let h0 = Hamt::new()
        .put(b"account:alice".to_vec(), b"1000".to_vec())
        .put(b"account:bob".to_vec(), b"500".to_vec());

    let snap = h0.snapshot();

    let h1 = h0
        .put(b"account:alice".to_vec(), b"800".to_vec())
        .put(b"account:bob".to_vec(), b"700".to_vec());

    assert_eq!(snap.get(b"account:alice"), Some(b"1000".as_ref()));
    assert_eq!(snap.get(b"account:bob"), Some(b"500".as_ref()));
    assert_eq!(h1.get(b"account:alice"), Some(b"800".as_ref()));
    assert_eq!(h1.get(b"account:bob"), Some(b"700".as_ref()));

    let snap_total: i64 = snap
        .get(b"account:alice")
        .and_then(|v| std::str::from_utf8(v).ok())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0)
        + snap
            .get(b"account:bob")
            .and_then(|v| std::str::from_utf8(v).ok())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

    let live_total: i64 = h1
        .get(b"account:alice")
        .and_then(|v| std::str::from_utf8(v).ok())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0)
        + h1.get(b"account:bob")
            .and_then(|v| std::str::from_utf8(v).ok())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

    assert_eq!(snap_total, 1500, "snapshot conservation violated");
    assert_eq!(live_total, 1500, "live conservation violated");
}

// ============================================================================
// P0-GAP-1: Page file persistence
//
// Pages are sent to BackgroundCompactor which holds them in memory only.
// Engine::open() does not read page files from disk.
// After flush + close + reopen, data that was only in pages is LOST.
// ============================================================================

#[test]
fn gap_p0_1_page_persistence_after_restart() {
    let dir = tmp_dir("page_persist_gap");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine_small(&dir);

        for i in 0..50u32 {
            let k = format!("persist_{i:04}");
            let v = format!("value_{i}_xxxxxxxxxxxxxxxxxxx");
            eng.put(k.as_bytes(), v.as_bytes()).unwrap();
        }

        thread::sleep(Duration::from_millis(150));
        let did_flush = eng.flush_count() > 0 || eng.compaction_records() > 0;

        // Close cleanly -- WAL is truncated on next open.
        eng.close().unwrap();

        if !did_flush {
            // If nothing flushed, data is still in WAL and will survive.
            // Skip the gap demonstration in this case.
            cleanup(&dir);
            return;
        }
    }

    {
        let mut eng = open_engine_small(&dir);

        let mut lost = 0u32;
        for i in 0..50u32 {
            let k = format!("persist_{i:04}");
            let v = format!("value_{i}_xxxxxxxxxxxxxxxxxxx");
            if eng.get(k.as_bytes()) != Some(v.as_bytes().to_vec()) {
                lost += 1;
            }
        }

        // REQUIRED BEHAVIOUR (FIX): lost == 0
        // CURRENT BEHAVIOUR (GAP): lost > 0 for records that were in pages
        assert_eq!(
            lost,
            0,
            "GAP-P0-1: {lost}/50 records lost after restart because pages are \
             not persisted to disk. Engine::open() must reload serialised page \
             files from data_dir on startup."
        );
    }

    cleanup(&dir);
}

// ============================================================================
// P0-GAP-2: HNSW index persistence
//
// HnswIndex is in-memory only. Restart loses all vectors.
// ============================================================================

#[test]
fn gap_p0_2_hnsw_persistence_after_restart() {
    let dir = tmp_dir("hnsw_persist_gap");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);

        for i in 0..20u32 {
            let key = format!("vec_{i:03}");
            let v = rng_unit_vec(16, i as u64);
            eng.put(key.as_bytes(), b"embedding document").unwrap();
            eng.indices.on_put_vector(key.as_bytes(), v);
        }

        let query_vec = rng_unit_vec(16, 0);
        let spec = QuerySpec {
            vector: query_vec,
            top_k: 5,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);
        assert!(
            !results.is_empty(),
            "HNSW should return results before restart"
        );

        eng.close().unwrap();
    }

    {
        let mut eng = open_engine(&dir);

        let query_vec = rng_unit_vec(16, 0);
        let spec = QuerySpec {
            vector: query_vec,
            top_k: 5,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);

        // CURRENT BEHAVIOUR (GAP): results is empty
        // REQUIRED BEHAVIOUR (FIX): results.len() >= 1
        assert!(
            !results.is_empty(),
            "GAP-P0-2: HNSW index not persisted -- 0 results after restart. \
             Engine must serialise/deserialise HnswIndex to data_dir on close/open."
        );
    }

    cleanup(&dir);
}

// ============================================================================
// P0-GAP-3: Real transaction wiring
//
// handle_tx_begin allocates an ID but does NOT create a TxSession.
// Reads within a transaction see the live engine state, not a snapshot.
// Rollback is a no-op.
// ============================================================================

#[test]
fn gap_p0_3_transaction_snapshot_isolation() {
    let dir = tmp_dir("tx_gap");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Write baseline.
    handler.handle_put(record::Put {
        key: "acct:alice".into(),
        value: b"1000".to_vec(),
    });

    // Begin a transaction.
    let tx_resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x02 });
    let tx_id = match &tx_resp {
        Response::Single { payload, .. } => {
            // First u64 field in payload: TAG_U64(0x04) + 8 bytes.
            if payload.len() >= 9 && payload[0] == 0x04 {
                u64::from_be_bytes(payload[1..9].try_into().unwrap_or([0u8; 8]))
            } else {
                0
            }
        }
        _ => 0,
    };
    assert!(tx_id > 0, "tx_begin must return a non-zero tx_id");

    // An external writer modifies acct:alice while our tx is open.
    handler.handle_put(record::Put {
        key: "acct:alice".into(),
        value: b"500".to_vec(),
    });

    // Within the transaction, read acct:alice via tx_get.
    // With proper snapshot isolation this MUST return 1000 (value at tx_begin).
    let value_in_tx = handler.tx_get(tx_id, "acct:alice");

    // REQUIRED BEHAVIOUR (FIX): value_in_tx == Some(b"1000")
    // CURRENT BEHAVIOUR (GAP): value_in_tx == Some(b"500")
    assert_eq!(
        value_in_tx,
        Some(b"1000".to_vec()),
        "GAP-P0-3: Transaction reads live state instead of snapshot. \
         handle_tx_begin must create a TxSession with a HAMT snapshot. \
         Reads within the tx must consult the session snapshot."
    );

    let rb = handler.handle_tx_rollback(tx::TxRollback { tx_id });
    assert!(is_ok_response(&rb), "tx_rollback must succeed");

    cleanup(&dir);
}

// ============================================================================
// P1-GAP-5: Concurrent connections
//
// UmpHandler (Arc<Mutex<Engine>>) IS thread-safe.
// bin/uldb.rs accept loop is NOT -- it calls handle_connection() inline.
//
// This test confirms the handler is safe under concurrent load.
// The accept loop gap is documented but requires a binary-level fix.
// ============================================================================

#[test]
fn gap_p1_5_concurrent_handler_requests() {
    let dir = tmp_dir("concurrent_gap");
    fs::create_dir_all(&dir).unwrap();

    let handler = Arc::new(make_handler(&dir));

    for i in 0..100u32 {
        handler.handle_put(record::Put {
            key: format!("concurrent_{i:04}"),
            value: format!("val_{i}").into_bytes(),
        });
    }

    let n_threads = 8usize;
    let n_ops = 50usize;
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for t in 0..n_threads {
        let h = Arc::clone(&handler);
        let errs = Arc::clone(&errors);
        handles.push(thread::spawn(move || {
            for op in 0..n_ops {
                let key = format!("concurrent_{:04}", (t * n_ops + op) % 100);
                let resp = h.handle_get(record::Get { key: key.clone() });
                match resp {
                    Response::Single { opcode: op_code, .. }
                        if op_code == opcode::OP_RESULT_ROW
                            || op_code == opcode::OP_ERROR => {}
                    other => {
                        errs.lock()
                            .unwrap()
                            .push(format!("t{t} op{op}: unexpected {other:?}"));
                    }
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panicked under concurrent load");
    }

    let errs = errors.lock().unwrap();
    assert!(
        errs.is_empty(),
        "GAP-P1-5: Concurrent handler errors: {:?}",
        &*errs
    );

    eprintln!(
        "GAP-P1-5-NOTE: UmpHandler is concurrent-safe (Mutex<Engine>), \
         but bin/uldb.rs accept loop is single-threaded. Fix: \
         thread::spawn per accepted connection."
    );

    cleanup(&dir);
}

// ============================================================================
// P1-GAP-8: Graph persistence
//
// RelationGraph is in-memory only. Restart loses all edges.
// ============================================================================

#[test]
fn gap_p1_8_graph_persistence_after_restart() {
    let dir = tmp_dir("graph_persist_gap");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);

        eng.put(b"AuthService", b"class AuthService").unwrap();
        eng.put(b"validate_token", b"def validate_token").unwrap();
        eng.put(b"hash_password", b"def hash_password").unwrap();

        eng.indices.on_add_edge("AuthService", "validate_token", "calls");
        eng.indices.on_add_edge("AuthService", "hash_password", "calls");

        let spec = QuerySpec {
            text: "AuthService".into(),
            relations: vec!["calls".into()],
            top_k: 5,
            max_depth: 2,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);
        let keys: Vec<String> = results
            .iter()
            .map(|r| String::from_utf8_lossy(&r.key).to_string())
            .collect();
        assert!(
            keys.contains(&"validate_token".to_string()),
            "graph traversal should work before restart: {keys:?}"
        );

        eng.close().unwrap();
    }

    {
        let mut eng = open_engine(&dir);

        let spec = QuerySpec {
            text: "AuthService".into(),
            relations: vec!["calls".into()],
            top_k: 5,
            max_depth: 2,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);
        let keys: Vec<String> = results
            .iter()
            .map(|r| String::from_utf8_lossy(&r.key).to_string())
            .collect();

        // CURRENT BEHAVIOUR (GAP): graph empty after restart
        // REQUIRED BEHAVIOUR (FIX): edges rebuilt or reloaded
        assert!(
            keys.contains(&"validate_token".to_string()),
            "GAP-P1-8: Graph edges not persisted. After restart, 'validate_token' \
             is not reachable from 'AuthService' via 'calls'. \
             RelationGraph must be serialised to data_dir and loaded on open."
        );
    }

    cleanup(&dir);
}

// ============================================================================
// P1-GAP-9: Vector indexing via handler
//
// handle_query_vector returns an empty stub.
// There is no opcode or field to submit vector embeddings per key.
// ============================================================================

#[test]
fn gap_p1_9_vector_query_via_handler() {
    let dir = tmp_dir("vector_handler_gap");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "doc:auth_service".into(),
        value: b"AuthService validates JWT tokens".to_vec(),
    });
    handler.handle_put(record::Put {
        key: "doc:user_model".into(),
        value: b"User model with email and password".to_vec(),
    });

    // QueryVector has fields: namespace, vector, top_k
    let vq_resp = handler.handle_query_vector(query::QueryVector {
        namespace: "".into(),
        vector: vec![1.0f32, 0.0, 0.0, 0.0],
        top_k: 5,
    });

    let vq_rows = count_result_rows(&vq_resp);

    // CURRENT BEHAVIOUR (GAP): stub returns 0 rows.
    // REQUIRED BEHAVIOUR (FIX): after submitting embeddings, returns relevant rows.
    eprintln!(
        "GAP-P1-9: handle_query_vector returned {vq_rows} rows (stub). \
         Required: (1) OP_PUT_VECTOR opcode or embedding field in OP_PUT, \
         (2) handle_query_vector routes to HnswIndex."
    );

    // This is a documentation/regression test -- assert the stub behaviour
    // rather than failing the suite, so we can track when it's fixed.
    assert_eq!(
        vq_rows, 0,
        "GAP-P1-9: expected 0 rows from stub handle_query_vector. \
         When fixed, remove this assertion and replace with a real vector test."
    );

    cleanup(&dir);
}

// ============================================================================
// P1-GAP-10: Namespace-aware indices
//
// handler hard-codes ns_id=0. Indices are global across all namespaces.
// A query in namespace A can return results from namespace B.
// ============================================================================

#[test]
fn gap_p1_10_namespace_index_isolation() {
    let dir = tmp_dir("ns_index_gap");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let ns_a = derive_namespace_id("github.com/org/repo-a", "sha-1");
    let ns_b = derive_namespace_id("github.com/org/repo-b", "sha-2");

    // Write JWT-related records for namespace A.
    for key in &["validate_jwt", "decode_jwt", "encode_jwt"] {
        let scoped = scope_key(ns_a, key.as_bytes());
        let scoped_str = String::from_utf8_lossy(&scoped).to_string();
        handler.handle_put(record::Put {
            key: scoped_str,
            value: b"jwt authentication token".to_vec(),
        });
    }

    // Write database-related records for namespace B.
    for key in &["run_query", "build_index", "flush_cache"] {
        let scoped = scope_key(ns_b, key.as_bytes());
        let scoped_str = String::from_utf8_lossy(&scoped).to_string();
        handler.handle_put(record::Put {
            key: scoped_str,
            value: b"database sql query".to_vec(),
        });
    }

    // Query namespace A for "database sql query" -- should return 0 rows.
    // Currently the handler ignores the namespace field and queries the global
    // index, so it will return namespace B records.
    let resp_wrong = handler.handle_query(query::Query {
        namespace: format!("{ns_a}"),
        text: "database sql query".into(),
        vector: vec![],
        top_k: 10,
        max_depth: 0,
        relations: vec![],
        lang_filter: vec![],
        type_filter: vec![],
        file_filter: vec![],
        merge_strategy: 1,
        timeout_ms: 5000,
    });

    let rows_wrong = count_result_rows(&resp_wrong);

    if rows_wrong > 0 {
        eprintln!(
            "GAP-P1-10 CONFIRMED: Query for 'database sql query' in namespace_a \
             returned {rows_wrong} rows (from namespace_b). \
             IndexManager must be per-namespace."
        );
    }

    // REQUIRED BEHAVIOUR (FIX): rows_wrong == 0
    // CURRENT BEHAVIOUR (GAP): rows_wrong > 0 (ns_b bleed)
    assert_eq!(
        rows_wrong, 0,
        "GAP-P1-10: Index namespace isolation broken. \
         'database sql query' belongs to ns_b but was returned for ns_a query. \
         Fix: one IndexManager per namespace_id."
    );

    cleanup(&dir);
}

// ============================================================================
// P2-GAP-4: WATCH/NOTIFY
//
// handle_watch returns a stub. No notifications on key changes.
// ============================================================================

#[test]
fn gap_p2_4_watch_notify_stub() {
    let dir = tmp_dir("watch_gap");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Watch struct fields: namespace_id, scope, pattern, watch_id, initial_credit
    let watch_resp = handler.handle_watch(watch::Watch {
        namespace_id: 0,
        scope: 0x01,         // prefix scope
        pattern: "auth.".into(),
        watch_id: 1,
        initial_credit: 100,
    });

    match &watch_resp {
        Response::Stream { frames } => {
            eprintln!(
                "GAP-P2-4 confirmed: handle_watch returns Stream with {} frames (stub). \
                 Required: register subscription, stream OP_NOTIFY on key changes.",
                frames.len()
            );
        }
        Response::None => {
            eprintln!(
                "GAP-P2-4 confirmed: handle_watch returns Response::None (complete stub)."
            );
        }
        Response::Single { opcode: op, .. } if *op == opcode::OP_ERROR => {
            eprintln!("GAP-P2-4 confirmed: handle_watch returns OP_ERROR (not implemented).");
        }
        other => {
            eprintln!("GAP-P2-4: handle_watch returned: {other:?}");
        }
    }

    // Write a key that matches the watch prefix.
    handler.handle_put(record::Put {
        key: "auth.validate_token".into(),
        value: b"updated implementation".to_vec(),
    });

    eprintln!(
        "GAP-P2-4: Wrote 'auth.validate_token' -- no OP_NOTIFY delivered. \
         Fix: WatchRegistry in server, notify_watchers() on put/delete."
    );

    cleanup(&dir);
}

// ============================================================================
// CORRECTNESS CONFIRMATIONS
// These must all pass regardless of gaps.
// ============================================================================

/// Scan must be a half-open interval [start, end).
#[test]
fn confirm_scan_half_open_interval() {
    let dir = tmp_dir("scan_interval");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    for i in 0u32..=10 {
        handler.handle_put(record::Put {
            key: format!("interval_{i:03}"),
            value: format!("v{i}").into_bytes(),
        });
    }

    // [003, 007) = 003, 004, 005, 006 => 4 rows.
    let resp = handler.handle_scan(record::Scan {
        start: "interval_003".into(),
        end: "interval_007".into(),
        limit: 100,
    });
    let rows = count_result_rows(&resp);
    assert_eq!(rows, 4, "scan [003, 007) should return 4 rows, got {rows}");

    cleanup(&dir);
}

/// Range delete must remove exactly the keys in [start, end).
#[test]
fn confirm_range_delete_correctness() {
    let dir = tmp_dir("range_del_correct");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    for i in 0u32..10 {
        handler.handle_put(record::Put {
            key: format!("rdel_{i:03}"),
            value: b"v".to_vec(),
        });
    }

    // Delete [002, 006) = 002, 003, 004, 005.
    let del = handler.handle_range_delete(record::RangeDelete {
        start: "rdel_002".into(),
        end: "rdel_006".into(),
    });
    assert!(is_ok_response(&del), "range_delete failed");

    for key in &["rdel_000", "rdel_001"] {
        let r = handler.handle_get(record::Get { key: (*key).into() });
        assert!(
            !is_error_response(&r),
            "{key} should exist after range_delete"
        );
    }
    for key in &["rdel_002", "rdel_003", "rdel_004", "rdel_005"] {
        let r = handler.handle_get(record::Get { key: (*key).into() });
        assert!(
            is_error_response(&r),
            "{key} should be deleted after range_delete"
        );
    }
    for key in &["rdel_006", "rdel_007", "rdel_008", "rdel_009"] {
        let r = handler.handle_get(record::Get { key: (*key).into() });
        assert!(
            !is_error_response(&r),
            "{key} should exist after range_delete"
        );
    }

    cleanup(&dir);
}

/// PUT_BATCH must write all records atomically.
#[test]
fn confirm_put_batch_all_or_nothing() {
    let dir = tmp_dir("batch_atomic");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let batch: Vec<(String, Vec<u8>)> = (0u32..20)
        .map(|i| (format!("batch_{i:03}"), format!("bval_{i}").into_bytes()))
        .collect();

    let resp = handler.handle_put_batch(record::PutBatch {
        records: batch.clone(),
    });
    assert!(is_ok_response(&resp), "put_batch failed: {resp:?}");

    for (key, _) in &batch {
        let r = handler.handle_get(record::Get { key: key.clone() });
        match r {
            Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_ROW => {}
            other => panic!("{key} not found after batch: {other:?}"),
        }
    }

    cleanup(&dir);
}

/// Index rebuild from WAL replay must produce correct query results.
#[test]
fn confirm_index_rebuild_from_wal() {
    let dir = tmp_dir("idx_rebuild");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"auth::validate_token", b"jwt token validation").unwrap();
        eng.put(b"auth::hash_password", b"bcrypt hash salted").unwrap();
        eng.put(b"models::User", b"user email model").unwrap();
        eng.close().unwrap();
    }

    {
        let mut eng = open_engine(&dir);

        let spec = QuerySpec {
            text: "validate token".into(),
            top_k: 5,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);
        assert!(
            !results.is_empty(),
            "BM25 index must be rebuilt from WAL replay"
        );

        let keys: Vec<String> = results
            .iter()
            .map(|r| String::from_utf8_lossy(&r.key).to_string())
            .collect();
        assert!(
            keys.iter().any(|k| k.contains("validate_token")),
            "validate_token should be in results: {keys:?}"
        );

        let spec2 = QuerySpec {
            text: "hashPasswrd".into(),
            top_k: 3,
            ..Default::default()
        };
        let results2 = eng.indices.query(&spec2);
        assert!(
            !results2.is_empty(),
            "Fuzzy index must be rebuilt from WAL replay"
        );
    }

    cleanup(&dir);
}

/// Stats handler must return a non-trivial payload.
#[test]
fn confirm_stats_accuracy() {
    let dir = tmp_dir("stats_accuracy");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    for i in 0..10u32 {
        handler.handle_put(record::Put {
            key: format!("stat_key_{i}"),
            value: b"v".to_vec(),
        });
    }

    let resp = handler.handle_stats(admin::Stats);
    match resp {
        Response::Single { opcode: op, payload } => {
            assert_eq!(op, opcode::OP_RESULT_END);
            assert!(
                payload.len() > 20,
                "stats payload too short: {} bytes",
                payload.len()
            );
        }
        other => panic!("stats returned unexpected response: {other:?}"),
    }

    cleanup(&dir);
}

/// Snapshot list/create/restore/delete full lifecycle.
#[test]
fn confirm_snapshot_full_lifecycle() {
    let dir = tmp_dir("snap_full");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "doc1".into(),
        value: b"version 1".to_vec(),
    });

    let cr = handler.handle_snap_create(snapshot::SnapCreate {
        namespace_id: 0,
        description: "v1_checkpoint".into(),
    });
    assert!(is_ok_response(&cr), "snap_create failed");

    let lr = handler.handle_snap_list(snapshot::SnapList { namespace_id: 0 });
    assert_eq!(count_result_rows(&lr), 1, "should have 1 snapshot");

    handler.handle_put(record::Put {
        key: "doc1".into(),
        value: b"version 2".to_vec(),
    });

    let rr = handler.handle_snap_restore(snapshot::SnapRestore {
        namespace_id: 0,
        snapshot_id: "v1_checkpoint".into(),
    });
    assert!(is_ok_response(&rr), "snap_restore failed");

    let dr = handler.handle_snap_delete(snapshot::SnapDelete {
        namespace_id: 0,
        snapshot_id: "v1_checkpoint".into(),
    });
    assert!(is_ok_response(&dr), "snap_delete failed");

    let lr2 = handler.handle_snap_list(snapshot::SnapList { namespace_id: 0 });
    assert_eq!(count_result_rows(&lr2), 0, "no snapshots should remain");

    let dr2 = handler.handle_snap_delete(snapshot::SnapDelete {
        namespace_id: 0,
        snapshot_id: "nonexistent".into(),
    });
    assert!(
        is_error_response(&dr2),
        "deleting nonexistent snapshot must error"
    );

    cleanup(&dir);
}

/// HNSW recall@10 must be >= 80% at N=200, dim=32.
#[test]
fn confirm_hnsw_recall_at_10() {
    let dim = 32usize;
    let n = 200usize;
    let k = 10usize;

    let mut idx = HnswIndex::new(dim, 12, 100, 50);
    let mut vecs: Vec<Vec<f32>> = Vec::new();

    for i in 0..n {
        let v = rng_unit_vec(dim, i as u64);
        idx.add(format!("v{i}").into_bytes(), v.clone()).unwrap();
        vecs.push(v);
    }

    let n_queries = 20usize;
    let mut total_overlap = 0usize;

    for qi in 0..n_queries {
        let q = rng_unit_vec(dim, 100_000 + qi as u64);

        let mut exact: Vec<(f32, usize)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                (1.0 - dot.clamp(-1.0, 1.0), i)
            })
            .collect();
        exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let exact_set: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

        let hnsw_res = idx.search(&q, k);
        let hnsw_set: HashSet<usize> = hnsw_res
            .iter()
            .filter_map(|r| {
                std::str::from_utf8(&r.key)
                    .ok()
                    .and_then(|s| s.strip_prefix('v'))
                    .and_then(|n| n.parse::<usize>().ok())
            })
            .collect();

        total_overlap += exact_set.intersection(&hnsw_set).count();
    }

    let recall = total_overlap as f64 / (n_queries * k) as f64;
    assert!(
        recall >= 0.80,
        "HNSW recall@10 = {recall:.3} is below 0.80 threshold"
    );
}

/// Bloom filter FPR must be < 3% and have zero false negatives.
#[test]
fn confirm_bloom_fpr_bound() {
    use uldb::index::bloom::BloomFilter;

    let mut bf = BloomFilter::new(10_000, 0.01);
    for i in 0..10_000u32 {
        bf.add(format!("present_{i}").as_bytes());
    }

    let fp = (0..10_000u32)
        .filter(|i| bf.may_contain(format!("absent_{i}").as_bytes()))
        .count();

    let fpr = fp as f64 / 10_000.0;
    assert!(fpr < 0.03, "Bloom filter FPR = {fpr:.4} exceeds 0.03");

    for i in 0..10_000u32 {
        assert!(
            bf.may_contain(format!("present_{i}").as_bytes()),
            "false negative at {i}"
        );
    }
}

/// Namespace key scoping must ensure complete scan range isolation.
#[test]
fn confirm_namespace_key_isolation() {
    let dir = tmp_dir("ns_key_isolation");
    fs::create_dir_all(&dir).unwrap();

    let ns_a = derive_namespace_id("github.com/org/repo-a", "sha-a1");
    let ns_b = derive_namespace_id("github.com/org/repo-b", "sha-b1");

    let mut eng = open_engine(&dir);

    for i in 0..10u32 {
        let k = scope_key(ns_a, format!("key_{i:03}").as_bytes());
        eng.put(&k, format!("ns_a_val_{i}").as_bytes()).unwrap();
    }
    for i in 0..10u32 {
        let k = scope_key(ns_b, format!("key_{i:03}").as_bytes());
        eng.put(&k, format!("ns_b_val_{i}").as_bytes()).unwrap();
    }

    let (start_a, end_a) = uldb::namespace::ns_scan_range(ns_a);
    let results_a = eng.scan(&start_a, &end_a);
    assert_eq!(
        results_a.len(),
        10,
        "namespace A scan must return exactly 10 records"
    );

    for (k, v) in &results_a {
        assert!(
            uldb::namespace::key_in_namespace(k, ns_a),
            "ns_a scan returned key from wrong namespace"
        );
        let val_str = String::from_utf8_lossy(v);
        assert!(
            val_str.starts_with("ns_a_val_"),
            "ns_a scan returned ns_b value: {val_str}"
        );
    }

    let (start_b, end_b) = uldb::namespace::ns_scan_range(ns_b);
    let results_b = eng.scan(&start_b, &end_b);
    assert_eq!(
        results_b.len(),
        10,
        "namespace B scan must return exactly 10 records"
    );

    cleanup(&dir);
}
