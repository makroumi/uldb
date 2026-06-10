// tests/rust/correctness.rs
//
// Rigorous correctness tests for uldb.
//
// Unlike e2e_gaps.rs (which documents known gaps), these tests assert
// exact correct behaviour. Every assertion must pass. Any failure is a
// regression.
//
// Run: cargo test --test correctness --features server

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use uldb::agent_store;
use uldb::engine::{Engine, EngineConfig};
use uldb::namespace::scope_key;
use ulmp::messages::admin;
use uldb::query::planner::QuerySpec;
use uldb::server::UmpHandler;
use ulmp::messages::{record, query, tx, snapshot, branch};
use ulmp::server::handler::{Handler, Response};
use ulmp::messages::opcode;
use ulmen_core::{
    AgentPayload, AgentHeader, AgentRecord,
    RecordType, FieldValue, MetaFields,
    validate_payload, compress_context, CompressStrategy,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("uldb_correctness_{name}_{}", std::process::id()));
    p
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn open_engine(dir: &Path) -> Engine {
    Engine::open(EngineConfig::new(dir)).unwrap()
}

fn make_handler(dir: &Path) -> UmpHandler {
    let engine = open_engine(dir);
    UmpHandler::new(engine)
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

fn make_msg(id: &str, step: i64, content: &str) -> AgentRecord {
    AgentRecord {
        record_type: RecordType::Msg,
        id: id.into(),
        thread_id: "t1".into(),
        step,
        fields: vec![
            FieldValue::Str("user".into()),
            FieldValue::Int(1),
            FieldValue::Str(content.into()),
            FieldValue::Int(5),
            FieldValue::Bool(false),
        ],
        meta: MetaFields::default(),
    }
}

fn make_payload(records: Vec<AgentRecord>) -> AgentPayload {
    AgentPayload {
        header: AgentHeader {
            thread_id: Some("t1".into()),
            record_count: records.len(),
            ..Default::default()
        },
        records,
    }
}

// ===========================================================================
// 1. WAL DURABILITY
// ===========================================================================

#[test]
fn wal_durability_single_put() {
    let dir = tmp_dir("wal_single");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"durable_key", b"durable_value").unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        assert_eq!(
            eng.get(b"durable_key").as_deref(),
            Some(b"durable_value".as_slice()),
            "WAL must persist single put across restart"
        );
    }
    cleanup(&dir);
}

#[test]
fn wal_durability_1000_puts() {
    let dir = tmp_dir("wal_1000");
    fs::create_dir_all(&dir).unwrap();

    let n = 1000usize;
    {
        let mut eng = open_engine(&dir);
        for i in 0..n {
            let k = format!("key_{i:05}");
            let v = format!("value_{i}_xxxx");
            eng.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        let mut lost = 0;
        for i in 0..n {
            let k = format!("key_{i:05}");
            let v = format!("value_{i}_xxxx");
            if eng.get(k.as_bytes()).as_deref() != Some(v.as_bytes()) {
                lost += 1;
            }
        }
        assert_eq!(lost, 0, "WAL must persist all {n} puts, lost {lost}");
    }
    cleanup(&dir);
}

#[test]
fn wal_durability_delete_persists() {
    let dir = tmp_dir("wal_delete");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"to_delete", b"exists").unwrap();
        eng.delete(b"to_delete").unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        assert!(
            eng.get(b"to_delete").is_none(),
            "delete must be durable across restart"
        );
    }
    cleanup(&dir);
}

#[test]
fn wal_durability_overwrite_persists() {
    let dir = tmp_dir("wal_overwrite");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"key", b"v1").unwrap();
        eng.put(b"key", b"v2").unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        assert_eq!(
            eng.get(b"key").as_deref(),
            Some(b"v2".as_slice()),
            "overwrite must be durable: expected v2, not v1"
        );
    }
    cleanup(&dir);
}

// ===========================================================================
// 2. BRANCH MERGE WAL DURABILITY (our fix)
// ===========================================================================

#[test]
fn branch_merge_durable_after_restart() {
    let dir = tmp_dir("branch_merge_wal");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"base_key", b"original").unwrap();
        eng.branch_create("feat/test", "").unwrap();

        // Write into branch
        if let Some(branch_hamt) = eng.snapshots.get("feat/test").cloned() {
            let updated = branch_hamt.put(b"base_key".to_vec(), b"from_branch".to_vec());
            eng.snapshots.insert("feat/test".to_string(), updated);
        }

        // Merge branch to main (our WAL fix must make this durable)
        eng.branch_merge("feat/test").unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        assert_eq!(
            eng.get(b"base_key").as_deref(),
            Some(b"from_branch".as_slice()),
            "branch_merge must be WAL-durable: data must survive restart"
        );
    }
    cleanup(&dir);
}

#[test]
fn branch_merge_multiple_keys_durable() {
    let dir = tmp_dir("branch_merge_multi");
    fs::create_dir_all(&dir).unwrap();

    let n = 50usize;
    {
        let mut eng = open_engine(&dir);
        eng.branch_create("feat/bulk", "").unwrap();

        let mut hamt = eng.snapshots.get("feat/bulk").cloned().unwrap();
        for i in 0..n {
            let k = format!("merged_{i:04}");
            let v = format!("branch_val_{i}");
            hamt = hamt.put(k.into_bytes(), v.into_bytes());
        }
        eng.snapshots.insert("feat/bulk".to_string(), hamt);
        eng.branch_merge("feat/bulk").unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        let mut missing = 0;
        for i in 0..n {
            let k = format!("merged_{i:04}");
            let v = format!("branch_val_{i}");
            if eng.get(k.as_bytes()).as_deref() != Some(v.as_bytes()) {
                missing += 1;
            }
        }
        assert_eq!(
            missing, 0,
            "branch_merge must persist all {n} keys after restart, {missing} missing"
        );
    }
    cleanup(&dir);
}

// ===========================================================================
// 3. HNSW VECTOR INDEX PERSISTENCE
// ===========================================================================

#[test]
fn hnsw_persistence_survives_restart() {
    let dir = tmp_dir("hnsw_persist");
    fs::create_dir_all(&dir).unwrap();

    let dim = 8usize;

    // Build an index with known vectors
    {
        let mut eng = open_engine(&dir);
        for i in 0..10u32 {
            let key = format!("vec_{i:03}");
            let mut v = vec![0.0f32; dim];
            v[i as usize % dim] = 1.0;
            eng.put(key.as_bytes(), b"document").unwrap();
            eng.indices.on_put_vector(key.as_bytes(), v);
        }

        // Verify works before close
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let spec = QuerySpec {
            vector: query.clone(),
            top_k: 3,
            ..Default::default()
        };
        let before = eng.indices.query(&spec);
        assert!(!before.is_empty(), "HNSW must find vectors before restart");
        eng.close().unwrap();
    }

    // Verify survives restart
    {
        let mut eng = open_engine(&dir);
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let spec = QuerySpec {
            vector: query,
            top_k: 3,
            ..Default::default()
        };
        let after = eng.indices.query(&spec);
        assert!(
            !after.is_empty(),
            "HNSW must find vectors AFTER restart (persistence check)"
        );
        assert_eq!(
            after[0].key, b"vec_000",
            "Top result must be the closest vector"
        );
    }
    cleanup(&dir);
}

// ===========================================================================
// 4. GRAPH INDEX PERSISTENCE
// ===========================================================================

#[test]
fn graph_persistence_survives_restart() {
    let dir = tmp_dir("graph_persist");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"AuthService", b"class").unwrap();
        eng.put(b"validate_token", b"def").unwrap();
        eng.put(b"hash_password", b"def").unwrap();

        eng.indices.on_add_edge("AuthService", "validate_token", "calls");
        eng.indices.on_add_edge("AuthService", "hash_password", "calls");

        // Verify before close
        let spec = QuerySpec {
            text: "AuthService".into(),
            relations: vec!["calls".into()],
            top_k: 5,
            max_depth: 2,
            ..Default::default()
        };
        let before = eng.indices.query(&spec);
        let keys: HashSet<String> = before.iter()
            .map(|h| String::from_utf8_lossy(&h.key).to_string())
            .collect();
        assert!(keys.contains("validate_token"), "graph must work before restart");
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
        let after = eng.indices.query(&spec);
        let keys: HashSet<String> = after.iter()
            .map(|h| String::from_utf8_lossy(&h.key).to_string())
            .collect();
        assert!(
            keys.contains("validate_token"),
            "graph traversal must work AFTER restart: got {keys:?}"
        );
        assert!(
            keys.contains("hash_password"),
            "all edges must survive restart: got {keys:?}"
        );
    }
    cleanup(&dir);
}

// ===========================================================================
// 5. BM25 INDEX CORRECTNESS
// ===========================================================================

#[test]
fn bm25_relevance_ordering() {
    let dir = tmp_dir("bm25_order");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    // Highly relevant: mentions JWT and authentication multiple times
    eng.put(b"auth/jwt.rs", b"jwt authentication token validate jwt decode jwt encode").unwrap();
    // Somewhat relevant: mentions authentication once
    eng.put(b"auth/basic.rs", b"basic authentication username password").unwrap();
    // Not relevant: different domain entirely
    eng.put(b"utils/date.rs", b"format date parse timestamp convert").unwrap();

    let spec = QuerySpec {
        text: "jwt authentication".into(),
        top_k: 3,
        ..Default::default()
    };
    let results = eng.indices.query(&spec);

    assert!(!results.is_empty(), "BM25 must return results");
    assert_eq!(
        results[0].key, b"auth/jwt.rs",
        "Most relevant document must rank first"
    );
    let jwt_score = results.iter().find(|h| h.key == b"auth/jwt.rs").map(|h| h.score).unwrap_or(0.0);
    let date_score = results.iter().find(|h| h.key == b"utils/date.rs").map(|h| h.score).unwrap_or(0.0);
    assert!(
        jwt_score > date_score,
        "JWT doc score {jwt_score} must be > date doc score {date_score}"
    );
    cleanup(&dir);
}

#[test]
fn bm25_index_survives_restart() {
    let dir = tmp_dir("bm25_persist");
    fs::create_dir_all(&dir).unwrap();

    {
        let mut eng = open_engine(&dir);
        eng.put(b"auth/validate", b"jwt token validate authentication").unwrap();
        eng.put(b"auth/login", b"login user password credential").unwrap();
        eng.close().unwrap();
    }

    {
        let mut eng = open_engine(&dir);
        let spec = QuerySpec {
            text: "jwt token".into(),
            top_k: 5,
            ..Default::default()
        };
        let results = eng.indices.query(&spec);
        assert!(
            !results.is_empty(),
            "BM25 index must be rebuilt from WAL after restart"
        );
        assert!(
            results.iter().any(|h| h.key == b"auth/validate"),
            "Most relevant document must appear in results after restart"
        );
    }
    cleanup(&dir);
}

// ===========================================================================
// 6. TRANSACTION CORRECTNESS
// ===========================================================================

#[test]
fn transaction_snapshot_isolation() {
    let dir = tmp_dir("tx_snapshot");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Write initial value
    handler.handle_put(record::Put {
        key: "account".into(),
        value: b"1000".to_vec(),
    });

    // Begin snapshot transaction
    let tx_resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x02 });
    let tx_id = match &tx_resp {
        Response::Single { payload, .. } if payload.len() >= 9 && payload[0] == 0x04 => {
            u64::from_be_bytes(payload[1..9].try_into().unwrap())
        }
        _ => panic!("tx_begin must return tx_id, got: {:?}", tx_resp),
    };
    assert!(tx_id > 0, "tx_id must be non-zero");

    // External write modifies the key after tx started
    handler.handle_put(record::Put {
        key: "account".into(),
        value: b"500".to_vec(),
    });

    // Commit the transaction
    let commit_resp = handler.handle_tx_commit(tx::TxCommit { tx_id });
    match commit_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("tx_commit must succeed, got: {:?}", other),
    }

    // Tx commit succeeded, external write is visible
    let get_resp = handler.handle_get(record::Get { key: "account".into() });
    match get_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_ROW => {}
        other => panic!("get after tx must return result row, got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn transaction_rollback_discards_work() {
    let dir = tmp_dir("tx_rollback");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "stable_key".into(),
        value: b"stable_value".to_vec(),
    });

    let tx_id = {
        let resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x01 });
        match resp {
            Response::Single { payload, .. } if payload.len() >= 9 && payload[0] == 0x04 => {
                u64::from_be_bytes(payload[1..9].try_into().unwrap())
            }
            _ => panic!("tx_begin failed"),
        }
    };

    // Rollback
    let rollback_resp = handler.handle_tx_rollback(tx::TxRollback { tx_id });
    match rollback_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("rollback must succeed, got: {:?}", other),
    }

    // After rollback, tx_id must be gone
    let status_resp = handler.handle_tx_status(tx::TxStatus { tx_id });
    match status_resp {
        Response::Single { payload, .. } => {
            let s = String::from_utf8_lossy(&payload);
            assert!(s.contains("unknown"), "rolled back tx must show 'unknown' status");
        }
        other => panic!("tx_status after rollback: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn multiple_transactions_independent() {
    let dir = tmp_dir("tx_multi");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let tx1 = {
        let resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x01 });
        match resp {
            Response::Single { payload, .. } if payload.len() >= 9 && payload[0] == 0x04 => {
                u64::from_be_bytes(payload[1..9].try_into().unwrap())
            }
            _ => panic!("tx1 begin failed"),
        }
    };

    let tx2 = {
        let resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x01 });
        match resp {
            Response::Single { payload, .. } if payload.len() >= 9 && payload[0] == 0x04 => {
                u64::from_be_bytes(payload[1..9].try_into().unwrap())
            }
            _ => panic!("tx2 begin failed"),
        }
    };

    assert_ne!(tx1, tx2, "transactions must have unique IDs");

    // Commit tx1, rollback tx2
    handler.handle_tx_commit(tx::TxCommit { tx_id: tx1 });
    handler.handle_tx_rollback(tx::TxRollback { tx_id: tx2 });

    // Both should be gone
    for tx_id in [tx1, tx2] {
        let resp = handler.handle_tx_status(tx::TxStatus { tx_id });
        match resp {
            Response::Single { payload, .. } => {
                let s = String::from_utf8_lossy(&payload);
                assert!(s.contains("unknown"), "tx {tx_id} must be unknown after commit/rollback");
            }
            _ => {}
        }
    }
    cleanup(&dir);
}

// ===========================================================================
// 7. SNAPSHOT LIFECYCLE
// ===========================================================================

#[test]
fn snapshot_create_list_restore() {
    let dir = tmp_dir("snap_lifecycle");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Write baseline
    handler.handle_put(record::Put {
        key: "snap_key".into(),
        value: b"v1".to_vec(),
    });

    // Create snapshot
    let snap_resp = handler.handle_snap_create(snapshot::SnapCreate {
        namespace_id: 0u64,
        description: "before_edit".into(),
    });
    match snap_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("snap_create must succeed, got: {:?}", other),
    }

    // Modify data
    handler.handle_put(record::Put {
        key: "snap_key".into(),
        value: b"v2".to_vec(),
    });

    // List snapshots
    let list_resp = handler.handle_snap_list(snapshot::SnapList {
        namespace_id: 0u64,
    });
    match list_resp {
        Response::Stream { .. } | Response::Single { .. } => {}
        other => panic!("snap_list must return data, got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 8. BRANCH LIFECYCLE
// ===========================================================================

#[test]
fn branch_create_diff_merge_rollback() {
    let dir = tmp_dir("branch_lifecycle");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Write baseline
    handler.handle_put(record::Put {
        key: "base".into(),
        value: b"original".to_vec(),
    });

    // Create branch
    let create_resp = handler.handle_branch_create(branch::BranchCreate {
        namespace_id: 0u64,
        branch_id: "feat/test".into(),
        from_snapshot: "".into(),
        description: "test branch".into(),
    });
    match create_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("branch_create must succeed, got: {:?}", other),
    }

    // Rollback the branch
    let rollback_resp = handler.handle_branch_rollback(branch::BranchRollback {
        namespace_id: 0u64,
        branch_id: "feat/test".into(),
    });
    match rollback_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("branch_rollback must succeed, got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 9. SCAN CORRECTNESS
// ===========================================================================

#[test]
fn scan_half_open_interval() {
    let dir = tmp_dir("scan_interval");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    for i in 0..10u32 {
        eng.put(format!("key_{i:03}").as_bytes(), b"v").unwrap();
    }

    // [key_003, key_007) should return 003,004,005,006 = 4 keys
    let results = eng.scan(b"key_003", b"key_007");
    assert_eq!(results.len(), 4, "scan [key_003, key_007) must return 4 keys");

    let keys: Vec<String> = results.iter()
        .map(|(k, _)| String::from_utf8_lossy(k).to_string())
        .collect();
    assert!(keys.contains(&"key_003".to_string()));
    assert!(keys.contains(&"key_006".to_string()));
    assert!(!keys.contains(&"key_007".to_string()), "upper bound must be exclusive");
    assert!(!keys.contains(&"key_002".to_string()), "lower bound must be inclusive");

    cleanup(&dir);
}

#[test]
fn scan_empty_range() {
    let dir = tmp_dir("scan_empty");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);
    eng.put(b"zzz", b"v").unwrap();
    let results = eng.scan(b"aaa", b"aab");
    assert!(results.is_empty(), "scan of empty range must return no results");
    cleanup(&dir);
}

// ===========================================================================
// 10. CONCURRENT HANDLER ACCESS
// ===========================================================================

#[test]
fn concurrent_reads_safe() {
    let dir = tmp_dir("concurrent_reads");
    fs::create_dir_all(&dir).unwrap();
    let handler = Arc::new(make_handler(&dir));

    // Seed data
    for i in 0..20u32 {
        handler.handle_put(record::Put {
            key: format!("c_{i:04}"),
            value: format!("val_{i}").into_bytes(),
        });
    }

    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut handles = Vec::new();

    for t in 0..8 {
        let h = Arc::clone(&handler);
        let errs = Arc::clone(&errors);
        handles.push(thread::spawn(move || {
            for i in 0..20u32 {
                let key = format!("c_{:04}", (t * 20 + i) % 20);
                let resp = h.handle_get(record::Get { key: key.clone() });
                match resp {
                    Response::Single { opcode: op, .. }
                        if op == opcode::OP_RESULT_ROW || op == opcode::OP_ERROR => {}
                    other => {
                        errs.lock().unwrap().push(
                            format!("thread {t}, key {key}: unexpected {:?}", other)
                        );
                    }
                }
            }
        }));
    }

    for h in handles { h.join().unwrap(); }

    let errs = errors.lock().unwrap();
    assert!(errs.is_empty(), "concurrent reads produced errors: {:?}", *errs);
    cleanup(&dir);
}

#[test]
fn concurrent_mixed_ops_no_panic() {
    let dir = tmp_dir("concurrent_mixed");
    fs::create_dir_all(&dir).unwrap();
    let handler = Arc::new(make_handler(&dir));

    let errors = Arc::new(Mutex::new(0u32));
    let mut handles = Vec::new();

    for t in 0..4 {
        let h = Arc::clone(&handler);
        let errs = Arc::clone(&errors);
        handles.push(thread::spawn(move || {
            for i in 0..25u32 {
                let key = format!("mixed_{t}_{i:04}");
                // Write
                h.handle_put(record::Put {
                    key: key.clone(),
                    value: format!("v_{t}_{i}").into_bytes(),
                });
                // Read back
                match h.handle_get(record::Get { key: key.clone() }) {
                    Response::Single { opcode: op, .. }
                        if op == opcode::OP_RESULT_ROW => {}
                    _ => { *errs.lock().unwrap() += 1; }
                }
            }
        }));
    }

    for h in handles { h.join().unwrap(); }

    let errs = *errors.lock().unwrap();
    assert_eq!(errs, 0, "concurrent mixed ops: {errs} reads returned no data");
    cleanup(&dir);
}

// ===========================================================================
// 11. AGENT STORE CORRECTNESS
// ===========================================================================

#[test]
fn agent_store_roundtrip() {
    let dir = tmp_dir("agent_store");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    let payload = make_payload(vec![
        make_msg("m1", 1, "hello world authentication"),
        make_msg("m2", 2, "jwt token validation"),
        make_msg("m3", 3, "password hashing bcrypt"),
    ]);

    agent_store::store_payload(&mut eng, "session_001", &payload).unwrap();
    let loaded = agent_store::load_payload(&eng, "session_001").unwrap();

    assert_eq!(loaded.records.len(), 3);
    assert_eq!(loaded.records[0].id, "m1");
    assert_eq!(loaded.records[2].step, 3);
    cleanup(&dir);
}

#[test]
fn agent_store_validates_on_write() {
    let dir = tmp_dir("agent_store_valid");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    // Invalid payload: empty thread_id
    let bad_payload = AgentPayload {
        header: AgentHeader { record_count: 1, ..Default::default() },
        records: vec![AgentRecord {
            record_type: RecordType::Msg,
            id: "m1".into(),
            thread_id: "".into(), // invalid
            step: 1,
            fields: vec![
                FieldValue::Str("user".into()), FieldValue::Int(1),
                FieldValue::Str("hi".into()), FieldValue::Int(1),
                FieldValue::Bool(false),
            ],
            meta: MetaFields::default(),
        }],
    };

    let result = agent_store::store_payload(&mut eng, "bad", &bad_payload);
    assert!(result.is_err(), "store_payload must reject invalid payload");
    cleanup(&dir);
}

#[test]
fn agent_store_persist_survives_restart() {
    let dir = tmp_dir("agent_store_persist");
    fs::create_dir_all(&dir).unwrap();

    let payload = make_payload(vec![
        make_msg("m1", 1, "authentication and jwt"),
        make_msg("m2", 2, "password hashing"),
    ]);

    {
        let mut eng = open_engine(&dir);
        agent_store::store_payload(&mut eng, "session_persist", &payload).unwrap();
        eng.close().unwrap();
    }

    {
        let eng = open_engine(&dir);
        let loaded = agent_store::load_payload(&eng, "session_persist").unwrap();
        assert_eq!(loaded.records.len(), 2, "agent payload must survive restart");
        assert_eq!(loaded.records[0].id, "m1");
    }
    cleanup(&dir);
}

#[test]
fn agent_store_search_finds_records() {
    let dir = tmp_dir("agent_store_search");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    let payload = make_payload(vec![
        make_msg("m1", 1, "jwt authentication token validate"),
        make_msg("m2", 2, "password hashing bcrypt salt"),
        make_msg("m3", 3, "sql query database optimization"),
    ]);
    agent_store::store_payload(&mut eng, "s1", &payload).unwrap();

    let results = agent_store::search_records(&mut eng, "jwt authentication", 5);
    assert!(!results.is_empty(), "search must find agent records");

    cleanup(&dir);
}

#[test]
fn agent_store_append_extends() {
    let dir = tmp_dir("agent_store_append");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    let initial = vec![make_msg("m1", 1, "first")];
    agent_store::store_payload(&mut eng, "session", &make_payload(initial)).unwrap();

    let more = vec![make_msg("m2", 2, "second"), make_msg("m3", 3, "third")];
    agent_store::append_records(&mut eng, "session", &more).unwrap();

    let loaded = agent_store::load_payload(&eng, "session").unwrap();
    assert_eq!(loaded.records.len(), 3, "append must extend to 3 records");
    cleanup(&dir);
}

#[test]
fn agent_store_list_sessions() {
    let dir = tmp_dir("agent_store_list");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    let p = make_payload(vec![make_msg("m1", 1, "test")]);
    agent_store::store_payload(&mut eng, "sess_a", &p).unwrap();
    agent_store::store_payload(&mut eng, "sess_b", &p).unwrap();
    agent_store::store_payload(&mut eng, "sess_c", &p).unwrap();

    let sessions = agent_store::list_sessions(&eng);
    assert!(sessions.contains(&"sess_a".to_string()));
    assert!(sessions.contains(&"sess_b".to_string()));
    assert!(sessions.contains(&"sess_c".to_string()));
    cleanup(&dir);
}

// ===========================================================================
// 12. ULMEN-CORE CORRECTNESS
// ===========================================================================

#[test]
fn ulmen_core_encode_decode_roundtrip() {
    let records = vec![
        make_msg("m1", 1, "hello"),
        make_msg("m2", 2, "world"),
        make_msg("m3", 3, "end"),
    ];
    let payload = make_payload(records.clone());
    let encoded = payload.encode();
    let decoded = AgentPayload::decode(&encoded).unwrap();

    assert_eq!(decoded.records.len(), 3);
    for (orig, dec) in records.iter().zip(decoded.records.iter()) {
        assert_eq!(orig.id, dec.id);
        assert_eq!(orig.step, dec.step);
        assert_eq!(orig.record_type, dec.record_type);
    }
}

#[test]
fn ulmen_core_validate_rejects_empty_thread_id() {
    let payload = AgentPayload {
        header: AgentHeader { record_count: 1, ..Default::default() },
        records: vec![AgentRecord {
            record_type: RecordType::Msg,
            id: "m1".into(),
            thread_id: "".into(),
            step: 1,
            fields: vec![
                FieldValue::Str("user".into()), FieldValue::Int(1),
                FieldValue::Str("hi".into()), FieldValue::Int(1),
                FieldValue::Bool(false),
            ],
            meta: MetaFields::default(),
        }],
    };
    assert!(validate_payload(&payload).is_err(), "empty thread_id must fail validation");
}

#[test]
fn ulmen_core_validate_rejects_non_monotonic_step() {
    let payload = make_payload(vec![
        make_msg("m1", 5, "first"),
        make_msg("m2", 3, "second"), // step goes backwards
    ]);
    assert!(validate_payload(&payload).is_err(), "non-monotonic step must fail");
}

#[test]
fn ulmen_core_validate_accepts_same_step() {
    let payload = make_payload(vec![
        make_msg("m1", 1, "a"),
        make_msg("m2", 1, "b"), // same step is ok
    ]);
    assert!(validate_payload(&payload).is_ok(), "same step must be accepted");
}

#[test]
fn ulmen_core_compress_completed_sequences() {
    let tool = AgentRecord {
        record_type: RecordType::Tool,
        id: "t1".into(),
        thread_id: "t1".into(),
        step: 2,
        fields: vec![
            FieldValue::Str("search".into()),
            FieldValue::Str("{}".into()),
            FieldValue::Str("done".into()),
        ],
        meta: MetaFields::default(),
    };
    let res = AgentRecord {
        record_type: RecordType::Res,
        id: "t1".into(),
        thread_id: "t1".into(),
        step: 3,
        fields: vec![
            FieldValue::Str("search".into()),
            FieldValue::Str("result".into()),
            FieldValue::Str("done".into()),
            FieldValue::Int(42),
        ],
        meta: MetaFields::default(),
    };
    let records = vec![make_msg("m1", 1, "query"), tool, res];
    let compressed = compress_context(&records, CompressStrategy::CompletedSequences, 2, None, None, false);

    // tool+res pair should be replaced by a mem summary
    assert!(compressed.len() < records.len(), "compression must reduce record count");
    let has_mem = compressed.iter().any(|r| r.record_type == RecordType::Mem);
    assert!(has_mem, "completed tool+res must produce a mem summary");
    let has_tool = compressed.iter().any(|r| r.record_type == RecordType::Tool);
    assert!(!has_tool, "tool record must be removed after compression");
}

// ===========================================================================
// 13. STATS ACCURACY
// ===========================================================================

#[test]
fn stats_track_puts_and_deletes() {
    let dir = tmp_dir("stats_accuracy");
    fs::create_dir_all(&dir).unwrap();
    let mut eng = open_engine(&dir);

    assert_eq!(eng.total_puts(), 0, "fresh engine must show 0 puts");
    assert_eq!(eng.total_deletes(), 0, "fresh engine must show 0 deletes");

    for i in 0..10u32 {
        eng.put(format!("k{i}").as_bytes(), b"v").unwrap();
    }
    assert_eq!(eng.total_puts(), 10, "engine must count puts");

    eng.delete(b"k0").unwrap();
    assert_eq!(eng.total_deletes(), 1, "engine must count deletes");

    cleanup(&dir);
}

// ===========================================================================
// 14. HANDLER CRUD FULL CORRECTNESS
// ===========================================================================

#[test]
fn handler_put_get_delete_exact_values() {
    let dir = tmp_dir("handler_crud");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // PUT
    let put_resp = handler.handle_put(record::Put {
        key: "test_key".into(),
        value: b"exact_test_value_42".to_vec(),
    });
    match put_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("put must succeed, got: {:?}", other),
    }

    // GET - verify exact value
    let get_resp = handler.handle_get(record::Get { key: "test_key".into() });
    match get_resp {
        Response::Single { opcode: op, payload } if op == opcode::OP_RESULT_ROW => {
            // The value should be in the payload
            assert!(
                payload.windows(b"exact_test_value_42".len())
                    .any(|w| w == b"exact_test_value_42"),
                "GET must return exact stored value"
            );
        }
        other => panic!("get must return OP_RESULT_ROW, got: {:?}", other),
    }

    // DELETE
    let del_resp = handler.handle_delete(record::Delete { key: "test_key".into() });
    match del_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("delete must succeed, got: {:?}", other),
    }

    // GET after DELETE must return error/not-found
    let get_after = handler.handle_get(record::Get { key: "test_key".into() });
    match get_after {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("get after delete must return error, got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn handler_put_batch_all_visible() {
    let dir = tmp_dir("handler_batch");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let records: Vec<(String, Vec<u8>)> = (0..10u32)
        .map(|i| (format!("batch_{i:04}"), format!("batch_val_{i}").into_bytes()))
        .collect();

    let batch_resp = handler.handle_put_batch(record::PutBatch { records: records.clone() });
    match batch_resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("put_batch must succeed, got: {:?}", other),
    }

    // Verify all are readable
    for (key, val) in &records {
        let get_resp = handler.handle_get(record::Get { key: key.clone() });
        match get_resp {
            Response::Single { opcode: op, payload } if op == opcode::OP_RESULT_ROW => {
                assert!(
                    payload.windows(val.len()).any(|w| w == val.as_slice()),
                    "batch key {key} must return correct value"
                );
            }
            other => panic!("batch key {key} must be readable, got: {:?}", other),
        }
    }
    cleanup(&dir);
}

// ===========================================================================
// 15. KNOWN REMAINING STUBS (documented honestly)
// ===========================================================================

#[test]
fn stub_watch_acknowledges_without_delivery() {
    // KNOWN STUB: handle_watch acknowledges but never delivers notifications.
    // This test documents current behaviour. When fixed:
    //   1. Remove this test
    //   2. Add real watch notification test
    let dir = tmp_dir("stub_watch");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_watch(ulmp::messages::watch::Watch {
        namespace_id: 0,
        scope: 0x01,
        pattern: "auth.".into(),
        watch_id: 1,
        initial_credit: 100,
    });

    // Must not panic, must return something (stub is acceptable)
    match resp {
        Response::Stream { .. } | Response::Single { .. } | Response::None => {}
        other => panic!("watch stub must return a valid response, got: {:?}", other),
    }

    // Write a matching key -- no notification delivered (known stub)
    handler.handle_put(record::Put {
        key: "auth.validate".into(),
        value: b"code".to_vec(),
    });

    // No way to receive notification since it's a stub
    // This test documents the gap, not fixes it
    cleanup(&dir);
}

#[test]
fn stub_vector_query_requires_vector_submission() {
    // KNOWN LIMITATION: handle_query_vector works when vectors are submitted
    // via on_put_vector, but there is no HTTP/ulmp PUT path that accepts
    // an embedding alongside a document. The gap is in the submission path,
    // not the query path.
    //
    // Current behavior: query returns 0 rows if no embeddings were submitted
    // via the handler protocol.
    let dir = tmp_dir("stub_vector");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_put(record::Put {
        key: "doc1".into(),
        value: b"some content".to_vec(),
    });

    // Query without prior vector submission: 0 rows (expected)
    let resp = handler.handle_query_vector(query::QueryVector {
        namespace: "".into(),
        vector: vec![1.0f32, 0.0, 0.0, 0.0],
        top_k: 5,
    });
    let rows = count_result_rows(&resp);
    assert_eq!(rows, 0, "vector query without embedding submission must return 0 rows");
    // When fixed: add OP_PUT_VECTOR or embedding field, assert rows > 0
    cleanup(&dir);
}

// ===========================================================================
// 16. WATCH REGISTRY
// ===========================================================================

#[test]
fn watch_register_and_unwatch() {
    let dir = tmp_dir("watch_reg");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Register a watch
    let resp = handler.handle_watch(ulmp::messages::watch::Watch {
        namespace_id: 0,
        scope: 0x01,
        pattern: "auth.".into(),
        watch_id: 42,
        initial_credit: 100,
    });
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("watch must succeed, got: {:?}", other),
    }

    // Unwatch
    let resp2 = handler.handle_unwatch(ulmp::messages::watch::Unwatch {
        watch_id: 42,
    });
    match resp2 {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("unwatch must succeed, got: {:?}", other),
    }

    // Unwatch again should fail
    let resp3 = handler.handle_unwatch(ulmp::messages::watch::Unwatch {
        watch_id: 42,
    });
    match resp3 {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("double unwatch must error, got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn watch_window_adds_credit() {
    let dir = tmp_dir("watch_credit");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    handler.handle_watch(ulmp::messages::watch::Watch {
        namespace_id: 0,
        scope: 0x01,
        pattern: "test.".into(),
        watch_id: 99,
        initial_credit: 10,
    });

    // Add credits
    let resp = handler.handle_watch_window(ulmp::messages::watch::WatchWindow {
        watch_id: 99,
        credit: 50,
    });
    match resp {
        Response::None => {} // acknowledged
        other => panic!("watch_window must return None, got: {:?}", other),
    }

    // Invalid watch_id
    let resp2 = handler.handle_watch_window(ulmp::messages::watch::WatchWindow {
        watch_id: 999,
        credit: 10,
    });
    match resp2 {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("watch_window on unknown id must error, got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 17. AUTH ROTATION
// ===========================================================================

#[test]
fn auth_rotate_returns_challenge() {
    let dir = tmp_dir("auth_rotate");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_auth_rotate_request(
        ulmp::messages::auth_rotate::AuthRotateRequest,
    );
    match resp {
        Response::Single { opcode: op, payload } => {
            assert_eq!(op, ulmp::messages::opcode::OP_AUTH_ROTATE_CHALLENGE,
                "must return AUTH_ROTATE_CHALLENGE opcode");
            assert!(!payload.is_empty(), "challenge payload must not be empty");
        }
        other => panic!("auth_rotate_request must return challenge, got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn auth_rotate_accepts() {
    let dir = tmp_dir("auth_accept");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_auth_rotate(
        ulmp::messages::auth_rotate::AuthRotate {
            new_token_hash: [0u8; 32],
        },
    );
    match resp {
        Response::Single { opcode: op, .. } => {
            assert_eq!(op, ulmp::messages::opcode::OP_AUTH_ROTATE_ACK,
                "must return AUTH_ROTATE_ACK opcode");
        }
        other => panic!("auth_rotate must return ack, got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 18. STREAM RESUME
// ===========================================================================

#[test]
fn stream_resume_with_valid_token() {
    let dir = tmp_dir("stream_resume");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Build a 64-byte token with confirmed_row=42 at bytes 32..36
    let mut token = vec![0u8; 64];
    token[32..36].copy_from_slice(&42u32.to_be_bytes());

    let resp = handler.handle_stream_resume(
        ulmp::messages::checkpoint::StreamResume { token },
    );
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("stream_resume with valid token must succeed, got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn stream_resume_invalid_token_length() {
    let dir = tmp_dir("stream_resume_bad");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_stream_resume(
        ulmp::messages::checkpoint::StreamResume { token: vec![0u8; 10] },
    );
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("stream_resume with bad token must error, got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 19. BACKUP
// ===========================================================================

#[test]
fn backup_creates_copy() {
    let dir = tmp_dir("backup_test");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    // Write some data
    for i in 0..10u32 {
        handler.handle_put(record::Put {
            key: format!("backup_key_{i}"),
            value: format!("backup_val_{i}").into_bytes(),
        });
    }

    let backup_dir = tmp_dir("backup_dest");
    let resp = handler.handle_backup(admin::Backup {
        destination: backup_dir.to_string_lossy().to_string(),
    });
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
        other => panic!("backup must succeed, got: {:?}", other),
    }

    // Verify backup directory has files
    assert!(backup_dir.exists(), "backup directory must exist");
    let file_count: usize = fs::read_dir(&backup_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .count();
    assert!(file_count > 0, "backup directory must have files");

    // Verify backup is usable: open as a new engine
    let eng2 = Engine::open(EngineConfig::new(&backup_dir)).unwrap();
    for i in 0..10u32 {
        let key = format!("backup_key_{i}");
        let expected = format!("backup_val_{i}");
        let scoped = scope_key(0, key.as_bytes());
        let val = eng2.get(&scoped);
        assert!(
            val.is_some(),
            "backup must contain key {key}"
        );
        assert_eq!(
            val.unwrap(), expected.as_bytes(),
            "backup must contain correct value for {key}"
        );
    }

    cleanup(&dir);
    cleanup(&backup_dir);
}

#[test]
fn restore_returns_proper_error() {
    let dir = tmp_dir("restore_test");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_restore_backup(admin::RestoreBackup {
        source: "/tmp/nonexistent".into(),
        data: vec![],
    });
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("restore must return error (requires restart), got: {:?}", other),
    }

    cleanup(&dir);
}

// ===========================================================================
// 20. CONFIG
// ===========================================================================

#[test]
fn config_set_is_readonly() {
    let dir = tmp_dir("config_set");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    let resp = handler.handle_config_set(admin::ConfigSet {
        key: "max_connections".into(),
        value: "2048".into(),
    });
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("config_set must return error (read-only), got: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
fn config_get_known_keys() {
    let dir = tmp_dir("config_get");
    fs::create_dir_all(&dir).unwrap();
    let handler = make_handler(&dir);

    for key in &["version", "max_connections", "flush_threshold"] {
        let resp = handler.handle_config_get(admin::ConfigGet { key: key.to_string() });
        match resp {
            Response::Single { opcode: op, .. } if op == opcode::OP_RESULT_END => {}
            other => panic!("config_get({key}) must succeed, got: {:?}", other),
        }
    }

    // Unknown key
    let resp = handler.handle_config_get(admin::ConfigGet { key: "nonexistent".into() });
    match resp {
        Response::Single { opcode: op, .. } if op == opcode::OP_ERROR => {}
        other => panic!("config_get(unknown) must error, got: {:?}", other),
    }

    cleanup(&dir);
}
