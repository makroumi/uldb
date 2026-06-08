// src/server.rs
//
// UMP protocol handler for uldb.
//
// Implements ulmp::server::handler::Handler by dispatching
// UMP messages to the Engine.
//
// Architecture:
//   UmpHandler owns a Mutex<Engine>.
//   Each Handler method locks the engine, performs the operation,
//   and returns a Response with the appropriate opcode and payload.
//
// Thread safety:
//   Handler trait requires Send + Sync.
//   Engine is wrapped in Mutex for exclusive access.
//   This is correct for a single-writer database.
//   Read concurrency can be improved later with RwLock.

use std::sync::{Mutex, RwLock};
use crate::namespace::{scope_key, unscope_key, derive_namespace_id};
use crate::tx::session::{TxManager, TxIsolation, TxOp};

use ulmp::messages::opcode;
use ulmp::messages::tx;
use ulmp::messages::snapshot;
use ulmp::messages::branch;
use ulmp::messages::namespace;
use ulmp::messages::record;
use ulmp::messages::query;
use ulmp::messages::admin;
use ulmp::messages::flow;
use ulmp::messages::session;
use ulmp::messages::watch;
use ulmp::messages::workspace;
use ulmp::messages::auth_rotate;
use ulmp::messages::checkpoint as chk_msg;
use ulmp::server::handler::{Handler, Response};

use crate::engine::Engine;

// Tag constants for payload building
const TAG_U8: u8 = 0x01;
const TAG_U32: u8 = 0x03;
const TAG_U64: u8 = 0x04;
const TAG_F64: u8 = 0x0A;
const TAG_BYTES: u8 = 0x0C;
const TAG_STRING: u8 = 0x0D;
const TAG_END: u8 = 0xFF;

/// Build a payload from raw tag+data pairs.
fn enc(fields: Vec<(u8, Vec<u8>)>) -> Vec<u8> {
    let mut buf = Vec::new();
    for (tag, data) in fields {
        buf.push(tag);
        match tag {
            TAG_U8 => {
                if !data.is_empty() {
                    buf.push(data[0]);
                }
            }
            TAG_U32 | TAG_U64 | TAG_F64 => {
                buf.extend_from_slice(&data);
            }
            TAG_BYTES | TAG_STRING => {
                buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
                buf.extend_from_slice(&data);
            }
            _ => {}
        }
    }
    buf.push(TAG_END);
    buf
}

fn string_field(s: &str) -> (u8, Vec<u8>) {
    (TAG_STRING, s.as_bytes().to_vec())
}

fn bytes_field(b: &[u8]) -> (u8, Vec<u8>) {
    (TAG_BYTES, b.to_vec())
}

fn u32_field(v: u32) -> (u8, Vec<u8>) {
    (TAG_U32, v.to_be_bytes().to_vec())
}

fn u64_field(v: u64) -> (u8, Vec<u8>) {
    (TAG_U64, v.to_be_bytes().to_vec())
}

fn u8_field(v: u8) -> (u8, Vec<u8>) {
    (TAG_U8, vec![v])
}

fn f64_field(v: f64) -> (u8, Vec<u8>) {
    (TAG_F64, v.to_bits().to_be_bytes().to_vec())
}

fn error_response(code: u8, msg: &str) -> Response {
    let payload = enc(vec![
        u8_field(code),
        u32_field(0), // stream_id placeholder
        string_field(msg),
    ]);
    Response::Single {
        opcode: opcode::OP_ERROR,
        payload,
    }
}

fn result_end_response(total: u32, elapsed_ms: u32) -> Response {
    let payload = enc(vec![
        u32_field(total),
        u32_field(elapsed_ms),
    ]);
    Response::Single {
        opcode: opcode::OP_RESULT_END,
        payload,
    }
}


/// Resolve a namespace string to a numeric namespace ID.
///
/// Formats:
///   ""              -> 0 (global namespace)
///   "42"            -> 42 (raw numeric)
///   "repo@sha"      -> fnv1a(repo || "::" || sha)
///   "repo::sha"     -> fnv1a(repo || "::" || sha)
#[allow(dead_code)]
fn resolve_ns(namespace: &str) -> u64 {
    if namespace.is_empty() {
        return 0;
    }
    if let Ok(n) = namespace.parse::<u64>() {
        return n;
    }
    // Try repo@sha format
    if let Some(pos) = namespace.find('@') {
        let repo = &namespace[..pos];
        let sha = &namespace[pos+1..];
        return derive_namespace_id(repo, sha);
    }
    // Fallback: hash the whole string
    derive_namespace_id(namespace, "")
}

/// UMP handler backed by the uldb storage engine.
///
/// Tracks active transactions per session.
/// Transaction isolation is enforced by MVCC inside the engine.
pub struct UmpHandler {
    engine: RwLock<Engine>,
    tx_mgr: Mutex<TxManager>,
}

impl UmpHandler {
    pub fn new(engine: Engine) -> Self {
        Self {
            engine: RwLock::new(engine),
            tx_mgr: Mutex::new(TxManager::new()),
        }
    }

    /// Begin a transaction. Snapshots the current HAMT state.
    /// Returns the tx_id.
    pub fn begin_tx(&self, isolation: u8) -> u64 {
        let eng = self.engine.read().unwrap();
        let snapshot = eng.state.snapshot();
        let iso = TxIsolation::from_byte(isolation);
        let mut mgr = self.tx_mgr.lock().unwrap();
        mgr.begin(iso, snapshot)
    }

    /// Commit a transaction. Applies all buffered writes to the engine.
    /// Returns Ok(count) with the number of operations applied,
    /// or Err(msg) on conflict (serializable isolation).
    pub fn commit_tx(&self, tx_id: u64) -> Result<usize, String> {
        let session = {
            let mut mgr = self.tx_mgr.lock().unwrap();
            mgr.remove(tx_id)
                .ok_or_else(|| format!("transaction not found: {tx_id}"))?
        };

        // Serializable conflict detection: check if any read key
        // was modified since the transaction began.
        if session.isolation == TxIsolation::Serializable {
            let eng = self.engine.read().unwrap();
            for key in &session.read_set {
                // If the live state has a different value than the snapshot,
                // another writer modified it.
                let live_val = eng.state.get(key);
                let snap_val = session.snapshot.get(key);
                if live_val != snap_val {
                    return Err(format!(
                        "serialization conflict on key={}",
                        String::from_utf8_lossy(key)
                    ));
                }
            }
        }

        // Apply the write buffer to the engine.
        let count = session.write_buffer.len();
        let mut eng = self.engine.write().unwrap();
        for op in session.write_buffer {
            match op {
                TxOp::Put(key, value) => {
                    if let Err(e) = eng.put(&key, &value) {
                        return Err(format!("tx commit put failed: {e}"));
                    }
                }
                TxOp::Delete(key) => {
                    if let Err(e) = eng.delete(&key) {
                        return Err(format!("tx commit delete failed: {e}"));
                    }
                }
            }
        }

        Ok(count)
    }

    /// Rollback a transaction. Discards all buffered writes.
    pub fn rollback_tx(&self, tx_id: u64) -> bool {
        let mut mgr = self.tx_mgr.lock().unwrap();
        mgr.remove(tx_id).is_some()
    }

    /// Read a key within a transaction (snapshot isolation).
    /// Falls back to the live engine if tx_id is not found.
    pub fn tx_get(&self, tx_id: u64, key: &str) -> Option<Vec<u8>> {
        let ns_id = 0u64;
        let scoped = scope_key(ns_id, key.as_bytes());

        let mut mgr = self.tx_mgr.lock().unwrap();
        if let Some(session) = mgr.get_mut(tx_id) {
            return session.get(&scoped);
        }
        drop(mgr);

        // Fallback: no active tx, read live.
        let eng = self.engine.write().unwrap();
        eng.get(&scoped)
    }

    /// Write a key within a transaction (buffered, not applied until commit).
    pub fn tx_put(&self, tx_id: u64, key: &str, value: Vec<u8>) -> bool {
        let ns_id = 0u64;
        let scoped = scope_key(ns_id, key.as_bytes());

        let mut mgr = self.tx_mgr.lock().unwrap();
        if let Some(session) = mgr.get_mut(tx_id) {
            session.put(scoped, value);
            return true;
        }
        false
    }

    /// Delete a key within a transaction (buffered, not applied until commit).
    pub fn tx_delete(&self, tx_id: u64, key: &str) -> bool {
        let ns_id = 0u64;
        let scoped = scope_key(ns_id, key.as_bytes());

        let mut mgr = self.tx_mgr.lock().unwrap();
        if let Some(session) = mgr.get_mut(tx_id) {
            session.delete(scoped);
            return true;
        }
        false
    }

    /// Build a streaming query response from index hits.
    fn build_query_response(
        &self,
        eng: &mut Engine,
        hits: &[crate::query::planner::RankedHit],
    ) -> Response {
        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(hits.len() as u32)]),
        ));

        for hit in hits {
            let value = eng.get(&hit.key).unwrap_or_default();
            let display_key = unscope_key(&hit.key)
                .map(|k| String::from_utf8_lossy(k).to_string())
                .unwrap_or_else(|| String::from_utf8_lossy(&hit.key).to_string());

            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![
                    string_field(&display_key),
                    bytes_field(&value),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(hit.score),
                    u32_field(hit.rank as u32),
                ]),
            ));
        }

        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![u32_field(hits.len() as u32), u32_field(0)]),
        ));

        Response::Stream { frames }
    }

}

impl Handler for UmpHandler {
    fn handle_put(&self, msg: record::Put) -> Response {
        let ns_id = 0u64; // namespace scoping applied at workspace level
        let scoped = scope_key(ns_id, msg.key.as_bytes());
        let mut eng = self.engine.write().unwrap();
        match eng.put(&scoped, &msg.value) {
            Ok(()) => result_end_response(1, 0),
            Err(e) => error_response(0xFF, &format!("put failed: {e}")),
        }
    }

    fn handle_get(&self, msg: record::Get) -> Response {
        let ns_id = 0u64;
        let scoped = scope_key(ns_id, msg.key.as_bytes());
        let eng = self.engine.write().unwrap();
        match eng.get(&scoped) {
            Some(value) => {
                let payload = enc(vec![
                    string_field(&msg.key),
                    bytes_field(&value),
                    f64_field(0.0), // vector score
                    f64_field(0.0), // text score
                    f64_field(0.0), // fuzzy score
                    f64_field(0.0), // graph score
                    f64_field(0.0), // final score
                    u32_field(0), // rank
                ]);
                Response::Single {
                    opcode: opcode::OP_RESULT_ROW,
                    payload,
                }
            }
            None => error_response(0x40, &format!("key not found: {}", msg.key)),
        }
    }

    fn handle_delete(&self, msg: record::Delete) -> Response {
        let ns_id = 0u64;
        let scoped = scope_key(ns_id, msg.key.as_bytes());
        let mut eng = self.engine.write().unwrap();
        match eng.delete(&scoped) {
            Ok(()) => result_end_response(1, 0),
            Err(e) => error_response(0xFF, &format!("delete failed: {e}")),
        }
    }

    fn handle_scan(&self, msg: record::Scan) -> Response {
        let ns_id = 0u64;
        let scoped_start = scope_key(ns_id, msg.start.as_bytes());
        let scoped_end = scope_key(ns_id, msg.end.as_bytes());
        let eng = self.engine.write().unwrap();
        let results = eng.scan(&scoped_start, &scoped_end);
        let truncated: Vec<_> = results.into_iter().take(msg.limit as usize).collect();

        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(truncated.len() as u32)]),
        ));

        for (i, (key, value)) in truncated.iter().enumerate() {
            let display_key = unscope_key(key)
                .map(|k| String::from_utf8_lossy(k).to_string())
                .unwrap_or_else(|| String::from_utf8_lossy(key).to_string());
            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![
                    string_field(&display_key),
                    bytes_field(value),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    u32_field(i as u32),
                ]),
            ));
        }

        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![
                u32_field(truncated.len() as u32),
                u32_field(0),
            ]),
        ));

        Response::Stream { frames }
    }

    fn handle_put_batch(&self, msg: record::PutBatch) -> Response {
        let ns_id = 0u64;
        let mut eng = self.engine.write().unwrap();
        let count = msg.records.len();

        // Scope all keys and collect for batch write.
        let scoped: Vec<(Vec<u8>, Vec<u8>)> = msg.records
            .into_iter()
            .map(|(key, value)| (scope_key(ns_id, key.as_bytes()), value))
            .collect();

        let refs: Vec<(&[u8], &[u8])> = scoped
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();

        match eng.put_batch(&refs) {
            Ok(()) => result_end_response(count as u32, 0),
            Err(e) => error_response(0xFF, &format!("batch put failed: {e}")),
        }
    }

    fn handle_range_delete(&self, msg: record::RangeDelete) -> Response {
        let ns_id = 0u64;
        let scoped_start = scope_key(ns_id, msg.start.as_bytes());
        let scoped_end = scope_key(ns_id, msg.end.as_bytes());
        let mut eng = self.engine.write().unwrap();
        let keys: Vec<Vec<u8>> = eng
            .scan(&scoped_start, &scoped_end)
            .into_iter()
            .map(|(k, _)| k)
            .collect();

        let count = keys.len();
        for key in keys {
            if let Err(e) = eng.delete(&key) {
                return error_response(0xFF, &format!("range delete failed: {e}"));
            }
        }

        result_end_response(count as u32, 0)
    }

    fn handle_query(&self, msg: query::Query) -> Response {
        let mut eng = self.engine.write().unwrap();

        let spec = crate::query::planner::QuerySpec {
            text: msg.text,
            vector: msg.vector,
            top_k: msg.top_k as usize,
            max_depth: msg.max_depth as usize,
            relations: msg.relations,
            lang_filter: msg.lang_filter,
            type_filter: msg.type_filter,
            file_filter: msg.file_filter,
            merge_strategy: msg.merge_strategy,
            timeout_ms: msg.timeout_ms,
        };

        let all_hits = eng.indices.query(&spec);

        // Filter hits by namespace if the query specifies a numeric namespace.
        // Keys stored via handle_put use an 8-byte ns_id=0 prefix. When
        // application-level namespace scoping is used (keys have a second
        // 8-byte prefix at bytes 8..16), the filter matches that prefix.
        //
        // Only activate filtering when the namespace string is a raw numeric
        // u64 (e.g. "12345"). String-named namespaces like "my_project" are
        // not yet used as storage prefixes (handle_put always uses ns_id=0),
        // so filtering by their hash would incorrectly reject all results.
        let query_ns: Option<u64> = msg.namespace.parse::<u64>().ok()
            .filter(|&ns| ns != 0);
        let hits: Vec<_> = if let Some(ns) = query_ns {
            let ns_prefix = ns.to_be_bytes();
            all_hits
                .into_iter()
                .filter(|hit| {
                    hit.key.len() >= 16 && hit.key[8..16] == ns_prefix
                })
                .collect()
        } else {
            all_hits
        };

        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(hits.len() as u32)]),
        ));

        for hit in &hits {
            // Hit keys are the full stored keys (with ns_id=0 prefix).
            // Look up the value directly.
            let value = eng.get(&hit.key).unwrap_or_default();

            // Display key: strip the handler's 8-byte ns_id=0 prefix.
            // If namespace filtering was active (double-prefixed keys),
            // also strip the application namespace prefix.
            let display_key = if query_ns.is_some() && hit.key.len() >= 16 {
                String::from_utf8_lossy(&hit.key[16..]).to_string()
            } else {
                unscope_key(&hit.key)
                    .map(|k| String::from_utf8_lossy(k).to_string())
                    .unwrap_or_else(|| String::from_utf8_lossy(&hit.key).to_string())
            };

            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![
                    string_field(&display_key),
                    bytes_field(&value),
                    f64_field(0.0), // vector score
                    f64_field(0.0), // text score
                    f64_field(0.0), // fuzzy score
                    f64_field(0.0), // graph score
                    f64_field(hit.score), // final score
                    u32_field(hit.rank as u32),
                ]),
            ));
        }

        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![
                u32_field(hits.len() as u32),
                u32_field(0),
            ]),
        ));

        Response::Stream { frames }
    }

    fn handle_get_batch(&self, msg: record::GetBatch) -> Response {
        let ns_id = 0u64;
        let mut frames = Vec::new();
        let eng = self.engine.write().unwrap();
        let mut found = 0u32;

        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(msg.keys.len() as u32)]),
        ));

        for (i, key) in msg.keys.iter().enumerate() {
            let scoped = scope_key(ns_id, key.as_bytes());
            let value = eng.get(&scoped).unwrap_or_default();
            if !value.is_empty() {
                found += 1;
            }
            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![
                    string_field(key),
                    bytes_field(&value),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    f64_field(0.0),
                    u32_field(i as u32),
                ]),
            ));
        }

        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![u32_field(found), u32_field(0)]),
        ));

        Response::Stream { frames }
    }

    fn handle_query_fuzzy(&self, msg: query::QueryFuzzy) -> Response {
        let mut eng = self.engine.write().unwrap();
        let spec = crate::query::planner::QuerySpec {
            text: msg.query,
            top_k: msg.top_k as usize,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        self.build_query_response(&mut eng, &hits)
    }

    fn handle_query_vector(&self, msg: query::QueryVector) -> Response {
        let mut eng = self.engine.write().unwrap();
        let spec = crate::query::planner::QuerySpec {
            vector: msg.vector,
            top_k: msg.top_k as usize,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        self.build_query_response(&mut eng, &hits)
    }

    fn handle_query_graph(&self, msg: query::QueryGraph) -> Response {
        let mut eng = self.engine.write().unwrap();
        let spec = crate::query::planner::QuerySpec {
            text: msg.start_key,
            relations: msg.relations,
            top_k: msg.top_k as usize,
            max_depth: msg.max_depth as usize,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        self.build_query_response(&mut eng, &hits)
    }



    fn handle_query_keyword(&self, msg: query::QueryKeyword) -> Response {
        self.handle_query(query::Query {
            namespace: msg.namespace,
            text: msg.query,
            vector: vec![],
            top_k: msg.top_k,
            max_depth: 0,
            relations: vec![],
            lang_filter: vec![],
            type_filter: vec![],
            file_filter: vec![],
            merge_strategy: 1,
            timeout_ms: 5000,
        })
    }

    fn handle_stats(&self, _msg: admin::Stats) -> Response {
        let eng = self.engine.read().unwrap();
        let payload = enc(vec![
            string_field("memtable_len"),
            u32_field(eng.memtable_len() as u32),
            string_field("memtable_bytes"),
            u32_field(eng.memtable_bytes() as u32),
            string_field("flush_count"),
            u64_field(eng.flush_count()),
            string_field("compaction_count"),
            u64_field(eng.compaction_count()),
            string_field("total_puts"),
            u64_field(eng.total_puts()),
            string_field("total_gets"),
            u64_field(eng.total_gets()),
            string_field("total_deletes"),
            u64_field(eng.total_deletes()),
        ]);

        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    fn handle_compact(&self, _msg: admin::Compact) -> Response {
        let mut eng = self.engine.write().unwrap();
        match eng.flush() {
            Ok(()) => result_end_response(0, 0),
            Err(e) => error_response(0xFF, &format!("compact failed: {e}")),
        }
    }

    // ========================================================================
    // Transaction operations
    // ========================================================================

    fn handle_tx_begin(&self, msg: tx::TxBegin) -> Response {
        let tx_id = self.begin_tx(msg.isolation);
        let payload = enc(vec![u64_field(tx_id)]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    fn handle_tx_commit(&self, msg: tx::TxCommit) -> Response {
        match self.commit_tx(msg.tx_id) {
            Ok(count) => result_end_response(count as u32, 0),
            Err(e) => error_response(0x61, &e), // ERR_TX_CONFLICT
        }
    }

    fn handle_tx_rollback(&self, msg: tx::TxRollback) -> Response {
        if self.rollback_tx(msg.tx_id) {
            result_end_response(0, 0)
        } else {
            error_response(0x60, &format!("transaction not found: {}", msg.tx_id))
        }
    }

    fn handle_tx_status(&self, msg: tx::TxStatus) -> Response {
        let mgr = self.tx_mgr.lock().unwrap();
        let status = if mgr.exists(msg.tx_id) { "active" } else { "unknown" };
        let payload = enc(vec![
            u64_field(msg.tx_id),
            string_field(status),
        ]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    // ========================================================================
    // Snapshot operations
    // ========================================================================

    fn handle_snap_create(&self, msg: snapshot::SnapCreate) -> Response {
        let mut eng = self.engine.write().unwrap();
        let id = eng.snapshot_create(&msg.description);
        let payload = enc(vec![string_field(&id)]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    fn handle_snap_restore(&self, msg: snapshot::SnapRestore) -> Response {
        let mut eng = self.engine.write().unwrap();
        match eng.snapshot_restore(&msg.snapshot_id) {
            Ok(()) => result_end_response(1, 0),
            Err(e) => error_response(0x82, &e),
        }
    }

    fn handle_snap_delete(&self, msg: snapshot::SnapDelete) -> Response {
        let mut eng = self.engine.write().unwrap();
        if eng.snapshot_delete(&msg.snapshot_id) {
            result_end_response(1, 0)
        } else {
            error_response(0x82, &format!("snapshot not found: {}", msg.snapshot_id))
        }
    }

    fn handle_snap_list(&self, _msg: snapshot::SnapList) -> Response {
        let eng = self.engine.read().unwrap();
        let names = eng.snapshot_list();
        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(names.len() as u32)]),
        ));
        for (i, name) in names.iter().enumerate() {
            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![string_field(name), u32_field(i as u32)]),
            ));
        }
        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![u32_field(names.len() as u32), u32_field(0)]),
        ));
        Response::Stream { frames }
    }

    // ========================================================================
    // Branch operations
    // ========================================================================

    fn handle_branch_create(&self, msg: branch::BranchCreate) -> Response {
        let mut eng = self.engine.write().unwrap();
        match eng.branch_create(&msg.branch_id, &msg.from_snapshot) {
            Ok(id) => {
                let payload = enc(vec![string_field(&id)]);
                Response::Single {
                    opcode: opcode::OP_RESULT_END,
                    payload,
                }
            }
            Err(e) => error_response(0x80, &e),
        }
    }

    fn handle_branch_merge(&self, msg: branch::BranchMerge) -> Response {
        let mut eng = self.engine.write().unwrap();
        match eng.branch_merge(&msg.branch_id) {
            Ok(count) => result_end_response(count as u32, 0),
            Err(e) => error_response(0x81, &e),
        }
    }

    fn handle_branch_rollback(&self, msg: branch::BranchRollback) -> Response {
        let mut eng = self.engine.write().unwrap();
        if eng.branch_rollback(&msg.branch_id) {
            result_end_response(1, 0)
        } else {
            error_response(0x80, &format!("branch not found: {}", msg.branch_id))
        }
    }

    fn handle_branch_diff(&self, msg: branch::BranchDiff) -> Response {
        let eng = self.engine.read().unwrap();
        match eng.branch_diff(&msg.branch_a) {
            Ok(diffs) => {
                let mut frames = Vec::new();
                frames.push((
                    opcode::OP_RESULT_START,
                    enc(vec![u32_field(diffs.len() as u32)]),
                ));
                for (i, (key, live_val, branch_val)) in diffs.iter().enumerate() {
                    let key_str = String::from_utf8_lossy(key);
                    frames.push((
                        opcode::OP_RESULT_ROW,
                        enc(vec![
                            string_field(&key_str),
                            bytes_field(live_val.as_deref().unwrap_or(&[])),
                            bytes_field(branch_val.as_deref().unwrap_or(&[])),
                            u32_field(i as u32),
                        ]),
                    ));
                }
                frames.push((
                    opcode::OP_RESULT_END,
                    enc(vec![u32_field(diffs.len() as u32), u32_field(0)]),
                ));
                Response::Stream { frames }
            }
            Err(e) => error_response(0x80, &e),
        }
    }

    fn handle_branch_list(&self, _msg: branch::BranchList) -> Response {
        // Branches are stored in the same map as snapshots.
        // For now, list all snapshots as potential branches.
        let eng = self.engine.read().unwrap();
        let names = eng.snapshot_list();
        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(names.len() as u32)]),
        ));
        for (i, name) in names.iter().enumerate() {
            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![string_field(name), u32_field(i as u32)]),
            ));
        }
        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![u32_field(names.len() as u32), u32_field(0)]),
        ));
        Response::Stream { frames }
    }

    // ========================================================================
    // Namespace operations
    // ========================================================================

    fn handle_ns_create(&self, msg: namespace::NsCreate) -> Response {
        // Store namespace metadata as a record.
        let mut eng = self.engine.write().unwrap();
        let ns_key = format!("__ns__::{}", msg.repo_url);
        let ns_val = format!("{}|{}", msg.commit_sha, msg.description);
        match eng.put(ns_key.as_bytes(), ns_val.as_bytes()) {
            Ok(()) => result_end_response(1, 0),
            Err(e) => error_response(0xFF, &format!("ns create failed: {e}")),
        }
    }

    fn handle_ns_list(&self, _msg: namespace::NsList) -> Response {
        let eng = self.engine.write().unwrap();
        let results = eng.scan(b"__ns__::", b"__ns__::\xff");
        let mut frames = Vec::new();
        frames.push((
            opcode::OP_RESULT_START,
            enc(vec![u32_field(results.len() as u32)]),
        ));
        for (i, (key, value)) in results.iter().enumerate() {
            let key_str = String::from_utf8_lossy(key);
            frames.push((
                opcode::OP_RESULT_ROW,
                enc(vec![
                    string_field(&key_str),
                    bytes_field(value),
                    u32_field(i as u32),
                ]),
            ));
        }
        frames.push((
            opcode::OP_RESULT_END,
            enc(vec![u32_field(results.len() as u32), u32_field(0)]),
        ));
        Response::Stream { frames }
    }

    fn handle_ns_stat(&self, _msg: namespace::NsStat) -> Response {
        let eng = self.engine.read().unwrap();
        let idx_stats = eng.indices.stats();
        let payload = enc(vec![
            string_field("memtable_len"),
            u32_field(eng.memtable_len() as u32),
            string_field("bm25_docs"),
            u32_field(idx_stats.bm25_docs as u32),
            string_field("fuzzy_symbols"),
            u32_field(idx_stats.fuzzy_symbols as u32),
            string_field("hnsw_vectors"),
            u32_field(idx_stats.hnsw_vectors as u32),
            string_field("graph_nodes"),
            u32_field(idx_stats.graph_nodes as u32),
            string_field("graph_edges"),
            u32_field(idx_stats.graph_edges as u32),
        ]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }


    // ========================================================================
    // Namespace management
    // ========================================================================

    fn handle_ns_open(&self, msg: namespace::NsOpen) -> Response {
        // Acknowledge namespace open. In a full implementation this would
        // set the active namespace for the session.
        let payload = enc(vec![u64_field(msg.namespace_id)]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    fn handle_ns_delete(&self, msg: namespace::NsDelete) -> Response {
        // Delete all keys in the namespace via range delete.
        let mut eng = self.engine.write().unwrap();
        let (start, end) = crate::namespace::ns_scan_range(msg.namespace_id);
        let keys: Vec<Vec<u8>> = eng.scan(&start, &end)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        let count = keys.len();
        for key in keys {
            let _ = eng.delete(&key);
        }
        result_end_response(count as u32, 0)
    }

    fn handle_ns_grant(&self, _msg: namespace::NsGrant) -> Response {
        // Permission management acknowledged. Full ACL enforcement
        // requires session context (token_index tracking).
        result_end_response(1, 0)
    }

    // ========================================================================
    // Workspace
    // ========================================================================

    fn handle_workspace_set(&self, _msg: workspace::WorkspaceSet) -> Response {
        // Acknowledge workspace selection. In a full implementation this
        // would set the active namespace/branch for the session.
        result_end_response(1, 0)
    }

    // ========================================================================
    // Intent / Observe
    // ========================================================================

    fn handle_intent(&self, _msg: session::Intent) -> Response {
        // Intent is a hint. Acknowledge silently.
        Response::None
    }

    // ========================================================================
    // Watch (stub with proper response format)
    // ========================================================================

    fn handle_watch(&self, _msg: watch::Watch) -> Response {
        // Watch registration acknowledged. Notifications not yet delivered.
        result_end_response(1, 0)
    }

    fn handle_unwatch(&self, _msg: watch::Unwatch) -> Response {
        result_end_response(1, 0)
    }

    fn handle_watch_window(&self, _msg: watch::WatchWindow) -> Response {
        // Credit grant acknowledged.
        Response::None
    }

    // ========================================================================
    // Flow control
    // ========================================================================

    fn handle_window_update(&self, _msg: flow::WindowUpdate) -> Response {
        // Connection-level flow control ack. No action needed until
        // we implement real backpressure.
        Response::None
    }

    // ========================================================================
    // Auth rotation (stub)
    // ========================================================================

    fn handle_auth_rotate_request(
        &self,
        _msg: auth_rotate::AuthRotateRequest,
    ) -> Response {
        // In-session token rotation not yet supported.
        error_response(0x20, "auth rotation not implemented")
    }

    fn handle_auth_rotate(&self, _msg: auth_rotate::AuthRotate) -> Response {
        error_response(0x20, "auth rotation not implemented")
    }

    // ========================================================================
    // Checkpoint / Stream resume (stub)
    // ========================================================================

    fn handle_stream_resume(&self, _msg: chk_msg::StreamResume) -> Response {
        error_response(0x20, "stream resume not implemented")
    }

    // ========================================================================
    // Admin: config and backup
    // ========================================================================

    fn handle_config_get(&self, msg: admin::ConfigGet) -> Response {
        let value = match msg.key.as_str() {
            "max_connections" => "1024".to_string(),
            "flush_threshold" => "4194304".to_string(),
            "version" => env!("CARGO_PKG_VERSION").to_string(),
            _ => return error_response(0x90, &format!("unknown config key: {}", msg.key)),
        };
        let payload = enc(vec![
            string_field(&msg.key),
            string_field(&value),
        ]);
        Response::Single {
            opcode: opcode::OP_RESULT_END,
            payload,
        }
    }

    fn handle_config_set(&self, _msg: admin::ConfigSet) -> Response {
        // Config is read-only in this version.
        error_response(0x91, "configuration is read-only")
    }

    fn handle_backup(&self, _msg: admin::Backup) -> Response {
        error_response(0x20, "backup not implemented")
    }

    fn handle_restore_backup(&self, _msg: admin::RestoreBackup) -> Response {
        error_response(0x20, "restore not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use ulmp::messages::{tx, snapshot, branch, namespace};
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!("uldb_server_{name}_{}", std::process::id()));
        p
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn make_handler(name: &str) -> (UmpHandler, PathBuf) {
        let dir = tmp_dir(name);
        let config = EngineConfig::new(&dir);
        let engine = Engine::open(config).unwrap();
        (UmpHandler::new(engine), dir)
    }

    #[test]
    fn handler_put_and_get() {
        let (handler, dir) = make_handler("put_get");

        let put_resp = handler.handle_put(record::Put {
            key: "test_key".into(),
            value: b"test_value".to_vec(),
        });
        match put_resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single response for PUT"),
        }

        let get_resp = handler.handle_get(record::Get {
            key: "test_key".into(),
        });
        match get_resp {
            Response::Single { opcode, payload } => {
                assert_eq!(opcode, opcode::OP_RESULT_ROW);
                assert!(!payload.is_empty());
            }
            _ => panic!("expected Single response for GET"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_get_missing_key() {
        let (handler, dir) = make_handler("get_missing");

        let resp = handler.handle_get(record::Get {
            key: "nonexistent".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_ERROR);
            }
            _ => panic!("expected error for missing key"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_delete() {
        let (handler, dir) = make_handler("delete");

        handler.handle_put(record::Put {
            key: "del_key".into(),
            value: b"val".to_vec(),
        });

        let del_resp = handler.handle_delete(record::Delete {
            key: "del_key".into(),
        });
        match del_resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for DELETE"),
        }

        let get_resp = handler.handle_get(record::Get {
            key: "del_key".into(),
        });
        match get_resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_ERROR);
            }
            _ => panic!("expected error after delete"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_scan() {
        let (handler, dir) = make_handler("scan");

        for i in 0..5u32 {
            handler.handle_put(record::Put {
                key: format!("scan_{i:03}"),
                value: format!("val_{i}").into_bytes(),
            });
        }

        let resp = handler.handle_scan(record::Scan {
            start: "scan_001".into(),
            end: "scan_004".into(),
            limit: 100,
        });

        match resp {
            Response::Stream { frames } => {
                assert!(frames.len() >= 3);
                assert_eq!(frames[0].0, opcode::OP_RESULT_START);
                assert_eq!(frames.last().unwrap().0, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Stream for SCAN"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_put_batch() {
        let (handler, dir) = make_handler("batch");

        let resp = handler.handle_put_batch(record::PutBatch {
            records: vec![
                ("batch_1".into(), b"v1".to_vec()),
                ("batch_2".into(), b"v2".to_vec()),
                ("batch_3".into(), b"v3".to_vec()),
            ],
        });

        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for PUT_BATCH"),
        }

        let r1 = handler.handle_get(record::Get { key: "batch_1".into() });
        match r1 {
            Response::Single { opcode, .. } => assert_eq!(opcode, opcode::OP_RESULT_ROW),
            _ => panic!("batch_1 not found"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_query_text_match() {
        let (handler, dir) = make_handler("query");

        handler.handle_put(record::Put {
            key: "auth.py::validate_token".into(),
            value: b"def validate_token(): ...".to_vec(),
        });
        handler.handle_put(record::Put {
            key: "auth.py::hash_password".into(),
            value: b"def hash_password(): ...".to_vec(),
        });
        handler.handle_put(record::Put {
            key: "models.py::User".into(),
            value: b"class User: ...".to_vec(),
        });

        let resp = handler.handle_query(query::Query {
            namespace: "".into(),
            text: "auth".into(),
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

        match resp {
            Response::Stream { frames } => {
                let row_count = frames
                    .iter()
                    .filter(|(op, _)| *op == opcode::OP_RESULT_ROW)
                    .count();
                assert_eq!(row_count, 2);
            }
            _ => panic!("expected Stream for QUERY"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_stats() {
        let (handler, dir) = make_handler("stats");

        handler.handle_put(record::Put {
            key: "k".into(),
            value: b"v".to_vec(),
        });

        let resp = handler.handle_stats(admin::Stats);
        match resp {
            Response::Single { opcode, payload } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
                assert!(!payload.is_empty());
            }
            _ => panic!("expected Single for STATS"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_range_delete() {
        let (handler, dir) = make_handler("range_del");

        for i in 0..5u32 {
            handler.handle_put(record::Put {
                key: format!("rd_{i:03}"),
                value: b"v".to_vec(),
            });
        }

        let resp = handler.handle_range_delete(record::RangeDelete {
            start: "rd_001".into(),
            end: "rd_004".into(),
        });

        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for RANGE_DELETE"),
        }

        let r0 = handler.handle_get(record::Get { key: "rd_000".into() });
        match r0 {
            Response::Single { opcode, .. } => assert_eq!(opcode, opcode::OP_RESULT_ROW),
            _ => panic!("rd_000 should exist"),
        }

        let r1 = handler.handle_get(record::Get { key: "rd_001".into() });
        match r1 {
            Response::Single { opcode, .. } => assert_eq!(opcode, opcode::OP_ERROR),
            _ => panic!("rd_001 should be deleted"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_tx_begin_returns_id() {
        let (handler, dir) = make_handler("tx_begin");
        let resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x02 });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for TX_BEGIN"),
        }
        cleanup(&dir);
    }

    #[test]
    fn handler_tx_commit() {
        let (handler, dir) = make_handler("tx_commit");

        // Must begin a transaction before committing it.
        let begin_resp = handler.handle_tx_begin(tx::TxBegin { isolation: 0x02 });
        let tx_id = match &begin_resp {
            Response::Single { payload, .. } if payload.len() >= 9 => {
                u64::from_be_bytes(payload[1..9].try_into().unwrap())
            }
            _ => panic!("tx_begin should return tx_id"),
        };

        let resp = handler.handle_tx_commit(tx::TxCommit { tx_id });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for TX_COMMIT"),
        }
        cleanup(&dir);
    }

    #[test]
    fn handler_snapshot_lifecycle() {
        let (handler, dir) = make_handler("snap_lifecycle");

        // PUT some data
        handler.handle_put(record::Put {
            key: "k1".into(),
            value: b"v1".to_vec(),
        });

        // Create snapshot
        let resp = handler.handle_snap_create(snapshot::SnapCreate {
            namespace_id: 0,
            description: "before_edit".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for SNAP_CREATE"),
        }

        // List snapshots
        let resp = handler.handle_snap_list(snapshot::SnapList { namespace_id: 0 });
        match resp {
            Response::Stream { frames } => {
                let rows = frames.iter().filter(|(op, _)| *op == opcode::OP_RESULT_ROW).count();
                assert_eq!(rows, 1);
            }
            _ => panic!("expected Stream for SNAP_LIST"),
        }

        // Delete snapshot
        let resp = handler.handle_snap_delete(snapshot::SnapDelete {
            namespace_id: 0,
            snapshot_id: "before_edit".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for SNAP_DELETE"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_branch_lifecycle() {
        let (handler, dir) = make_handler("branch_lifecycle");

        handler.handle_put(record::Put {
            key: "shared".into(),
            value: b"original".to_vec(),
        });

        // Create branch
        let resp = handler.handle_branch_create(branch::BranchCreate {
            namespace_id: 0,
            branch_id: "feat/test".into(),
            from_snapshot: "".into(),
            description: "test branch".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for BRANCH_CREATE"),
        }

        // Merge branch
        let resp = handler.handle_branch_merge(branch::BranchMerge {
            namespace_id: 0,
            branch_id: "feat/test".into(),
            into_branch: "".into(),
            resolutions: vec![],
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for BRANCH_MERGE"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_branch_rollback() {
        let (handler, dir) = make_handler("branch_rollback");

        handler.handle_put(record::Put {
            key: "k".into(),
            value: b"v".to_vec(),
        });

        handler.handle_branch_create(branch::BranchCreate {
            namespace_id: 0,
            branch_id: "bad_idea".into(),
            from_snapshot: "".into(),
            description: "".into(),
        });

        let resp = handler.handle_branch_rollback(branch::BranchRollback {
            namespace_id: 0,
            branch_id: "bad_idea".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for BRANCH_ROLLBACK"),
        }

        cleanup(&dir);
    }

    #[test]
    fn handler_ns_create_and_list() {
        let (handler, dir) = make_handler("ns_ops");

        let resp = handler.handle_ns_create(namespace::NsCreate {
            repo_url: "github.com/org/repo".into(),
            commit_sha: "abc123".into(),
            description: "test namespace".into(),
        });
        match resp {
            Response::Single { opcode, .. } => {
                assert_eq!(opcode, opcode::OP_RESULT_END);
            }
            _ => panic!("expected Single for NS_CREATE"),
        }

        let resp = handler.handle_ns_list(namespace::NsList);
        match resp {
            Response::Stream { frames } => {
                let rows = frames.iter().filter(|(op, _)| *op == opcode::OP_RESULT_ROW).count();
                assert!(rows >= 1);
            }
            _ => panic!("expected Stream for NS_LIST"),
        }

        cleanup(&dir);
    }
}
