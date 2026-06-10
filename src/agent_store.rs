// src/agent_store.rs
//
// Typed agent payload storage layer for uldb.
//
// Provides convenience functions to store, retrieve, search, and
// manage ULMEN-AGENT payloads in the uldb engine. Values are stored
// as encoded ULMEN-AGENT v1 strings (UTF-8 bytes). The engine's
// existing BM25/fuzzy indices automatically index the text content.
//
// This module does NOT change the engine's core API. It is a typed
// wrapper that serializes AgentRecord/AgentPayload to/from the raw
// byte interface.
//
// Usage from Rust:
//
//   use uldb::agent_store;
//   use ulmen_core::*;
//
//   let payload = AgentPayload { ... };
//   agent_store::store_payload(&mut engine, "session:001", &payload)?;
//   let restored = agent_store::load_payload(&engine, "session:001")?;

use std::io;

use ulmen_core::{
    AgentError, AgentPayload, AgentRecord, RecordType,
    validate_payload,
};

use crate::engine::Engine;

/// Key prefix for agent payloads in the engine.
const AGENT_PREFIX: &str = "agent:";

/// Key prefix for individual agent records (indexed separately).
const RECORD_PREFIX: &str = "arec:";

// ---------------------------------------------------------------------------
// Store / Load full payloads
// ---------------------------------------------------------------------------

/// Store an agent payload, validating it first.
/// The payload is stored as its encoded string under key "agent:{session_key}".
/// Individual records are also stored separately for search indexing.
pub fn store_payload(
    engine: &mut Engine,
    session_key: &str,
    payload: &AgentPayload,
) -> io::Result<()> {
    // Validate before storing
    if let Err(e) = validate_payload(payload) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("validation failed: {}", e),
        ));
    }

    // Store the full encoded payload
    let encoded = payload.encode();
    let key = format!("{}{}", AGENT_PREFIX, session_key);
    engine.put(key.as_bytes(), encoded.as_bytes())?;

    // Store individual records for indexing.
    // Key format: "arec:{session_key}:{step}:{type}:{id}"
    // Value: the single record line (searchable by BM25).
    for rec in &payload.records {
        let rec_key = format!(
            "{}{}:{}:{}:{}",
            RECORD_PREFIX, session_key, rec.step,
            rec.record_type.as_str(), rec.id
        );
        let rec_line = rec.encode(&[]);
        engine.put(rec_key.as_bytes(), rec_line.as_bytes())?;
    }

    Ok(())
}

/// Load an agent payload by session key.
pub fn load_payload(engine: &Engine, session_key: &str) -> Result<AgentPayload, AgentError> {
    let key = format!("{}{}", AGENT_PREFIX, session_key);
    match engine.get(key.as_bytes()) {
        None => Err(AgentError::Parse(format!("no payload for key {:?}", session_key))),
        Some(bytes) => {
            let text = String::from_utf8(bytes)
                .map_err(|e| AgentError::Parse(format!("invalid UTF-8: {}", e)))?;
            AgentPayload::decode(&text)
        }
    }
}

/// Delete an agent payload and all its indexed records.
pub fn delete_payload(
    engine: &mut Engine,
    session_key: &str,
) -> io::Result<()> {
    // Delete the full payload
    let key = format!("{}{}", AGENT_PREFIX, session_key);
    engine.delete(key.as_bytes())?;

    // Delete indexed records by scanning the prefix
    let prefix = format!("{}{}", RECORD_PREFIX, session_key);
    let start = prefix.as_bytes().to_vec();
    let mut end = start.clone();
    end.push(0xFF);
    let records = engine.scan(&start, &end);
    for (k, _) in records {
        engine.delete(&k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Search agent records by content. Returns matching records with scores.
/// Uses the engine's BM25 index which automatically indexed the record lines.
pub fn search_records(
    engine: &mut Engine,
    query: &str,
    limit: usize,
) -> Vec<(AgentRecord, f64)> {
    use crate::query::planner::QuerySpec;

    let spec = QuerySpec {
        text: query.to_string(),
        top_k: limit,
        ..Default::default()
    };

    let hits = engine.indices.query(&spec);
    let mut results = Vec::new();

    for hit in hits {
        let key_str = String::from_utf8_lossy(&hit.key);
        // Only match agent record keys
        if !key_str.starts_with(RECORD_PREFIX) {
            continue;
        }
        if let Some(value) = engine.get(&hit.key) {
            let line = String::from_utf8_lossy(&value);
            // Try to decode the record line
            if let Ok(rec) = ulmen_core::decode_record_public(&line, &[]) {
                results.push((rec, hit.score));
            }
        }
    }

    results
}

/// List all stored session keys.
pub fn list_sessions(engine: &Engine) -> Vec<String> {
    let start = AGENT_PREFIX.as_bytes();
    let mut end = start.to_vec();
    end.push(0xFF);
    engine.scan(start, &end)
        .into_iter()
        .map(|(k, _)| {
            let full = String::from_utf8_lossy(&k);
            full.strip_prefix(AGENT_PREFIX)
                .unwrap_or(&full)
                .to_string()
        })
        .collect()
}

/// Get all records for a session, filtered by type.
pub fn get_records_by_type(
    engine: &Engine,
    session_key: &str,
    record_type: RecordType,
) -> Vec<AgentRecord> {
    let prefix = format!("{}{}:", RECORD_PREFIX, session_key);
    let start = prefix.as_bytes().to_vec();
    let mut end = start.clone();
    end.push(0xFF);

    engine.scan(&start, &end)
        .into_iter()
        .filter_map(|(k, v)| {
            let key_str = String::from_utf8_lossy(&k);
            // Key format: arec:{session}:{step}:{type}:{id}
            // Check if the type segment matches
            let parts: Vec<&str> = key_str.splitn(5, ':').collect();
            if parts.len() >= 4 && parts[3] == record_type.as_str() {
                let line = String::from_utf8_lossy(&v);
                ulmen_core::decode_record_public(&line, &[]).ok()
            } else {
                None
            }
        })
        .collect()
}

/// Get the latest N records for a session (by step, descending).
pub fn get_recent_records(
    engine: &Engine,
    session_key: &str,
    limit: usize,
) -> Vec<AgentRecord> {
    match load_payload(engine, session_key) {
        Ok(payload) => {
            let n = payload.records.len();
            if n <= limit {
                payload.records
            } else {
                payload.records[n - limit..].to_vec()
            }
        }
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Append (add records to existing session)
// ---------------------------------------------------------------------------

/// Append records to an existing session payload.
/// If the session does not exist, creates a new payload.
pub fn append_records(
    engine: &mut Engine,
    session_key: &str,
    new_records: &[AgentRecord],
) -> io::Result<()> {
    let mut payload = match load_payload(engine, session_key) {
        Ok(p) => p,
        Err(_) => AgentPayload {
            header: ulmen_core::AgentHeader::default(),
            records: Vec::new(),
        },
    };

    payload.records.extend_from_slice(new_records);
    payload.header.record_count = payload.records.len();

    store_payload(engine, session_key, &payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ulmen_core::*;
    use tempfile::TempDir;

    fn tmp_engine() -> (Engine, tempfile::TempDir) {
        let dir = TempDir::new().unwrap();
        let engine = Engine::open(
            crate::engine::EngineConfig::new(dir.path())
        ).unwrap();
        (engine, dir)
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

    #[test]
    fn store_and_load_roundtrip() {
        let (mut engine, _dir) = tmp_engine();
        let payload = make_payload(vec![
            make_msg("m1", 1, "hello world"),
            make_msg("m2", 2, "goodbye world"),
        ]);

        store_payload(&mut engine, "sess_001", &payload).unwrap();
        let loaded = load_payload(&engine, "sess_001").unwrap();

        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records[0].id, "m1");
        assert_eq!(loaded.records[1].id, "m2");
    }

    #[test]
    fn store_validates() {
        let (mut engine, _dir) = tmp_engine();
        // Create invalid payload (empty thread_id)
        let bad = AgentPayload {
            header: AgentHeader { record_count: 1, ..Default::default() },
            records: vec![AgentRecord {
                record_type: RecordType::Msg,
                id: "m1".into(),
                thread_id: "".into(), // invalid
                step: 1,
                fields: vec![
                    FieldValue::Str("user".into()),
                    FieldValue::Int(1),
                    FieldValue::Str("hi".into()),
                    FieldValue::Int(1),
                    FieldValue::Bool(false),
                ],
                meta: MetaFields::default(),
            }],
        };

        let result = store_payload(&mut engine, "bad", &bad);
        assert!(result.is_err());
    }

    #[test]
    fn load_missing_returns_error() {
        let (engine, _dir) = tmp_engine();
        assert!(load_payload(&engine, "nonexistent").is_err());
    }

    #[test]
    fn delete_payload_removes_all() {
        let (mut engine, _dir) = tmp_engine();
        let payload = make_payload(vec![make_msg("m1", 1, "test")]);
        store_payload(&mut engine, "sess_del", &payload).unwrap();

        delete_payload(&mut engine, "sess_del").unwrap();
        assert!(load_payload(&engine, "sess_del").is_err());
    }

    #[test]
    fn list_sessions_works() {
        let (mut engine, _dir) = tmp_engine();
        store_payload(&mut engine, "a", &make_payload(vec![make_msg("m1", 1, "x")])).unwrap();
        store_payload(&mut engine, "b", &make_payload(vec![make_msg("m2", 1, "y")])).unwrap();

        let sessions = list_sessions(&engine);
        assert!(sessions.contains(&"a".to_string()));
        assert!(sessions.contains(&"b".to_string()));
    }

    #[test]
    fn append_records_creates_new() {
        let (mut engine, _dir) = tmp_engine();
        let rec = make_msg("m1", 1, "first");
        append_records(&mut engine, "new_sess", &[rec]).unwrap();

        let loaded = load_payload(&engine, "new_sess").unwrap();
        assert_eq!(loaded.records.len(), 1);
    }

    #[test]
    fn append_records_extends_existing() {
        let (mut engine, _dir) = tmp_engine();
        let payload = make_payload(vec![make_msg("m1", 1, "first")]);
        store_payload(&mut engine, "append_test", &payload).unwrap();

        let new_rec = make_msg("m2", 2, "second");
        append_records(&mut engine, "append_test", &[new_rec]).unwrap();

        let loaded = load_payload(&engine, "append_test").unwrap();
        assert_eq!(loaded.records.len(), 2);
    }

    #[test]
    fn get_records_by_type_filters() {
        let (mut engine, _dir) = tmp_engine();
        let payload = make_payload(vec![
            make_msg("m1", 1, "hello"),
            AgentRecord {
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
            },
        ]);
        store_payload(&mut engine, "type_test", &payload).unwrap();

        let msgs = get_records_by_type(&engine, "type_test", RecordType::Msg);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].id, "m1");

        let tools = get_records_by_type(&engine, "type_test", RecordType::Tool);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].id, "t1");
    }

    #[test]
    fn get_recent_records_limits() {
        let (mut engine, _dir) = tmp_engine();
        let recs: Vec<AgentRecord> = (1..=10)
            .map(|i| make_msg(&format!("m{}", i), i, &format!("msg {}", i)))
            .collect();
        store_payload(&mut engine, "recent_test", &make_payload(recs)).unwrap();

        let recent = get_recent_records(&engine, "recent_test", 3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].step, 8);
        assert_eq!(recent[2].step, 10);
    }
}
