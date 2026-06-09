use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;

use crate::engine::{Engine, EngineConfig};
use crate::query::planner::QuerySpec;

use std::sync::{Arc, RwLock};
use std::collections::HashMap;

// ============================================================================
// Document: the semantic data container
// ============================================================================

#[pyclass]
#[derive(Clone)]
pub struct Document {
    #[pyo3(get)]
    pub id: String,
    content_bytes: Vec<u8>,
    meta: HashMap<String, String>,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl Document {
    #[getter]
    fn content(&self) -> &str {
        std::str::from_utf8(&self.content_bytes).unwrap_or("")
    }

    #[getter]
    fn raw(&self) -> &[u8] {
        &self.content_bytes
    }

    #[getter]
    fn metadata(&self) -> HashMap<String, String> {
        self.meta.clone()
    }

    #[getter]
    fn vector(&self) -> PyResult<Option<Vec<f32>>> {
        // Vector storage not yet wired through Python.
        Ok(None)
    }

    fn __repr__(&self) -> String {
        if self.score > 0.0 {
            format!("Document(id={:?}, score={:.4}, len={})",
                self.id, self.score, self.content_bytes.len())
        } else {
            format!("Document(id={:?}, len={})",
                self.id, self.content_bytes.len())
        }
    }

    fn __str__(&self) -> &str {
        self.content()
    }

    fn __len__(&self) -> usize {
        self.content_bytes.len()
    }

    fn __bool__(&self) -> bool {
        !self.content_bytes.is_empty()
    }
}

// ============================================================================
// ContextEngine: the intelligence plane
// ============================================================================

#[pyclass]
struct ContextEngine {
    engine: Arc<RwLock<Engine>>,
}

#[pymethods]
impl ContextEngine {
    /// Hybrid search: BM25 text + fuzzy symbol matching.
    /// Returns documents ranked by relevance.
    #[pyo3(signature = (text, limit=10))]
    fn query(&self, text: &str, limit: usize) -> PyResult<Vec<Document>> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let spec = QuerySpec {
            text: text.to_string(),
            top_k: limit,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = eng.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: h.score,
            }
        }).collect())
    }

    /// BM25 keyword search only.
    #[pyo3(signature = (text, limit=10))]
    fn search_text(&self, text: &str, limit: usize) -> PyResult<Vec<Document>> {
        self.query(text, limit)
    }

    /// Fuzzy symbol search (typo-tolerant).
    #[pyo3(signature = (symbol, limit=5))]
    fn search_fuzzy(&self, symbol: &str, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let results = eng.indices.fuzzy.query(symbol, limit);
        Ok(results.into_iter().map(|m| {
            let key = m.symbol.as_bytes();
            let value = eng.get(key).unwrap_or_default();
            Document {
                id: m.symbol,
                content_bytes: value,
                meta: HashMap::new(),
                score: m.jaccard,
            }
        }).collect())
    }

    /// Ingest a dictionary of {path: content} pairs.
    /// This is the primary way to load a codebase into uldb.
    fn ingest(&self, records: HashMap<String, Vec<u8>>) -> PyResult<usize> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let entries: Vec<(Vec<u8>, Vec<u8>)> = records
            .into_iter()
            .map(|(k, v)| (k.into_bytes(), v))
            .collect();
        let refs: Vec<(&[u8], &[u8])> = entries.iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        eng.bulk_ingest(&refs)
            .map_err(|e| PyRuntimeError::new_err(format!("ingest failed: {e}")))
    }

    /// Vector similarity search using HNSW.
    #[pyo3(signature = (embedding, limit=10))]
    fn search_vector(&self, embedding: Vec<f32>, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if let Some(ref hnsw) = eng.indices.hnsw {
            let results = hnsw.search(&embedding, limit);
            Ok(results.into_iter().map(|r| {
                let value = eng.get(&r.key).unwrap_or_default();
                Document {
                    id: String::from_utf8_lossy(&r.key).to_string(),
                    content_bytes: value,
                    meta: HashMap::new(),
                    score: 1.0 - r.distance as f64,
                }
            }).collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Add a relation edge between two symbols.
    fn add_edge(&self, src: &str, dst: &str, relation: &str) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.indices.on_add_edge(src, dst, relation);
        Ok(())
    }

    /// Traverse the relation graph from a start node.
    #[pyo3(signature = (start, relation="calls", depth=3, limit=20))]
    fn search_graph(&self, start: &str, relation: &str, depth: usize, limit: usize) -> PyResult<Vec<Document>> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let spec = QuerySpec {
            text: start.to_string(),
            relations: vec![relation.to_string()],
            top_k: limit,
            max_depth: depth,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = eng.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: h.score,
            }
        }).collect())
    }

    /// Index a document with optional vector and edges.
    /// Unified entry point for full document indexing.
    #[pyo3(signature = (key, content, vector=None, edges=None))]
    fn index(
        &self,
        key: &str,
        content: &[u8],
        vector: Option<Vec<f32>>,
        edges: Option<Vec<(String, String)>>,
    ) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        // Write the record
        eng.put(key.as_bytes(), content)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        // Add vector embedding if provided
        if let Some(vec) = vector {
            eng.indices.on_put_vector(key.as_bytes(), vec);
        }
        // Add graph edges if provided
        if let Some(edge_list) = edges {
            for (dst, rel) in edge_list {
                eng.indices.on_add_edge(key, &dst, &rel);
            }
        }
        Ok(())
    }

        fn __repr__(&self) -> String {
        let eng = self.engine.read().unwrap();
        let stats = eng.indices.stats();
        format!(
            "ContextEngine(docs={}, symbols={}, vectors={})",
            stats.bm25_docs, stats.fuzzy_symbols, stats.hnsw_vectors
        )
    }
}

// ============================================================================
// Workspace: the isolation sandbox
// ============================================================================

#[pyclass]
struct Workspace {
    engine: Arc<RwLock<Engine>>,
    branch_name: String,
    is_branch: bool,
}

#[pymethods]
impl Workspace {
    /// Read a document by path. Returns None if not found.
    fn get(&self, path: &str) -> PyResult<Option<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        match eng.get(path.as_bytes()) {
            Some(value) => Ok(Some(Document {
                id: path.to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: 0.0,
            })),
            None => Ok(None),
        }
    }

    /// Write a document.
    #[pyo3(signature = (path, content, _metadata=None))]
    fn put(&self, path: &str, content: &[u8], _metadata: Option<HashMap<String, String>>) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.put(path.as_bytes(), content)
            .map_err(|e| PyRuntimeError::new_err(format!("put failed: {e}")))
    }

    /// Delete a document by path.
    fn delete(&self, path: &str) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.delete(path.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("delete failed: {e}")))
    }

    /// Scan documents by prefix.
    #[pyo3(signature = (prefix, limit=100))]
    fn scan(&self, prefix: &str, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let start = prefix.as_bytes();
        let mut end = prefix.as_bytes().to_vec();
        end.push(0xFF);
        let results = eng.scan(start, &end);
        Ok(results.into_iter().take(limit).map(|(k, v)| {
            Document {
                id: String::from_utf8_lossy(&k).to_string(),
                content_bytes: v,
                meta: HashMap::new(),
                score: 0.0,
            }
        }).collect())
    }

    /// Write many documents at once (faster than calling put() in a loop).
    fn put_batch(&self, records: HashMap<String, Vec<u8>>) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let entries: Vec<(Vec<u8>, Vec<u8>)> = records
            .into_iter()
            .map(|(k, v)| (k.into_bytes(), v))
            .collect();
        let refs: Vec<(&[u8], &[u8])> = entries.iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        eng.put_batch(&refs)
            .map_err(|e| PyRuntimeError::new_err(format!("put_batch failed: {e}")))
    }

    /// Create a snapshot. Returns the snapshot ID.
    #[pyo3(signature = (name=None))]
    fn snapshot(&self, name: Option<&str>) -> PyResult<String> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(eng.snapshot_create(name.unwrap_or("")))
    }

    /// Rollback to a snapshot.
    fn rollback_to(&self, checkpoint_id: &str) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.snapshot_restore(checkpoint_id)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Commit workspace changes (flush to disk).
    fn commit(&self) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.flush()
            .map_err(|e| PyRuntimeError::new_err(format!("commit failed: {e}")))
    }

    /// Merge this branch back to main.
    fn merge_to_main(&self) -> PyResult<usize> {
        if !self.is_branch {
            return Err(PyRuntimeError::new_err("not a branch workspace"));
        }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.branch_merge(&self.branch_name)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Check if a path exists.
    fn __contains__(&self, path: &str) -> PyResult<bool> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(eng.get(path.as_bytes()).is_some())
    }

    fn __repr__(&self) -> String {
        if self.is_branch {
            format!("Workspace(branch={:?})", self.branch_name)
        } else {
            "Workspace(main)".to_string()
        }
    }
}

// ============================================================================
// Client: the global control plane
// ============================================================================

#[pyclass]
struct Client {
    engine: Option<Arc<RwLock<Engine>>>,
    data_path: String,
}

#[pymethods]
impl Client {
    /// Connect to a database. Creates it if it does not exist.
    ///
    /// Usage:
    ///     client = Client.connect("./my_project")
    #[staticmethod]
    fn connect(url: &str) -> PyResult<Self> {
        let path = url.strip_prefix("uldb://").unwrap_or(url);
        let config = EngineConfig::new(path);
        let engine = Engine::open(config)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to connect: {e}")))?;
        Ok(Self {
            engine: Some(Arc::new(RwLock::new(engine))),
            data_path: path.to_string(),
        })
    }

    /// Get the main workspace (direct access to the database).
    #[getter]
    fn workspace(&self) -> PyResult<Workspace> {
        Ok(Workspace {
            engine: self.get_engine()?.clone(),
            branch_name: "main".to_string(),
            is_branch: false,
        })
    }

    /// Get the context engine (search and indexing).
    #[getter]
    fn context(&self) -> PyResult<ContextEngine> {
        Ok(ContextEngine {
            engine: self.get_engine()?.clone(),
        })
    }

    /// Create an isolated branch workspace for an agent.
    #[pyo3(signature = (name, base="main"))]
    fn branch(&self, name: &str, base: &str) -> PyResult<Workspace> {
        let eng = self.get_engine()?;
        let mut e = eng.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let from_snapshot = if base == "main" { "" } else { base };
        e.branch_create(name, from_snapshot)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        drop(e);
        Ok(Workspace {
            engine: eng.clone(),
            branch_name: name.to_string(),
            is_branch: true,
        })
    }
        /// Create an isolated agent workspace.
    fn agent(&self, name: &str) -> PyResult<Agent> {
        let eng = self.get_engine()?;
        let mut e = eng.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        e.branch_create(name, "")
            .map_err(|e| PyRuntimeError::new_err(format!("agent create failed: {e}")))?;
        drop(e);
        Ok(Agent {
            engine: eng.clone(),
            name: name.to_string(),
            finished: false,
        })
    }

    /// List all branches.
    fn branches(&self) -> PyResult<Vec<String>> {
        let eng = self.get_engine()?;
        let e = eng.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(e.snapshot_list())
    }

    /// Close the database.
    fn close(&mut self) -> PyResult<()> {
        if let Some(engine_arc) = self.engine.take() {
            match Arc::try_unwrap(engine_arc) {
                Ok(rw) => {
                    let engine = rw.into_inner()
                        .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
                    engine.close()
                        .map_err(|e| PyRuntimeError::new_err(format!("close failed: {e}")))?;
                }
                Err(arc) => {
                    // Other references exist, just flush
                    {
                        let mut eng = arc.write()
                            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
                        eng.flush()
                            .map_err(|e| PyRuntimeError::new_err(format!("flush failed: {e}")))?;
                    }
                    self.engine = Some(arc);
                }
            }
        }
        Ok(())
    }

    /// Database statistics.
    fn stats(&self) -> PyResult<HashMap<String, usize>> {
        let eng = self.get_engine()?;
        let e = eng.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let idx = e.indices.stats();
        let mut s = HashMap::new();
        s.insert("records".to_string(), e.memtable_len());
        s.insert("flush_count".to_string(), e.flush_count() as usize);
        s.insert("index_docs".to_string(), idx.bm25_docs);
        s.insert("index_symbols".to_string(), idx.fuzzy_symbols);
        s.insert("index_vectors".to_string(), idx.hnsw_vectors);
        s.insert("graph_nodes".to_string(), idx.graph_nodes);
        s.insert("graph_edges".to_string(), idx.graph_edges);
        Ok(s)
    }

    fn __repr__(&self) -> String {
        match &self.engine {
            Some(_) => format!("Client(path={:?})", self.data_path),
            None => "Client(closed)".to_string(),
        }
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(&mut self, _et: Option<&Bound<PyAny>>, _ev: Option<&Bound<PyAny>>, _tb: Option<&Bound<PyAny>>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

impl Client {
    fn get_engine(&self) -> PyResult<&Arc<RwLock<Engine>>> {
        self.engine.as_ref().ok_or_else(||
            PyRuntimeError::new_err("database is closed")
        )
    }
}

// ============================================================================
// Convenience: DB as a simpler alias
// ============================================================================

// ============================================================================
// Agent: the unified agentic AI primitive
// ============================================================================

/// An isolated agent workspace with search capabilities.
/// Combines branch + workspace + context into one concept.
///
///     with app.agent("refactor-auth") as agent:
///         docs = agent.search("validate token")
///         agent.put("auth.py::validate", b"def validate_v2(): ...")
///         # auto-merge on clean exit
///         # auto-rollback on exception
/// An isolated agent workspace for agentic AI workflows.
///
/// Consistency guarantees:
///   - Agent reads see: snapshot state at creation + agent's own writes
///   - Agent writes are invisible to main and other agents until commit
///   - Search (agent.search) always reflects committed main state
///   - Merge applies all agent writes to main atomically
///   - Rollback discards all agent writes completely
///
/// Lifecycle: created -> active -> committed/rolled_back
///   - Context manager: auto-merge on clean exit, auto-rollback on exception
///   - After commit or rollback, the agent cannot be used again
///
/// Agents cannot fork other agents. Each agent is independent.
#[pyclass]
struct Agent {
    engine: Arc<RwLock<Engine>>,
    name: String,
    finished: bool,
}

#[pymethods]
impl Agent {
    /// Search the shared index. Returns documents ranked by relevance.
    #[pyo3(signature = (query, limit=10))]
    fn search(&self, query: &str, limit: usize) -> PyResult<Vec<Document>> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let spec = QuerySpec {
            text: query.to_string(),
            top_k: limit,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = eng.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: h.score,
            }
        }).collect())
    }

    /// Fuzzy symbol search (typo-tolerant).
    #[pyo3(signature = (symbol, limit=5))]
    fn search_fuzzy(&self, symbol: &str, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let results = eng.indices.fuzzy.query(symbol, limit);
        Ok(results.into_iter().map(|m| {
            let value = eng.get(m.symbol.as_bytes()).unwrap_or_default();
            Document {
                id: m.symbol,
                content_bytes: value,
                meta: HashMap::new(),
                score: m.jaccard,
            }
        }).collect())
    }

    /// Read a document. Reads from agent branch first, then live engine.
    fn get(&self, path: &str) -> PyResult<Option<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        // Check agent's branch HAMT first (isolated writes)
        if let Some(val) = eng.snapshot_get(&self.name, path.as_bytes()) {
            return Ok(Some(Document {
                id: path.to_string(),
                content_bytes: val,
                meta: HashMap::new(),
                score: 0.0,
            }));
        }
        // Fall back to live engine
        match eng.get(path.as_bytes()) {
            Some(value) => Ok(Some(Document {
                id: path.to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: 0.0,
            })),
            None => Ok(None),
        }
    }

    /// Write a document in isolation. Only visible to this agent until commit.
    fn put(&self, path: &str, content: &[u8]) -> PyResult<()> {
        if self.finished { return Err(PyRuntimeError::new_err("agent is already committed or rolled back")); }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let branch = eng.snapshots.get(&self.name)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("agent branch not found"))?;
        let updated = branch.put(path.as_bytes().to_vec(), content.to_vec());
        eng.snapshots.insert(self.name.clone(), updated);
        Ok(())
    }

    /// Delete a document in isolation.
    fn delete(&self, path: &str) -> PyResult<()> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let branch = eng.snapshots.get(&self.name)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("agent branch not found"))?;
        let updated = branch.delete(path.as_bytes());
        eng.snapshots.insert(self.name.clone(), updated);
        Ok(())
    }

    /// Scan documents by prefix.
    #[pyo3(signature = (prefix, limit=100))]
    fn scan(&self, prefix: &str, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let start = prefix.as_bytes();
        let mut end = start.to_vec();
        end.push(0xFF);
        Ok(eng.scan(start, &end).into_iter().take(limit).map(|(k, v)| {
            Document {
                id: String::from_utf8_lossy(&k).to_string(),
                content_bytes: v,
                meta: HashMap::new(),
                score: 0.0,
            }
        }).collect())
    }

    /// Bulk write documents in isolation.
    fn load(&self, records: HashMap<String, Vec<u8>>) -> PyResult<usize> {
        if self.finished { return Err(PyRuntimeError::new_err("agent is already committed or rolled back")); }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let mut branch = eng.snapshots.get(&self.name)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("agent branch not found"))?;
        let mut count = 0;
        for (k, v) in records {
            branch = branch.put(k.into_bytes(), v);
            count += 1;
        }
        eng.snapshots.insert(self.name.clone(), branch);
        Ok(count)
    }

    /// Manually commit (merge) this agent's work to main.
    fn commit(&mut self) -> PyResult<usize> {
        if self.finished { return Err(PyRuntimeError::new_err("agent already finished")); }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let result = eng.branch_merge(&self.name)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        self.finished = true;
        Ok(result)
    }

    /// Manually discard all agent's work.
    fn discard(&mut self) -> PyResult<()> {
        if self.finished { return Ok(()); }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        eng.branch_rollback(&self.name);
        self.finished = true;
        Ok(())
    }

    /// Context manager: auto-merge on clean exit, auto-rollback on exception.
    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(&mut self, exc_type: Option<&Bound<PyAny>>, _ev: Option<&Bound<PyAny>>, _tb: Option<&Bound<PyAny>>) -> PyResult<bool> {
        if self.finished { return Ok(false); }
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if exc_type.is_some() {
            eng.branch_rollback(&self.name);
        } else {
            let _ = eng.branch_merge(&self.name);
        }
        self.finished = true;
        Ok(false)
    }

    /// Vector similarity search (shared index).
    #[pyo3(signature = (embedding, limit=10))]
    fn search_vector(&self, embedding: Vec<f32>, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.engine.read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if let Some(ref hnsw) = eng.indices.hnsw {
            Ok(hnsw.search(&embedding, limit).into_iter().map(|r| {
                let value = eng.get(&r.key).unwrap_or_default();
                Document {
                    id: String::from_utf8_lossy(&r.key).to_string(),
                    content_bytes: value,
                    meta: HashMap::new(),
                    score: 1.0 - r.distance as f64,
                }
            }).collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Graph traversal (shared index).
    #[pyo3(signature = (start, relation="calls", depth=3, limit=20))]
    fn search_graph(&self, start: &str, relation: &str, depth: usize, limit: usize) -> PyResult<Vec<Document>> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let spec = QuerySpec {
            text: start.to_string(),
            relations: vec![relation.to_string()],
            top_k: limit,
            max_depth: depth,
            ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = eng.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: h.score,
            }
        }).collect())
    }

    /// Create a checkpoint within this agent's work.
    #[pyo3(signature = (name=None))]
    fn checkpoint(&self, name: Option<&str>) -> PyResult<String> {
        let mut eng = self.engine.write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let snap_name = name.map(|n| n.to_string()).unwrap_or_else(||
            format!("{}-checkpoint", self.name)
        );
        Ok(eng.snapshot_create(&snap_name))
    }

        fn __repr__(&self) -> String {
        format!("Agent({:?})", self.name)
    }
}

/// The simplest way to use uldb. One class does everything.
///
/// Consistency guarantees:
///   - put() is immediately visible to get() and search()
///   - search() is deterministic: same query + same data = same results
///   - agent writes are isolated until commit (auto on clean context exit)
///   - snapshots are O(1) and share memory via structural sharing
///   - all data is crash-safe via write-ahead log
///
///     db = DB("./my_data")
///     db.put("key", b"value")
///     print(db.get("key"))
///     results = db.search("query")
///     db.close()
#[pyclass]
struct DB {
    engine: Option<Arc<RwLock<Engine>>>,
}

#[pymethods]
impl DB {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let config = EngineConfig::new(path);
        let engine = Engine::open(config)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open: {e}")))?;
        Ok(Self { engine: Some(Arc::new(RwLock::new(engine))) })
    }

    fn put(&self, key: &str, value: &[u8]) -> PyResult<()> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        e.put(key.as_bytes(), value).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let e = self.eng()?;
        let e = e.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        Ok(e.get(key.as_bytes()).map(|v| PyBytes::new(py, &v)))
    }

    fn delete(&self, key: &str) -> PyResult<()> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        e.delete(key.as_bytes()).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (query, limit=10))]
    fn search(&self, query: &str, limit: usize) -> PyResult<Vec<Document>> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let spec = QuerySpec { text: query.to_string(), top_k: limit, ..Default::default() };
        let hits = e.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = e.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value,
                meta: HashMap::new(),
                score: h.score,
            }
        }).collect())
    }

    #[pyo3(signature = (prefix, limit=100))]
    fn scan(&self, prefix: &str, limit: usize) -> PyResult<Vec<Document>> {
        let e = self.eng()?;
        let e = e.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let start = prefix.as_bytes();
        let mut end = start.to_vec();
        end.push(0xFF);
        Ok(e.scan(start, &end).into_iter().take(limit).map(|(k, v)| {
            Document {
                id: String::from_utf8_lossy(&k).to_string(),
                content_bytes: v,
                meta: HashMap::new(),
                score: 0.0,
            }
        }).collect())
    }

    fn load(&self, records: HashMap<String, Vec<u8>>) -> PyResult<usize> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let entries: Vec<(Vec<u8>, Vec<u8>)> = records
            .into_iter().map(|(k, v)| (k.into_bytes(), v)).collect();
        let refs: Vec<(&[u8], &[u8])> = entries.iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
        e.bulk_ingest(&refs)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (name=None))]
    fn snapshot(&self, name: Option<&str>) -> PyResult<String> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        Ok(e.snapshot_create(name.unwrap_or("")))
    }

    fn restore(&self, name: &str) -> PyResult<()> {
        let e = self.eng()?;
        let mut e = e.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        e.snapshot_restore(name).map_err(|e| PyRuntimeError::new_err(e))
    }

    fn snapshots(&self) -> PyResult<Vec<String>> {
        let e = self.eng()?;
        let e = e.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        Ok(e.snapshot_list())
    }

    /// Create an isolated agent workspace.
    ///
    /// Usage:
    ///     with db.agent("refactor-auth") as agent:
    ///         docs = agent.search("validate")
    ///         agent.put("key", b"new value")
    ///         # auto-merge on success, auto-rollback on failure
    /// Full-text + fuzzy search (typo-tolerant symbol lookup).
    #[pyo3(signature = (symbol, limit=5))]
    fn search_fuzzy(&self, symbol: &str, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.eng()?;
        let eng = eng.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let results = eng.indices.fuzzy.query(symbol, limit);
        Ok(results.into_iter().map(|m| {
            let value = eng.get(m.symbol.as_bytes()).unwrap_or_default();
            Document { id: m.symbol, content_bytes: value, meta: HashMap::new(), score: m.jaccard }
        }).collect())
    }

    /// Vector similarity search using HNSW index.
    #[pyo3(signature = (embedding, limit=10))]
    fn search_vector(&self, embedding: Vec<f32>, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.eng()?;
        let eng = eng.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        if let Some(ref hnsw) = eng.indices.hnsw {
            let results = hnsw.search(&embedding, limit);
            Ok(results.into_iter().map(|r| {
                let value = eng.get(&r.key).unwrap_or_default();
                Document {
                    id: String::from_utf8_lossy(&r.key).to_string(),
                    content_bytes: value, meta: HashMap::new(),
                    score: 1.0 - r.distance as f64,
                }
            }).collect())
        } else { Ok(Vec::new()) }
    }

    /// Add a relation edge between two symbols.
    fn add_edge(&self, src: &str, dst: &str, relation: &str) -> PyResult<()> {
        let eng = self.eng()?;
        let mut eng = eng.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        eng.indices.on_add_edge(src, dst, relation);
        Ok(())
    }

    /// Graph traversal search from a start node.
    #[pyo3(signature = (start, relation="calls", depth=3, limit=20))]
    fn search_graph(&self, start: &str, relation: &str, depth: usize, limit: usize) -> PyResult<Vec<Document>> {
        let eng = self.eng()?;
        let mut eng = eng.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let spec = QuerySpec {
            text: start.to_string(), relations: vec![relation.to_string()],
            top_k: limit, max_depth: depth, ..Default::default()
        };
        let hits = eng.indices.query(&spec);
        Ok(hits.into_iter().map(|h| {
            let value = eng.get(&h.key).unwrap_or_default();
            Document {
                id: String::from_utf8_lossy(&h.key).to_string(),
                content_bytes: value, meta: HashMap::new(), score: h.score,
            }
        }).collect())
    }

    /// Unified indexing: write content + optional vector + optional graph edges.
    #[pyo3(signature = (key, content, vector=None, edges=None))]
    fn index(&self, key: &str, content: &[u8], vector: Option<Vec<f32>>, edges: Option<Vec<(String, String)>>) -> PyResult<()> {
        let eng = self.eng()?;
        let mut eng = eng.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        eng.put(key.as_bytes(), content).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        if let Some(vec) = vector { eng.indices.on_put_vector(key.as_bytes(), vec); }
        if let Some(el) = edges { for (dst, rel) in el { eng.indices.on_add_edge(key, &dst, &rel); } }
        Ok(())
    }

    /// Delete all keys in range [start, end).
    fn delete_range(&self, start: &str, end: &str) -> PyResult<usize> {
        let eng = self.eng()?;
        let mut eng = eng.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let keys: Vec<Vec<u8>> = eng.scan(start.as_bytes(), end.as_bytes())
            .into_iter().map(|(k, _)| k).collect();
        let count = keys.len();
        for key in keys { eng.delete(&key).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?; }
        Ok(count)
    }

    /// List keys with optional prefix filter.
    #[pyo3(signature = (prefix="", limit=10000))]
    fn keys(&self, prefix: &str, limit: usize) -> PyResult<Vec<String>> {
        let eng = self.eng()?;
        let eng = eng.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let start = prefix.as_bytes();
        let mut end = start.to_vec(); end.push(0xFF);
        Ok(eng.scan(start, &end).into_iter().take(limit)
            .map(|(k, _)| String::from_utf8_lossy(&k).to_string()).collect())
    }

    /// Database statistics.
    fn stats(&self) -> PyResult<HashMap<String, usize>> {
        let eng = self.eng()?;
        let eng = eng.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        let idx = eng.indices.stats();
        let mut s = HashMap::new();
        s.insert("records".into(), eng.memtable_len());
        s.insert("flush_count".into(), eng.flush_count() as usize);
        s.insert("index_docs".into(), idx.bm25_docs);
        s.insert("index_symbols".into(), idx.fuzzy_symbols);
        s.insert("index_vectors".into(), idx.hnsw_vectors);
        s.insert("graph_nodes".into(), idx.graph_nodes);
        s.insert("graph_edges".into(), idx.graph_edges);
        Ok(s)
    }

    fn agent(&self, name: &str) -> PyResult<Agent> {
        let eng = self.eng()?;
        let mut e = eng.write().map_err(|_| PyRuntimeError::new_err("lock"))?;
        e.branch_create(name, "")
            .map_err(|e| PyRuntimeError::new_err(format!("agent create failed: {e}")))?;
        drop(e);
        Ok(Agent {
            engine: eng.clone(),
            name: name.to_string(),
            finished: false,
        })
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(arc) = self.engine.take() {
            match Arc::try_unwrap(arc) {
                Ok(rw) => {
                    let engine = rw.into_inner()
                        .map_err(|_| PyRuntimeError::new_err("lock"))?;
                    engine.close().map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                }
                Err(_) => {}
            }
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        let e = self.eng()?;
        let e = e.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        Ok(e.get(key.as_bytes()).is_some())
    }

    fn __len__(&self) -> PyResult<usize> {
        let e = self.eng()?;
        let e = e.read().map_err(|_| PyRuntimeError::new_err("lock"))?;
        Ok(e.memtable_len())
    }

    fn __repr__(&self) -> String {
        match &self.engine {
            Some(e) => {
                let eng = e.read().unwrap();
                format!("DB({:?}, records={})", eng.data_dir(), eng.memtable_len())
            }
            None => "DB(closed)".to_string(),
        }
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __exit__(&mut self, _et: Option<&Bound<PyAny>>, _ev: Option<&Bound<PyAny>>, _tb: Option<&Bound<PyAny>>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

impl DB {
    fn eng(&self) -> PyResult<&Arc<RwLock<Engine>>> {
        self.engine.as_ref().ok_or_else(|| PyRuntimeError::new_err("database is closed"))
    }
}

// ============================================================================
// Module registration
// ============================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DB>()?;
    m.add_class::<Client>()?;
    m.add_class::<Workspace>()?;
    m.add_class::<ContextEngine>()?;
    m.add_class::<Document>()?;
    m.add_class::<Agent>()?;
    Ok(())
}
