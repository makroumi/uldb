//! uldb engine microbenchmarks.
//!
//! Run: cargo bench --bench engine_bench
//!
//! Measures raw storage engine performance without network/TLS overhead.
//! All operations are against a local Engine instance with NVMe-backed WAL.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use uldb::engine::{Engine, EngineConfig};
use uldb::query::planner::QuerySpec;

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = env::temp_dir();
    p.push(format!("uldb_bench_{name}_{}", std::process::id()));
    p
}

fn cleanup(dir: &std::path::Path) {
    let _ = fs::remove_dir_all(dir);
}

fn format_ops(ops: u64, elapsed: Duration) -> String {
    let per_sec = ops as f64 / elapsed.as_secs_f64();
    if per_sec >= 1_000_000.0 {
        format!("{:.2}M ops/sec", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.1}K ops/sec", per_sec / 1_000.0)
    } else {
        format!("{:.0} ops/sec", per_sec)
    }
}

fn format_latency(elapsed: Duration, ops: u64) -> String {
    let ns_per_op = elapsed.as_nanos() as f64 / ops as f64;
    if ns_per_op >= 1_000_000.0 {
        format!("{:.2}ms", ns_per_op / 1_000_000.0)
    } else if ns_per_op >= 1_000.0 {
        format!("{:.1}us", ns_per_op / 1_000.0)
    } else {
        format!("{:.0}ns", ns_per_op)
    }
}

// ============================================================================
// PUT benchmarks
// ============================================================================

fn bench_put_sequential(n: u64) {
    let dir = tmp_dir("put_seq");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Warmup
    for i in 0..1000u64 {
        let key = format!("warmup_{i:08}");
        engine.put(key.as_bytes(), b"warmup_value").unwrap();
    }

    let start = Instant::now();
    for i in 0..n {
        let key = format!("bench_key_{i:08}");
        let val = format!("bench_value_{i}_padding_for_realistic_size_xxxxxxxxxx");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }
    let elapsed = start.elapsed();

    println!("  PUT sequential ({n} ops):     {} | {}",
        format_ops(n, elapsed), format_latency(elapsed, n));

    drop(engine);
    cleanup(&dir);
}

fn bench_put_batch(n: u64, batch_size: usize) {
    let dir = tmp_dir("put_batch");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    let batches = n as usize / batch_size;
    let total = (batches * batch_size) as u64;

    let start = Instant::now();
    for b in 0..batches {
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..batch_size)
            .map(|i| {
                let idx = b * batch_size + i;
                (
                    format!("batch_key_{idx:08}").into_bytes(),
                    format!("batch_val_{idx}_padding_xxxxxxxxxx").into_bytes(),
                )
            })
            .collect();
        let refs: Vec<(&[u8], &[u8])> = entries.iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        engine.put_batch(&refs).unwrap();
    }
    let elapsed = start.elapsed();

    println!("  PUT batch ({}x{} = {} ops): {} | {}",
        batches, batch_size, total,
        format_ops(total, elapsed), format_latency(elapsed, total));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// GET benchmarks
// ============================================================================

fn bench_get_hit(n: u64) {
    let dir = tmp_dir("get_hit");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Seed data
    for i in 0..n {
        let key = format!("get_key_{i:08}");
        let val = format!("get_val_{i}_padding_for_realistic_size_xxxxxxxxxx");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    // Benchmark reads (all hits, reading from memtable)
    let start = Instant::now();
    for i in 0..n {
        let key = format!("get_key_{i:08}");
        let _ = engine.get(key.as_bytes());
    }
    let elapsed = start.elapsed();

    println!("  GET hit ({n} ops):            {} | {}",
        format_ops(n, elapsed), format_latency(elapsed, n));

    drop(engine);
    cleanup(&dir);
}

fn bench_get_miss(n: u64) {
    let dir = tmp_dir("get_miss");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Seed some data so bloom filters are active
    for i in 0..1000u64 {
        engine.put(format!("exists_{i:06}").as_bytes(), b"v").unwrap();
    }
    engine.flush().unwrap();
    std::thread::sleep(Duration::from_millis(100));

    // Benchmark misses (bloom filter should skip pages)
    let start = Instant::now();
    for i in 0..n {
        let key = format!("missing_{i:08}");
        let _ = engine.get(key.as_bytes());
    }
    let elapsed = start.elapsed();

    println!("  GET miss ({n} ops):           {} | {}",
        format_ops(n, elapsed), format_latency(elapsed, n));

    drop(engine);
    cleanup(&dir);
}

fn bench_get_cached(n: u64) {
    let dir = tmp_dir("get_cached");
    let config = EngineConfig::new(&dir).with_flush_threshold(1024);
    let mut engine = Engine::open(config).unwrap();

    // Seed and flush so data is in pages
    for i in 0..100u64 {
        let key = format!("cached_{i:04}");
        let val = format!("cached_val_{i}_xxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }
    engine.flush().unwrap();
    std::thread::sleep(Duration::from_millis(100));

    // First read populates LRU cache
    for i in 0..100u64 {
        engine.get(format!("cached_{i:04}").as_bytes());
    }

    // Benchmark cached reads (hot path)
    let start = Instant::now();
    for _ in 0..n {
        for i in 0..100u64 {
            let _ = engine.get(format!("cached_{i:04}").as_bytes());
        }
    }
    let total = n * 100;
    let elapsed = start.elapsed();

    println!("  GET cached ({total} ops):       {} | {}",
        format_ops(total, elapsed), format_latency(elapsed, total));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// SCAN benchmarks
// ============================================================================

fn bench_scan(n_records: u64, scan_range: u64) {
    let dir = tmp_dir("scan");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    for i in 0..n_records {
        let key = format!("scan_{i:08}");
        engine.put(key.as_bytes(), b"scan_value_padding_xxxx").unwrap();
    }

    let iterations = 1000u64;
    let start = Instant::now();
    for _ in 0..iterations {
        let s = format!("scan_{:08}", n_records / 4);
        let e = format!("scan_{:08}", n_records / 4 + scan_range);
        let _ = engine.scan(s.as_bytes(), e.as_bytes());
    }
    let elapsed = start.elapsed();

    println!("  SCAN {scan_range} from {n_records} ({iterations} iters): {} | {}",
        format_ops(iterations, elapsed), format_latency(elapsed, iterations));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// DELETE benchmark
// ============================================================================

fn bench_delete(n: u64) {
    let dir = tmp_dir("delete");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    for i in 0..n {
        engine.put(format!("del_{i:08}").as_bytes(), b"value").unwrap();
    }

    let start = Instant::now();
    for i in 0..n {
        engine.delete(format!("del_{i:08}").as_bytes()).unwrap();
    }
    let elapsed = start.elapsed();

    println!("  DELETE ({n} ops):             {} | {}",
        format_ops(n, elapsed), format_latency(elapsed, n));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// Query benchmarks
// ============================================================================

fn bench_bm25_query(n_docs: usize, n_queries: u64) {
    let dir = tmp_dir("bm25");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Index realistic code documents
    let functions = [
        "validate_token", "hash_password", "send_email", "parse_json",
        "connect_db", "query_users", "update_profile", "delete_account",
        "generate_report", "encrypt_data", "decrypt_payload", "log_event",
        "authenticate_user", "authorize_request", "create_session",
        "invalidate_cache", "compress_data", "decompress_stream",
    ];

    for i in 0..n_docs {
        let func = functions[i % functions.len()];
        let key = format!("module_{:04}::{}_{}", i / 20, func, i);
        let val = format!(
            "def {func}(arg{i}): return process({func}, arg{i}) # line {i}"
        );
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    let queries = [
        "validate token authentication",
        "hash password bcrypt",
        "send email notification",
        "connect database query",
        "encrypt decrypt data",
    ];

    let start = Instant::now();
    for i in 0..n_queries {
        let q = queries[i as usize % queries.len()];
        let spec = QuerySpec {
            text: q.to_string(),
            top_k: 10,
            ..Default::default()
        };
        let _ = engine.indices.query(&spec);
    }
    let elapsed = start.elapsed();

    println!("  BM25 query ({n_docs} docs, {n_queries} queries): {} | {}",
        format_ops(n_queries, elapsed), format_latency(elapsed, n_queries));

    drop(engine);
    cleanup(&dir);
}

fn bench_fuzzy_query(n_symbols: usize, n_queries: u64) {
    let dir = tmp_dir("fuzzy");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    let prefixes = ["get", "set", "create", "delete", "update", "find", "validate", "parse"];
    let suffixes = ["User", "Token", "Email", "Password", "Session", "Config", "Cache", "Data"];

    for i in 0..n_symbols {
        let name = format!("{}{}_{}", prefixes[i % prefixes.len()], suffixes[i % suffixes.len()], i);
        engine.put(name.as_bytes(), b"function body").unwrap();
    }

    let typo_queries = [
        "getUsrById", "vallidateToken", "createSesion",
        "updaetProfile", "deleteAccont",
    ];

    let start = Instant::now();
    for i in 0..n_queries {
        let q = typo_queries[i as usize % typo_queries.len()];
        let spec = QuerySpec {
            text: q.to_string(),
            top_k: 5,
            ..Default::default()
        };
        let _ = engine.indices.query(&spec);
    }
    let elapsed = start.elapsed();

    println!("  Fuzzy query ({n_symbols} symbols, {n_queries} queries): {} | {}",
        format_ops(n_queries, elapsed), format_latency(elapsed, n_queries));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// WAL recovery benchmark
// ============================================================================

fn bench_wal_recovery(n: u64) {
    let dir = tmp_dir("wal_recovery");

    // Phase 1: write records
    {
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();
        for i in 0..n {
            let key = format!("wal_{i:08}");
            let val = format!("wal_val_{i}_padding_xxxxxxxxxxxx");
            engine.put(key.as_bytes(), val.as_bytes()).unwrap();
        }
        // Drop without close (simulate crash)
    }

    // Phase 2: measure recovery time
    let start = Instant::now();
    let config = EngineConfig::new(&dir);
    let engine = Engine::open(config).unwrap();
    let elapsed = start.elapsed();

    println!("  WAL recovery ({n} records):   {:.1}ms | {} records/sec",
        elapsed.as_secs_f64() * 1000.0,
        format_ops(n, elapsed));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// Snapshot benchmark
// ============================================================================

fn bench_snapshot(n: u64) {
    let dir = tmp_dir("snapshot");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    for i in 0..n {
        engine.put(format!("snap_{i:06}").as_bytes(), b"value").unwrap();
    }

    let iterations = 10_000u64;
    let start = Instant::now();
    for i in 0..iterations {
        let name = format!("snap_{i}");
        engine.snapshot_create(&name);
        engine.snapshot_delete(&name);
    }
    let elapsed = start.elapsed();

    println!("  Snapshot create+delete ({n} keys, {iterations} iters): {} | {}",
        format_ops(iterations, elapsed), format_latency(elapsed, iterations));

    drop(engine);
    cleanup(&dir);
}

// ============================================================================
// Size benchmark
// ============================================================================

fn bench_disk_footprint(n: u64) {
    let dir = tmp_dir("footprint");
    let config = EngineConfig::new(&dir).with_flush_threshold(64 * 1024);
    let mut engine = Engine::open(config).unwrap();

    let mut raw_bytes = 0u64;
    for i in 0..n {
        let key = format!("fp_{i:08}");
        let val = format!("def function_{i}(x): return validate(x, token_{i}) # {i}");
        raw_bytes += key.len() as u64 + val.len() as u64;
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }
    engine.flush().unwrap();
    std::thread::sleep(Duration::from_millis(200));
    engine.close().unwrap();

    // Measure disk usage
    let mut total_disk = 0u64;
    for entry in fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        if meta.is_file() {
            total_disk += meta.len();
        }
        if meta.is_dir() {
            for sub in fs::read_dir(entry.path()).unwrap() {
                let sub = sub.unwrap();
                total_disk += sub.metadata().unwrap().len();
            }
        }
    }

    let ratio = total_disk as f64 / raw_bytes as f64;
    println!("  Disk footprint ({n} records):");
    println!("    Raw data:     {:.1} KB", raw_bytes as f64 / 1024.0);
    println!("    On disk:      {:.1} KB", total_disk as f64 / 1024.0);
    println!("    Ratio:        {ratio:.2}x (includes WAL + pages + indices)");

    cleanup(&dir);
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("======================================================");
    println!("  uldb Engine Microbenchmarks");
    println!("  AMD Ryzen 7 4700U | 8 cores | NVMe SSD");
    println!("======================================================\n");

    println!("--- WRITE ---");
    bench_put_sequential(100_000);
    bench_put_batch(100_000, 100);
    bench_put_batch(100_000, 1000);

    println!("\n--- READ ---");
    bench_get_hit(100_000);
    bench_get_miss(100_000);
    bench_get_cached(1_000);

    println!("\n--- SCAN ---");
    bench_scan(10_000, 100);
    bench_scan(10_000, 1000);

    println!("\n--- DELETE ---");
    bench_delete(50_000);

    println!("\n--- QUERY ---");
    bench_bm25_query(10_000, 1_000);
    bench_fuzzy_query(5_000, 500);

    println!("\n--- RECOVERY ---");
    bench_wal_recovery(100_000);

    println!("\n--- SNAPSHOT ---");
    bench_snapshot(10_000);

    println!("\n--- STORAGE ---");
    bench_disk_footprint(10_000);

    println!("\n--- PUT BREAKDOWN ---");
    {
        let dir = tmp_dir("put_profile");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        // Warmup
        for i in 0..1000u64 {
            engine.put(format!("w{i:06}").as_bytes(), b"warmup_val_xxxx").unwrap();
        }

        let n = 10_000u64;
        let mut acc = [0u128; 5];
        for i in 0..n {
            let key = format!("prof_{i:08}");
            let val = format!("def function_{i}(x): return validate(x) # {i}");
            engine.put_profiled(key.as_bytes(), val.as_bytes(), &mut acc).unwrap();
        }

        let labels = ["WAL", "Memtable", "Index", "HAMT", "Flush"];
        let total: u128 = acc.iter().sum();
        println!("  Per-put breakdown ({n} ops, avg per op):");
        for (i, label) in labels.iter().enumerate() {
            let ns = acc[i] / n as u128;
            let pct = (acc[i] as f64 / total as f64) * 100.0;
            println!("    {label:10} {ns:>7}ns  ({pct:.1}%)");
        }
        println!("    {:<10} {:>7}ns", "TOTAL", total / n as u128);

        drop(engine);
        cleanup(&dir);
    }

        println!("\n--- INDEX BREAKDOWN ---");
    {
        let dir = tmp_dir("idx_profile");
        let config = EngineConfig::new(&dir);
        let mut engine = Engine::open(config).unwrap();

        let n = 10_000u64;
        let mut t_contains = 0u128;
        let mut t_format = 0u128;
        let mut t_add_doc = 0u128;
        let mut t_fuzzy = 0u128;

        for i in 0..n {
            let key = format!("prof_{i:08}");
            let val = format!("def function_{i}(x): return validate(x) # {i}");
            let kb = key.as_bytes();
            let vb = val.as_bytes();

            let s0 = std::time::Instant::now();
            let exists = engine.indices.bm25.contains_key(kb);
            let s1 = std::time::Instant::now();
            if exists { engine.indices.bm25.remove_document(kb); }

            let ks = String::from_utf8_lossy(kb);
            let content = format!("{} {}", ks, String::from_utf8_lossy(vb));
            let s2 = std::time::Instant::now();

            engine.indices.bm25.add_document(kb.to_vec(), &content);
            let s3 = std::time::Instant::now();

            engine.indices.fuzzy.add(&ks);
            let s4 = std::time::Instant::now();

            t_contains += (s1 - s0).as_nanos();
            t_format += (s2 - s1).as_nanos();
            t_add_doc += (s3 - s2).as_nanos();
            t_fuzzy += (s4 - s3).as_nanos();

            // Also do the actual put for realistic state
            engine.put(kb, vb).unwrap();
        }

        let total = t_contains + t_format + t_add_doc + t_fuzzy;
        println!("  Index per-op breakdown ({n} ops):");
        println!("    contains_key: {:>6}ns  ({:.1}%)", t_contains / n as u128, t_contains as f64 / total as f64 * 100.0);
        println!("    format+lossy: {:>6}ns  ({:.1}%)", t_format / n as u128, t_format as f64 / total as f64 * 100.0);
        println!("    add_document: {:>6}ns  ({:.1}%)", t_add_doc / n as u128, t_add_doc as f64 / total as f64 * 100.0);
        println!("    fuzzy.add:    {:>6}ns  ({:.1}%)", t_fuzzy / n as u128, t_fuzzy as f64 / total as f64 * 100.0);
        println!("    TOTAL:        {:>6}ns", total / n as u128);

        drop(engine);
        cleanup(&dir);
    }

        println!("\n--- AGENTIC WORKLOAD ---");
    bench_codebase_ingest(10_000);
    bench_codebase_ingest(50_000);
    bench_agent_query_session(10_000);
    bench_concurrent_agents(10_000, 4);

    println!("\n======================================================");
    println!("  Benchmark complete");
    println!("======================================================");
}

// ============================================================================
// Agentic workload benchmarks
// ============================================================================

fn bench_codebase_ingest(n: usize) {
    let dir = tmp_dir("ingest");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Build realistic code records
    let functions = [
        "validate_token", "hash_password", "send_email", "parse_json",
        "connect_db", "query_users", "update_profile", "delete_account",
        "generate_report", "encrypt_data", "decrypt_payload", "log_event",
        "authenticate_user", "authorize_request", "create_session",
        "invalidate_cache", "compress_data", "decompress_stream",
    ];

    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let func = functions[i % functions.len()];
            let key = format!("module_{:04}::{}_{}", i / 20, func, i);
            let val = format!(
                "def {func}(arg{i}): return process({func}, arg{i}) # line {i} \
                 validates input and returns processed result for the {func} operation"
            );
            (key.into_bytes(), val.into_bytes())
        })
        .collect();

    let refs: Vec<(&[u8], &[u8])> = entries.iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();

    let start = Instant::now();
    engine.bulk_ingest(&refs).unwrap();
    let elapsed = start.elapsed();

    println!("  Codebase ingest ({n} symbols): {:.1}ms | {} | {:.1} symbols/sec",
        elapsed.as_secs_f64() * 1000.0,
        format_ops(n as u64, elapsed),
        n as f64 / elapsed.as_secs_f64(),
    );

    drop(engine);
    cleanup(&dir);
}

fn bench_agent_query_session(n_docs: usize) {
    let dir = tmp_dir("agent_session");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Ingest a codebase
    let functions = [
        "validate_token", "hash_password", "send_email", "parse_json",
        "connect_db", "query_users", "update_profile", "delete_account",
    ];

    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..n_docs)
        .map(|i| {
            let func = functions[i % functions.len()];
            let key = format!("src/{}_{:04}.py::{}", func, i / 8, func);
            let val = format!("def {func}(x): return validate(x) # {func} implementation {i}");
            (key.into_bytes(), val.into_bytes())
        })
        .collect();

    let refs: Vec<(&[u8], &[u8])> = entries.iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    engine.bulk_ingest(&refs).unwrap();

    // Simulate an agent conversation: mix of queries and reads
    let queries = [
        "validate token authentication JWT",
        "hash password bcrypt salt",
        "send email notification SMTP",
        "database connection query",
        "encrypt decrypt data payload",
    ];

    let n_ops = 1000u64;
    let start = Instant::now();
    for i in 0..n_ops {
        match i % 5 {
            0..=2 => {
                // 60% queries
                let q = queries[i as usize % queries.len()];
                let spec = QuerySpec {
                    text: q.to_string(),
                    top_k: 10,
                    ..Default::default()
                };
                let _ = engine.indices.query(&spec);
            }
            3 => {
                // 20% point reads
                let key = format!("src/validate_token_{:04}.py::validate_token", i % (n_docs as u64 / 8));
                let _ = engine.get(key.as_bytes());
            }
            _ => {
                // 20% scan
                let _ = engine.scan(b"src/validate", b"src/validate~");
            }
        }
    }
    let elapsed = start.elapsed();

    println!("  Agent session ({n_docs} docs, {n_ops} mixed ops): {:.1}ms | {} | {:.1}us/op",
        elapsed.as_secs_f64() * 1000.0,
        format_ops(n_ops, elapsed),
        elapsed.as_secs_f64() * 1_000_000.0 / n_ops as f64,
    );

    drop(engine);
    cleanup(&dir);
}

fn bench_concurrent_agents(n_docs: usize, n_agents: usize) {
    let dir = tmp_dir("concurrent_agents");
    let config = EngineConfig::new(&dir);
    let mut engine = Engine::open(config).unwrap();

    // Ingest shared codebase
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..n_docs)
        .map(|i| {
            let key = format!("code_{i:06}");
            let val = format!("def function_{i}(x): return process(x) # implementation");
            (key.into_bytes(), val.into_bytes())
        })
        .collect();

    let refs: Vec<(&[u8], &[u8])> = entries.iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    engine.bulk_ingest(&refs).unwrap();

    // Each agent creates a snapshot (simulating branch)
    let start = Instant::now();
    for agent in 0..n_agents {
        let snap_name = format!("agent_{agent}_workspace");
        engine.snapshot_create(&snap_name);

        // Agent reads 100 symbols from snapshot
        for i in 0..100 {
            let key = format!("code_{:06}", (agent * 100 + i) % n_docs);
            let _ = engine.snapshot_get(&snap_name, key.as_bytes());
        }

        // Agent queries
        for _ in 0..10 {
            let spec = QuerySpec {
                text: "validate process function".to_string(),
                top_k: 5,
                ..Default::default()
            };
            let _ = engine.indices.query(&spec);
        }

        engine.snapshot_delete(&snap_name);
    }
    let elapsed = start.elapsed();

    let total_ops = n_agents as u64 * (1 + 100 + 10 + 1); // create + reads + queries + delete
    println!("  {n_agents} agents ({n_docs} shared docs): {:.1}ms | {} total ops | {:.1}us/op",
        elapsed.as_secs_f64() * 1000.0,
        format_ops(total_ops, elapsed),
        elapsed.as_secs_f64() * 1_000_000.0 / total_ops as f64,
    );

    drop(engine);
    cleanup(&dir);
}
