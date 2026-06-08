use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use uldb::engine::{Engine, EngineConfig};
use uldb::storage::compaction::CompactionManager;
use uldb::storage::page::Page;

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = env::temp_dir();
    p.push(format!("uldb_scan_read_through_{name}_{}", std::process::id()));
    p
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn make_page(entries: &[(&str, &str, bool)]) -> Page {
    let mut p = Page::new(0);
    for (k, v, tombstone) in entries {
        p.push(k.as_bytes().to_vec(), v.as_bytes().to_vec(), *tombstone);
    }
    p.sort();
    p
}

#[test]
fn page_range_is_half_open() {
    let mut page = Page::new(0);
    for i in 0..10u32 {
        page.push(
            format!("key_{i:02}").into_bytes(),
            format!("val_{i}").into_bytes(),
            false,
        );
    }
    page.sort();

    let keys: Vec<Vec<u8>> = page
        .range(b"key_03", b"key_07")
        .map(|rec| rec.key.clone())
        .collect();

    assert_eq!(
        keys,
        vec![
            b"key_03".to_vec(),
            b"key_04".to_vec(),
            b"key_05".to_vec(),
            b"key_06".to_vec(),
        ]
    );
}

#[test]
fn compaction_scan_prefers_newer_data_and_honors_tombstones() {
    let mut cm = CompactionManager::new();

    // First 4 pages trigger L0 -> L1 compaction.
    cm.add_l0_page(make_page(&[("key_001", "old_1", false)]));
    cm.add_l0_page(make_page(&[("key_002", "old_2", false)]));
    cm.add_l0_page(make_page(&[("key_003", "old_3", false)]));
    cm.add_l0_page(make_page(&[("key_004", "old_4", false)]));

    // Newer L0 updates.
    cm.add_l0_page(make_page(&[("key_002", "new_2", false)]));
    cm.add_l0_page(make_page(&[("key_003", "", true)]));

    let rows = cm.scan(b"key_001", b"key_005");

    assert_eq!(rows.get(b"key_001".as_ref()), Some(&b"old_1".to_vec()));
    assert_eq!(rows.get(b"key_002".as_ref()), Some(&b"new_2".to_vec()));
    assert_eq!(rows.get(b"key_004".as_ref()), Some(&b"old_4".to_vec()));
    assert!(!rows.contains_key(b"key_003".as_ref()));
    assert_eq!(rows.len(), 3);
}

#[test]
fn engine_scan_reads_compacted_pages_and_memtable() {
    let dir = tmp_dir("engine");
    let config = EngineConfig::new(&dir).with_flush_threshold(10_000);
    let mut engine = Engine::open(config).unwrap();

    // First batch stays in memtable until explicit flush.
    for i in 0..20u32 {
        let key = format!("key_{i:03}");
        let val = format!("compacted_val_{i}");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    // Force first batch into compaction.
    engine.flush().unwrap();
    thread::sleep(Duration::from_millis(100));

    // Second batch remains in memtable.
    for i in 20..25u32 {
        let key = format!("key_{i:03}");
        let val = format!("memtable_val_{i}");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    // Tombstone one compacted key in memtable; memtable tombstone must win.
    engine.delete(b"key_005").unwrap();

    let rows = engine.scan(b"key_000", b"key_999");
    let keys: Vec<Vec<u8>> = rows.iter().map(|(k, _)| k.clone()).collect();

    assert_eq!(rows.len(), 24);
    assert!(keys.contains(&b"key_000".to_vec()));
    assert!(keys.contains(&b"key_024".to_vec()));
    assert!(!keys.contains(&b"key_005".to_vec()));

    cleanup(&dir);
}
