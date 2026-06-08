// src/storage/bg_compaction.rs
//
// Background compaction worker with disk persistence.
//
// Moves page compaction off the write path.
// The Engine sends flushed memtable pages to this worker via channel.
// After each compaction mutation, pages are persisted to disk.
//
// Architecture:
//   main thread: WAL + memtable
//   background thread: CompactionManager::add_l0_page(page) + PageStore::save_all()
//
// Complexity:
//   submit(page): O(1) channel send
//   worker loop: O(V + E) per compaction + O(total_records) disk write
//
// Thread safety:
//   CompactionManager wrapped in Arc<Mutex<...>>
//   Worker owns the receiver end of the channel

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use crate::storage::page::Page;
use crate::storage::compaction::CompactionManager;
use crate::storage::page_store::PageStore;

/// Messages sent to the compaction worker.
enum Command {
    AddPage(Page),
    Shutdown,
}

/// Background compaction service with disk persistence.
pub struct BackgroundCompactor {
    tx: mpsc::Sender<Command>,
    handle: Option<JoinHandle<()>>,
    compaction: Arc<Mutex<CompactionManager>>,
}

impl BackgroundCompactor {
    /// Start the background compaction thread.
    ///
    /// If data_dir is Some, pages are persisted to disk after each
    /// compaction mutation. If None, pages are in-memory only (tests).
    pub fn new(data_dir: Option<PathBuf>) -> Self {
        let (tx, rx) = mpsc::channel::<Command>();
        let compaction = Arc::new(Mutex::new(CompactionManager::new()));
        let compaction_thread = Arc::clone(&compaction);

        let handle = thread::spawn(move || {
            let page_store = data_dir.as_ref().and_then(|dir| {
                PageStore::new(dir).ok()
            });

            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Command::AddPage(page) => {
                        let mut c = compaction_thread.lock().unwrap();
                        c.add_l0_page(page);

                        // Persist current page state to disk.
                        if let Some(ref store) = page_store {
                            if let Err(e) = store.save_all(c.all_pages()) {
                                eprintln!("[uldb] page persistence failed: {e}");
                            }
                        }
                    }
                    Command::Shutdown => {
                        break;
                    }
                }
            }
        });

        Self {
            tx,
            handle: Some(handle),
            compaction,
        }
    }

    /// Submit a page for asynchronous compaction.
    pub fn submit(&self, page: Page) -> Result<(), String> {
        self.tx
            .send(Command::AddPage(page))
            .map_err(|e| format!("failed to submit page to compactor: {e}"))
    }

    /// Access the shared compaction state for reads.
    pub fn state(&self) -> Arc<Mutex<CompactionManager>> {
        Arc::clone(&self.compaction)
    }

    /// Shut down the compaction worker and wait for thread exit.
    pub fn shutdown(mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::page::Page;
    use std::time::Duration;

    fn make_page(key: &str, val: &str) -> Page {
        let mut p = Page::new(0);
        p.push(key.as_bytes().to_vec(), val.as_bytes().to_vec(), false);
        p.sort();
        p
    }

    #[test]
    fn submit_page_and_compact() {
        let bg = BackgroundCompactor::new(None);
        let state = bg.state();

        bg.submit(make_page("k1", "v1")).unwrap();
        bg.submit(make_page("k2", "v2")).unwrap();

        // Give the worker thread a moment to process.
        std::thread::sleep(Duration::from_millis(50));

        let c = state.lock().unwrap();
        assert!(c.total_records() >= 2);

        drop(c);
        bg.shutdown();
    }

    #[test]
    fn shutdown_cleanly() {
        let bg = BackgroundCompactor::new(None);
        bg.shutdown();
    }

    #[test]
    fn multiple_pages() {
        let bg = BackgroundCompactor::new(None);
        let state = bg.state();

        for i in 0..10u32 {
            bg.submit(make_page(&format!("k{i}"), &format!("v{i}"))).unwrap();
        }

        std::thread::sleep(Duration::from_millis(100));

        let c = state.lock().unwrap();
        assert!(c.total_records() >= 10);

        drop(c);
        bg.shutdown();
    }

    #[test]
    fn pages_persisted_to_disk() {
        let dir = std::env::temp_dir().join(format!(
            "uldb_bg_persist_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let bg = BackgroundCompactor::new(Some(dir.clone()));

        for i in 0..5u32 {
            bg.submit(make_page(&format!("pk{i}"), &format!("pv{i}"))).unwrap();
        }

        std::thread::sleep(Duration::from_millis(150));

        // Verify page files exist on disk.
        let store = PageStore::new(&dir).unwrap();
        assert!(store.file_count() > 0, "page files should exist on disk");

        let loaded = store.load_all().unwrap();
        let total: usize = loaded.iter().map(|l| l.len()).sum();
        assert!(total > 0, "should load at least one page from disk");

        bg.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
