// src/storage/bg_compaction.rs
//
// Background compaction worker.
//
// Moves page compaction off the write path.
// The Engine sends flushed memtable pages to this worker via channel.
//
// Architecture:
//   main thread: WAL + memtable
//   background thread: CompactionManager::add_l0_page(page)
//
// Complexity:
//   submit(page): O(1) channel send
//   worker loop: O(V + E) per compaction, same as synchronous version
//
// Thread safety:
//   CompactionManager wrapped in Arc<Mutex<...>>
//   Worker owns the receiver end of the channel

use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use crate::storage::page::Page;
use crate::storage::compaction::CompactionManager;

/// Messages sent to the compaction worker.
enum Command {
    AddPage(Page),
    Shutdown,
}

/// Background compaction service.
pub struct BackgroundCompactor {
    tx: mpsc::Sender<Command>,
    handle: Option<JoinHandle<()>>,
    compaction: Arc<Mutex<CompactionManager>>,
}

impl BackgroundCompactor {
    /// Start the background compaction thread.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Command>();
        let compaction = Arc::new(Mutex::new(CompactionManager::new()));
        let compaction_thread = Arc::clone(&compaction);

        let handle = thread::spawn(move || {
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Command::AddPage(page) => {
                        let mut c = compaction_thread.lock().unwrap();
                        c.add_l0_page(page);
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
        let bg = BackgroundCompactor::new();
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
        let bg = BackgroundCompactor::new();
        bg.shutdown();
    }

    #[test]
    fn multiple_pages() {
        let bg = BackgroundCompactor::new();
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
}
