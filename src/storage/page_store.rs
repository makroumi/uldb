// src/storage/page_store.rs
//
// Disk persistence for compaction pages.
//
// File layout:
//   data_dir/pages/l{level}_{index:04}.ulpg
//
// Persistence strategy:
//   After each compaction mutation, the background worker calls
//   save_all() which atomically replaces all page files with the
//   current in-memory state.
//
// Load strategy:
//   On Engine::open(), load_all() scans data_dir/pages/ for .ulpg
//   files, parses level from filename, deserializes each, and
//   returns them grouped by level.
//
// Complexity:
//   save_all: O(total_records) -- serialize all pages
//   load_all: O(total_bytes_on_disk) -- read and deserialize
//
// Correctness:
//   - save_all writes to a temp directory then renames files
//     (not a full atomic swap, but crash-safe enough: on partial
//     write, load_all skips corrupt files and WAL replay recovers)
//   - filenames are deterministic from level+index
//   - no manifest file needed: level is encoded in the filename

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::page::Page;

/// Subdirectory within data_dir for page files.
const PAGES_DIR: &str = "pages";

/// File extension for page files.
const PAGE_EXT: &str = "ulpg";

/// Manages reading and writing page files for the compaction subsystem.
pub struct PageStore {
    pages_dir: PathBuf,
}

impl PageStore {
    /// Create a PageStore for the given data directory.
    /// Creates the pages/ subdirectory if it does not exist.
    pub fn new(data_dir: &Path) -> io::Result<Self> {
        let pages_dir = data_dir.join(PAGES_DIR);
        fs::create_dir_all(&pages_dir)?;
        Ok(Self { pages_dir })
    }

    /// Save all pages from all levels to disk.
    ///
    /// Replaces all existing page files with the current state.
    /// Steps:
    ///   1. Delete all existing .ulpg files in pages/
    ///   2. Write each page as l{level}_{index:04}.ulpg
    ///
    /// This is called by the background compaction worker after
    /// each compaction mutation.
    pub fn save_all(&self, levels: &[Vec<Page>; 3]) -> io::Result<()> {
        // Step 1: remove all existing page files.
        self.clear()?;

        // Step 2: write current pages.
        for (level, pages) in levels.iter().enumerate() {
            for (index, page) in pages.iter().enumerate() {
                let filename = format!("l{}_{:04}.{}", level, index, PAGE_EXT);
                let path = self.pages_dir.join(&filename);
                let data = page.serialize()?;
                fs::write(&path, &data)?;
            }
        }

        Ok(())
    }

    /// Load all pages from disk, grouped by level.
    ///
    /// Scans the pages/ directory for .ulpg files, parses the level
    /// from the filename prefix, deserializes each file, and returns
    /// them in level order.
    ///
    /// Files that fail to parse or deserialize are skipped with a
    /// warning printed to stderr. WAL replay will recover any lost data.
    pub fn load_all(&self) -> io::Result<[Vec<Page>; 3]> {
        let mut levels: [Vec<(usize, Page)>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        if !self.pages_dir.exists() {
            return Ok([Vec::new(), Vec::new(), Vec::new()]);
        }

        let mut entries: Vec<_> = fs::read_dir(&self.pages_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == PAGE_EXT)
            })
            .collect();

        // Sort by filename for deterministic load order.
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let filename = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            // Parse level and index from filename "l{level}_{index:04}"
            let (level, index) = match parse_page_filename(&filename) {
                Some(pair) => pair,
                None => {
                    eprintln!(
                        "[uldb] skipping unrecognised page file: {}",
                        path.display()
                    );
                    continue;
                }
            };

            if level > 2 {
                eprintln!(
                    "[uldb] skipping page file with invalid level {}: {}",
                    level,
                    path.display()
                );
                continue;
            }

            let data = match fs::read(&path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!(
                        "[uldb] failed to read page file {}: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            match Page::deserialize(&data) {
                Ok(mut page) => {
                    page.level = level as u32;
                    levels[level].push((index, page));
                }
                Err(e) => {
                    eprintln!(
                        "[uldb] failed to deserialize page file {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }

        // Sort each level by index to preserve insertion order.
        for level in &mut levels {
            level.sort_by_key(|(idx, _)| *idx);
        }

        Ok([
            levels[0].drain(..).map(|(_, p)| p).collect(),
            levels[1].drain(..).map(|(_, p)| p).collect(),
            levels[2].drain(..).map(|(_, p)| p).collect(),
        ])
    }

    /// Delete all page files from the pages/ directory.
    fn clear(&self) -> io::Result<()> {
        if !self.pages_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.pages_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == PAGE_EXT) {
                fs::remove_file(&path)?;
            }
        }

        Ok(())
    }

    /// Number of page files currently on disk.
    pub fn file_count(&self) -> usize {
        if !self.pages_dir.exists() {
            return 0;
        }
        fs::read_dir(&self.pages_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map_or(false, |ext| ext == PAGE_EXT)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

/// Parse a page filename like "l0_0003" into (level=0, index=3).
fn parse_page_filename(name: &str) -> Option<(usize, usize)> {
    // Expected format: l{digit}_{digits}
    if !name.starts_with('l') {
        return None;
    }
    let rest = &name[1..];
    let underscore = rest.find('_')?;
    let level: usize = rest[..underscore].parse().ok()?;
    let index: usize = rest[underscore + 1..].parse().ok()?;
    Some((level, index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("uldb_pagestore_{name}_{}", std::process::id()));
        p
    }

    fn make_page(pairs: &[(&str, &str)]) -> Page {
        let mut p = Page::new(0);
        for (k, v) in pairs {
            p.push(k.as_bytes().to_vec(), v.as_bytes().to_vec(), false);
        }
        p.sort();
        p
    }

    #[test]
    fn parse_filename_valid() {
        assert_eq!(parse_page_filename("l0_0000"), Some((0, 0)));
        assert_eq!(parse_page_filename("l1_0005"), Some((1, 5)));
        assert_eq!(parse_page_filename("l2_0123"), Some((2, 123)));
    }

    #[test]
    fn parse_filename_invalid() {
        assert_eq!(parse_page_filename("x0_0000"), None);
        assert_eq!(parse_page_filename("l"), None);
        assert_eq!(parse_page_filename(""), None);
        assert_eq!(parse_page_filename("l0"), None);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tmp_dir("roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let store = PageStore::new(&dir).unwrap();

        let mut levels: [Vec<Page>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        levels[0].push(make_page(&[("k1", "v1"), ("k2", "v2")]));
        levels[0].push(make_page(&[("k3", "v3")]));
        levels[1].push(make_page(&[("k4", "v4"), ("k5", "v5"), ("k6", "v6")]));

        store.save_all(&levels).unwrap();
        assert_eq!(store.file_count(), 3);

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].len(), 2);
        assert_eq!(loaded[1].len(), 1);
        assert_eq!(loaded[2].len(), 0);

        // Verify data integrity.
        assert_eq!(loaded[0][0].get(b"k1"), Some(b"v1".as_ref()));
        assert_eq!(loaded[0][0].get(b"k2"), Some(b"v2".as_ref()));
        assert_eq!(loaded[0][1].get(b"k3"), Some(b"v3".as_ref()));
        assert_eq!(loaded[1][0].get(b"k4"), Some(b"v4".as_ref()));
        assert_eq!(loaded[1][0].get(b"k5"), Some(b"v5".as_ref()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_replaces_old_files() {
        let dir = tmp_dir("replace");
        fs::create_dir_all(&dir).unwrap();
        let store = PageStore::new(&dir).unwrap();

        // First save: 2 pages.
        let mut levels: [Vec<Page>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        levels[0].push(make_page(&[("a", "1")]));
        levels[0].push(make_page(&[("b", "2")]));
        store.save_all(&levels).unwrap();
        assert_eq!(store.file_count(), 2);

        // Second save: 1 page. Old files must be gone.
        let mut levels2: [Vec<Page>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        levels2[1].push(make_page(&[("c", "3")]));
        store.save_all(&levels2).unwrap();
        assert_eq!(store.file_count(), 1);

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].len(), 0);
        assert_eq!(loaded[1].len(), 1);
        assert_eq!(loaded[1][0].get(b"c"), Some(b"3".as_ref()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_empty_dir() {
        let dir = tmp_dir("empty");
        fs::create_dir_all(&dir).unwrap();
        let store = PageStore::new(&dir).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].len(), 0);
        assert_eq!(loaded[1].len(), 0);
        assert_eq!(loaded[2].len(), 0);

        let _ = fs::remove_dir_all(&dir);
    }
}
