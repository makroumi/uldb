// src/storage/page.rs
//
// Compressed sorted page store.
//
// Validated: Cell 4
//   Compression: 15.6% of JSON size
//   Write: 55,097 records/sec (Python) -- Rust target: 1M+
//   Read:  184,789 records/sec (Python) -- Rust target: 5M+
//
// Page format (binary):
//   [4B magic "ULPG"]
//   [1B version]
//   [4B record_count]
//   [4B uncompressed_size]
//   [4B compressed_size]
//   [body...]
//
// Record format (inside body):
//   [4B magic "ULRC"]
//   [2B key_len]
//   [4B val_len]
//   [key bytes]
//   [val bytes]
//   [1B tombstone flag]

use std::io;

use crate::index::bloom::BloomFilter;

const PAGE_MAGIC: &[u8; 4] = b"ULPG";
const PAGE_VERSION: u8 = 1;
const RECORD_MAGIC: &[u8; 4] = b"ULRC";

/// A single record within a page.
#[derive(Debug, Clone)]
pub struct PageRecord {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub tombstone: bool,
}

/// A sorted page of key-value records.
#[derive(Debug)]
pub struct Page {
    pub records: Vec<PageRecord>,
    pub level: u32,
    bloom: Option<BloomFilter>,
}

impl Page {
    pub fn new(level: u32) -> Self {
        Self {
            records: Vec::new(),
            level,
            bloom: None,
        }
    }

    pub fn push(&mut self, key: Vec<u8>, value: Vec<u8>, tombstone: bool) {
        self.records.push(PageRecord { key, value, tombstone });
    }

    pub fn sort(&mut self) {
        self.records.sort_by(|a, b| a.key.cmp(&b.key));
    }

    /// Build a bloom filter from the current records.
    /// Call after sort() and before using get() for optimal performance.
    pub fn build_bloom(&mut self) {
        if self.records.is_empty() {
            return;
        }
        let mut bf = BloomFilter::new(self.records.len().max(16), 0.01);
        for rec in &self.records {
            if !rec.tombstone {
                bf.add(&rec.key);
            }
        }
        self.bloom = Some(bf);
    }

    /// Check bloom filter for a key. Returns true if key might exist,
    /// false if key definitely does not exist in this page.
    pub fn bloom_may_contain(&self, key: &[u8]) -> bool {
        match &self.bloom {
            Some(bf) => bf.may_contain(key),
            None => true, // no bloom filter, assume key might exist
        }
    }

    /// Serialize to binary format.
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut body = Vec::new();
        for rec in &self.records {
            body.extend_from_slice(RECORD_MAGIC);
            body.extend_from_slice(&(rec.key.len() as u16).to_be_bytes());
            body.extend_from_slice(&(rec.value.len() as u32).to_be_bytes());
            body.extend_from_slice(&rec.key);
            body.extend_from_slice(&rec.value);
            body.push(if rec.tombstone { 1 } else { 0 });
        }

        // Store uncompressed. compressed_size == uncompressed_size signals
        // identity encoding. zlib will be added as an optional feature.
        let body_len = body.len() as u32;

        let mut out = Vec::with_capacity(17 + body.len());
        out.extend_from_slice(PAGE_MAGIC);
        out.push(PAGE_VERSION);
        out.extend_from_slice(&(self.records.len() as u32).to_be_bytes());
        out.extend_from_slice(&body_len.to_be_bytes()); // uncompressed_size
        out.extend_from_slice(&body_len.to_be_bytes()); // compressed_size
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 17 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "page too short"));
        }

        if &data[0..4] != PAGE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad page magic"));
        }

        let _version = data[4];
        let record_count =
            u32::from_be_bytes([data[5], data[6], data[7], data[8]]) as usize;
        let _uncompressed =
            u32::from_be_bytes([data[9], data[10], data[11], data[12]]);
        let compressed_size =
            u32::from_be_bytes([data[13], data[14], data[15], data[16]]) as usize;

        if data.len() < 17 + compressed_size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "page truncated"));
        }

        let body = &data[17..17 + compressed_size];
        let mut page = Page::new(0);
        // bloom filter will be built after loading if needed
        let mut pos = 0;

        while pos < body.len() {
            if pos + 4 > body.len() { break; }
            if &body[pos..pos + 4] != RECORD_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bad record magic",
                ));
            }
            pos += 4;

            if pos + 6 > body.len() { break; }
            let key_len =
                u16::from_be_bytes([body[pos], body[pos + 1]]) as usize;
            let val_len = u32::from_be_bytes([
                body[pos + 2], body[pos + 3],
                body[pos + 4], body[pos + 5],
            ]) as usize;
            pos += 6;

            if pos + key_len + val_len + 1 > body.len() { break; }

            let key   = body[pos..pos + key_len].to_vec(); pos += key_len;
            let value = body[pos..pos + val_len].to_vec(); pos += val_len;
            let tombstone = body[pos] != 0;                pos += 1;

            page.records.push(PageRecord { key, value, tombstone });
        }

        assert_eq!(page.records.len(), record_count);
        Ok(page)
    }

    /// Binary search for a key. Records must be sorted.
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.records
            .binary_search_by(|r| r.key.as_slice().cmp(key))
            .ok()
            .and_then(|idx| {
                let rec = &self.records[idx];
                if rec.tombstone { None } else { Some(rec.value.as_slice()) }
            })
    }

    /// Range scan within this page over the half-open interval [start, end).
    ///
    /// Records are sorted by key, so we binary-search to the first candidate
    /// and then iterate until we reach `end`.
    pub fn range<'a>(
        &'a self,
        start: &'a [u8],
        end: &'a [u8],
    ) -> impl Iterator<Item = &'a PageRecord> + 'a {
        let first = self.records
            .binary_search_by(|r| r.key.as_slice().cmp(start))
            .unwrap_or_else(|idx| idx);

        self.records[first..]
            .iter()
            .take_while(move |rec| rec.key.as_slice() < end)
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    pub fn tombstone_count(&self) -> usize {
        self.records.iter().filter(|r| r.tombstone).count()
    }
    pub fn live_count(&self) -> usize {
        self.records.iter().filter(|r| !r.tombstone).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_roundtrip() {
        let mut page = Page::new(0);
        for i in 0..100u32 {
            page.push(
                format!("key_{i:04}").into_bytes(),
                format!("val_{i}").into_bytes(),
                false,
            );
        }
        page.sort();

        let data = page.serialize().unwrap();
        let page2 = Page::deserialize(&data).unwrap();

        assert_eq!(page2.len(), 100);
        for i in 0..100u32 {
            let key = format!("key_{i:04}");
            let val = format!("val_{i}");
            assert_eq!(page2.get(key.as_bytes()), Some(val.as_bytes()));
        }
    }

    #[test]
    fn tombstone_hidden() {
        let mut page = Page::new(0);
        page.push(b"alive".to_vec(), b"yes".to_vec(), false);
        page.push(b"dead".to_vec(), b"".to_vec(), true);
        page.sort();
        assert_eq!(page.get(b"alive"), Some(b"yes".as_ref()));
        assert_eq!(page.get(b"dead"), None);
        assert_eq!(page.tombstone_count(), 1);
        assert_eq!(page.live_count(), 1);
    }

    #[test]
    fn binary_search_works() {
        let mut page = Page::new(0);
        for i in 0..1000u32 {
            page.push(
                format!("k{i:06}").into_bytes(),
                format!("v{i}").into_bytes(),
                false,
            );
        }
        page.sort();
        assert_eq!(page.get(b"k000500"), Some(b"v500".as_ref()));
        assert_eq!(page.get(b"k999999"), None);
    }
}
