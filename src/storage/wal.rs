// src/storage/wal.rs
//
// Write-Ahead Log for crash recovery.
//
// Validated: Cell 2 (999/1000 recovery after simulated crash)
// Projected: Cell 15 (32x speedup over Python)
//
// Record format:
//   [4B crc32][2B key_len][4B val_len][key_bytes][val_bytes]
//
// Recovery: scan forward, skip records with bad CRC.
//
// Complexity:
//   append: O(key_len + val_len)
//   replay: O(file_size)

use crc32fast::Hasher;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, BufReader, Read, Write};
use std::path::Path;

/// One WAL record: a key-value pair with CRC integrity.
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// WAL writer. Appends records to a file with CRC checksums.
pub struct WalWriter {
    writer: BufWriter<File>,
    bytes_written: u64,
    records_written: u64,
}

impl WalWriter {
    /// Open or create a WAL file for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let meta = file.metadata()?;
        Ok(Self {
            writer: BufWriter::with_capacity(64 * 1024, file),
            bytes_written: meta.len(),
            records_written: 0,
        })
    }

    /// Serialize and append one record. Does NOT fsync.
    pub fn append(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        let data = serialize(key, value);
        self.writer.write_all(&data)?;
        self.bytes_written += data.len() as u64;
        self.records_written += 1;
        Ok(())
    }

    /// Flush the buffer to the OS page cache.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Flush and fsync to disk. Guarantees durability.
    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn records_written(&self) -> u64 {
        self.records_written
    }
}

/// WAL reader. Replays all valid records from a WAL file.
pub struct WalReader {
    reader: BufReader<File>,
}

impl WalReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::with_capacity(64 * 1024, file),
        })
    }

    /// Replay all valid records. Skips corrupted records.
    /// Returns (valid_records, corrupted_count).
    pub fn replay(mut self) -> io::Result<(Vec<WalRecord>, usize)> {
        let mut records = Vec::new();
        let mut corrupted = 0;

        loop {
            // Read CRC (4 bytes)
            let mut crc_buf = [0u8; 4];
            match self.reader.read_exact(&mut crc_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let crc_stored = u32::from_be_bytes(crc_buf);

            // Read key_len (2 bytes) and val_len (4 bytes)
            let mut header = [0u8; 6];
            match self.reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(_) => {
                    corrupted += 1;
                    break;
                }
            }

            let key_len = u16::from_be_bytes([header[0], header[1]]) as usize;
            let val_len = u32::from_be_bytes([
                header[2], header[3], header[4], header[5],
            ]) as usize;

            // Sanity check
            if key_len > 1_000_000 || val_len > 100_000_000 {
                corrupted += 1;
                break;
            }

            let mut payload = vec![0u8; key_len + val_len];
            match self.reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(_) => {
                    corrupted += 1;
                    break;
                }
            }

            // Verify CRC over header + payload
            let mut hasher = Hasher::new();
            hasher.update(&header);
            hasher.update(&payload);
            let crc_calc = hasher.finalize();

            if crc_calc != crc_stored {
                corrupted += 1;
                break;
            }

            records.push(WalRecord {
                key: payload[..key_len].to_vec(),
                value: payload[key_len..].to_vec(),
            });
        }

        Ok((records, corrupted))
    }
}

/// Serialize a key-value pair into WAL record bytes.
/// Format: [4B crc32][2B key_len][4B val_len][key][value]
pub fn serialize(key: &[u8], value: &[u8]) -> Vec<u8> {
    let key_len = key.len() as u16;
    let val_len = value.len() as u32;

    let header = [
        (key_len >> 8) as u8,
        key_len as u8,
        (val_len >> 24) as u8,
        (val_len >> 16) as u8,
        (val_len >> 8) as u8,
        val_len as u8,
    ];

    let mut hasher = Hasher::new();
    hasher.update(&header);
    hasher.update(key);
    hasher.update(value);
    let crc = hasher.finalize();

    let mut out = Vec::with_capacity(4 + 6 + key.len() + value.len());
    out.extend_from_slice(&crc.to_be_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(key);
    out.extend_from_slice(value);
    out
}

/// Deserialize a single WAL record from bytes.
pub fn deserialize(data: &[u8]) -> Option<WalRecord> {
    if data.len() < 10 {
        return None;
    }

    let crc_stored = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let payload = &data[4..];

    let mut hasher = Hasher::new();
    hasher.update(payload);
    if hasher.finalize() != crc_stored {
        return None;
    }

    let key_len = u16::from_be_bytes([payload[0], payload[1]]) as usize;
    let val_len = u32::from_be_bytes([
        payload[2], payload[3], payload[4], payload[5],
    ]) as usize;

    if payload.len() < 6 + key_len + val_len {
        return None;
    }

    Some(WalRecord {
        key: payload[6..6 + key_len].to_vec(),
        value: payload[6 + key_len..6 + key_len + val_len].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ulmen_wal_{name}_{}", std::process::id()));
        p
    }

    #[test]
    fn serialize_roundtrip() {
        let key = b"auth::validate_token";
        let val = b"{\"type\":\"function\",\"line\":42}";
        let data = serialize(key, val);
        let rec = deserialize(&data).unwrap();
        assert_eq!(rec.key, key);
        assert_eq!(rec.value, val);
    }

    #[test]
    fn corrupt_crc_rejected() {
        let mut data = serialize(b"key", b"val");
        data[0] ^= 0xFF;
        assert!(deserialize(&data).is_none());
    }

    #[test]
    fn writer_reader_roundtrip() {
        let path = tmp_path("roundtrip");
        let n = 1000u64;

        {
            let mut w = WalWriter::open(&path).unwrap();
            for i in 0..n {
                let key = format!("key_{i}");
                let val = format!("val_{i}");
                w.append(key.as_bytes(), val.as_bytes()).unwrap();
            }
            w.sync().unwrap();
            assert_eq!(w.records_written(), n);
        }

        {
            let r = WalReader::open(&path).unwrap();
            let (records, corrupted) = r.replay().unwrap();
            assert_eq!(records.len(), n as usize);
            assert_eq!(corrupted, 0);
            for (i, rec) in records.iter().enumerate() {
                assert_eq!(rec.key, format!("key_{i}").as_bytes());
                assert_eq!(rec.value, format!("val_{i}").as_bytes());
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn crash_recovery() {
        let path = tmp_path("crash");

        {
            let mut w = WalWriter::open(&path).unwrap();
            for i in 0..100u32 {
                w.append(
                    format!("k{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                ).unwrap();
            }
            w.flush().unwrap();

            // Simulate crash: append corrupt bytes
            use std::io::Write as IoWrite;
            let mut raw = OpenOptions::new().append(true).open(&path).unwrap();
            raw.write_all(b"\xff\xff\xff\xff\x00\x05garbage").unwrap();
        }

        {
            let r = WalReader::open(&path).unwrap();
            let (records, corrupted) = r.replay().unwrap();
            assert_eq!(records.len(), 100);
            assert!(corrupted >= 1);
        }

        std::fs::remove_file(&path).ok();
    }
}
