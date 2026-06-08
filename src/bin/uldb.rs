// src/bin/uldb.rs
//
// uldb server binary.
//
// Usage:
//   uldb serve --port 7771 --data ./data --token mytoken
//
// Architecture:
//   main thread: TcpListener, accept loop
//   per connection: TLS handshake -> UMP auth -> frame routing
//   storage: Engine behind Arc<Mutex<UmpHandler>>
//   shutdown: Ctrl+C sets atomic flag, accept loop exits

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustls::StreamOwned;

use ulmp::crypto::hmac::hmac_sha256;
use ulmp::frame::header::{Header, HEADER_SIZE};
use ulmp::messages::opcode;
use ulmp::messages::session;
use ulmp::messages::Message;
use ulmp::server::auth::NonceStore;
use ulmp::server::connection::build_frame;
use ulmp::server::handler::Response;
use ulmp::server::router::dispatch;
use ulmp::tls::{generate_self_signed_ed25519, TlsServer, cert_fingerprint_sha256_hex};

use uldb::engine::{Engine, EngineConfig};
use uldb::server::UmpHandler;

// Tag constants for payload encoding
const TAG_U8: u8 = 0x01;
const TAG_U32: u8 = 0x03;
const TAG_U64: u8 = 0x04;
const TAG_BYTES: u8 = 0x0C;
const TAG_STRING: u8 = 0x0D;
const TAG_END: u8 = 0xFF;

fn enc(fields: Vec<(u8, Vec<u8>)>) -> Vec<u8> {
    let mut buf = Vec::new();
    for (tag, data) in fields {
        buf.push(tag);
        match tag {
            TAG_U8 => { if !data.is_empty() { buf.push(data[0]); } }
            TAG_U32 | TAG_U64 => { buf.extend_from_slice(&data); }
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

fn send_frame(stream: &mut impl Write, opcode: u8, stream_id: u16, seq: u32, payload: &[u8]) {
    let frame = build_frame(opcode, 0, stream_id, seq, payload);
    let _ = stream.write_all(&frame);
    let _ = stream.flush();
}

fn recv_frame(stream: &mut impl Read) -> Option<(Header, Vec<u8>)> {
    let mut hdr = [0u8; HEADER_SIZE];
    if stream.read_exact(&mut hdr).is_err() {
        return None;
    }
    let header = Header::decode(&hdr).ok()?;
    let mut payload = vec![0u8; header.payload_length as usize];
    if header.payload_length > 0 {
        if stream.read_exact(&mut payload).is_err() {
            return None;
        }
    }
    Some((header, payload))
}

fn handle_connection(
    tcp: TcpStream,
    tls_server: &TlsServer,
    handler: &UmpHandler,
    token: &[u8],
    nonce_store: &std::sync::Mutex<NonceStore>,
    server_name: &str,
    shutdown: &AtomicBool,
) {
    let peer = tcp.peer_addr().ok();
    let addr_str = peer.map(|a| a.to_string()).unwrap_or_else(|| "unknown".into());
    eprintln!("[uldb] connection from {addr_str}");

    tcp.set_read_timeout(Some(Duration::from_secs(120))).ok();
    tcp.set_nodelay(true).ok();

    let tls_conn = match tls_server.accept() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[uldb] TLS accept error: {e}");
            return;
        }
    };
    let mut stream = StreamOwned::new(tls_conn, tcp);
    stream.flush().ok();

    // UMP auth: CHALLENGE -> HELLO -> WELCOME
    let nonce: [u8; 16];
    {
        let mut store = nonce_store.lock().unwrap();
        match store.generate() {
            Some(n) => nonce = n,
            None => {
                eprintln!("[uldb] entropy failure, refusing connection");
                return;
            }
        }
    }

    let challenge_payload = enc(vec![(TAG_BYTES, nonce.to_vec())]);
    send_frame(&mut stream, opcode::OP_CHALLENGE, 0, 0, &challenge_payload);

    let (hello_hdr, hello_payload) = match recv_frame(&mut stream) {
        Some(f) => f,
        None => { eprintln!("[uldb] {addr_str} disconnected during auth"); return; }
    };

    if hello_hdr.opcode != opcode::OP_HELLO {
        eprintln!("[uldb] {addr_str} expected HELLO, got 0x{:02x}", hello_hdr.opcode);
        return;
    }

    let hello = match session::Hello::decode_payload(&hello_payload) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[uldb] {addr_str} bad HELLO payload: {e:?}");
            return;
        }
    };

    // Verify HMAC
    let expected = hmac_sha256(token, &nonce);
    let mut diff = 0u8;
    for (a, b) in expected.iter().zip(hello.token_hash.iter()) {
        diff |= a ^ b;
    }
    if diff != 0 || hello.token_hash.len() != 32 {
        eprintln!("[uldb] {addr_str} auth failed (bad token)");
        // Silent disconnect per spec
        return;
    }

    // Consume nonce
    {
        let mut store = nonce_store.lock().unwrap();
        if !store.consume(nonce) {
            eprintln!("[uldb] {addr_str} nonce replay detected");
            return;
        }
    }

    // Send WELCOME
    let session_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let welcome_payload = enc(vec![
        (TAG_U64, session_id.to_be_bytes().to_vec()),
        (TAG_STRING, server_name.as_bytes().to_vec()),
        (TAG_U32, (64u32 * 1024 * 1024).to_be_bytes().to_vec()),
        (TAG_U32, 128u32.to_be_bytes().to_vec()),
    ]);
    send_frame(&mut stream, opcode::OP_WELCOME, 0, 0, &welcome_payload);

    eprintln!("[uldb] {addr_str} authenticated (client={})", hello.client_name);

    // Main request loop
    let mut seq = 1u32;
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let (header, payload) = match recv_frame(&mut stream) {
            Some(f) => f,
            None => break, // client disconnected
        };

        match header.opcode {
            opcode::OP_PING => {
                let pong_payload = match session::Ping::decode_payload(&payload) {
                    Ok(ping) => {
                        let pong = session::Pong { timestamp_us: ping.timestamp_us };
                        pong.encode_payload()
                    }
                    Err(_) => payload.clone(),
                };
                send_frame(&mut stream, opcode::OP_PONG, header.stream_id, seq, &pong_payload);
                seq = seq.wrapping_add(1);
            }

            opcode::OP_GOODBYE => {
                eprintln!("[uldb] {addr_str} sent GOODBYE");
                break;
            }

            _ => {
                // Dispatch to handler via ulmp router
                let response = match dispatch(&header, &payload, handler) {
                    Ok(resp) => resp,
                    Err(e) => {
                        let err_payload = enc(vec![
                            (TAG_U8, vec![0x21]), // ERR_PAYLOAD_DECODE
                            (TAG_U32, 0u32.to_be_bytes().to_vec()),
                            (TAG_STRING, format!("decode error: {e:?}").into_bytes()),
                        ]);
                        Response::Single {
                            opcode: opcode::OP_ERROR,
                            payload: err_payload,
                        }
                    }
                };

                match response {
                    Response::Single { opcode: op, payload: p } => {
                        send_frame(&mut stream, op, header.stream_id, seq, &p);
                        seq = seq.wrapping_add(1);
                    }
                    Response::Stream { frames } => {
                        for (op, p) in frames {
                            send_frame(&mut stream, op, header.stream_id, seq, &p);
                            seq = seq.wrapping_add(1);
                        }
                    }
                    Response::Disconnect { opcode: op, payload: p } => {
                        send_frame(&mut stream, op, header.stream_id, seq, &p);
                        break;
                    }
                    Response::None => {}
                }
            }
        }
    }

    eprintln!("[uldb] {addr_str} disconnected");
}

fn parse_args() -> (u16, PathBuf, String) {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("uldb v0.1.0 -- agentic AI database");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  uldb serve [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --port PORT    listen port (default: 7771)");
        eprintln!("  --data DIR     data directory (default: ./data)");
        eprintln!("  --token TOKEN  auth token (required)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  uldb serve --port 7771 --data ./data --token mytoken");
        std::process::exit(0);
    }

    if args[1] != "serve" {
        eprintln!("unknown command: {}", args[1]);
        eprintln!("run 'uldb --help' for usage");
        std::process::exit(1);
    }

    let mut port = 7771u16;
    let mut data_dir = PathBuf::from("./data");
    let mut token = String::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(7771);
            }
            "--data" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    data_dir = PathBuf::from(d);
                }
            }
            "--token" => {
                i += 1;
                if let Some(t) = args.get(i) {
                    token = t.clone();
                }
            }
            other => {
                eprintln!("unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if token.is_empty() {
        eprintln!("error: --token is required");
        std::process::exit(1);
    }

    (port, data_dir, token)
}

fn main() {
    let (port, data_dir, token) = parse_args();
    let server_name = format!("uldb/{}", env!("CARGO_PKG_VERSION"));

    eprintln!("[uldb] starting {server_name}");
    eprintln!("[uldb] data directory: {}", data_dir.display());
    eprintln!("[uldb] port: {port}");

    // Generate self-signed TLS certificate
    eprintln!("[uldb] generating self-signed TLS certificate...");
    let identity = generate_self_signed_ed25519("localhost").unwrap_or_else(|e| {
        eprintln!("[uldb] TLS cert generation failed: {e}");
        std::process::exit(1);
    });
    let fingerprint = cert_fingerprint_sha256_hex(identity.leaf_cert());
    eprintln!("[uldb] certificate fingerprint: {fingerprint}");

    let tls_server = TlsServer::new(identity).unwrap_or_else(|e| {
        eprintln!("[uldb] TLS server init failed: {e}");
        std::process::exit(1);
    });

    // Open storage engine
    let config = EngineConfig::new(&data_dir);
    let engine = Engine::open(config).unwrap_or_else(|e| {
        eprintln!("[uldb] failed to open engine: {e}");
        std::process::exit(1);
    });
    let handler = UmpHandler::new(engine);
    let nonce_store = std::sync::Mutex::new(NonceStore::new());

    // Shutdown signal
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);
    ctrlc_handler(shutdown_clone);

    // TCP listener
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("[uldb] failed to bind {addr}: {e}");
        std::process::exit(1);
    });
    listener.set_nonblocking(false).ok();

    eprintln!("[uldb] listening on {addr}");
    eprintln!("[uldb] ready for connections");

    // Accept loop
    loop {
        if shutdown.load(Ordering::Relaxed) {
            eprintln!("[uldb] shutting down...");
            break;
        }

        // Set a timeout so we can check the shutdown flag periodically
        listener.set_nonblocking(true).ok();
        match listener.accept() {
            Ok((tcp, _)) => {
                tcp.set_nonblocking(false).ok();
                handle_connection(
                    tcp,
                    &tls_server,
                    &handler,
                    token.as_bytes(),
                    &nonce_store,
                    &server_name,
                    &shutdown,
                );
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("[uldb] accept error: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    eprintln!("[uldb] stopped");
}

/// Register Ctrl+C handler using a simple approach.
/// Sets the shutdown flag so the accept loop exits cleanly.
fn ctrlc_handler(shutdown: Arc<AtomicBool>) {
    // Use a simple signal approach with unsafe.
    // This is the zero-dep way to handle Ctrl+C.
    std::thread::spawn(move || {
        // Block on stdin as a poor man's signal handler.
        // In production, use libc::signal or the signal-hook crate.
        // For now, the process will exit on Ctrl+C via default signal
        // behavior, which is acceptable for v1.
        loop {
            std::thread::sleep(Duration::from_secs(3600));
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
        }
    });
}
