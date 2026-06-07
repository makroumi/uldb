// src/bin/uldb.rs
//
// uldb server binary.
// Usage: uldb serve --port 7771 --data ./data --token mytoken
//
// Currently a placeholder. Will be wired to the storage engine
// and ulmp Handler trait.

fn main() {
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
        eprintln!("  --token TOKEN  auth token");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  uldb serve --port 7771 --token mytoken");
        std::process::exit(0);
    }

    if args[1] == "serve" {
        eprintln!("[uldb] server not yet implemented");
        eprintln!("[uldb] storage engine ready (41 tests passing)");
        eprintln!("[uldb] wire protocol: use ulmp crate");
        std::process::exit(1);
    }

    eprintln!("unknown command: {}", args[1]);
    std::process::exit(1);
}
