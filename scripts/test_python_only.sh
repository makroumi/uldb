#!/usr/bin/env zsh
# Test UlmenDB using only the Python fallback (no Rust toolchain needed)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Testing UlmenDB (Python fallback only)"
echo ""
PYTHONPATH="python" python3 tests/python/test_all.py
