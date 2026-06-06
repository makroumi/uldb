#!/usr/bin/env zsh
# UlmenDB build and test script
# Usage: ./scripts/build.sh

set -euo pipefail

echo "============================================"
echo "UlmenDB Build Pipeline"
echo "============================================"

cd "$(dirname "$0")/.."

# Step 1: Rust tests
echo ""
echo "-- Rust unit tests --"
cargo test --lib 2>&1

# Step 2: Build Python extension
echo ""
echo "-- Building Python extension --"
if command -v maturin &> /dev/null; then
    maturin develop --release 2>&1
    echo "Rust extension built."
else
    echo "maturin not found. Using Python fallback."
fi

# Step 3: Python integration tests
echo ""
echo "-- Python integration tests --"
PYTHONPATH="python:$PYTHONPATH" python3 tests/python/test_all.py

echo ""
echo "============================================"
echo "BUILD COMPLETE"
echo "============================================"
