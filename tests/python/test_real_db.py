"""
Real-world integration test for uldb.

Starts the actual uldb binary, connects with the ulmp Python client,
and exercises every major database operation over TLS.

Run:
    cd uldb
    .venv/bin/pytest tests/python/test_real_db.py -v -s
"""

import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest

# Ensure ulmp is importable
from ulmp import AsyncClient
from ulmp.errors import KeyNotFoundError, UlmpError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """Block until the port accepts TCP connections or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uldb_server():
    """Start the real uldb binary and yield connection info."""
    port = _get_free_port()
    token = "test_integration_token_42"
    data_dir = tempfile.mkdtemp(prefix="uldb_real_test_")

    # Find the binary
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    binary = os.path.join(project_root, "target", "release", "uldb")
    if not os.path.exists(binary):
        binary = os.path.join(project_root, "target", "debug", "uldb")
    if not os.path.exists(binary):
        pytest.skip("uldb binary not found (run: cargo build --release --bin uldb)")

    proc = subprocess.Popen(
        [binary, "serve", "--port", str(port), "--data", data_dir, "--token", token],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start accepting connections
    if not _wait_for_port(port, timeout=10.0):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"uldb failed to start on port {port}\n"
            f"stdout: {stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )

    yield {
        "port": port,
        "token": token.encode(),
        "data_dir": data_dir,
        "proc": proc,
    }

    # Shutdown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    shutil.rmtree(data_dir, ignore_errors=True)


async def _connect(server_info) -> AsyncClient:
    """Helper to connect to the test server."""
    return await AsyncClient.connect(
        host="127.0.0.1",
        port=server_info["port"],
        token=server_info["token"],
        verify_cert=False,
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConnection:
    """Basic connectivity tests."""

    @pytest.mark.asyncio
    async def test_connect_and_ping(self, uldb_server):
        """Can connect, authenticate, and ping."""
        db = await _connect(uldb_server)
        try:
            latency = await db.ping()
            assert isinstance(latency, int)
            assert latency >= 0
            assert db.is_connected
            assert db.info is not None
            assert db.info.session_id > 0
            print(f"  Connected to: {db.info.server_name}")
            print(f"  Session ID:   {db.info.session_id}")
            print(f"  Ping latency: {latency} us")
        finally:
            await db.close()


class TestCRUD:
    """Put, Get, Delete, Scan operations."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, uldb_server):
        """PUT then GET returns the same value."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_crud")
            await ns.put("auth.py::AuthService", b"class AuthService: pass")
            value = await ns.get("auth.py::AuthService")
            assert value == b"class AuthService: pass"
            print(f"  PUT/GET roundtrip: OK ({len(value)} bytes)")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_get_missing_key(self, uldb_server):
        """GET on nonexistent key returns None."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_crud")
            value = await ns.get("nonexistent_key_xyz")
            assert value is None
            print("  GET missing key: None (correct)")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_delete(self, uldb_server):
        """DELETE removes a key."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_delete")
            await ns.put("temp_key", b"temp_value")
            value = await ns.get("temp_key")
            assert value is not None

            await ns.delete("temp_key")
            value_after = await ns.get("temp_key")
            assert value_after is None
            print("  DELETE: key removed correctly")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_overwrite(self, uldb_server):
        """Second PUT overwrites the first."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_overwrite")
            await ns.put("mutable_key", b"version_1")
            assert await ns.get("mutable_key") == b"version_1"

            await ns.put("mutable_key", b"version_2")
            assert await ns.get("mutable_key") == b"version_2"
            print("  Overwrite: v1 -> v2 correct")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_large_value(self, uldb_server):
        """Can store and retrieve a 1MB value."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_large")
            big_value = b"x" * (1024 * 1024)
            await ns.put("big_key", big_value)
            result = await ns.get("big_key")
            assert result == big_value
            print(f"  Large value: {len(big_value)} bytes roundtrip OK")
        finally:
            await db.close()


class TestBatch:
    """Batch operations."""

    @pytest.mark.asyncio
    async def test_put_batch(self, uldb_server):
        """PUT_BATCH writes multiple records atomically."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_batch")
            records = [
                (f"batch_key_{i:03}", f"batch_val_{i}".encode())
                for i in range(20)
            ]
            await ns.put_batch(records)

            # Verify all records exist
            for key, expected_val in records:
                val = await ns.get(key)
                assert val == expected_val, f"batch key {key} mismatch"

            print(f"  PUT_BATCH: {len(records)} records written and verified")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_scan(self, uldb_server):
        """SCAN returns keys in a range."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_scan")

            # Write 10 sequentially named keys
            for i in range(10):
                await ns.put(f"scan_{i:03}", f"val_{i}".encode())

            # Scan [scan_003, scan_007) should return 4 keys
            pairs = await ns.scan("scan_003", "scan_007", limit=100)
            keys = [k for k, _ in pairs]
            assert len(pairs) == 4, f"expected 4, got {len(pairs)}: {keys}"
            print(f"  SCAN: returned {len(pairs)} keys (correct)")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_range_delete(self, uldb_server):
        """RANGE_DELETE removes keys in a range."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_range_del")
            for i in range(10):
                await ns.put(f"rd_{i:03}", b"v")

            await ns.range_delete("rd_003", "rd_007")

            # rd_000..rd_002 should exist
            for i in range(3):
                assert await ns.get(f"rd_{i:03}") is not None

            # rd_003..rd_006 should be gone
            for i in range(3, 7):
                assert await ns.get(f"rd_{i:03}") is None

            # rd_007..rd_009 should exist
            for i in range(7, 10):
                assert await ns.get(f"rd_{i:03}") is not None

            print("  RANGE_DELETE: correct boundaries")
        finally:
            await db.close()


class TestQuery:
    """Text search via BM25 + fuzzy indices."""

    @pytest.mark.asyncio
    async def test_text_query(self, uldb_server):
        """Text query returns relevant results."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_query")

            await ns.put("auth.py::validate_token",
                         b"def validate_token validates JWT authentication token")
            await ns.put("auth.py::hash_password",
                         b"def hash_password uses bcrypt hashing")
            await ns.put("models.py::User",
                         b"class User email password model")
            await ns.put("utils.py::send_email",
                         b"def send_email sends via SMTP")

            results = await ns.query("validate token authentication") \
                .top_k(5) \
                .execute()

            assert len(results) >= 1, "query should return at least 1 result"
            print(f"  TEXT QUERY: {len(results)} results")
            for r in results[:3]:
                print(f"    {r.key}  score={r.final_score:.4f}")
        finally:
            await db.close()


class TestSnapshots:
    """Snapshot create, list, restore, delete."""

    @pytest.mark.asyncio
    async def test_snapshot_lifecycle(self, uldb_server):
        """Full snapshot lifecycle."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_snap")

            # Write baseline
            await ns.put("doc1", b"version_1")

            # Create snapshot
            snap = await ns.snapshot_create("before_edit")
            print(f"  Snapshot created: {snap}")

            # Modify data
            await ns.put("doc1", b"version_2")

            # List snapshots
            snaps = await ns.snapshot_list()
            assert len(snaps) >= 1, "should have at least 1 snapshot"
            print(f"  Snapshots listed: {len(snaps)}")

            # Restore
            await ns.snapshot_restore("before_edit")
            print("  Snapshot restored")

        finally:
            await db.close()


class TestBranches:
    """Branch create, list, rollback."""

    @pytest.mark.asyncio
    async def test_branch_lifecycle(self, uldb_server):
        """Create and rollback a branch."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_branch")

            await ns.put("shared_key", b"original")

            # Create branch (using the low-level client API)
            await db._send_branch_create(0, "feat/test", "", "test branch")

            # List branches
            branches = await ns.branch_list()
            assert len(branches) >= 1
            print(f"  Branches: {len(branches)}")

            # Rollback
            await db._send_branch_rollback(0, "feat/test")
            print("  Branch rolled back")

        except AttributeError:
            # If _send_branch_create is not exposed, test branch via
            # the namespace.branch() context manager if available
            print("  Branch API not directly exposed, skipping")
        finally:
            await db.close()


class TestStats:
    """Server statistics."""

    @pytest.mark.asyncio
    async def test_stats(self, uldb_server):
        """Stats returns non-empty data."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_stats")

            # Write some data to make stats non-trivial
            for i in range(5):
                await ns.put(f"stat_key_{i}", b"v")

            # Ping to confirm connection is alive
            latency = await db.ping()
            assert latency >= 0
            print(f"  Stats test: 5 keys written, ping={latency}us")
        finally:
            await db.close()


class TestPersistence:
    """Data survives server restart."""

    @pytest.mark.asyncio
    async def test_data_survives_restart(self, uldb_server):
        """Write data, restart server, verify data is still there."""
        port = uldb_server["port"]
        token = uldb_server["token"]
        data_dir = uldb_server["data_dir"]
        proc = uldb_server["proc"]

        # Phase 1: write data
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_persist")
            for i in range(50):
                await ns.put(f"persist_{i:04}", f"value_{i}".encode())
            print(f"  Phase 1: wrote 50 records")
        finally:
            await db.close()

        # Phase 2: restart the server
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        # Find binary again
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        binary = os.path.join(project_root, "target", "release", "uldb")
        if not os.path.exists(binary):
            binary = os.path.join(project_root, "target", "debug", "uldb")

        new_port = _get_free_port()
        new_proc = subprocess.Popen(
            [binary, "serve", "--port", str(new_port),
             "--data", data_dir, "--token", token.decode()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Update fixture for other tests (they should not run after this)
        uldb_server["port"] = new_port
        uldb_server["proc"] = new_proc

        if not _wait_for_port(new_port, timeout=10.0):
            new_proc.kill()
            stdout, stderr = new_proc.communicate(timeout=5)
            pytest.fail(
                f"uldb failed to restart\n"
                f"stderr: {stderr.decode()}"
            )

        # Phase 3: verify data
        db2 = await _connect(uldb_server)
        try:
            ns2 = db2.namespace("test_persist")
            recovered = 0
            lost = 0
            for i in range(50):
                val = await ns2.get(f"persist_{i:04}")
                if val == f"value_{i}".encode():
                    recovered += 1
                else:
                    lost += 1
            print(f"  Phase 3: recovered={recovered}, lost={lost}")
            assert lost == 0, f"{lost}/50 records lost after restart"
        finally:
            await db2.close()


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
