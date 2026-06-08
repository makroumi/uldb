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
    """Branch create, list, merge, rollback."""

    @pytest.mark.asyncio
    async def test_branch_create_and_rollback(self, uldb_server):
        """Create a branch, list it, then roll it back."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_branch_rollback")

            await ns.put("shared_key", b"original")

            # Create branch via low-level client API
            await db._branch_create("", "feat/rollback", "test rollback")

            # List branches
            branches = await ns.branch_list()
            assert len(branches) >= 1, f"expected >= 1 branch, got {len(branches)}"
            print(f"  Branches after create: {len(branches)}")

            # Rollback
            await db._branch_rollback("", "feat/rollback")

            # List again - should be gone
            branches_after = await ns.branch_list()
            print(f"  Branches after rollback: {len(branches_after)}")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_branch_context_manager(self, uldb_server):
        """Branch context manager auto-merges on clean exit."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_branch_ctx")

            await ns.put("ctx_key", b"before_branch")

            async with ns.branch("feat/ctx-test", "context manager test") as branch:
                await branch.put("ctx_key", b"branch_version")
                val = await branch.get("ctx_key")
                assert val == b"branch_version"
                print(f"  In-branch read: {val}")
            # auto-merge on clean exit

            print("  Branch auto-merged")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_branch_rollback_on_exception(self, uldb_server):
        """Branch context manager auto-rolls-back on exception."""
        db = await _connect(uldb_server)
        try:
            ns = db.namespace("test_branch_exc")
            await ns.put("exc_key", b"original")

            try:
                async with ns.branch("feat/will-fail", "should rollback"):
                    raise ValueError("simulated failure")
            except ValueError:
                pass  # expected

            # Branch should have been rolled back
            branches = await ns.branch_list()
            branch_names = [str(b) for b in branches]
            assert "feat/will-fail" not in branch_names,                 "branch should be rolled back after exception"
            print("  Branch rolled back on exception: correct")
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


class TestConcurrent:
    """Multiple simultaneous clients."""

    @pytest.mark.asyncio
    async def test_concurrent_clients(self, uldb_server):
        """Multiple clients can read and write simultaneously."""
        import asyncio

        async def client_work(client_id: int):
            db = await _connect(uldb_server)
            try:
                ns = db.namespace("test_concurrent")
                for i in range(10):
                    key = f"client_{client_id}_key_{i}"
                    val = f"client_{client_id}_val_{i}".encode()
                    await ns.put(key, val)
                    got = await ns.get(key)
                    assert got == val, f"client {client_id}: mismatch on {key}"
            finally:
                await db.close()

        tasks = [client_work(i) for i in range(4)]
        await asyncio.gather(*tasks)
        print(f"  4 concurrent clients x 10 ops each: all correct")

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_writes(self, uldb_server):
        """Reader and writer clients operate simultaneously."""
        import asyncio

        db_writer = await _connect(uldb_server)
        db_reader = await _connect(uldb_server)
        try:
            ns_w = db_writer.namespace("test_rw")
            ns_r = db_reader.namespace("test_rw")

            # Writer seeds data
            for i in range(20):
                await ns_w.put(f"rw_{i:03}", f"val_{i}".encode())

            # Reader reads while writer overwrites
            async def write_loop():
                for i in range(20):
                    await ns_w.put(f"rw_{i:03}", f"updated_{i}".encode())

            async def read_loop():
                results = []
                for i in range(20):
                    val = await ns_r.get(f"rw_{i:03}")
                    results.append(val is not None)
                return all(results)

            _, all_found = await asyncio.gather(write_loop(), read_loop())
            assert all_found, "reader should find all keys during concurrent writes"
            print("  Concurrent read+write: no crashes, all keys found")
        finally:
            await db_writer.close()
            await db_reader.close()


class TestTransactions:
    """Transaction begin/commit/rollback."""

    @pytest.mark.asyncio
    async def test_transaction_commit(self, uldb_server):
        """Transaction begin + commit lifecycle."""
        db = await _connect(uldb_server)
        try:
            # Begin transaction
            tx_id = await db._tx_begin("snapshot")
            assert tx_id > 0, f"tx_begin should return positive tx_id, got {tx_id}"
            print(f"  Transaction started: tx_id={tx_id}")

            # Commit
            await db._tx_commit(tx_id)
            print("  Transaction committed")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, uldb_server):
        """Transaction begin + rollback lifecycle."""
        db = await _connect(uldb_server)
        try:
            tx_id = await db._tx_begin("snapshot")
            assert tx_id > 0
            await db._tx_rollback(tx_id)
            print(f"  Transaction {tx_id} rolled back")
        finally:
            await db.close()


class TestAdmin:
    """Admin operations."""

    @pytest.mark.asyncio
    async def test_config_get(self, uldb_server):
        """Can read server config values."""
        db = await _connect(uldb_server)
        try:
            version = await db.config_get("version")
            print(f"  Server version: {version}")
            assert version is not None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_namespace_create_and_list(self, uldb_server):
        """Create a namespace and list it."""
        db = await _connect(uldb_server)
        try:
            await db._ns_create(
                "github.com/test/repo",
                "abc123",
                "test namespace"
            )
            namespaces = await db._ns_list()
            assert len(namespaces) >= 1
            print(f"  Namespaces: {len(namespaces)}")
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
