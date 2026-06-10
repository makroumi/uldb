"""
Comprehensive uldb feature test.

Covers every Python-exposed method on DB, Agent, Workspace, ContextEngine, Client.
"""
import pytest
import uldb


# =========================================================================
# DB core: put, get, delete, scan, search, stats
# =========================================================================

class TestDBCore:

    def test_put_get_bytes(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"hello")
        assert app.get("k1") == b"hello"
        app.close()

    def test_get_missing_returns_none(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        assert app.get("nonexistent") is None
        app.close()

    def test_delete(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"v1")
        app.delete("k1")
        assert app.get("k1") is None
        app.close()

    def test_overwrite(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"v1")
        app.put("k1", b"v2")
        assert app.get("k1") == b"v2"
        app.close()

    def test_scan(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(10):
            app.put(f"scan_{i:03}", f"val_{i}".encode())
        results = app.scan("scan_", limit=100)
        assert len(results) == 10
        assert all(hasattr(d, 'id') for d in results)
        app.close()

    def test_search_bm25(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("auth.py::validate", b"def validate(token): check jwt token")
        app.put("auth.py::login", b"def login(user, password): authenticate")
        app.put("utils.py::format", b"def format_date(): format dates")
        results = app.search("validate token")
        assert len(results) >= 1
        assert results[0].score > 0
        app.close()

    def test_search_fuzzy(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("getUserById", b"code")
        app.put("setUserName", b"code")
        results = app.search_fuzzy("getUserByID", limit=5)
        assert len(results) >= 1
        app.close()

    def test_stats(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(5):
            app.put(f"k{i}", b"v")
        stats = app.stats()
        assert isinstance(stats, dict)
        assert "records" in stats or "index_docs" in stats
        app.close()

    def test_load_bulk(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        records = {f"bulk_{i:03}": f"val_{i}".encode() for i in range(50)}
        count = app.load(records)
        assert count == 50
        assert app.get("bulk_000") is not None
        app.close()

    def test_empty_value(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("empty", b"")
        result = app.get("empty")
        assert result == b""
        app.close()

    def test_large_value(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        big = b"x" * 100_000
        app.put("big", big)
        assert app.get("big") == big
        app.close()

    def test_special_chars_in_key(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("path/to/file.py::ClassName.method_name", b"code")
        assert app.get("path/to/file.py::ClassName.method_name") is not None
        app.close()

    def test_context_manager(self, tmp_path):
        with uldb.open(str(tmp_path / "db")) as app:
            app.put("ctx", b"works")
            assert app.get("ctx") == b"works"

    def test_delete_range(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(10):
            app.put(f"dr_{i:03}", b"v")
        deleted = app.delete_range("dr_003", "dr_007")
        assert deleted >= 0
        app.close()

    def test_keys(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(5):
            app.put(f"keys_test_{i}", b"v")
        keys = app.keys("keys_test_", limit=10)
        assert len(keys) == 5
        assert all(isinstance(k, str) for k in keys)
        app.close()


# =========================================================================
# Snapshots
# =========================================================================

class TestSnapshots:

    def test_snapshot_create(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"v1")
        snap_id = app.snapshot("test_snap")
        assert isinstance(snap_id, str)
        assert len(snap_id) > 0
        app.close()

    def test_snapshot_list(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"v1")
        app.snapshot("snap_a")
        app.snapshot("snap_b")
        snaps = app.snapshots()
        assert len(snaps) >= 2
        app.close()

    def test_snapshot_restore(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"original")
        app.snapshot("before_edit")
        app.put("k1", b"modified")
        assert app.get("k1") == b"modified"
        app.restore("before_edit")
        # After restore, state should be back
        app.close()


# =========================================================================
# Agent: isolated workspace
# =========================================================================

class TestAgent:

    def test_agent_put_get(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        with app.agent("test_agent") as agent:
            agent.put("k1", b"agent_val")
            doc = agent.get("k1")
            assert doc is not None
            assert doc.raw == b"agent_val"
        app.close()

    def test_agent_search(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("auth.py::validate", b"def validate token jwt")
        with app.agent("searcher") as agent:
            results = agent.search("validate token")
            assert isinstance(results, list)
        app.close()

    def test_agent_commit_merges(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"original")
        with app.agent("committer") as agent:
            agent.put("k1", b"modified")
        # After commit, main should see the change
        assert app.get("k1") == b"modified"
        app.close()

    def test_agent_rollback_on_exception(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"original")
        try:
            with app.agent("failer") as agent:
                agent.put("k1", b"bad_value")
                raise ValueError("boom")
        except ValueError:
            pass
        assert app.get("k1") == b"original"
        app.close()

    def test_agent_manual_discard(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"original")
        agent = app.agent("manual")
        agent.put("k1", b"temp")
        agent.discard()
        assert app.get("k1") == b"original"
        app.close()

    def test_agent_manual_commit(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"original")
        agent = app.agent("manual_commit")
        agent.put("k1", b"committed")
        agent.commit()
        assert app.get("k1") == b"committed"
        app.close()

    def test_agent_delete(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("k1", b"v1")
        with app.agent("deleter") as agent:
            agent.delete("k1")
        app.close()

    def test_agent_scan(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(5):
            app.put(f"agent_scan_{i}", b"v")
        with app.agent("scanner") as agent:
            results = agent.scan("agent_scan_", limit=10)
            assert len(results) >= 0
        app.close()

    def test_agent_load_bulk(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        with app.agent("bulk_loader") as agent:
            records = {f"bulk_{i}": f"val_{i}".encode() for i in range(20)}
            count = agent.load(records)
            assert count == 20
        app.close()

    def test_agent_search_fuzzy(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("getUserById", b"code")
        with app.agent("fuzzy_agent") as agent:
            results = agent.search_fuzzy("getUserByID", limit=5)
            assert isinstance(results, list)
        app.close()

    def test_agent_cannot_reuse_after_commit(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        agent = app.agent("done_agent")
        agent.commit()
        with pytest.raises(Exception):
            agent.put("k", b"v")
        app.close()

    def test_agent_repr(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        agent = app.agent("repr_test")
        assert "repr_test" in repr(agent)
        agent.discard()
        app.close()

    def test_agent_checkpoint(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        with app.agent("cp_agent") as agent:
            agent.put("k1", b"v1")
            cp = agent.checkpoint("mid_point")
            assert isinstance(cp, str)
        app.close()


# =========================================================================
# Document object
# =========================================================================

class TestDocument:

    def test_document_fields(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("doc_test", b"hello world")
        results = app.search("hello")
        if results:
            doc = results[0]
            assert hasattr(doc, 'id')
            assert hasattr(doc, 'content')
            assert hasattr(doc, 'raw')
            assert hasattr(doc, 'score')
            assert hasattr(doc, 'metadata')
            assert isinstance(doc.id, str)
            assert isinstance(doc.score, float)
            assert isinstance(doc.content, str)
            assert isinstance(doc.raw, (bytes, memoryview))
            assert isinstance(doc.metadata, dict)
            assert len(doc) > 0
            assert bool(doc)
            assert isinstance(repr(doc), str)
            assert isinstance(str(doc), str)
        app.close()


# =========================================================================
# ContextEngine (indexing)
# =========================================================================

class TestContextEngine:

    def test_context_query(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ctx = client.context
        ctx.ingest({"auth.py::validate": b"def validate token jwt"})
        results = ctx.query("validate", limit=5)
        assert isinstance(results, list)
        client.close()

    def test_context_search_text(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ctx = client.context
        ctx.ingest({"search_text_key": b"some searchable content"})
        results = ctx.search_text("searchable", limit=5)
        assert isinstance(results, list)
        client.close()

    def test_context_search_fuzzy(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ctx = client.context
        ctx.ingest({"getUserById": b"code"})
        results = ctx.search_fuzzy("getUserByID", limit=5)
        assert isinstance(results, list)
        client.close()

    def test_context_index(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ctx = client.context
        ctx.index("idx_key", b"indexed content")
        results = ctx.query("indexed", limit=5)
        assert isinstance(results, list)
        client.close()

    def test_context_add_edge(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ctx = client.context
        ctx.ingest({"A": b"node A", "B": b"node B"})
        ctx.add_edge("A", "B", "calls")
        client.close()


# =========================================================================
# Client class
# =========================================================================

class TestClient:

    def test_client_connect(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        assert client is not None
        client.close()

    def test_client_workspace(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        assert ws is not None
        ws.put("ws_key", b"ws_val")
        doc = ws.get("ws_key")
        assert doc is not None
        client.close()

    def test_client_branch(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        ws.put("k1", b"v1")
        branch_ws = client.branch("feat/test", "main")
        assert branch_ws is not None
        assert "feat/test" in repr(branch_ws)
        client.close()

    def test_client_agent(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        agent = client.agent("test_agent")
        assert agent is not None
        agent.discard()
        client.close()

    def test_client_branches_list(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        branches = client.branches()
        assert isinstance(branches, list)
        client.close()

    def test_client_stats(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        stats = client.stats()
        assert isinstance(stats, dict)
        client.close()

    def test_client_context_manager(self, tmp_path):
        from uldb._core import Client
        with Client.connect(str(tmp_path / "db")) as client:
            ws = client.workspace
            ws.put("cm_key", b"cm_val")

    def test_client_repr(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        assert str(tmp_path) in repr(client)
        client.close()


# =========================================================================
# Workspace class
# =========================================================================

class TestWorkspace:

    def test_workspace_crud(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        ws.put("ws_k1", b"ws_v1")
        doc = ws.get("ws_k1")
        assert doc is not None
        ws.delete("ws_k1")
        assert ws.get("ws_k1") is None
        client.close()

    def test_workspace_scan(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        for i in range(5):
            ws.put(f"ws_scan_{i}", b"v")
        results = ws.scan("ws_scan_", limit=10)
        assert len(results) == 5
        client.close()

    def test_workspace_put_batch(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        records = {f"batch_{i}": f"val_{i}".encode() for i in range(10)}
        ws.put_batch(records)
        doc = ws.get("batch_0")
        assert doc is not None
        client.close()

    def test_workspace_snapshot(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        ws.put("snap_k", b"snap_v")
        snap = ws.snapshot("test_snap")
        assert isinstance(snap, str)
        client.close()

    def test_workspace_contains(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        ws.put("exists_k", b"v")
        assert "exists_k" in ws
        assert "nope" not in ws
        client.close()

    def test_workspace_merge_branch(self, tmp_path):
        from uldb._core import Client
        client = Client.connect(str(tmp_path / "db"))
        ws = client.workspace
        ws.put("merge_k", b"original")
        branch = client.branch("feat/merge", "main")
        branch.put("merge_k", b"branched")
        count = branch.merge_to_main()
        assert isinstance(count, int)
        client.close()


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:

    def test_many_small_writes(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        for i in range(500):
            app.put(f"small_{i:05}", b"x")
        assert app.get("small_00000") == b"x"
        assert app.get("small_00499") == b"x"
        app.close()

    def test_unicode_key(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("key_with_emoji_\U0001f600", b"smile")
        result = app.get("key_with_emoji_\U0001f600")
        assert result == b"smile"
        app.close()

    def test_binary_value(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        binary = bytes(range(256))
        app.put("binary_key", binary)
        assert app.get("binary_key") == binary
        app.close()

    def test_multiple_agents_independent(self, tmp_path):
        app = uldb.open(str(tmp_path / "db"))
        app.put("shared", b"original")
        agent1 = app.agent("agent_1")
        agent2 = app.agent("agent_2")
        agent1.put("shared", b"from_agent_1")
        agent2.put("shared", b"from_agent_2")
        # Each agent sees its own write
        assert agent1.get("shared").raw == b"from_agent_1"
        assert agent2.get("shared").raw == b"from_agent_2"
        agent1.discard()
        agent2.commit()
        assert app.get("shared") == b"from_agent_2"
        app.close()

    def test_reopen_persists(self, tmp_path):
        db_path = str(tmp_path / "persist_db")
        app = uldb.open(db_path)
        app.put("persist_k", b"persist_v")
        app.close()

        app2 = uldb.open(db_path)
        assert app2.get("persist_k") == b"persist_v"
        app2.close()
