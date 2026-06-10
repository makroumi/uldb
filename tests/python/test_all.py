"""uldb integration tests."""

import uldb


def test_backend():
    print(f"Backend: {uldb._BACKEND}")
    assert uldb._BACKEND in ("rust", "python-fallback")


def test_open_and_close(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    assert app is not None
    app.close()


def test_put_and_get(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    app.put("hello", b"world")
    doc = app.get("hello")
    assert doc is not None
    assert doc == b"world"
    app.close()


def test_search(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    app.put("auth.py::validate", b"def validate(token): pass")
    app.put("auth.py::login", b"def login(user, password): pass")
    results = app.search("validate token")
    assert len(results) > 0
    app.close()


def test_delete(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    app.put("key1", b"value1")
    app.delete("key1")
    doc = app.get("key1")
    assert doc is None
    app.close()


def test_agent_workflow(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    app.put("auth.py::validate", b"def validate(token): pass")

    with app.agent("refactor-auth") as agent:
        docs = agent.search("validate")
        assert len(docs) >= 0
        agent.put("auth.py::validate", b"def validate_v2(token): pass")

    # After commit, main should have the new version
    doc = app.get("auth.py::validate")
    assert doc is not None
    assert b"validate_v2" in doc
    app.close()


def test_agent_rollback(tmp_path):
    app = uldb.open(str(tmp_path / "test_db"))
    app.put("key1", b"original")

    try:
        with app.agent("bad-agent") as agent:
            agent.put("key1", b"modified")
            raise ValueError("simulated failure")
    except ValueError:
        pass

    # After rollback, main should still have original
    doc = app.get("key1")
    assert doc is not None
    assert doc == b"original"
    app.close()


def test_context_manager(tmp_path):
    with uldb.open(str(tmp_path / "test_db")) as app:
        app.put("key", b"value")
        doc = app.get("key")
        assert doc is not None
