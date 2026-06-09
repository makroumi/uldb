"""
uldb - The database for AI agents.

Quick start (30 seconds):

    from uldb import DB

    db = DB("./my_project")
    db.put("auth.py::validate", b"def validate(token): ...")
    print(db.get("auth.py::validate"))
    results = db.search("validate token")
    db.close()

For multi-agent workflows:

    from uldb import Client

    client = Client.connect("./my_project")
    ws = client.workspace           # main workspace
    ctx = client.context            # search engine

    # Index a codebase
    ctx.ingest({
        "auth.py::validate": b"def validate(token): ...",
        "auth.py::hash":     b"def hash(password): ...",
    })

    # Search
    docs = ctx.query("validate token")
    for doc in docs:
        print(f"{doc.id}: {doc.score:.4f}")

    # Branch for an agent task
    branch = client.branch("feat/refactor")
    branch.put("auth.py::validate", b"def validate_v2(token): ...")
    branch.merge_to_main()

    client.close()
"""

__version__ = "0.1.0"

try:
    from uldb._core import (
        DB,
        Client,
        Workspace,
        ContextEngine,
        Document,
    )
    _BACKEND = "rust"
except ImportError:
    _BACKEND = "python-fallback"

    class DB:
        def __init__(self, path):
            raise RuntimeError(
                "uldb requires the Rust extension. "
                "Install with: pip install uldb"
            )

    class Client:
        @staticmethod
        def connect(url):
            raise RuntimeError("uldb requires the Rust extension.")

    class Workspace:
        pass

    class ContextEngine:
        pass

    class Document:
        pass

__all__ = ["DB", "Client", "Workspace", "ContextEngine", "Document"]
