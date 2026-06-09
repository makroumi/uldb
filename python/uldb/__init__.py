"""
uldb -- The database for AI agents.

    import uldb

    app = uldb.open("./my_project")

    # Store and search
    app.put("auth.py::validate", b"def validate(token): ...")
    app.search("validate token")

    # Agent with isolated workspace
    with app.agent("refactor-auth") as agent:
        docs = agent.search("validate token")
        agent.put("auth.py::validate", b"def validate_v2(token): ...")
        # auto-merge on success, auto-rollback on failure

    app.close()
"""

__version__ = "0.1.0"

try:
    from uldb._core import (
        DB,
        Client,
        Workspace,
        ContextEngine,
        Document,
        Agent,
    )
    _BACKEND = "rust"
except ImportError:
    _BACKEND = "python-fallback"
    class DB:
        def __init__(self, path):
            raise RuntimeError("uldb requires the Rust extension. Install with: pip install uldb")
    class Client:
        @staticmethod
        def connect(url):
            raise RuntimeError("uldb requires the Rust extension.")
    class Workspace: pass
    class ContextEngine: pass
    class Document: pass
    class Agent: pass


def open(path: str) -> DB:
    """Open a database. The primary entry point for uldb.

    Args:
        path: Directory to store data. Created if it does not exist.

    Returns:
        A DB instance with put/get/search/agent methods.

    Usage:
        app = uldb.open("./my_project")
        app.put("key", b"value")
        app.search("query")
        app.close()
    """
    return DB(path)


__all__ = ["open", "DB", "Client", "Agent", "Document"]
