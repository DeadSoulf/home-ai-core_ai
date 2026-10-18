"""Isolated SQLite data store. Provider credentials are never returned by list APIs."""
from __future__ import annotations

import contextlib
import os
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

class Store:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "ai.sqlite3"
        self.lock = threading.RLock()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS providers (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
                    base_url TEXT NOT NULL, model TEXT NOT NULL, api_key TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY, action TEXT NOT NULL, target TEXT NOT NULL,
                    reason TEXT NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL
                );
            """)
        os.chmod(self.path, 0o600)

    @contextlib.contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _id() -> str:
        return uuid.uuid4().hex

    def providers(self) -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.execute("SELECT id,name,kind,base_url,model,(length(api_key)>0) AS has_api_key FROM providers ORDER BY name").fetchall()
            return [dict(row) for row in rows]

    def provider(self, provider_id: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.execute("SELECT * FROM providers WHERE id=?", (provider_id,)).fetchone()
            return dict(row) if row else None

    def add_provider(self, values: dict[str, str]) -> str:
        identifier = self._id()
        with self.db() as db:
            db.execute("INSERT INTO providers(id,name,kind,base_url,model,api_key) VALUES(?,?,?,?,?,?)", (
                identifier, values["name"], values["kind"], values["base_url"], values["model"], values["api_key"]
            ))
        return identifier

    def ensure_local_provider(self, name: str, base_url: str, model: str) -> str:
        """Register a ready bundled model without duplicating the stored provider.

        Preserve existing user-defined providers and their API keys.
        """
        with self.lock, self.db() as db:
            row = db.execute(
                "SELECT id FROM providers WHERE kind='ollama' AND base_url=? AND model=?",
                (base_url, model),
            ).fetchone()
            if row:
                return str(row["id"])
            identifier = self._id()
            db.execute(
                "INSERT INTO providers(id,name,kind,base_url,model,api_key) VALUES(?,?,?,?,?,?)",
                (identifier, name, "ollama", base_url, model, ""),
            )
            return identifier

    def delete_provider(self, provider_id: str) -> bool:
        with self.db() as db:
            return db.execute("DELETE FROM providers WHERE id=?", (provider_id,)).rowcount > 0

    def conversations(self) -> list[dict[str, Any]]:
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 100")]

    def conversation_exists(self, identifier: str) -> bool:
        with self.db() as db:
            return db.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone() is not None

    def delete_conversation(self, identifier: str) -> bool:
        """Permanently delete a conversation and its messages in one transaction."""
        with self.lock, self.db() as db:
            # Explicit deletion also handles databases created before FK support.
            existing = db.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone()
            if existing is None:
                return False
            db.execute("DELETE FROM messages WHERE conversation_id=?", (identifier,))
            db.execute("DELETE FROM conversations WHERE id=?", (identifier,))
            return True

    def new_conversation(self, title: str) -> str:
        identifier = self._id()
        now = int(time.time())
        with self.db() as db:
            db.execute("INSERT INTO conversations VALUES(?,?,?,?)", (identifier, title[:100], now, now))
        return identifier

    def add_message(self, identifier: str, role: str, content: str) -> None:
        with self.db() as db:
            db.execute("INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)", (identifier, role, content))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (int(time.time()), identifier))

    def messages(self, identifier: str, limit: int = 40) -> list[dict[str, str]]:
        with self.db() as db:
            rows = db.execute("SELECT role,content FROM (SELECT id,role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?) ORDER BY id", (identifier, limit)).fetchall()
            return [dict(row) for row in rows]

    def add_approval(self, values: dict[str, str]) -> str:
        identifier = self._id()
        with self.db() as db:
            db.execute("INSERT INTO approvals VALUES(?,?,?,?,?,?)", (identifier, values["action"], values["target"], values["reason"], "pending", int(time.time())))
        return identifier

    def approvals(self) -> list[dict[str, Any]]:
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM approvals ORDER BY created_at DESC LIMIT 100")]

    def decide(self, identifier: str, decision: str) -> bool:
        status = "approved_not_executed" if decision == "approve" else "rejected"
        with self.lock, self.db() as db:
            changed = db.execute("UPDATE approvals SET status=? WHERE id=? AND status='pending'", (status, identifier))
            return changed.rowcount > 0


def access_token(data_dir: str) -> str:
    from_env = os.environ.get("AI_ACCESS_TOKEN")
    if from_env:
        if len(from_env) < 24:
            raise ValueError("AI_ACCESS_TOKEN must be at least 24 characters")
        return from_env
    path = Path(data_dir) / "access-token"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(40)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(value + "\n")
    return value
