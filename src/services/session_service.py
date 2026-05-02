from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    sources     TEXT,
    timestamp   TEXT    NOT NULL
)
"""


class SessionService:
    """
    Persistent conversation management backed by SQLite.

    Uses a context-manager connection pattern: each operation opens,
    commits, and closes its own connection so no connection is held
    open between calls.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_TURNS)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(self, name: Optional[str] = None) -> str:
        """Create a new session and return its UUID."""
        sid = str(uuid.uuid4())
        now = self._now()
        label = name or f"Session {now[:10]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, label, now, now),
            )
        return sid

    def get_history(self, session_id: str, max_turns: int = 3) -> List[dict]:
        """Return the last *max_turns* full exchanges as role/content dicts."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM turns
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, max_turns * 2),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def record(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[str],
    ) -> None:
        """Append a user/assistant turn pair to *session_id*."""
        now = self._now()
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO turns (session_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, "user", question, None, now),
                    (session_id, "assistant", answer, json.dumps(sources), now),
                ],
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def list_all(self) -> List[dict]:
        """Return all sessions ordered by most recently updated."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
