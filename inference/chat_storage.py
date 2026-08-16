"""
chat_storage.py
─────────────────────────────────────────────────────────────────────────────
SQLite storage backend for Synapse chat conversation history.

Design goals:
  • Zero external dependencies — stdlib sqlite3 only
  • Thread-safe via per-request connection pattern (no global cursor)
  • Conversations are keyed by (client_ip, id) so each IP has its own namespace
  • Messages are stored normalised in a separate table for efficient queries
  • WAL mode for concurrent reads during writes

Tables:
  conversations — id, client_ip, title, created_at, updated_at
  messages      — id, conversation_id, role, content, timestamp

Usage:
  storage = ChatStorage("synapse_chats.db")
  await storage.init()
  await storage.save_conversation(ip, conv_dict)
  convs = await storage.list_conversations(ip)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("synapse.storage")


class ConversationOwnershipError(Exception):
    """Raised when a client tries to write a conversation id owned by another client_ip.

    Conversations are namespaced by (client_ip, id). save_conversation must refuse to
    overwrite/destroy a row owned by a different IP — otherwise knowing another user's
    conversation id would let you wipe and replace their message history.
    """


# Maximum number of conversations per IP (prevents abuse)
MAX_CONVERSATIONS_PER_IP = 50

# Maximum number of messages per conversation persisted
MAX_MESSAGES_PER_CONVERSATION = 100


class ChatStorage:
    """Synchronous SQLite storage wrapped in a thread-safe API.

    SQLite doesn't support true async I/O, so we use a simple synchronous
    pattern with per-call connections (autocommit transactions). This is
    sufficient for the expected load (single-user inference server).
    For production scale, swap for asyncpg / Turso.
    """

    def __init__(self, db_path: str | Path = "synapse_chats.db") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._initialized = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def init(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        if self._initialized:
            return

        conn = self._connect()
        try:
            conn.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;

                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT    NOT NULL,
                    client_ip   TEXT    NOT NULL,
                    title       TEXT    NOT NULL DEFAULT 'New Chat',
                    created_at  INTEGER NOT NULL,
                    updated_at  INTEGER NOT NULL,
                    PRIMARY KEY (id)
                );

                CREATE INDEX IF NOT EXISTS idx_conv_ip_updated
                    ON conversations (client_ip, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id              TEXT    NOT NULL,
                    conversation_id TEXT    NOT NULL,
                    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content         TEXT    NOT NULL,
                    timestamp       INTEGER NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_msg_conv
                    ON messages (conversation_id, timestamp ASC);
            """)
            conn.commit()
            self._initialized = True
            logger.info("ChatStorage initialised: %s", self._db_path)
        finally:
            conn.close()
        self._restrict_permissions()

    def _restrict_permissions(self) -> None:
        """chmod 600 the DB (+ WAL/SHM sidecars) so conversation transcripts aren't
        world-readable on a shared/rented host. Best-effort; a no-op on Windows."""
        if os.name == "nt":
            return
        for suffix in ("", "-wal", "-shm"):
            p = self._db_path + suffix
            try:
                if os.path.exists(p):
                    os.chmod(p, 0o600)
            except OSError as e:
                logger.warning("Could not chmod %s: %s", p, e)

    def purge_old(self, max_age_days: int) -> int:
        """Delete conversations (and, via cascade, their messages) not updated within
        max_age_days. Returns the number removed. Retention hygiene for a public store
        of user chat content. max_age_days <= 0 disables purging."""
        if max_age_days <= 0:
            return 0
        cutoff = int(time.time() * 1000) - max_age_days * 86_400_000  # updated_at is ms
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))
                conn.commit()
                n = cur.rowcount
            finally:
                conn.close()
        if n:
            logger.info("ChatStorage: purged %d conversation(s) older than %d days", n, max_age_days)
        return n

    # ── Connection helper ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save_conversation(
        self,
        client_ip: str,
        conv: dict[str, Any],
    ) -> None:
        """Upsert a conversation with its messages.

        Expected `conv` shape:
        {
            "id": "uuid-string",
            "title": "First user message...",
            "createdAt": 1718...,
            "updatedAt": 1718...,
            "messages": [
                {"id": "msg-uuid", "role": "user", "content": "...", "timestamp": 1718...},
                ...
            ]
        }
        """
        conv_id = conv["id"]
        title = conv.get("title", "New Chat")
        created_at = conv.get("createdAt", conv.get("created_at", 0))
        updated_at = conv.get("updatedAt", conv.get("updated_at", 0))
        messages: list[dict[str, Any]] = conv.get("messages", [])

        # Enforce message limit (keep most recent)
        if len(messages) > MAX_MESSAGES_PER_CONVERSATION:
            messages = messages[-MAX_MESSAGES_PER_CONVERSATION:]

        with self._lock:
            conn = self._connect()
            try:
                # ── Ownership guard ──────────────────────────────────────────
                # Conversation ids are globally unique (PRIMARY KEY on id), so an
                # upsert on a known id would otherwise let ANY client overwrite the
                # title and DELETE+replace the messages of a conversation owned by a
                # DIFFERENT client_ip. Every other CRUD method here filters by
                # client_ip; this one must too. Refuse cross-owner writes.
                owner = conn.execute(
                    "SELECT client_ip FROM conversations WHERE id = ?",
                    (conv_id,),
                ).fetchone()
                if owner is not None and owner["client_ip"] != client_ip:
                    raise ConversationOwnershipError(conv_id)

                # Upsert conversation (now guaranteed same-owner or brand-new id)
                conn.execute(
                    """
                    INSERT INTO conversations (id, client_ip, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        updated_at = excluded.updated_at
                    """,
                    (conv_id, client_ip, title, created_at, updated_at),
                )

                # Replace all messages (simple strategy — delete + insert). Safe now:
                # we've verified this client_ip owns conv_id (or just created it).
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conv_id,),
                )

                if messages:
                    conn.executemany(
                        """
                        INSERT INTO messages (id, conversation_id, role, content, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                # `or`, not a default: ConversationMessageItem.id
                                # defaults to "" so the key is ALWAYS present, which
                                # made `m.get("id", fallback)` dead — a blank id then
                                # squatted the GLOBAL `messages.id` primary key and an
                                # IntegrityError blocked every other owner's save of an
                                # id-less message. `or` sends "" to the per-conversation
                                # fallback instead.
                                (m.get("id") or f"{conv_id}-{i}"),
                                conv_id,
                                m["role"],
                                m["content"],
                                m.get("timestamp", 0),
                            )
                            for i, m in enumerate(messages)
                        ],
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        # Enforce per-IP limit (evict oldest)
        self._enforce_ip_limit(client_ip)

    def get_conversation(
        self,
        client_ip: str,
        conv_id: str,
    ) -> dict[str, Any] | None:
        """Fetch a single conversation with messages. Returns None if not found
        or if the conversation doesn't belong to this IP."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND client_ip = ?",
                (conv_id, client_ip),
            ).fetchone()

            if not row:
                return None

            messages = conn.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conv_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "title": row["title"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
                "messages": [
                    {
                        "id": m["id"],
                        "role": m["role"],
                        "content": m["content"],
                        "timestamp": m["timestamp"],
                    }
                    for m in messages
                ],
            }
        finally:
            conn.close()

    def list_conversations(self, client_ip: str) -> list[dict[str, Any]]:
        """Return all conversations for an IP, sorted by updatedAt descending.
        Does NOT include messages (lightweight listing)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE client_ip = ? ORDER BY updated_at DESC",
                (client_ip,),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "createdAt": r["created_at"],
                    "updatedAt": r["updated_at"],
                    "messageCount": self._count_messages(conn, r["id"]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def delete_conversation(self, client_ip: str, conv_id: str) -> bool:
        """Delete a conversation (cascade deletes messages). Returns True if found."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ? AND client_ip = ?",
                    (conv_id, client_ip),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def rename_conversation(
        self,
        client_ip: str,
        conv_id: str,
        new_title: str,
    ) -> bool:
        """Rename a conversation. Returns True if found."""
        with self._lock:
            conn = self._connect()
            try:
                import time
                cursor = conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? "
                    "WHERE id = ? AND client_ip = ?",
                    (new_title, int(time.time() * 1000), conv_id, client_ip),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _count_messages(conn: sqlite3.Connection, conv_id: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def _enforce_ip_limit(self, client_ip: str) -> None:
        """If an IP has more than MAX_CONVERSATIONS_PER_IP, delete the oldest."""
        conn = self._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversations WHERE client_ip = ?",
                (client_ip,),
            ).fetchone()

            if count and count["cnt"] > MAX_CONVERSATIONS_PER_IP:
                excess = count["cnt"] - MAX_CONVERSATIONS_PER_IP
                conn.execute(
                    """
                    DELETE FROM conversations WHERE id IN (
                        SELECT id FROM conversations
                        WHERE client_ip = ?
                        ORDER BY updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (client_ip, excess),
                )
                conn.commit()
                logger.info(
                    "Evicted %d old conversations for IP %s",
                    excess,
                    client_ip,
                )
        finally:
            conn.close()
