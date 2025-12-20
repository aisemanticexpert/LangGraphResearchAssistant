"""
Persistence utilities for the Research Assistant.

Provides checkpointer factory for different backends (memory, sqlite).
"""

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from ..config import settings

logger = logging.getLogger(__name__)


def get_checkpointer() -> Any:
    """
    Get the appropriate checkpointer based on configuration.

    Returns:
        Checkpointer instance (MemorySaver or SqliteSaver)
    """
    backend = settings.checkpoint_backend.lower()

    if backend == "sqlite":
        return _get_sqlite_checkpointer()
    elif backend == "postgres":
        return _get_postgres_checkpointer()
    else:
        logger.info("Using in-memory checkpointer")
        return MemorySaver()


def _get_sqlite_checkpointer() -> Any:
    """Create SQLite checkpointer."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        # Ensure directory exists
        db_dir = os.path.dirname(settings.sqlite_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Create connection
        conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        logger.info(f"Using SQLite checkpointer: {settings.sqlite_path}")
        return checkpointer

    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite not installed, falling back to memory")
        return MemorySaver()
    except Exception as e:
        logger.error(f"SQLite setup failed: {e}, falling back to memory")
        return MemorySaver()


def _get_postgres_checkpointer() -> Any:
    """Create PostgreSQL checkpointer."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        if not settings.postgres_url:
            logger.warning("POSTGRES_URL not configured, falling back to memory")
            return MemorySaver()

        checkpointer = PostgresSaver.from_conn_string(settings.postgres_url)
        logger.info("Using PostgreSQL checkpointer")
        return checkpointer

    except ImportError:
        logger.warning("langgraph-checkpoint-postgres not installed, falling back to memory")
        return MemorySaver()
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}, falling back to memory")
        return MemorySaver()


class ConversationStore:
    """
    Simple conversation metadata storage using SQLite.

    Stores conversation metadata separately from LangGraph checkpoints
    for easier querying and management.
    """

    def __init__(self, db_path: str = None):
        import sqlite3

        self.db_path = db_path or settings.sqlite_path.replace(".db", "_meta.db")

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()
        logger.info(f"ConversationStore initialized: {self.db_path}")

    def _init_tables(self):
        """Initialize database tables."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                detected_company TEXT,
                final_response TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                role TEXT,
                content TEXT,
                agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES conversations(thread_id)
            )
        """)
        self.conn.commit()

    def save_conversation(
        self,
        thread_id: str,
        user_query: str,
        detected_company: str = None,
        final_response: str = None,
        status: str = "active"
    ):
        """Save or update conversation metadata."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO conversations
            (thread_id, user_query, detected_company, final_response, status, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (thread_id, user_query, detected_company, final_response, status))
        self.conn.commit()

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        agent: str = None
    ):
        """Add a message to conversation history."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_messages (thread_id, role, content, agent)
            VALUES (?, ?, ?, ?)
        """, (thread_id, role, content, agent))
        self.conn.commit()

    def get_conversation(self, thread_id: str) -> dict:
        """Get conversation metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM conversations WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "thread_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "user_query": row[3],
                "detected_company": row[4],
                "final_response": row[5],
                "status": row[6],
            }
        return None

    def get_messages(self, thread_id: str) -> list:
        """Get all messages for a conversation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content, agent, timestamp FROM conversation_messages WHERE thread_id = ? ORDER BY id",
            (thread_id,)
        )
        return [
            {"role": row[0], "content": row[1], "agent": row[2], "timestamp": row[3]}
            for row in cursor.fetchall()
        ]

    def list_conversations(self, limit: int = 50) -> list:
        """List recent conversations."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT thread_id, created_at, user_query, detected_company, status FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        )
        return [
            {
                "thread_id": row[0],
                "created_at": row[1],
                "user_query": row[2],
                "detected_company": row[3],
                "status": row[4],
            }
            for row in cursor.fetchall()
        ]

    def delete_conversation(self, thread_id: str):
        """Delete a conversation and its messages."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM conversation_messages WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()
