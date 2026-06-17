"""SQLite state management for pipeline agents.

This module provides persistence for PipelineAgent instances, allowing them
to be saved and restored across pipeline runs.
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict

from .agent import PipelineAgent


class AgentStateManager:
    """Manages agent state persistence in SQLite.

    Stores agent configurations, conversation history, and lifecycle state
    in a separate database from pipeline runs.

    Usage:
        manager = AgentStateManager.create("./agents.db")
        manager.initialize_schema()

        agent = PipelineAgent(name="main_agent", steps=5)
        manager.save_agent("run-123", agent)

        restored = manager.load_agent("run-123", "main_agent")
    """

    SCHEMA_SQL = '''
    CREATE TABLE IF NOT EXISTS pipeline_agents (
        run_id TEXT NOT NULL,
        name TEXT NOT NULL,
        steps INTEGER,
        steps_remaining INTEGER,
        model TEXT,
        thinking INTEGER,
        conversation_history TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, name)
    );

    CREATE INDEX IF NOT EXISTS idx_run_id ON pipeline_agents(run_id);
    CREATE INDEX IF NOT EXISTS idx_agent_name ON pipeline_agents(name);
    '''

    def __init__(self, db_path: str):
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure parent directory exists (skip for in-memory databases)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def create(cls, db_path: str = "./agents.db") -> AgentStateManager:
        """Create an agent state manager.

        Args:
            db_path: Path to SQLite database file

        Returns:
            Configured AgentStateManager instance
        """
        return cls(db_path)

    def initialize_schema(self) -> None:
        """Create the required database tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def save_agent(self, run_id: str, agent: PipelineAgent) -> None:
        """Save or update an agent's state.

        Args:
            run_id: UUID of the pipeline run
            agent: PipelineAgent instance to save
        """
        conn = self._get_connection()
        try:
            # Convert boolean to integer for SQLite
            thinking_int = None if agent.thinking is None else (1 if agent.thinking else 0)

            conn.execute("""
                INSERT OR REPLACE INTO pipeline_agents
                (run_id, name, steps, steps_remaining, model, thinking, conversation_history, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                run_id,
                agent.name,
                agent.steps,
                agent.steps_remaining,
                agent.model,
                thinking_int,
                json.dumps(agent.conversation_history),
            ))
            conn.commit()
        finally:
            conn.close()

    def load_agent(self, run_id: str, agent_name: str) -> Optional[PipelineAgent]:
        """Load an agent's state.

        Args:
            run_id: UUID of the pipeline run
            agent_name: Name of the agent to load

        Returns:
            PipelineAgent instance or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM pipeline_agents
                WHERE run_id = ? AND name = ?
            """, (run_id, agent_name))
            row = cursor.fetchone()

            if not row:
                return None

            # Convert integer back to boolean
            thinking = None if row['thinking'] is None else bool(row['thinking'])

            agent = PipelineAgent(
                name=row['name'],
                steps=row['steps'],
                model=row['model'],
                thinking=thinking,
            )
            agent.steps_remaining = row['steps_remaining']
            agent.conversation_history = json.loads(row['conversation_history']) if row['conversation_history'] else []

            return agent
        finally:
            conn.close()

    def delete_agent(self, run_id: str, agent_name: str) -> None:
        """Delete an agent's state.

        Args:
            run_id: UUID of the pipeline run
            agent_name: Name of the agent to delete
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                DELETE FROM pipeline_agents
                WHERE run_id = ? AND name = ?
            """, (run_id, agent_name))
            conn.commit()
        finally:
            conn.close()

