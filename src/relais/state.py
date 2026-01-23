"""SQLite state management for pipeline persistence."""

from __future__ import annotations
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class PipelineRunState:
    """Represents the state of a pipeline run.

    Attributes:
        id: Unique identifier for the run
        pipeline_name: Name of the pipeline definition
        current_step: Name of the current step
        status: Current status (running, completed, failed, paused)
        args: Pipeline arguments
        conversation_history: Message history for main agent
        step_results: Results from completed steps
        created_at: When the run started
        updated_at: Last update time
    """
    id: str
    pipeline_name: str
    current_step: str
    status: str
    args: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    step_results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SQLiteStateManager:
    """Manages pipeline state persistence in SQLite.

    Uses a local SQLite database file for simple, zero-config persistence.

    Usage:
        state_manager = SQLiteStateManager.create("./pipeline.db")

        run_id = state_manager.create_pipeline_run("my_pipeline", "start_step")
        state_manager.update_pipeline_step(run_id, "next_step", messages, result)
        state_manager.complete_pipeline(run_id)
    """

    SCHEMA_SQL = '''
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id TEXT PRIMARY KEY,
        pipeline_name TEXT NOT NULL,
        current_step TEXT NOT NULL,
        status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed', 'paused')),
        args TEXT,
        conversation_history TEXT,
        step_results TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_pipeline_name ON pipeline_runs(pipeline_name);
    CREATE INDEX IF NOT EXISTS idx_status ON pipeline_runs(status);
    CREATE INDEX IF NOT EXISTS idx_created_at ON pipeline_runs(created_at);

    CREATE TABLE IF NOT EXISTS subagent_logs (
        id TEXT PRIMARY KEY,
        parent_pipeline_id TEXT NOT NULL,
        step_name TEXT NOT NULL,
        status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed')),
        conversation_history TEXT,
        result TEXT,
        turns_used INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_parent ON subagent_logs(parent_pipeline_id);
    '''

    def __init__(self, db_path: str):
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def create(cls, db_path: str = "./pipeline.db") -> SQLiteStateManager:
        """Create a state manager.

        Args:
            db_path: Path to SQLite database file

        Returns:
            Configured SQLiteStateManager instance
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

    def create_pipeline_run(
        self,
        pipeline_name: str,
        start_step: str,
        args: dict = None
    ) -> str:
        """Create a new pipeline run.

        Args:
            pipeline_name: Name of the pipeline definition
            start_step: Initial step name
            args: Pipeline arguments

        Returns:
            UUID of the created run
        """
        run_id = str(uuid.uuid4())

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO pipeline_runs
                (id, pipeline_name, current_step, args, conversation_history, step_results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                pipeline_name,
                start_step,
                json.dumps(args or {}),
                json.dumps([]),
                json.dumps({})
            ))
            conn.commit()
        finally:
            conn.close()

        return run_id

    def get_pipeline_run(self, run_id: str) -> Optional[PipelineRunState]:
        """Load a pipeline run state.

        Args:
            run_id: UUID of the run

        Returns:
            PipelineRunState or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM pipeline_runs WHERE id = ?",
                (run_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return PipelineRunState(
                id=row['id'],
                pipeline_name=row['pipeline_name'],
                current_step=row['current_step'],
                status=row['status'],
                args=json.loads(row['args']) if row['args'] else {},
                conversation_history=json.loads(row['conversation_history']) if row['conversation_history'] else [],
                step_results=json.loads(row['step_results']) if row['step_results'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        finally:
            conn.close()

    def update_pipeline_step(
        self,
        run_id: str,
        current_step: str,
        conversation_history: List[dict],
        step_result: dict = None
    ) -> None:
        """Update pipeline state after step completion.

        Args:
            run_id: UUID of the run
            current_step: Name of the new current step
            conversation_history: Updated message history
            step_result: Result data from the completed step
        """
        conn = self._get_connection()
        try:
            # Get current step_results to merge
            cursor = conn.execute(
                "SELECT step_results, current_step FROM pipeline_runs WHERE id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            step_results = json.loads(row['step_results']) if row and row['step_results'] else {}
            previous_step = row['current_step'] if row else None

            if step_result and previous_step:
                step_results[previous_step] = step_result

            conn.execute("""
                UPDATE pipeline_runs
                SET current_step = ?, conversation_history = ?, step_results = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                current_step,
                json.dumps(conversation_history),
                json.dumps(step_results),
                run_id
            ))
            conn.commit()
        finally:
            conn.close()

    def update_args(self, run_id: str, args: dict) -> None:
        """Update pipeline arguments.

        Args:
            run_id: UUID of the run
            args: New arguments to merge
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT args FROM pipeline_runs WHERE id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            current_args = json.loads(row['args']) if row and row['args'] else {}
            current_args.update(args)

            conn.execute(
                "UPDATE pipeline_runs SET args = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(current_args), run_id)
            )
            conn.commit()
        finally:
            conn.close()

    def complete_pipeline(self, run_id: str, status: str = 'completed') -> None:
        """Mark a pipeline as completed.

        Args:
            run_id: UUID of the run
            status: Final status (completed, failed)
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE pipeline_runs
                SET status = ?, completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, run_id))
            conn.commit()
        finally:
            conn.close()

    def pause_pipeline(self, run_id: str) -> None:
        """Pause a running pipeline.

        Args:
            run_id: UUID of the run
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE pipeline_runs SET status = 'paused', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,)
            )
            conn.commit()
        finally:
            conn.close()

    def resume_pipeline(self, run_id: str) -> None:
        """Resume a paused pipeline.

        Args:
            run_id: UUID of the run
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE pipeline_runs SET status = 'running', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,)
            )
            conn.commit()
        finally:
            conn.close()

    def log_subagent_spawn(
        self,
        parent_pipeline_id: str,
        subagent_id: str,
        step_name: str
    ) -> None:
        """Log a subagent spawn for auditing.

        Args:
            parent_pipeline_id: UUID of the parent pipeline run
            subagent_id: UUID for the subagent
            step_name: Name of the step being executed
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO subagent_logs
                (id, parent_pipeline_id, step_name, conversation_history)
                VALUES (?, ?, ?, ?)
            """, (subagent_id, parent_pipeline_id, step_name, json.dumps([])))
            conn.commit()
        finally:
            conn.close()

    def log_subagent_complete(
        self,
        subagent_id: str,
        result: dict,
        turns_used: int
    ) -> None:
        """Log subagent completion.

        Args:
            subagent_id: UUID of the subagent
            result: Execution result data
            turns_used: Number of API turns used
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE subagent_logs
                SET status = 'completed',
                    result = ?,
                    turns_used = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(result), turns_used, subagent_id))
            conn.commit()
        finally:
            conn.close()

    def get_pipeline_runs(
        self,
        pipeline_name: str = None,
        status: str = None,
        limit: int = 100
    ) -> List[PipelineRunState]:
        """Query pipeline runs with optional filters.

        Args:
            pipeline_name: Filter by pipeline name
            status: Filter by status
            limit: Maximum results to return

        Returns:
            List of matching pipeline run states
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM pipeline_runs WHERE 1=1"
            params = []

            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                PipelineRunState(
                    id=row['id'],
                    pipeline_name=row['pipeline_name'],
                    current_step=row['current_step'],
                    status=row['status'],
                    args=json.loads(row['args']) if row['args'] else {},
                    conversation_history=json.loads(row['conversation_history']) if row['conversation_history'] else [],
                    step_results=json.loads(row['step_results']) if row['step_results'] else {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]
        finally:
            conn.close()

    def delete_pipeline_run(self, run_id: str) -> None:
        """Delete a pipeline run and its associated data.

        Args:
            run_id: UUID of the run to delete
        """
        conn = self._get_connection()
        try:
            # Delete subagent logs first
            conn.execute(
                "DELETE FROM subagent_logs WHERE parent_pipeline_id = ?",
                (run_id,)
            )
            conn.execute(
                "DELETE FROM pipeline_runs WHERE id = ?",
                (run_id,)
            )
            conn.commit()
        finally:
            conn.close()
