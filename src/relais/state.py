"""SQLite persistence for pipeline runs.

Records each run and its per-step results so they can be inspected after the
fact (get_run / list_runs). Runs execute start-to-finish in one process; there
is no pause/resume. Step-level detail is logged to spool, not stored here.
"""

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
    """The recorded state of a pipeline run.

    Attributes:
        id: Unique identifier for the run
        pipeline_name: Name of the pipeline definition
        current_step: Name of the current (or final) step
        status: 'running', 'completed', or 'failed'
        args: Pipeline arguments
        step_results: Results from completed steps, keyed by step name
        created_at: When the run started
        updated_at: Last update time
    """
    id: str
    pipeline_name: str
    current_step: str
    status: str
    args: Dict[str, Any]
    step_results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SQLiteStateManager:
    """Persists pipeline runs to a local SQLite file (zero-config)."""

    SCHEMA_SQL = '''
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id TEXT PRIMARY KEY,
        pipeline_name TEXT NOT NULL,
        current_step TEXT NOT NULL,
        status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed')),
        args TEXT,
        step_results TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_pipeline_name ON pipeline_runs(pipeline_name);
    CREATE INDEX IF NOT EXISTS idx_status ON pipeline_runs(status);
    CREATE INDEX IF NOT EXISTS idx_created_at ON pipeline_runs(created_at);
    '''

    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def create(cls, db_path: str = "./pipeline.db") -> SQLiteStateManager:
        return cls(db_path)

    def initialize_schema(self) -> None:
        """Create the pipeline_runs table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _row_to_state(self, row: sqlite3.Row) -> PipelineRunState:
        return PipelineRunState(
            id=row['id'],
            pipeline_name=row['pipeline_name'],
            current_step=row['current_step'],
            status=row['status'],
            args=json.loads(row['args']) if row['args'] else {},
            step_results=json.loads(row['step_results']) if row['step_results'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def create_pipeline_run(
        self,
        pipeline_name: str,
        start_step: str,
        args: dict = None,
    ) -> str:
        """Create a new pipeline run and return its UUID."""
        run_id = str(uuid.uuid4())
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO pipeline_runs (id, pipeline_name, current_step, args, step_results)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, pipeline_name, start_step, json.dumps(args or {}), json.dumps({})))
            conn.commit()
        finally:
            conn.close()
        return run_id

    def get_pipeline_run(self, run_id: str) -> Optional[PipelineRunState]:
        """Load a run's state, or None if not found."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            return self._row_to_state(row) if row else None
        finally:
            conn.close()

    def update_pipeline_step(
        self,
        run_id: str,
        current_step: str,
        step_result: dict = None,
    ) -> None:
        """Advance a run to current_step and record the completed step's result."""
        conn = self._get_connection()
        try:
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
                SET current_step = ?, step_results = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_step, json.dumps(step_results), run_id))
            conn.commit()
        finally:
            conn.close()

    def complete_pipeline(self, run_id: str, status: str = 'completed') -> None:
        """Mark a run as completed or failed."""
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

    def get_pipeline_runs(
        self,
        pipeline_name: str = None,
        status: str = None,
        limit: int = 100,
    ) -> List[PipelineRunState]:
        """Query runs with optional filters, newest first."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM pipeline_runs WHERE 1=1"
            params: list = []
            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [self._row_to_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_pipeline_run(self, run_id: str) -> None:
        """Delete a run."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM pipeline_runs WHERE id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()
