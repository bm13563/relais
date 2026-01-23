"""Integration tests for SQLiteStateManager with real SQLite database.

These tests verify SQLite functionality with a real database file.
They focus on scenarios that benefit from integration-level testing.

Run with: pytest tests/integration/ -m integration
"""

import pytest
import os
import uuid
from datetime import datetime

from relais.state import SQLiteStateManager, PipelineRunState

# Mark all tests as integration
pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def state_manager(tmp_path):
    """Create and initialize state manager for each test."""
    db_path = str(tmp_path / f"integration_test_{uuid.uuid4().hex[:8]}.db")
    manager = SQLiteStateManager.create(db_path)
    manager.initialize_schema()
    return manager


class TestSQLiteStateManagerIntegration:
    """Integration tests for SQLite state management."""

    def test_create_and_get_pipeline_run(self, state_manager):
        """Test creating and retrieving a pipeline run."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_create_get",
            start_step="step1",
            args={"key": "value"}
        )

        state = state_manager.get_pipeline_run(run_id)

        assert state is not None
        assert state.id == run_id
        assert state.pipeline_name == "test_create_get"
        assert state.current_step == "step1"
        assert state.status == "running"
        assert state.args == {"key": "value"}

    def test_update_pipeline_step(self, state_manager):
        """Test updating pipeline step."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_update_step",
            start_step="step1"
        )

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            conversation_history=[{"role": "user", "content": "test"}],
            step_result={"output": "result1"}
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.current_step == "step2"
        assert "step1" in state.step_results
        assert state.step_results["step1"]["output"] == "result1"

    def test_update_args_merges(self, state_manager):
        """Test that updating args merges with existing."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_update_args",
            start_step="s",
            args={"a": 1, "b": 2}
        )

        state_manager.update_args(run_id, {"b": 20, "c": 3})

        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {"a": 1, "b": 20, "c": 3}

    def test_complete_pipeline(self, state_manager):
        """Test marking pipeline as completed."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_complete",
            start_step="s"
        )

        state_manager.complete_pipeline(run_id, status="completed")

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "completed"

    def test_complete_pipeline_failed(self, state_manager):
        """Test marking pipeline as failed."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_failed",
            start_step="s"
        )

        state_manager.complete_pipeline(run_id, status="failed")

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "failed"

    def test_pause_and_resume_pipeline(self, state_manager):
        """Test pausing and resuming pipeline."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_pause_resume",
            start_step="s"
        )

        state_manager.pause_pipeline(run_id)
        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "paused"

        state_manager.resume_pipeline(run_id)
        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "running"

    def test_subagent_logging(self, state_manager):
        """Test logging subagent spawn and completion."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_subagent",
            start_step="s"
        )

        subagent_id = str(uuid.uuid4())
        state_manager.log_subagent_spawn(
            parent_pipeline_id=run_id,
            subagent_id=subagent_id,
            step_name="research"
        )

        state_manager.log_subagent_complete(
            subagent_id=subagent_id,
            result={"findings": ["a", "b"]},
            turns_used=3
        )

        # The test passes if no exceptions are raised

    def test_get_pipeline_runs_filtering(self, state_manager):
        """Test querying runs with filters."""
        # Create runs with different statuses
        for i, status in enumerate(["completed", "failed", "running"]):
            run_id = state_manager.create_pipeline_run(
                pipeline_name=f"test_filter_{i}",
                start_step="s"
            )
            if status != "running":
                state_manager.complete_pipeline(run_id, status=status)

        # Filter by status
        completed = state_manager.get_pipeline_runs(status="completed")
        running = state_manager.get_pipeline_runs(status="running")

        assert any(r.status == "completed" for r in completed)
        assert any(r.status == "running" for r in running)

    def test_delete_pipeline_run(self, state_manager):
        """Test deleting a pipeline run."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_delete",
            start_step="s"
        )

        # Log a subagent to ensure cascade delete
        subagent_id = str(uuid.uuid4())
        state_manager.log_subagent_spawn(run_id, subagent_id, "step")

        state_manager.delete_pipeline_run(run_id)

        state = state_manager.get_pipeline_run(run_id)
        assert state is None

    def test_step_results_accumulate(self, state_manager):
        """Test that step results accumulate across updates."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_accumulate",
            start_step="step1"
        )

        # Complete step1, move to step2
        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            conversation_history=[],
            step_result={"data": "from_step1"}
        )

        # Complete step2, move to step3
        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step3",
            conversation_history=[],
            step_result={"data": "from_step2"}
        )

        state = state_manager.get_pipeline_run(run_id)
        assert "step1" in state.step_results
        assert "step2" in state.step_results
        assert state.step_results["step1"]["data"] == "from_step1"
        assert state.step_results["step2"]["data"] == "from_step2"

    def test_large_conversation_history(self, state_manager):
        """Test handling large conversation history."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_large_history",
            start_step="s"
        )

        # Create large conversation history
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}" * 100}
            for i in range(100)
        ]

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="s",
            conversation_history=history
        )

        state = state_manager.get_pipeline_run(run_id)
        assert len(state.conversation_history) == 100

    def test_special_characters_in_args(self, state_manager):
        """Test handling special characters in args."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_special_chars",
            start_step="s",
            args={
                "unicode": "日本語 한국어",
                "quotes": 'He said "Hello"',
                "newlines": "line1\nline2\nline3",
                "backslash": "path\\to\\file"
            }
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.args["unicode"] == "日本語 한국어"
        assert state.args["quotes"] == 'He said "Hello"'
        assert state.args["newlines"] == "line1\nline2\nline3"
        assert state.args["backslash"] == "path\\to\\file"

    def test_concurrent_access_same_db(self, tmp_path):
        """Test that multiple managers can access the same database."""
        db_path = str(tmp_path / "shared.db")

        manager1 = SQLiteStateManager.create(db_path)
        manager1.initialize_schema()

        manager2 = SQLiteStateManager.create(db_path)
        # Schema already exists, should work fine

        # Create run with manager1
        run_id = manager1.create_pipeline_run(
            pipeline_name="test_shared",
            start_step="s"
        )

        # Read with manager2
        state = manager2.get_pipeline_run(run_id)
        assert state is not None
        assert state.pipeline_name == "test_shared"

    def test_database_persistence(self, tmp_path):
        """Test that data persists across manager instances."""
        db_path = str(tmp_path / "persistent.db")

        # Create and populate
        manager1 = SQLiteStateManager.create(db_path)
        manager1.initialize_schema()
        run_id = manager1.create_pipeline_run(
            pipeline_name="test_persist",
            start_step="s",
            args={"key": "value"}
        )
        manager1.update_pipeline_step(
            run_id=run_id,
            current_step="s2",
            conversation_history=[{"msg": "test"}],
            step_result={"result": "data"}
        )

        # Create new manager instance
        manager2 = SQLiteStateManager.create(db_path)

        # Data should persist
        state = manager2.get_pipeline_run(run_id)
        assert state is not None
        assert state.args == {"key": "value"}
        assert state.current_step == "s2"
        assert "s" in state.step_results
