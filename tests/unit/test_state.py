"""Unit tests for state.py - SQLiteStateManager class."""

import pytest
import json
from datetime import datetime

from relais.state import SQLiteStateManager, PipelineRunState


class TestPipelineRunState:
    """Tests for PipelineRunState dataclass."""

    def test_create_pipeline_run_state(self):
        """Test creating a PipelineRunState."""
        now = datetime.now()
        state = PipelineRunState(
            id="test-id-123",
            pipeline_name="my_pipeline",
            current_step="step1",
            status="running",
            args={"key": "value"},
            conversation_history=[{"role": "user", "content": "hi"}],
            step_results={"step0": {"result": "done"}},
            created_at=now,
            updated_at=now
        )
        assert state.id == "test-id-123"
        assert state.pipeline_name == "my_pipeline"
        assert state.current_step == "step1"
        assert state.status == "running"
        assert state.args == {"key": "value"}
        assert state.conversation_history == [{"role": "user", "content": "hi"}]
        assert state.step_results == {"step0": {"result": "done"}}

    def test_pipeline_run_state_equality(self):
        """Test PipelineRunState equality."""
        now = datetime.now()
        state1 = PipelineRunState(
            id="id", pipeline_name="p", current_step="s",
            status="running", args={}, conversation_history=[],
            step_results={}, created_at=now, updated_at=now
        )
        state2 = PipelineRunState(
            id="id", pipeline_name="p", current_step="s",
            status="running", args={}, conversation_history=[],
            step_results={}, created_at=now, updated_at=now
        )
        assert state1 == state2


class TestSQLiteStateManagerCreate:
    """Tests for SQLiteStateManager.create class method."""

    def test_create_with_path(self, tmp_path):
        """Test creating manager with path."""
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        assert manager.db_path == db_path

    def test_create_with_default_path(self):
        """Test creating manager with default path."""
        manager = SQLiteStateManager.create()
        assert manager.db_path == "./pipeline.db"


class TestInitializeSchema:
    """Tests for initialize_schema method."""

    def test_initialize_schema_creates_tables(self, tmp_path):
        """Test that schema creates required tables."""
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        manager.initialize_schema()

        # Verify tables exist by trying to query them
        conn = manager._get_connection()
        try:
            conn.execute("SELECT * FROM pipeline_runs LIMIT 1")
            conn.execute("SELECT * FROM subagent_logs LIMIT 1")
        finally:
            conn.close()

    def test_initialize_schema_idempotent(self, tmp_path):
        """Test that initialize_schema can be called multiple times."""
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        manager.initialize_schema()
        manager.initialize_schema()  # Should not raise


@pytest.fixture
def state_manager(tmp_path):
    """Create a state manager with initialized schema."""
    db_path = str(tmp_path / "test.db")
    manager = SQLiteStateManager.create(db_path)
    manager.initialize_schema()
    return manager


class TestCreatePipelineRun:
    """Tests for create_pipeline_run method."""

    def test_create_pipeline_run_returns_uuid(self, state_manager):
        """Test that create_pipeline_run returns a UUID."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_pipeline",
            start_step="step1"
        )
        # Should be a valid UUID format
        assert len(run_id) == 36
        assert run_id.count("-") == 4

    def test_create_pipeline_run_with_args(self, state_manager):
        """Test creating run with arguments."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="my_pipeline",
            start_step="initial",
            args={"user": "test", "mode": "debug"}
        )
        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {"user": "test", "mode": "debug"}

    def test_create_pipeline_run_without_args(self, state_manager):
        """Test creating run without arguments."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")
        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {}


class TestGetPipelineRun:
    """Tests for get_pipeline_run method."""

    def test_get_pipeline_run_found(self, state_manager):
        """Test getting an existing pipeline run."""
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_pipeline",
            start_step="step1",
            args={"key": "value"}
        )
        state = state_manager.get_pipeline_run(run_id)

        assert state is not None
        assert state.id == run_id
        assert state.pipeline_name == "test_pipeline"
        assert state.current_step == "step1"
        assert state.status == "running"
        assert state.args == {"key": "value"}

    def test_get_pipeline_run_not_found(self, state_manager):
        """Test getting a non-existent pipeline run."""
        state = state_manager.get_pipeline_run("nonexistent-id")
        assert state is None


class TestUpdatePipelineStep:
    """Tests for update_pipeline_step method."""

    def test_update_pipeline_step_basic(self, state_manager):
        """Test basic step update."""
        run_id = state_manager.create_pipeline_run("pipeline", "step1")

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            conversation_history=[{"role": "assistant", "content": "done"}],
            step_result={"output": "result"}
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.current_step == "step2"
        assert "step1" in state.step_results
        assert state.step_results["step1"]["output"] == "result"

    def test_update_pipeline_step_accumulates_results(self, state_manager):
        """Test that step results accumulate."""
        run_id = state_manager.create_pipeline_run("pipeline", "step1")

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            conversation_history=[],
            step_result={"data": "from_step1"}
        )

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step3",
            conversation_history=[],
            step_result={"data": "from_step2"}
        )

        state = state_manager.get_pipeline_run(run_id)
        assert "step1" in state.step_results
        assert "step2" in state.step_results


class TestUpdateArgs:
    """Tests for update_args method."""

    def test_update_args_merges(self, state_manager):
        """Test that args are merged, not replaced."""
        run_id = state_manager.create_pipeline_run(
            "pipeline", "step", args={"a": 1, "b": 2}
        )

        state_manager.update_args(run_id, {"b": 20, "c": 3})

        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {"a": 1, "b": 20, "c": 3}


class TestCompletePipeline:
    """Tests for complete_pipeline method."""

    def test_complete_pipeline_success(self, state_manager):
        """Test marking pipeline as completed."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        state_manager.complete_pipeline(run_id, status="completed")

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "completed"

    def test_complete_pipeline_failed(self, state_manager):
        """Test marking pipeline as failed."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        state_manager.complete_pipeline(run_id, status="failed")

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "failed"


class TestPausePipeline:
    """Tests for pause_pipeline method."""

    def test_pause_pipeline(self, state_manager):
        """Test pausing a pipeline."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        state_manager.pause_pipeline(run_id)

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "paused"


class TestResumePipeline:
    """Tests for resume_pipeline method."""

    def test_resume_pipeline(self, state_manager):
        """Test resuming a pipeline."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")
        state_manager.pause_pipeline(run_id)

        state_manager.resume_pipeline(run_id)

        state = state_manager.get_pipeline_run(run_id)
        assert state.status == "running"


class TestSubagentLogging:
    """Tests for subagent logging methods."""

    def test_log_subagent_spawn_and_complete(self, state_manager):
        """Test logging subagent spawn and completion."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        # Log spawn
        state_manager.log_subagent_spawn(
            parent_pipeline_id=run_id,
            subagent_id="sub-123",
            step_name="research"
        )

        # Log completion
        state_manager.log_subagent_complete(
            subagent_id="sub-123",
            result={"findings": ["a", "b"]},
            turns_used=3
        )

        # Verify by direct query
        conn = state_manager._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM subagent_logs WHERE id = ?",
                ("sub-123",)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["status"] == "completed"
            assert row["turns_used"] == 3
        finally:
            conn.close()


class TestGetPipelineRuns:
    """Tests for get_pipeline_runs query method."""

    def test_get_pipeline_runs_no_filters(self, state_manager):
        """Test getting runs without filters."""
        state_manager.create_pipeline_run("p1", "s")
        state_manager.create_pipeline_run("p2", "s")

        runs = state_manager.get_pipeline_runs()
        assert len(runs) == 2

    def test_get_pipeline_runs_filter_by_name(self, state_manager):
        """Test filtering by pipeline name."""
        state_manager.create_pipeline_run("target", "s")
        state_manager.create_pipeline_run("other", "s")

        runs = state_manager.get_pipeline_runs(pipeline_name="target")
        assert len(runs) == 1
        assert runs[0].pipeline_name == "target"

    def test_get_pipeline_runs_filter_by_status(self, state_manager):
        """Test filtering by status."""
        run1 = state_manager.create_pipeline_run("p1", "s")
        run2 = state_manager.create_pipeline_run("p2", "s")
        state_manager.complete_pipeline(run1)

        completed = state_manager.get_pipeline_runs(status="completed")
        running = state_manager.get_pipeline_runs(status="running")

        assert len(completed) == 1
        assert len(running) == 1

    def test_get_pipeline_runs_with_limit(self, state_manager):
        """Test limiting results."""
        for i in range(10):
            state_manager.create_pipeline_run(f"p{i}", "s")

        runs = state_manager.get_pipeline_runs(limit=5)
        assert len(runs) == 5


class TestDeletePipelineRun:
    """Tests for delete_pipeline_run method."""

    def test_delete_pipeline_run(self, state_manager):
        """Test deleting a pipeline run."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        state_manager.delete_pipeline_run(run_id)

        state = state_manager.get_pipeline_run(run_id)
        assert state is None

    def test_delete_pipeline_run_cascades_subagents(self, state_manager):
        """Test that deleting run also deletes subagent logs."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")
        state_manager.log_subagent_spawn(run_id, "sub-123", "research")

        state_manager.delete_pipeline_run(run_id)

        # Verify subagent log is also deleted
        conn = state_manager._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM subagent_logs WHERE parent_pipeline_id = ?",
                (run_id,)
            )
            assert cursor.fetchone() is None
        finally:
            conn.close()


class TestSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_large_conversation_history(self, state_manager):
        """Test handling large conversation history."""
        run_id = state_manager.create_pipeline_run("pipeline", "step")

        # Create large conversation history
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}" * 100}
            for i in range(100)
        ]

        state_manager.update_pipeline_step(run_id, "step", history)

        state = state_manager.get_pipeline_run(run_id)
        assert len(state.conversation_history) == 100

    def test_special_characters_in_args(self, state_manager):
        """Test handling special characters in args."""
        run_id = state_manager.create_pipeline_run(
            "pipeline", "step",
            args={
                "unicode": "日本語 한국어 🎉",
                "quotes": 'He said "Hello"',
                "newlines": "line1\nline2\nline3",
                "backslash": "path\\to\\file"
            }
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.args["unicode"] == "日本語 한국어 🎉"
        assert state.args["quotes"] == 'He said "Hello"'
        assert state.args["newlines"] == "line1\nline2\nline3"

    def test_nested_json_in_step_results(self, state_manager):
        """Test deeply nested JSON in step results."""
        run_id = state_manager.create_pipeline_run("pipeline", "step1")

        state_manager.update_pipeline_step(
            run_id, "step2", [],
            step_result={
                "deeply": {
                    "nested": {
                        "structure": {
                            "with": ["arrays", "too"]
                        }
                    }
                }
            }
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.step_results["step1"]["deeply"]["nested"]["structure"]["with"] == ["arrays", "too"]
