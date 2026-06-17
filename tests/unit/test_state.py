"""Unit tests for state.py - SQLiteStateManager and PipelineRunState."""

import pytest
from datetime import datetime

from relais.state import SQLiteStateManager, PipelineRunState


class TestPipelineRunState:
    """Tests for PipelineRunState dataclass."""

    def test_create_pipeline_run_state(self):
        now = datetime.now()
        state = PipelineRunState(
            id="test-id-123",
            pipeline_name="my_pipeline",
            current_step="step1",
            status="running",
            args={"key": "value"},
            step_results={"step0": {"result": "done"}},
            created_at=now,
            updated_at=now,
        )
        assert state.id == "test-id-123"
        assert state.pipeline_name == "my_pipeline"
        assert state.current_step == "step1"
        assert state.status == "running"
        assert state.args == {"key": "value"}
        assert state.step_results == {"step0": {"result": "done"}}

    def test_pipeline_run_state_equality(self):
        now = datetime.now()
        state1 = PipelineRunState(
            id="id", pipeline_name="p", current_step="s", status="running",
            args={}, step_results={}, created_at=now, updated_at=now,
        )
        state2 = PipelineRunState(
            id="id", pipeline_name="p", current_step="s", status="running",
            args={}, step_results={}, created_at=now, updated_at=now,
        )
        assert state1 == state2


class TestSQLiteStateManagerCreate:
    def test_create_with_path(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        assert manager.db_path == db_path

    def test_create_with_default_path(self):
        manager = SQLiteStateManager.create()
        assert manager.db_path == "./pipeline.db"


class TestInitializeSchema:
    def test_initialize_schema_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        manager.initialize_schema()

        conn = manager._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_initialize_schema_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        manager = SQLiteStateManager.create(db_path)
        manager.initialize_schema()
        manager.initialize_schema()  # Should not raise


@pytest.fixture
def state_manager(tmp_path):
    manager = SQLiteStateManager.create(str(tmp_path / "test.db"))
    manager.initialize_schema()
    return manager


class TestCreatePipelineRun:
    def test_create_pipeline_run_returns_uuid(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step1")
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID

    def test_create_pipeline_run_with_args(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step1", args={"k": "v"})
        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {"k": "v"}

    def test_create_pipeline_run_without_args(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step1")
        state = state_manager.get_pipeline_run(run_id)
        assert state.args == {}


class TestGetPipelineRun:
    def test_get_pipeline_run_found(self, state_manager):
        run_id = state_manager.create_pipeline_run("my_pipeline", "start")
        state = state_manager.get_pipeline_run(run_id)
        assert state is not None
        assert state.id == run_id
        assert state.pipeline_name == "my_pipeline"
        assert state.current_step == "start"
        assert state.status == "running"

    def test_get_pipeline_run_not_found(self, state_manager):
        assert state_manager.get_pipeline_run("nonexistent") is None


class TestUpdatePipelineStep:
    def test_update_pipeline_step_basic(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step1")

        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            step_result={"output": "result"},
        )

        state = state_manager.get_pipeline_run(run_id)
        assert state.current_step == "step2"
        assert "step1" in state.step_results
        assert state.step_results["step1"]["output"] == "result"

    def test_update_pipeline_step_accumulates_results(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step1")
        state_manager.update_pipeline_step(run_id, "step2", step_result={"data": "from_step1"})
        state_manager.update_pipeline_step(run_id, "step3", step_result={"data": "from_step2"})

        state = state_manager.get_pipeline_run(run_id)
        assert "step1" in state.step_results
        assert "step2" in state.step_results


class TestCompletePipeline:
    def test_complete_pipeline_success(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step")
        state_manager.complete_pipeline(run_id, status="completed")
        assert state_manager.get_pipeline_run(run_id).status == "completed"

    def test_complete_pipeline_failed(self, state_manager):
        run_id = state_manager.create_pipeline_run("pipeline", "step")
        state_manager.complete_pipeline(run_id, status="failed")
        assert state_manager.get_pipeline_run(run_id).status == "failed"


class TestGetPipelineRuns:
    def test_get_pipeline_runs_no_filters(self, state_manager):
        state_manager.create_pipeline_run("p1", "s")
        state_manager.create_pipeline_run("p2", "s")
        assert len(state_manager.get_pipeline_runs()) == 2

    def test_get_pipeline_runs_filter_by_name(self, state_manager):
        state_manager.create_pipeline_run("alpha", "s")
        state_manager.create_pipeline_run("beta", "s")
        runs = state_manager.get_pipeline_runs(pipeline_name="alpha")
        assert len(runs) == 1
        assert runs[0].pipeline_name == "alpha"

    def test_get_pipeline_runs_filter_by_status(self, state_manager):
        r1 = state_manager.create_pipeline_run("p", "s")
        state_manager.create_pipeline_run("p", "s")
        state_manager.complete_pipeline(r1)
        completed = state_manager.get_pipeline_runs(status="completed")
        assert len(completed) == 1
        assert completed[0].id == r1

    def test_get_pipeline_runs_with_limit(self, state_manager):
        for _ in range(5):
            state_manager.create_pipeline_run("p", "s")
        assert len(state_manager.get_pipeline_runs(limit=3)) == 3


class TestDeletePipelineRun:
    def test_delete_pipeline_run(self, state_manager):
        run_id = state_manager.create_pipeline_run("p", "s")
        state_manager.delete_pipeline_run(run_id)
        assert state_manager.get_pipeline_run(run_id) is None


class TestSpecialCases:
    def test_special_characters_in_args(self, state_manager):
        args = {"text": "quotes \" and ' and \n newline", "emoji": "🎉"}
        run_id = state_manager.create_pipeline_run("p", "s", args=args)
        state = state_manager.get_pipeline_run(run_id)
        assert state.args == args

    def test_nested_json_in_step_results(self, state_manager):
        run_id = state_manager.create_pipeline_run("p", "s")
        nested = {"a": {"b": {"c": [1, 2, {"d": "deep"}]}}}
        state_manager.update_pipeline_step(run_id, "s2", step_result=nested)
        state = state_manager.get_pipeline_run(run_id)
        assert state.step_results["s"] == nested
