"""Integration tests for SQLiteStateManager against a real SQLite database.

These exercise real round-trips, multi-manager access, and on-disk persistence.

Run with: pytest tests/integration/ -m integration
"""

import pytest
import uuid

from relais.state import SQLiteStateManager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def state_manager(tmp_path):
    db_path = str(tmp_path / f"integration_test_{uuid.uuid4().hex[:8]}.db")
    manager = SQLiteStateManager.create(db_path)
    manager.initialize_schema()
    return manager


class TestSQLiteStateManagerIntegration:

    def test_create_and_get_pipeline_run(self, state_manager):
        run_id = state_manager.create_pipeline_run(
            pipeline_name="test_create", start_step="start", args={"a": 1}
        )
        state = state_manager.get_pipeline_run(run_id)
        assert state is not None
        assert state.pipeline_name == "test_create"
        assert state.current_step == "start"
        assert state.status == "running"
        assert state.args == {"a": 1}

    def test_update_pipeline_step(self, state_manager):
        run_id = state_manager.create_pipeline_run(pipeline_name="test_update", start_step="step1")
        state_manager.update_pipeline_step(
            run_id=run_id,
            current_step="step2",
            step_result={"output": "result1"},
        )
        state = state_manager.get_pipeline_run(run_id)
        assert state.current_step == "step2"
        assert state.step_results["step1"]["output"] == "result1"

    def test_complete_pipeline(self, state_manager):
        run_id = state_manager.create_pipeline_run(pipeline_name="test_complete", start_step="s")
        state_manager.complete_pipeline(run_id, status="completed")
        assert state_manager.get_pipeline_run(run_id).status == "completed"

    def test_complete_pipeline_failed(self, state_manager):
        run_id = state_manager.create_pipeline_run(pipeline_name="test_fail", start_step="s")
        state_manager.complete_pipeline(run_id, status="failed")
        assert state_manager.get_pipeline_run(run_id).status == "failed"

    def test_get_pipeline_runs_filtering(self, state_manager):
        a = state_manager.create_pipeline_run("alpha", "s")
        state_manager.create_pipeline_run("beta", "s")
        state_manager.complete_pipeline(a)

        assert len(state_manager.get_pipeline_runs(pipeline_name="alpha")) == 1
        completed = state_manager.get_pipeline_runs(status="completed")
        assert len(completed) == 1 and completed[0].id == a

    def test_delete_pipeline_run(self, state_manager):
        run_id = state_manager.create_pipeline_run(pipeline_name="test_delete", start_step="s")
        state_manager.delete_pipeline_run(run_id)
        assert state_manager.get_pipeline_run(run_id) is None

    def test_step_results_accumulate(self, state_manager):
        run_id = state_manager.create_pipeline_run(pipeline_name="accum", start_step="step1")
        state_manager.update_pipeline_step(run_id, "step2", step_result={"data": "from_step1"})
        state_manager.update_pipeline_step(run_id, "step3", step_result={"data": "from_step2"})
        state = state_manager.get_pipeline_run(run_id)
        assert "step1" in state.step_results
        assert "step2" in state.step_results

    def test_special_characters_in_args(self, state_manager):
        args = {"text": "quotes \" and ' and \n newline", "emoji": "🎉"}
        run_id = state_manager.create_pipeline_run("special", "s", args=args)
        assert state_manager.get_pipeline_run(run_id).args == args

    def test_concurrent_access_same_db(self, tmp_path):
        db_path = str(tmp_path / "shared.db")
        manager1 = SQLiteStateManager.create(db_path)
        manager1.initialize_schema()
        manager2 = SQLiteStateManager.create(db_path)

        run_id = manager1.create_pipeline_run(pipeline_name="test_shared", start_step="s")
        state = manager2.get_pipeline_run(run_id)
        assert state is not None
        assert state.pipeline_name == "test_shared"

    def test_database_persistence(self, tmp_path):
        db_path = str(tmp_path / "persistent.db")
        manager1 = SQLiteStateManager.create(db_path)
        manager1.initialize_schema()
        run_id = manager1.create_pipeline_run(
            pipeline_name="test_persist", start_step="s", args={"key": "value"}
        )
        manager1.update_pipeline_step(run_id, "s2", step_result={"result": "data"})

        manager2 = SQLiteStateManager.create(db_path)
        state = manager2.get_pipeline_run(run_id)
        assert state is not None
        assert state.args == {"key": "value"}
        assert state.current_step == "s2"
        assert "s" in state.step_results
