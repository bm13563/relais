"""Unit tests for agent persistence on SQLiteStateManager.

Agent runtime state (conversation history, steps consumed) is stored in the main
pipeline database alongside run state, via save_agent / load_agent / delete_agent.
"""

import tempfile
from pathlib import Path

from relais.agent import PipelineAgent
from relais.state import SQLiteStateManager


def _manager(tmpdir):
    manager = SQLiteStateManager.create(str(Path(tmpdir) / "pipeline.db"))
    manager.initialize_schema()
    return manager


class TestAgentSchema:
    def test_schema_creates_pipeline_agents_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            conn = manager._get_connection()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_agents'"
            )
            assert cursor.fetchone() is not None
            conn.close()


class TestSaveAndLoadAgent:
    def test_save_agent_creates_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            agent = PipelineAgent(name="test_agent", steps=5, model="opus")
            manager.save_agent("run-123", agent)

            loaded = manager.load_agent("run-123", "test_agent")
            assert loaded is not None
            assert loaded.name == "test_agent"

    def test_load_nonexistent_agent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            assert manager.load_agent("run-123", "nonexistent") is None

    def test_save_and_load_agent_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            agent = PipelineAgent(name="full_agent", steps=10, model="opus", thinking=False)
            agent.add_messages([
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ])
            agent.consume_step()
            agent.consume_step()

            manager.save_agent("run-456", agent)
            loaded = manager.load_agent("run-456", "full_agent")

            assert loaded.name == agent.name
            assert loaded.steps == agent.steps
            assert loaded.steps_remaining == agent.steps_remaining
            assert loaded.model == agent.model
            assert loaded.thinking == agent.thinking
            assert loaded.conversation_history == agent.conversation_history

    def test_save_updates_existing_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            agent = PipelineAgent(name="updatable", steps=5)
            manager.save_agent("run-789", agent)

            agent.consume_step()
            agent.add_message({"role": "user", "content": "New message"})
            manager.save_agent("run-789", agent)

            loaded = manager.load_agent("run-789", "updatable")
            assert loaded.steps_remaining == 4
            assert len(loaded.conversation_history) == 1

    def test_agents_are_scoped_per_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.save_agent("run-A", PipelineAgent(name="shared", steps=3))
            manager.save_agent("run-B", PipelineAgent(name="shared", steps=7))

            assert manager.load_agent("run-A", "shared").steps == 3
            assert manager.load_agent("run-B", "shared").steps == 7


class TestDeleteAgent:
    def test_delete_agent_removes_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.save_agent("run-111", PipelineAgent(name="deletable"))
            assert manager.load_agent("run-111", "deletable") is not None

            manager.delete_agent("run-111", "deletable")
            assert manager.load_agent("run-111", "deletable") is None

    def test_delete_nonexistent_agent_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.delete_agent("run-999", "nonexistent")  # should not raise

    def test_delete_pipeline_run_also_deletes_its_agents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            run_id = manager.create_pipeline_run("p", "start")
            manager.save_agent(run_id, PipelineAgent(name="a1"))

            manager.delete_pipeline_run(run_id)
            assert manager.load_agent(run_id, "a1") is None
