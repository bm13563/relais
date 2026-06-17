"""Unit tests for agent_state.py - AgentStateManager class."""

import pytest
import tempfile
from pathlib import Path

from relais.agent import PipelineAgent
from relais.agent_state import AgentStateManager


class TestAgentStateManagerCreation:
    """Tests for AgentStateManager instantiation."""

    def test_create_with_db_path(self):
        """Test creating state manager with database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))

            assert manager.db_path == str(db_path)

    def test_create_initializes_schema(self):
        """Test that create initializes database schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            # Verify tables exist by querying
            conn = manager._get_connection()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_agents'"
            )
            assert cursor.fetchone() is not None
            conn.close()


class TestSaveAndLoadAgent:
    """Tests for saving and loading agents."""

    def test_save_agent_creates_record(self):
        """Test that save_agent creates a database record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="test_agent", steps=5, model="opus")
            manager.save_agent("run-123", agent)

            # Verify record exists
            loaded = manager.load_agent("run-123", "test_agent")
            assert loaded is not None
            assert loaded.name == "test_agent"

    def test_load_nonexistent_agent_returns_none(self):
        """Test that loading nonexistent agent returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            result = manager.load_agent("run-123", "nonexistent")

            assert result is None

    def test_save_and_load_agent_preserves_state(self):
        """Test that save/load preserves agent state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(
                name="full_agent",
                steps=10,
                model="opus",
                thinking=False
            )
            agent.add_messages([
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}
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
        """Test that saving again updates existing record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="updatable", steps=5)
            manager.save_agent("run-789", agent)

            # Modify and save again
            agent.consume_step()
            agent.add_message({"role": "user", "content": "New message"})
            manager.save_agent("run-789", agent)

            loaded = manager.load_agent("run-789", "updatable")

            assert loaded.steps_remaining == 4
            assert len(loaded.conversation_history) == 1


class TestDeleteAgent:
    """Tests for deleting agents."""

    def test_delete_agent_removes_record(self):
        """Test that delete_agent removes the record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="deletable")
            manager.save_agent("run-111", agent)

            # Verify it exists
            assert manager.load_agent("run-111", "deletable") is not None

            # Delete it
            manager.delete_agent("run-111", "deletable")

            # Verify it's gone
            assert manager.load_agent("run-111", "deletable") is None

    def test_delete_nonexistent_agent_no_error(self):
        """Test that deleting nonexistent agent doesn't error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            # Should not raise
            manager.delete_agent("run-999", "nonexistent")
