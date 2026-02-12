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


class TestLoadAllAgents:
    """Tests for loading all agents for a run."""

    def test_load_all_agents_empty_when_none_exist(self):
        """Test that load_all_agents returns empty dict when no agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agents = manager.load_all_agents("run-999")

            assert agents == {}

    def test_load_all_agents_returns_all_agents_for_run(self):
        """Test that load_all_agents returns all agents for a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent1 = PipelineAgent(name="agent1", steps=3)
            agent2 = PipelineAgent(name="agent2", steps=5)
            agent3 = PipelineAgent(name="agent3")  # Persistent

            manager.save_agent("run-abc", agent1)
            manager.save_agent("run-abc", agent2)
            manager.save_agent("run-abc", agent3)

            # Save agent for different run (should not be loaded)
            manager.save_agent("run-xyz", PipelineAgent(name="other"))

            agents = manager.load_all_agents("run-abc")

            assert len(agents) == 3
            assert "agent1" in agents
            assert "agent2" in agents
            assert "agent3" in agents
            assert agents["agent1"].steps == 3
            assert agents["agent2"].steps == 5
            assert agents["agent3"].steps is None


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


class TestDeleteAllAgentsForRun:
    """Tests for deleting all agents for a run."""

    def test_delete_all_agents_for_run_removes_all(self):
        """Test that delete_all_agents_for_run removes all agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            manager.save_agent("run-222", PipelineAgent(name="a1"))
            manager.save_agent("run-222", PipelineAgent(name="a2"))
            manager.save_agent("run-222", PipelineAgent(name="a3"))
            # Different run
            manager.save_agent("run-333", PipelineAgent(name="b1"))

            manager.delete_all_agents_for_run("run-222")

            # Verify run-222 agents are gone
            agents_222 = manager.load_all_agents("run-222")
            assert len(agents_222) == 0

            # Verify run-333 agents still exist
            agents_333 = manager.load_all_agents("run-333")
            assert len(agents_333) == 1

    def test_delete_all_agents_for_nonexistent_run_no_error(self):
        """Test deleting agents for nonexistent run doesn't error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            # Should not raise
            manager.delete_all_agents_for_run("run-nonexistent")


class TestConversationHistory:
    """Tests for conversation history updates."""

    def test_update_conversation_history(self):
        """Test updating just the conversation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="chat_agent")
            manager.save_agent("run-444", agent)

            # Update history
            new_history = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
            manager.update_conversation_history("run-444", "chat_agent", new_history)

            loaded = manager.load_agent("run-444", "chat_agent")
            assert loaded.conversation_history == new_history

    def test_update_history_for_nonexistent_agent_no_error(self):
        """Test updating history for nonexistent agent creates record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            # Update history for agent that doesn't exist yet
            # This might create a partial record or be a no-op depending on design
            # For now, let's say it's a no-op
            history = [{"role": "user", "content": "test"}]
            manager.update_conversation_history("run-555", "ghost_agent", history)

            # Should still be None since agent was never properly created
            loaded = manager.load_agent("run-555", "ghost_agent")
            assert loaded is None


class TestStepsRemaining:
    """Tests for steps_remaining updates."""

    def test_update_steps_remaining(self):
        """Test updating steps_remaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="countdown", steps=5)
            manager.save_agent("run-666", agent)

            # Update steps_remaining
            manager.update_steps_remaining("run-666", "countdown", 3)

            loaded = manager.load_agent("run-666", "countdown")
            assert loaded.steps_remaining == 3

    def test_update_steps_remaining_to_zero_marks_expired(self):
        """Test that updating to zero makes agent expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agents.db"
            manager = AgentStateManager.create(str(db_path))
            manager.initialize_schema()

            agent = PipelineAgent(name="expiring", steps=2)
            manager.save_agent("run-777", agent)

            manager.update_steps_remaining("run-777", "expiring", 0)

            loaded = manager.load_agent("run-777", "expiring")
            assert loaded.is_expired()
