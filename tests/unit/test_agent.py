"""Unit tests for agent.py - PipelineAgent class."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from relais.agent import PipelineAgent


class TestPipelineAgentCreation:
    """Tests for PipelineAgent instantiation and defaults."""

    def test_minimal_agent(self):
        """Test creating agent with minimal required fields."""
        agent = PipelineAgent(name="main_agent")

        assert agent.name == "main_agent"
        assert agent.steps is None  # Persistent across all steps
        assert agent.model == "opus"  # Default model
        assert agent.thinking is False  # Default thinking
        assert agent.conversation_history == []
        assert agent.steps_remaining is None
        assert agent.client is None

    def test_agent_with_limited_steps(self):
        """Test creating agent with step limit."""
        agent = PipelineAgent(name="temp_agent", steps=3)

        assert agent.name == "temp_agent"
        assert agent.steps == 3
        assert agent.steps_remaining == 3  # Initialized from steps

    def test_agent_with_all_options(self):
        """Test creating agent with all options specified."""
        agent = PipelineAgent(
            name="custom_agent",
            steps=5,
            model="sonnet",
            thinking=True
        )

        assert agent.name == "custom_agent"
        assert agent.steps == 5
        assert agent.model == "sonnet"
        assert agent.thinking is True

    def test_persistent_agent_steps_none(self):
        """Test that steps=None means persistent agent."""
        agent = PipelineAgent(name="persistent", steps=None)

        assert agent.steps is None
        assert agent.steps_remaining is None
        assert agent.is_persistent()

    def test_temporary_agent_not_persistent(self):
        """Test that agent with steps limit is not persistent."""
        agent = PipelineAgent(name="temp", steps=2)

        assert not agent.is_persistent()


class TestAgentLifecycle:
    """Tests for agent lifecycle management."""

    def test_consume_step_decrements_remaining(self):
        """Test that consuming a step decrements steps_remaining."""
        agent = PipelineAgent(name="temp", steps=3)

        assert agent.steps_remaining == 3
        agent.consume_step()
        assert agent.steps_remaining == 2
        agent.consume_step()
        assert agent.steps_remaining == 1

    def test_consume_step_on_persistent_agent_no_effect(self):
        """Test that consuming step on persistent agent has no effect."""
        agent = PipelineAgent(name="persistent", steps=None)

        assert agent.steps_remaining is None
        agent.consume_step()
        assert agent.steps_remaining is None

    def test_is_expired_returns_true_when_steps_exhausted(self):
        """Test that agent is expired when steps reach 0."""
        agent = PipelineAgent(name="temp", steps=2)

        assert not agent.is_expired()
        agent.consume_step()
        assert not agent.is_expired()
        agent.consume_step()
        assert agent.is_expired()

    def test_is_expired_returns_false_for_persistent(self):
        """Test that persistent agents never expire."""
        agent = PipelineAgent(name="persistent")

        assert not agent.is_expired()
        # Even after consuming steps
        agent.consume_step()
        agent.consume_step()
        assert not agent.is_expired()

    def test_reset_resets_steps_remaining(self):
        """Test that reset restores steps_remaining to original steps."""
        agent = PipelineAgent(name="temp", steps=3)

        agent.consume_step()
        agent.consume_step()
        assert agent.steps_remaining == 1

        agent.reset()
        assert agent.steps_remaining == 3

    def test_reset_on_persistent_agent_no_effect(self):
        """Test that reset on persistent agent has no effect."""
        agent = PipelineAgent(name="persistent")

        agent.reset()
        assert agent.steps_remaining is None


class TestConversationHistory:
    """Tests for conversation history management."""

    def test_add_message_appends_to_history(self):
        """Test adding message to conversation history."""
        agent = PipelineAgent(name="agent")

        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "assistant", "content": "Hi there"}

        agent.add_message(msg1)
        agent.add_message(msg2)

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0] == msg1
        assert agent.conversation_history[1] == msg2

    def test_add_messages_appends_multiple(self):
        """Test adding multiple messages at once."""
        agent = PipelineAgent(name="agent")

        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

        agent.add_messages(messages)

        assert len(agent.conversation_history) == 3
        assert agent.conversation_history == messages

    def test_clear_history_empties_list(self):
        """Test clearing conversation history."""
        agent = PipelineAgent(name="agent")

        agent.add_message({"role": "user", "content": "Hello"})
        agent.add_message({"role": "assistant", "content": "Hi"})

        assert len(agent.conversation_history) == 2

        agent.clear_history()

        assert len(agent.conversation_history) == 0

    def test_get_history_returns_copy(self):
        """Test that get_history returns a copy, not reference."""
        agent = PipelineAgent(name="agent")

        agent.add_message({"role": "user", "content": "Hello"})

        history = agent.get_history()
        history.append({"role": "assistant", "content": "Modified"})

        # Original should be unchanged
        assert len(agent.conversation_history) == 1


class TestClientManagement:
    """Tests for SDK client management."""

    @pytest.mark.asyncio
    async def test_set_client_assigns_client(self):
        """Test setting the SDK client."""
        agent = PipelineAgent(name="agent")
        mock_client = MagicMock()

        agent.set_client(mock_client)

        assert agent.client == mock_client

    def test_has_client_returns_true_when_client_set(self):
        """Test has_client returns True when client is set."""
        agent = PipelineAgent(name="agent")

        assert not agent.has_client()

        agent.set_client(MagicMock())

        assert agent.has_client()

    @pytest.mark.asyncio
    async def test_disconnect_calls_client_disconnect(self):
        """Test that disconnect calls client.disconnect()."""
        agent = PipelineAgent(name="agent")
        mock_client = AsyncMock()
        agent.set_client(mock_client)

        await agent.disconnect()

        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_client_no_error(self):
        """Test that disconnect without client doesn't error."""
        agent = PipelineAgent(name="agent")

        # Should not raise
        await agent.disconnect()


class TestAgentEquality:
    """Tests for agent equality and identity."""

    def test_agents_with_same_name_are_equal(self):
        """Test that agents with same name are equal."""
        agent1 = PipelineAgent(name="agent", steps=3)
        agent2 = PipelineAgent(name="agent", steps=3)

        assert agent1 == agent2

    def test_agents_with_different_names_not_equal(self):
        """Test that agents with different names are not equal."""
        agent1 = PipelineAgent(name="agent1")
        agent2 = PipelineAgent(name="agent2")

        assert agent1 != agent2

    def test_agents_with_different_config_not_equal(self):
        """Test that agents with different config are not equal."""
        agent1 = PipelineAgent(name="agent", steps=3)
        agent2 = PipelineAgent(name="agent", steps=5)

        assert agent1 != agent2


class TestAgentSerialization:
    """Tests for agent state serialization."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all agent fields."""
        agent = PipelineAgent(
            name="agent",
            steps=5,
            model="opus",
            thinking=False
        )
        agent.add_message({"role": "user", "content": "test"})
        agent.consume_step()

        data = agent.to_dict()

        assert data["name"] == "agent"
        assert data["steps"] == 5
        assert data["steps_remaining"] == 4
        assert data["model"] == "opus"
        assert data["thinking"] is False
        assert len(data["conversation_history"]) == 1

    def test_from_dict_restores_agent(self):
        """Test that from_dict restores agent from dict."""
        data = {
            "name": "restored_agent",
            "steps": 3,
            "steps_remaining": 2,
            "model": "opus",
            "thinking": True,
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        }

        agent = PipelineAgent.from_dict(data)

        assert agent.name == "restored_agent"
        assert agent.steps == 3
        assert agent.steps_remaining == 2
        assert agent.model == "opus"
        assert agent.thinking is True
        assert len(agent.conversation_history) == 2

    def test_roundtrip_serialization(self):
        """Test that agent survives to_dict -> from_dict roundtrip."""
        original = PipelineAgent(
            name="test",
            steps=10,
            model="haiku",
            thinking=False
        )
        original.add_messages([
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"}
        ])
        original.consume_step()
        original.consume_step()

        data = original.to_dict()
        restored = PipelineAgent.from_dict(data)

        assert restored.name == original.name
        assert restored.steps == original.steps
        assert restored.steps_remaining == original.steps_remaining
        assert restored.model == original.model
        assert restored.thinking == original.thinking
        assert restored.conversation_history == original.conversation_history
