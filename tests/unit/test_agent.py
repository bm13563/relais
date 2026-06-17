"""Unit tests for agent.py - PipelineAgent class."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from relais.agent import PipelineAgent


class TestPipelineAgentCreation:
    """Tests for PipelineAgent instantiation and defaults."""

    def test_minimal_agent(self):
        agent = PipelineAgent(name="main_agent")

        assert agent.name == "main_agent"
        assert agent.tools == []
        assert agent.steps is None  # Persistent by default
        assert agent.steps_remaining is None
        assert agent.max_turns == 10  # Default turn budget
        assert agent.model == "opus"
        assert agent.thinking is False
        assert agent.client is None

    def test_max_turns_default(self):
        """A fresh agent gets the documented default turn budget."""
        assert PipelineAgent(name="a").max_turns == 10

    def test_agent_with_all_options(self):
        agent = PipelineAgent(
            name="custom_agent",
            tools=["t1", "t2"],
            steps=3,
            max_turns=5,
            model="sonnet",
            thinking=True,
        )
        assert agent.name == "custom_agent"
        assert agent.tools == ["t1", "t2"]
        assert agent.steps == 3
        assert agent.steps_remaining == 3
        assert agent.max_turns == 5
        assert agent.model == "sonnet"
        assert agent.thinking is True


class TestStepBudget:
    """Tests for the step budget (steps / steps_remaining / expiry)."""

    def test_persistent_agent_never_expires(self):
        agent = PipelineAgent(name="p")  # steps=None
        assert not agent.is_expired()
        agent.consume_step()  # no-op
        assert agent.steps_remaining is None
        assert not agent.is_expired()

    def test_budgeted_agent_expires_after_n_steps(self):
        agent = PipelineAgent(name="w", steps=2)
        assert agent.steps_remaining == 2
        assert not agent.is_expired()

        agent.consume_step()
        assert agent.steps_remaining == 1
        assert not agent.is_expired()

        agent.consume_step()
        assert agent.steps_remaining == 0
        assert agent.is_expired()

    def test_steps_one_expires_after_single_step(self):
        agent = PipelineAgent(name="solo", steps=1)
        assert not agent.is_expired()
        agent.consume_step()
        assert agent.is_expired()


class TestClientManagement:
    """Tests for SDK client management."""

    def test_set_client_assigns_client(self):
        agent = PipelineAgent(name="agent")
        mock_client = MagicMock()
        agent.set_client(mock_client)
        assert agent.client == mock_client

    def test_has_client(self):
        agent = PipelineAgent(name="agent")
        assert not agent.has_client()
        agent.set_client(MagicMock())
        assert agent.has_client()

    @pytest.mark.asyncio
    async def test_disconnect_calls_client_disconnect(self):
        agent = PipelineAgent(name="agent")
        mock_client = AsyncMock()
        agent.set_client(mock_client)
        await agent.disconnect()
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_client_no_error(self):
        agent = PipelineAgent(name="agent")
        await agent.disconnect()  # Should not raise


class TestAgentEquality:
    """Tests for agent equality based on static configuration."""

    def test_agents_with_same_config_are_equal(self):
        a = PipelineAgent(name="agent", tools=["t"], model="opus")
        b = PipelineAgent(name="agent", tools=["t"], model="opus")
        assert a == b

    def test_agents_with_different_names_not_equal(self):
        assert PipelineAgent(name="agent1") != PipelineAgent(name="agent2")

    def test_agents_with_different_model_not_equal(self):
        a = PipelineAgent(name="agent", model="opus")
        b = PipelineAgent(name="agent", model="haiku")
        assert a != b
