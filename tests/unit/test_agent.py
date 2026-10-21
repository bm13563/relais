"""Unit tests for agent.py - PipelineAgent class."""

import pytest
from unittest.mock import MagicMock

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
        assert agent.thinking is None  # the model's default
        assert agent.engine == "sdk"
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
    """Tests for the engine-state slot (closing it is the engine's job)."""

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



class TestAgentEquality:
    """Tests for agent equality based on static configuration."""

    def test_agents_with_same_config_are_equal(self):
        a = PipelineAgent(name="agent", tools=["t"], model="opus")
        b = PipelineAgent(name="agent", tools=["t"], model="opus")
        assert a == b

    def test_agents_with_different_names_not_equal(self):
        assert PipelineAgent(name="agent1") != PipelineAgent(name="agent2")

    def test_thinking_none_is_its_own_value(self):
        assert PipelineAgent(name="a") == PipelineAgent(name="a", thinking=None)
        assert PipelineAgent(name="a") != PipelineAgent(name="a", thinking=False)
        assert PipelineAgent(name="a", thinking=False) != PipelineAgent(name="a", thinking=True)

    def test_agents_with_different_model_not_equal(self):
        a = PipelineAgent(name="agent", model="opus")
        b = PipelineAgent(name="agent", model="haiku")
        assert a != b


class TestEngine:
    """PipelineAgent.engine — which engine the agent's steps run on."""

    def test_api_engine(self):
        assert PipelineAgent(name="a", engine="api").engine == "api"

    def test_agents_on_different_engines_not_equal(self):
        assert PipelineAgent(name="a", engine="sdk") != PipelineAgent(name="a", engine="api")

    def test_unknown_engine_is_refused_at_definition(self):
        with pytest.raises(ValueError, match="engine 'gpt'"):
            PipelineAgent(name="a", engine="gpt")
