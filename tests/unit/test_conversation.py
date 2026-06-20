"""Unit tests for conversational pipelines (no live model calls).

These cover the suspend/resume routing logic in _run_segment, the await_input
flag, the Turn object, and the conversation registry/eviction. The full live
round-trip (real agents, RAM-held memory across turns, sync dispatch) is verified
separately as a live test.
"""

import time

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from relais.step import PipelineStep
from relais.agent import PipelineAgent
from relais.executor import PipelineOrchestrator, PipelineConfig, StepExecutionResult
from relais.conversation import (
    Turn, Conversation, register, get, active_ids, is_active, _conversations, _evict_idle,
)
from pathlib import Path


test_agent = PipelineAgent(name="advisor", model="opus")


class TestAwaitInputStep:
    def test_pure_park_step_needs_no_response_tool(self):
        step = PipelineStep(name="await", instruction="x", await_input=True, next={"default": "next"})
        assert step.await_input is True
        assert step.agent is None

    def test_await_input_with_agent_still_needs_response_tool(self):
        with pytest.raises(ValueError, match="response_tool"):
            PipelineStep(name="s", instruction="x", await_input=True, agent=test_agent)

    def test_normal_step_unaffected(self):
        step = PipelineStep(name="s", instruction="x", response_tool="t", agent=test_agent)
        assert step.await_input is False


class TestTurn:
    def test_turn_fields(self):
        t = Turn(output={"a": 1}, step="summarize", awaiting=True)
        assert t.output == {"a": 1}
        assert t.step == "summarize"
        assert t.awaiting is True
        assert t.to_dict() == {"output": {"a": 1}, "step": "summarize", "awaiting": True}


def _orchestrator():
    return PipelineOrchestrator(
        tool_registry=MagicMock(),
        state_manager=MagicMock(),
        instructions_dir=Path("/instructions"),
    )


class TestRunSegmentSuspend:
    """_run_segment suspend/resume routing, with _execute_step mocked."""

    @pytest.mark.asyncio
    async def test_pure_park_entry_suspends_immediately_with_no_input(self, test_instructions_dir):
        orch = _orchestrator()
        steps = {
            "await": PipelineStep(name="await", instruction="test_step", await_input=True, next={"default": "work"}),
            "work": PipelineStep(name="work", instruction="test_step", response_tool="t", agent=test_agent, next={"default": None}),
        }
        config = PipelineConfig(name="c", steps=steps, start_step="await", instructions_dir=str(test_instructions_dir))

        seg = await orch._run_segment(
            "run", config, "await", None, {}, {}, MagicMock(), conversational=True,
        )
        # Pure-park entry with no input -> suspends right away at the entry, next is 'work'.
        assert seg["suspended"] is True
        assert seg["next_step"] == "work"
        assert seg["output"] is None

    @pytest.mark.asyncio
    async def test_input_flows_past_park_into_work(self, test_instructions_dir):
        orch = _orchestrator()
        steps = {
            "await": PipelineStep(name="await", instruction="test_step", await_input=True, next={"default": "work"}),
            "work": PipelineStep(name="work", instruction="test_step", response_tool="t", agent=test_agent, next={"default": None}),
        }
        config = PipelineConfig(name="c", steps=steps, start_step="await", instructions_dir=str(test_instructions_dir))

        async def fake_step(step, context, mcp_server, agent, images=None):
            return StepExecutionResult(step.name, "", [], 1, "success", routing_data={"ok": True})

        with patch.object(orch, "_execute_step", side_effect=fake_step):
            seg = await orch._run_segment(
                "run", config, "await", "hello", {}, {}, MagicMock(), conversational=True,
            )
        # Input present -> park is skipped, 'work' runs, pipeline ends (no suspend).
        assert seg["suspended"] is False
        assert seg["step"] == "work"

    @pytest.mark.asyncio
    async def test_mid_pipeline_await_runs_agent_then_suspends(self, test_instructions_dir):
        orch = _orchestrator()
        steps = {
            "summarize": PipelineStep(
                name="summarize", instruction="test_step", response_tool="t",
                agent=test_agent, await_input=True, next={"default": "decide"},
            ),
            "decide": PipelineStep(name="decide", instruction="test_step", response_tool="t", agent=test_agent, next={"default": None}),
        }
        config = PipelineConfig(name="c", steps=steps, start_step="summarize", instructions_dir=str(test_instructions_dir))

        async def fake_step(step, context, mcp_server, agent, images=None):
            return StepExecutionResult(step.name, "", [], 1, "success", routing_data={"summary": "done"})

        with patch.object(orch, "_execute_step", side_effect=fake_step):
            seg = await orch._run_segment(
                "run", config, "summarize", "go", {}, {}, MagicMock(), conversational=True,
            )
        # summarize runs (produces output), THEN suspends; next is 'decide'.
        assert seg["suspended"] is True
        assert seg["step"] == "summarize"
        assert seg["output"] == {"summary": "done"}
        assert seg["next_step"] == "decide"

    @pytest.mark.asyncio
    async def test_non_conversational_ignores_await_input(self, test_instructions_dir):
        orch = _orchestrator()
        steps = {
            "summarize": PipelineStep(
                name="summarize", instruction="test_step", response_tool="t",
                agent=test_agent, await_input=True, next={"default": None},
            ),
        }
        config = PipelineConfig(name="c", steps=steps, start_step="summarize", instructions_dir=str(test_instructions_dir))

        async def fake_step(step, context, mcp_server, agent, images=None):
            return StepExecutionResult(step.name, "", [], 1, "success", routing_data={})

        with patch.object(orch, "_execute_step", side_effect=fake_step):
            seg = await orch._run_segment(
                "run", config, "summarize", "go", {}, {}, MagicMock(), conversational=False,
            )
        # Fire-and-forget: await_input is ignored, runs to the end.
        assert seg["suspended"] is False


class TestRegistry:
    def setup_method(self):
        _conversations.clear()

    def test_register_and_get(self):
        convo = MagicMock(spec=Conversation)
        convo.id = "abc"
        convo.is_idle.return_value = False
        register(convo)
        assert get("abc") is convo

    def test_idle_eviction(self):
        live = MagicMock(spec=Conversation)
        live.id = "live"
        live.is_idle.return_value = False
        idle = MagicMock(spec=Conversation)
        idle.id = "idle"
        idle.is_idle.return_value = True
        _conversations["live"] = live
        _conversations["idle"] = idle

        _evict_idle()
        idle.end_conversation.assert_called_once()
        live.end_conversation.assert_not_called()

    def test_active_ids_and_is_active(self):
        convo = MagicMock(spec=Conversation)
        convo.id = "c1"
        convo.is_idle.return_value = False
        _conversations["c1"] = convo
        assert "c1" in active_ids()
        assert is_active("c1") is True
        assert is_active("nope") is False
