"""astream and inline instructions — the pieces a streaming host uses."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from relais import Pipeline, PipelineAgent, PipelineStep
from relais.executor import PipelineOrchestrator


def _orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator(
        tool_registry=MagicMock(), state_manager=MagicMock(), instructions_dir=Path("/nowhere")
    )


@pytest.mark.asyncio
async def test_inline_instruction_text_is_used_instead_of_a_file():
    step = PipelineStep(
        name="s", instruction="s", instruction_text="Do the thing.", response_tool="t",
        agent=PipelineAgent(name="a"),
    )
    context = await _orchestrator()._build_step_context(step, None, "hi", Path("/nowhere"), {})
    assert "[Instructions]\nDo the thing." in context


@pytest.mark.asyncio
async def test_astream_yields_the_run_events_then_run_done():
    orchestrator = _orchestrator()

    async def fake_run(pipeline_name, initial_input, args=None):
        await orchestrator._emit({"type": "text_delta", "step": "s", "text": "Hel"})
        await orchestrator._emit({"type": "tool_use", "step": "s", "name": "t", "input": {}})
        return "run-1"

    orchestrator.start_pipeline_async = fake_run
    state = MagicMock(
        current_step="s",
        step_results={"s": {"routing_data": {"ok": True}, "session_id": "sess-1"}},
    )
    pipeline = Pipeline(
        name="p", steps={}, start_step="s", instructions_dir=Path("/nowhere"),
        tool_registry=MagicMock(),
        state_manager=MagicMock(get_pipeline_run=MagicMock(return_value=state)),
        orchestrator=orchestrator,
    )

    events = [e async for e in pipeline.astream("hi")]

    assert [e["type"] for e in events] == ["text_delta", "tool_use", "run_done"]
    assert events[-1] == {"type": "run_done", "run": "run-1", "output": {"ok": True}, "session_id": "sess-1"}
    assert orchestrator._on_event is None


@pytest.mark.asyncio
async def test_astream_raises_the_runs_error():
    orchestrator = _orchestrator()

    async def failing_run(pipeline_name, initial_input, args=None):
        raise RuntimeError("boom")

    orchestrator.start_pipeline_async = failing_run
    pipeline = Pipeline(
        name="p", steps={}, start_step="s", instructions_dir=Path("/nowhere"),
        tool_registry=MagicMock(), state_manager=MagicMock(), orchestrator=orchestrator,
    )
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in pipeline.astream("hi"):
            pass
    assert orchestrator._on_event is None


@pytest.mark.asyncio
async def test_max_visits_bounds_a_revise_loop():
    """analyse → check → (revise) → analyse → check → … is bounded by capping BOTH
    steps of the loop: past its cap a step is skipped and its default route taken,
    so the run falls through to compose. (Capping only the step routed back to
    would not terminate — the router would keep routing to a skipped step.)"""
    from relais.executor import PipelineConfig, StepExecutionResult

    orchestrator = _orchestrator()
    agent = PipelineAgent(name="a")
    steps = {
        "analyse": PipelineStep(name="analyse", instruction="x", response_tool="t", agent=agent,
                                max_visits=1, next={"default": "check"}),
        "check": PipelineStep(name="check", instruction="x", response_tool="t", agent=agent, max_visits=1,
                              next={"field": "verdict", "routes": [{"equals": "revise", "goto": "analyse"}],
                                    "default": "compose"}),
        "compose": PipelineStep(name="compose", instruction="x", response_tool="t", agent=agent,
                                next={"default": None}),
    }
    config = PipelineConfig(name="p", steps=steps, start_step="analyse", instructions_dir="/nowhere",
                            agents={"a": agent})
    ran = []

    async def fake_step(step, context, mcp_server, agent, images=None):
        ran.append(step.name)
        data = {"verdict": "revise"} if step.name == "check" else {}
        return StepExecutionResult(step_name=step.name, final_response="", tool_results=[], turns_used=1,
                                   stop_reason="success", routing_data=data)

    async def fake_context(**kwargs):
        return "ctx"

    orchestrator._execute_step = fake_step
    orchestrator._build_step_context = fake_context
    result = await orchestrator._run_segment("run-1", config, "analyse", "hi", {}, {}, MagicMock(),
                                             conversational=False)
    assert ran == ["analyse", "check", "compose"]
    assert result["suspended"] is False
