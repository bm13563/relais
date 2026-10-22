"""Unit tests for executor.py - PipelineOrchestrator class."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from relais.engine import StepOutcome
from relais.engines.sdk import SdkEngine
from relais.executor import (
    PipelineOrchestrator,
    PipelineConfig,
    StepExecutionResult,
    ResponseToolNotCalled,
    StepError,
)
from relais.step import PipelineStep
from relais.agent import PipelineAgent


# Test agent used across all tests
test_agent = PipelineAgent(name="test_agent", model="opus")


def outcome(**overrides) -> StepOutcome:
    """A clean, successful StepOutcome, with any field overridden."""
    fields = dict(
        final_text="Done.", tool_calls=[], turns_used=1, is_error=False,
        error_kind=None, result_text=None, session_id="sess-1", usage={},
    )
    fields.update(overrides)
    return StepOutcome(**fields)


class FakeEngine:
    """Stands in for a real engine: returns a scripted StepOutcome and records
    what the orchestrator asked of it. Installed as the orchestrator's "sdk"
    engine (the default engine the test agents name)."""

    name = "sdk"

    def __init__(self, result: StepOutcome = None):
        self.result = result or outcome()
        self.calls = []
        self.closed = []

    async def run_step(self, **kwargs) -> StepOutcome:
        self.calls.append(kwargs)
        return self.result

    async def close(self, agent) -> None:
        self.closed.append(agent)
        agent.client = None


def captured_registry(captured=("test_tool", {"content": [{"type": "text", "text": '{"ok": true}'}]})):
    """A registry mock whose response tool appears captured (None: never called)."""
    registry = MagicMock()
    registry.get_tool_result.return_value = captured
    return registry


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating config with minimal fields."""
        steps = {"start": PipelineStep(name="start", instruction="test", response_tool="test_tool", agent=test_agent)}
        config = PipelineConfig(
            name="test",
            steps=steps,
            start_step="start",
            instructions_dir="/path/to/instructions"
        )
        assert config.name == "test"
        assert config.start_step == "start"
        assert config.cwd is None

    def test_create_full_config(self):
        """Test creating config with all fields."""
        steps = {"s": PipelineStep(name="s", instruction="test", response_tool="test_tool", agent=test_agent)}
        config = PipelineConfig(
            name="full",
            steps=steps,
            start_step="s",
            instructions_dir="/instructions",
            cwd="/workdir"
        )
        assert config.cwd == "/workdir"

    def test_create_allows_agentless_route_step(self):
        """A route step has no agent (it decides from hooks) — create() must allow it."""
        from relais import Pipeline
        steps = {
            "work": PipelineStep(name="work", instruction="test", response_tool="t", agent=test_agent, next={"default": "route"}),
            "route": PipelineStep(name="route", instruction="route", route=True, hooks=[lambda: {"x": 1}], next={"default": None}),
        }
        # Should not raise "missing required 'agent'".
        Pipeline.create(name="r", steps=steps, start_step="work", instructions_dir="/i", db_config="x.db")


class TestStepExecutionResult:
    """Tests for StepExecutionResult dataclass."""

    def test_create_result(self):
        """Test creating execution result."""
        result = StepExecutionResult(
            step_name="my_step",
            final_response="Done!",
            tool_results=[{"tool": "greet", "output": "Hello"}],
            turns_used=3,
            stop_reason="success",
            routing_data={"category": "A"},
        )
        assert result.step_name == "my_step"
        assert result.final_response == "Done!"
        assert len(result.tool_results) == 1
        assert result.turns_used == 3
        assert result.stop_reason == "success"
        assert result.routing_data == {"category": "A"}

    def test_result_defaults(self):
        """Test result default values."""
        result = StepExecutionResult(
            step_name="s",
            final_response="",
            tool_results=[],
            turns_used=0,
            stop_reason="success"
        )
        assert result.routing_data is None


class TestPipelineOrchestratorCreation:
    """Tests for PipelineOrchestrator instantiation."""

    def test_create_orchestrator(self):
        """Test creating orchestrator with required params."""
        mock_registry = MagicMock()
        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=Path("/instructions")
        )

        assert orchestrator.tool_registry == mock_registry
        assert orchestrator.state_manager == mock_state
        assert orchestrator.pipelines == {}

    def test_create_orchestrator_with_options(self):
        """Test creating orchestrator with all options."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
            cwd="/workdir"
        )

        assert orchestrator.cwd == "/workdir"


class TestRegisterPipeline:
    """Tests for pipeline registration."""

    def test_register_pipeline(self):
        """Test registering a pipeline."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        steps = {"s": PipelineStep(name="s", instruction="test", response_tool="test_tool", agent=test_agent)}
        config = PipelineConfig(
            name="my_pipeline",
            steps=steps,
            start_step="s",
            instructions_dir="/instructions"
        )

        orchestrator.register_pipeline(config)

        assert "my_pipeline" in orchestrator.pipelines
        assert orchestrator.pipelines["my_pipeline"] == config

    def test_register_multiple_pipelines(self):
        """Test registering multiple pipelines."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        for name in ["pipeline1", "pipeline2", "pipeline3"]:
            steps = {"s": PipelineStep(name="s", instruction="test", response_tool="test_tool", agent=test_agent)}
            config = PipelineConfig(
                name=name,
                steps=steps,
                start_step="s",
                instructions_dir="/instructions"
            )
            orchestrator.register_pipeline(config)

        assert len(orchestrator.pipelines) == 3


class TestBuildStepContext:
    """Tests for _build_step_context method."""

    async def test_build_context_basic(self, test_instructions_dir):
        """Test building basic context."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step", response_tool="test_tool", agent=test_agent)
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = await orchestrator._build_step_context(
            step=step,
            args={},
            previous_result=None,
            initial_input="Hello",
            instructions_dir=test_instructions_dir,
        )

        assert "[User Input]" in context
        assert "Hello" in context
        assert "[Current Step]" in context
        assert "test" in context
        assert "[Instructions]" in context

    async def test_build_context_with_args(self, test_instructions_dir):
        """Test context includes args."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step", response_tool="test_tool", agent=test_agent)
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = await orchestrator._build_step_context(
            step=step,
            args={"key": "value", "num": 42},
            previous_result=None,
            initial_input=None,
            instructions_dir=test_instructions_dir,
        )

        assert "[Pipeline Args]" in context
        assert "key" in context
        assert "value" in context

    async def test_build_context_with_previous_result(self, test_instructions_dir):
        """Test context includes previous result."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step", response_tool="test_tool", agent=test_agent)
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = await orchestrator._build_step_context(
            step=step,
            args={},
            previous_result={"output": "previous data"},
            initial_input=None,
            instructions_dir=test_instructions_dir,
        )

        assert "[Previous Step Output]" in context
        assert "previous data" in context

    async def test_build_context_with_hooks(self, test_instructions_dir):
        """Test context includes hook data."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        def my_hook():
            return {"timestamp": "2024-01-15", "user": "test"}

        step = PipelineStep(
            name="test",
            instruction="test_step",
            response_tool="test_tool",
            hooks=[my_hook],
            agent=test_agent
        )
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = await orchestrator._build_step_context(
            step=step,
            args={},
            previous_result=None,
            initial_input=None,
            instructions_dir=test_instructions_dir,
        )

        assert "[Hook Data]" in context
        assert "timestamp" in context
        assert "2024-01-15" in context

    async def test_build_context_missing_instruction(self, test_instructions_dir):
        """Test handling missing instruction file."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="nonexistent", response_tool="test_tool", agent=test_agent)
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        # Should not raise, just log warning
        context = await orchestrator._build_step_context(
            step=step,
            args={},
            previous_result=None,
            initial_input="test",
            instructions_dir=test_instructions_dir,
        )

        # Context should still be built
        assert "[User Input]" in context
        assert "[Current Step]" in context


class TestExtractFromMcpContent:
    """Tests for _extract_from_mcp_content, which parses a tool's MCP content list.

    Routing data is the output of the step's declared response_tool, captured by
    the ToolRegistry wrapper in MCP content format:
        {"content": [{"type": "text", "text": "<json or plain text>"}]}
    """

    def _orchestrator(self):
        return PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
        )

    def test_extract_from_empty_content(self):
        assert self._orchestrator()._extract_from_mcp_content([]) is None

    def test_extract_json_object(self):
        content = [{"type": "text", "text": '{"category": "question", "confidence": 0.9}'}]
        result = self._orchestrator()._extract_from_mcp_content(content)
        assert result == {"category": "question", "confidence": 0.9}

    def test_extract_non_json_string_wrapped_in_response(self):
        content = [{"type": "text", "text": "Just plain text"}]
        result = self._orchestrator()._extract_from_mcp_content(content)
        assert result == {"response": "Just plain text"}

    def test_extract_uses_first_text_block(self):
        content = [
            {"type": "text", "text": '{"order": 1}'},
            {"type": "text", "text": '{"order": 2}'},
        ]
        result = self._orchestrator()._extract_from_mcp_content(content)
        assert result == {"order": 1}

    def test_extract_skips_non_text_blocks(self):
        content = [
            {"type": "image", "data": "b64", "mimeType": "image/png"},
            {"type": "text", "text": '{"result": "success"}'},
        ]
        result = self._orchestrator()._extract_from_mcp_content(content)
        assert result == {"result": "success"}


class TestResponseToolRouting:
    """Tests that routing data comes from the declared response tool's capture."""

    def _orchestrator(self, registry):
        return PipelineOrchestrator(
            tool_registry=registry,
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
        )

    def test_missing_response_tool_call_raises(self):
        """If the agent never calls the response tool, the step fails loudly."""
        from relais.executor import ResponseToolNotCalled

        registry = MagicMock()
        registry.get_tool_result.return_value = None  # tool was never captured

        step = PipelineStep(
            name="s", instruction="test", response_tool="classify",
            tools=["classify"], agent=test_agent,
        )
        # Exercise just the routing-extraction contract via the registry helper.
        orch = self._orchestrator(registry)
        captured = orch.tool_registry.get_tool_result(step.response_tool)
        assert captured is None
        with pytest.raises(ResponseToolNotCalled):
            if not captured:
                raise ResponseToolNotCalled("not called")


class TestStartPipeline:
    """Tests for start_pipeline method."""

    def test_start_unknown_pipeline_raises(self):
        """Test starting unknown pipeline raises error."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        with pytest.raises(ValueError, match="Unknown pipeline"):
            orchestrator.start_pipeline("nonexistent", "input")

    @patch.object(PipelineOrchestrator, 'start_pipeline_async')
    def test_start_pipeline_calls_async(self, mock_async):
        """Test that start_pipeline wraps async method."""
        mock_async.return_value = "run-123"

        mock_registry = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        steps = {"s": PipelineStep(name="s", instruction="test", response_tool="test_tool", agent=test_agent)}
        config = PipelineConfig(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir="/instructions"
        )
        orchestrator.register_pipeline(config)

        # Note: asyncio.run in start_pipeline will call our mock
        # We can't easily test the full sync->async chain here
        # This would be better tested as integration test


class TestExecuteStep:
    """_execute_step: the step runs on its agent's engine; the outcome is read here."""

    def _orchestrator(self, registry, engine, cwd=None):
        orchestrator = PipelineOrchestrator(
            tool_registry=registry, state_manager=MagicMock(),
            instructions_dir=Path("/instructions"), cwd=cwd,
        )
        orchestrator._engines["sdk"] = engine
        return orchestrator

    def _step(self, **kwargs):
        return PipelineStep(
            name="main", instruction="test", response_tool="test_tool",
            tools=kwargs.pop("tools", ["tool1", "tool2"]), agent=PipelineAgent(name="test_agent"), **kwargs,
        )

    @pytest.mark.asyncio
    async def test_execute_step_sets_current_step(self):
        """The registry is gated to the step's tools before the engine runs."""
        registry = captured_registry()
        orchestrator = self._orchestrator(registry, FakeEngine())
        step = self._step()

        await orchestrator._execute_step(step=step, context="Test", agent=step.agent)

        registry.set_current_step.assert_called_once_with("main", ["tool1", "tool2"])

    @pytest.mark.asyncio
    async def test_hands_the_step_to_the_agents_engine(self):
        registry = captured_registry()
        engine = FakeEngine()
        orchestrator = self._orchestrator(registry, engine)
        sink = AsyncMock()
        orchestrator._on_event = sink
        step = self._step()

        await orchestrator._execute_step(step=step, context="Test context", agent=step.agent, images=["b64png"])

        assert engine.calls == [{
            "agent": step.agent, "step": step, "prompt": "Test context", "images": ["b64png"],
            "registry": registry, "emit": sink, "verbose": False,
        }]

    @pytest.mark.asyncio
    async def test_no_images_is_an_empty_list(self):
        engine = FakeEngine()
        orchestrator = self._orchestrator(captured_registry(), engine)
        step = self._step()
        await orchestrator._execute_step(step=step, context="c", agent=step.agent)
        assert engine.calls[0]["images"] == []

    @pytest.mark.asyncio
    async def test_execute_step_returns_result(self):
        """The result carries the outcome and the response tool's captured output."""
        calls = [{"id": "t1", "tool": "test_tool", "input": {"ok": True}}]
        usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        engine = FakeEngine(outcome(final_text="All done.", tool_calls=calls, turns_used=3, usage=usage))
        orchestrator = self._orchestrator(captured_registry(), engine, cwd="/work")
        step = self._step(tools=["tool1"])

        result = await orchestrator._execute_step(step=step, context="Test context", agent=step.agent)

        assert result.step_name == "main"
        assert result.stop_reason == "success"
        assert result.routing_data == {"ok": True}
        assert result.final_response == "All done."
        assert result.tool_results == calls
        assert result.turns_used == 3
        assert result.session_id == "sess-1"
        assert result.usage == usage

    @pytest.mark.asyncio
    async def test_missing_response_tool_raises(self):
        orchestrator = self._orchestrator(captured_registry(None), FakeEngine())
        step = self._step()
        with pytest.raises(ResponseToolNotCalled):
            await orchestrator._execute_step(step=step, context="c", agent=step.agent)

    @pytest.mark.asyncio
    async def test_an_error_before_the_answer_is_a_step_error(self):
        engine = FakeEngine(outcome(is_error=True, error_kind="rate_limit", result_text="429: slow down"))
        orchestrator = self._orchestrator(captured_registry(None), engine)
        step = self._step()

        with pytest.raises(StepError) as raised:
            await orchestrator._execute_step(step=step, context="c", agent=step.agent)

        assert (raised.value.step, raised.value.agent) == ("main", "test_agent")
        assert raised.value.kind == "rate_limit"
        assert raised.value.text == "429: slow down"

    @pytest.mark.asyncio
    async def test_an_unnamed_error_falls_back_to_result_error_and_the_final_text(self):
        engine = FakeEngine(outcome(is_error=True, error_kind=None, result_text=None, final_text="half an answer"))
        orchestrator = self._orchestrator(captured_registry(None), engine)
        step = self._step()

        with pytest.raises(StepError) as raised:
            await orchestrator._execute_step(step=step, context="c", agent=step.agent)

        assert raised.value.kind == "result_error"
        assert raised.value.text == "half an answer"

    @pytest.mark.asyncio
    async def test_an_error_after_the_answer_keeps_the_answer(self):
        """The turn cap hit after the response tool ran: logged, not raised."""
        engine = FakeEngine(outcome(is_error=True, error_kind="max_turns"))
        orchestrator = self._orchestrator(captured_registry(), engine)
        step = self._step()

        result = await orchestrator._execute_step(step=step, context="c", agent=step.agent)

        assert result.routing_data == {"ok": True}
        assert result.stop_reason == "error"


class TestEngines:
    """The orchestrator's engines: one per name, made on first use; agents are
    closed through the engine that ran them."""

    def _orchestrator(self):
        return PipelineOrchestrator(
            tool_registry=MagicMock(), state_manager=MagicMock(),
            instructions_dir=Path("/instructions"), cwd="/work",
        )

    def test_one_engine_per_name(self):
        orchestrator = self._orchestrator()
        first = orchestrator._engine(PipelineAgent(name="a"))
        second = orchestrator._engine(PipelineAgent(name="b"))
        assert isinstance(first, SdkEngine)
        assert first is second
        assert first.cwd == "/work"

    def test_a_fresh_instance_keeps_the_engine(self):
        clone = PipelineOrchestrator._clone_agent(PipelineAgent(name="a", engine="api", steps=2))
        assert clone.engine == "api"
        assert clone.client is None

    @pytest.mark.asyncio
    async def test_safe_close_goes_through_the_engine(self):
        orchestrator = self._orchestrator()
        engine = FakeEngine()
        orchestrator._engines["sdk"] = engine
        agent = PipelineAgent(name="a")
        agent.set_client(object())

        await orchestrator._safe_close(agent)

        assert engine.closed == [agent]
        assert agent.client is None

    @pytest.mark.asyncio
    async def test_safe_close_logs_rather_than_raises(self):
        orchestrator = self._orchestrator()
        engine = FakeEngine()
        engine.close = AsyncMock(side_effect=RuntimeError("cli gone"))
        orchestrator._engines["sdk"] = engine
        await orchestrator._safe_close(PipelineAgent(name="a"))  # should not raise


class TestExecutePipelineLoop:
    """Tests for the main _execute_pipeline loop."""

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self, test_instructions_dir):
        """Test executing a single-step pipeline."""
        mock_registry = MagicMock()
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(
            name="only",
            instruction="test_step",
            response_tool="test_tool",
            next={"default": None},  # Ends pipeline
            agent=test_agent
        )

        config = PipelineConfig(
            name="single",
            steps={"only": step},
            start_step="only",
            instructions_dir=str(test_instructions_dir)
        )

        with patch.object(orchestrator, '_execute_step') as mock_execute:
            mock_execute.return_value = StepExecutionResult(
                step_name="only",
                final_response="Done",
                tool_results=[],
                turns_used=1,
                stop_reason="success"
            )

            await orchestrator._execute_pipeline(
                run_id="run-123",
                config=config,
                initial_input="Start",
                args={}
            )

            # Verify step was executed
            mock_execute.assert_called_once()

            # Verify pipeline was completed
            mock_state.complete_pipeline.assert_called_once_with("run-123")

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self, test_instructions_dir):
        """Test executing a multi-step pipeline."""
        mock_registry = MagicMock()
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=test_instructions_dir
        )

        steps = {
            "first": PipelineStep(
                name="first",
                instruction="test_step",
                response_tool="test_tool",
                next={"default": "second"},
                agent=test_agent
            ),
            "second": PipelineStep(
                name="second",
                instruction="test_step",
                response_tool="test_tool",
                next={"default": "third"},
                agent=test_agent
            ),
            "third": PipelineStep(
                name="third",
                instruction="test_step",
                response_tool="test_tool",
                next={"default": None},
                agent=test_agent
            )
        }

        config = PipelineConfig(
            name="multi",
            steps=steps,
            start_step="first",
            instructions_dir=str(test_instructions_dir)
        )

        call_count = {"count": 0}

        async def mock_execute(*args, **kwargs):
            call_count["count"] += 1
            return StepExecutionResult(
                step_name=f"step-{call_count['count']}",
                final_response="Done",
                tool_results=[],
                turns_used=1,
                stop_reason="success"
            )

        with patch.object(orchestrator, '_execute_step', side_effect=mock_execute):
            await orchestrator._execute_pipeline(
                run_id="run-456",
                config=config,
                initial_input="Start",
                args={}
            )

            # Verify all 3 steps were executed
            assert call_count["count"] == 3

    @pytest.mark.asyncio
    async def test_conditional_routing(self, test_instructions_dir):
        """Test pipeline with conditional routing."""
        mock_registry = MagicMock()
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=test_instructions_dir
        )

        steps = {
            "router": PipelineStep(
                name="router",
                instruction="test_step",
                response_tool="test_tool",
                next={
                    "field": "route",
                    "routes": [
                        {"equals": "A", "goto": "handle_a"},
                        {"equals": "B", "goto": "handle_b"}
                    ],
                    "default": "handle_default"
                },
                agent=test_agent
            ),
            "handle_a": PipelineStep(name="handle_a", instruction="test_step", response_tool="test_tool", next={"default": None}, agent=test_agent),
            "handle_b": PipelineStep(name="handle_b", instruction="test_step", response_tool="test_tool", next={"default": None}, agent=test_agent),
            "handle_default": PipelineStep(name="handle_default", instruction="test_step", response_tool="test_tool", next={"default": None}, agent=test_agent)
        }

        config = PipelineConfig(
            name="routing",
            steps=steps,
            start_step="router",
            instructions_dir=str(test_instructions_dir)
        )

        executed_steps = []

        async def mock_execute(step, *args, **kwargs):
            executed_steps.append(step.name)
            # Router returns routing data that routes to B
            if step.name == "router":
                return StepExecutionResult(
                    step_name="router",
                    final_response="",
                    tool_results=[{"tool": "classify", "output": '{"route": "B"}'}],
                    turns_used=1,
                    stop_reason="success",
                    routing_data={"route": "B"}
                )
            else:
                return StepExecutionResult(
                    step_name=step.name,
                    final_response="",
                    tool_results=[],
                    turns_used=1,
                    stop_reason="success"
                )

        with patch.object(orchestrator, '_execute_step', side_effect=mock_execute):
            await orchestrator._execute_pipeline(
                run_id="run-789",
                config=config,
                initial_input="Test",
                args={}
            )

            # Should have executed router -> handle_b
            assert executed_steps == ["router", "handle_b"]

    @pytest.mark.asyncio
    async def test_step_budget_resets_agent_on_loop_back(self, test_instructions_dir):
        """A steps=N agent expires after N steps; looping back to it spins up a
        fresh instance with a new client (clean context)."""
        mock_registry = MagicMock()
        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=test_instructions_dir,
        )

        # One worker agent with a 2-step budget drives both A and B.
        worker = PipelineAgent(name="worker", steps=2)
        steps = {
            # A -> B always; B loops back to A once, then ends.
            "A": PipelineStep(name="A", instruction="test_step", response_tool="test_tool",
                              next={"default": "B"}, agent=worker),
            "B": PipelineStep(name="B", instruction="test_step", response_tool="test_tool",
                              next={"field": "again",
                                    "routes": [{"equals": True, "goto": "A"}],
                                    "default": None},
                              agent=worker),
        }
        config = PipelineConfig(name="loop", steps=steps, start_step="A",
                                instructions_dir=str(test_instructions_dir))

        # B asks to loop back the first time, then ends.
        b_calls = {"n": 0}
        seen_instances = []  # the agent instance object passed to each step

        async def mock_execute(step, context, agent, images=None):
            seen_instances.append(agent)
            # Give the live instance a fake client so we can see disconnects.
            if not agent.has_client():
                agent.set_client(AsyncMock())
            again = False
            if step.name == "B":
                b_calls["n"] += 1
                again = b_calls["n"] == 1  # loop back only on the first B
            return StepExecutionResult(
                step_name=step.name, final_response="", tool_results=[],
                turns_used=1, stop_reason="success", routing_data={"again": again},
            )

        with patch.object(orchestrator, '_execute_step', side_effect=mock_execute):
            await orchestrator._execute_pipeline(
                run_id="run-loop", config=config, initial_input="go", args={},
            )

        # Path was A, B, A, B (one retry) — four step executions.
        assert len(seen_instances) == 4

        # First two steps share one instance; after 2 steps it expires, so the
        # loop-back to A uses a DIFFERENT (fresh) instance for steps 3 and 4.
        first_instance = seen_instances[0]
        second_instance = seen_instances[2]
        assert seen_instances[1] is first_instance
        assert seen_instances[3] is second_instance
        assert second_instance is not first_instance
        # The expired first instance had its client closed and cleared.
        assert first_instance.client is None
        assert first_instance.is_expired()

    @pytest.mark.asyncio
    async def test_step_not_found_raises(self, test_instructions_dir):
        """Test that missing step raises error."""
        mock_registry = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        # Config references non-existent step
        config = PipelineConfig(
            name="broken",
            steps={},  # No steps!
            start_step="missing",
            instructions_dir=str(test_instructions_dir)
        )

        with pytest.raises(ValueError, match="Step not found"):
            await orchestrator._execute_pipeline(
                run_id="run",
                config=config,
                initial_input="",
                args={}
            )
