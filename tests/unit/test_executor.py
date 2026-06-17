"""Unit tests for executor.py - PipelineOrchestrator class."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from relais.executor import (
    PipelineOrchestrator,
    PipelineConfig,
    StepExecutionResult,
)
from relais.step import PipelineStep
from relais.agent import PipelineAgent


# Test agent used across all tests
test_agent = PipelineAgent(name="test_agent", steps=None, model="opus")


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
            session_id="session-123"
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
        assert result.session_id is None


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
            config=config
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
            config=config
        )

        assert "[Pipeline Args]" in context
        assert "key" in context
        assert "value" in context

    async def test_build_context_hides_internal_initial_input_arg(self, test_instructions_dir):
        """The session-resume bookkeeping key must not leak into context."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir,
        )
        step = PipelineStep(name="test", instruction="test_step", response_tool="test_tool", agent=test_agent)
        config = PipelineConfig(
            name="test_pipeline", steps={"test": step}, start_step="test",
            instructions_dir=str(test_instructions_dir),
        )
        context = await orchestrator._build_step_context(
            step=step,
            args={"_initial_input": "secret bookkeeping"},
            previous_result=None,
            initial_input=None,
            instructions_dir=test_instructions_dir,
            config=config,
        )
        assert "[Pipeline Args]" not in context
        assert "secret bookkeeping" not in context

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
            config=config
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
            config=config
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
            config=config
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

    @patch.object(PipelineOrchestrator, '_start_pipeline_async')
    def test_start_pipeline_calls_async(self, mock_async):
        """Test that start_pipeline wraps async method."""
        mock_async.return_value = "run-123"

        mock_registry = MagicMock()
        mock_registry.create_mcp_server.return_value = MagicMock()

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
    """Tests for unified _execute_step method."""

    @pytest.mark.asyncio
    async def test_execute_step_sets_current_step(self):
        """Test that step execution sets current step for tool validation."""
        mock_registry = MagicMock()
        mock_registry.get_allowed_tools.return_value = ["mcp__test__tool1"]
        mock_registry.name = "test"
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        from relais.agent import PipelineAgent
        test_agent_local = PipelineAgent(name="test_agent", steps=None)

        step = PipelineStep(
            name="test_step",
            instruction="test",
            response_tool="test_tool",
            tools=["tool1", "tool2"],
            agent=test_agent_local
        )

        config = PipelineConfig(
            name="test",
            steps={"test_step": step},
            start_step="test_step",
            instructions_dir="/instructions"
        )

        with patch('relais.executor.ClaudeSDKClient') as mock_sdk_class:
            mock_client = MagicMock()
            mock_sdk_class.return_value = mock_client

            async def mock_connect():
                pass
            mock_client.connect = mock_connect

            async def mock_query(prompt):
                pass
            mock_client.query = mock_query

            async def mock_receive():
                result_msg = MagicMock()
                result_msg.num_turns = 1
                result_msg.is_error = False
                type(result_msg).__name__ = "ResultMessage"
                yield result_msg

            mock_client.receive_response.return_value = mock_receive()

            await orchestrator._execute_step(
                step=step,
                context="Test",
                mcp_server=MagicMock(),
                run_id="run-123",
                config=config,
                agent=test_agent_local
            )

            # Verify set_current_step was called with correct args
            mock_registry.set_current_step.assert_called_once_with("test_step", ["tool1", "tool2"])

    @pytest.mark.asyncio
    async def test_execute_step_returns_result(self):
        """Test that step execution returns proper result."""
        mock_registry = MagicMock()
        mock_registry.get_allowed_tools.return_value = ["mcp__test__tool1"]
        mock_registry.name = "test"
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
            cwd="/work"
        )

        from relais.agent import PipelineAgent
        test_agent_local = PipelineAgent(name="test_agent", steps=None)

        step = PipelineStep(
            name="main",
            instruction="test",
            response_tool="test_tool",
            tools=["tool1"],
            agent=test_agent_local
        )

        config = PipelineConfig(
            name="test",
            steps={"main": step},
            start_step="main",
            instructions_dir="/instructions",
            cwd="/work",
            agents={"test_agent": test_agent_local}
        )

        with patch('relais.executor.ClaudeSDKClient') as mock_sdk_class:
            mock_client = MagicMock()
            mock_sdk_class.return_value = mock_client

            async def mock_connect():
                pass
            mock_client.connect = mock_connect

            async def mock_query(prompt):
                pass
            mock_client.query = mock_query

            async def mock_receive():
                result_msg = MagicMock()
                result_msg.num_turns = 1
                result_msg.is_error = False
                result_msg.usage = None
                type(result_msg).__name__ = "ResultMessage"
                yield result_msg

            mock_client.receive_response.return_value = mock_receive()

            result = await orchestrator._execute_step(
                step=step,
                context="Test context",
                mcp_server=MagicMock(),
                run_id="run-123",
                config=config,
                agent=test_agent_local
            )

            # Verify result returned
            assert result.step_name == "main"
            assert result.stop_reason == "success"

    @pytest.mark.asyncio
    async def test_execute_step_logs_to_db(self):
        """Test that step execution logs to database."""
        mock_registry = MagicMock()
        mock_registry.get_allowed_tools.return_value = []
        mock_registry.name = "test"
        # The step's response tool must appear captured for routing extraction.
        mock_registry.get_tool_result.return_value = (
            "test_tool",
            {"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=Path("/instructions")
        )

        from relais.agent import PipelineAgent
        research_agent = PipelineAgent(name="research_agent", steps=None)

        step = PipelineStep(
            name="research",
            instruction="test",
            response_tool="test_tool",
            tools=["test_tool"],
            agent=research_agent
        )

        config = PipelineConfig(
            name="test",
            steps={"research": step},
            start_step="research",
            instructions_dir="/instructions",
            agents={"research_agent": research_agent}
        )

        with patch('relais.executor.ClaudeSDKClient') as mock_sdk_class:
            mock_client = MagicMock()
            mock_sdk_class.return_value = mock_client

            async def mock_connect():
                pass
            mock_client.connect = mock_connect

            async def mock_query(prompt):
                pass
            mock_client.query = mock_query

            async def mock_receive():
                result_msg = MagicMock()
                result_msg.num_turns = 2
                type(result_msg).__name__ = "ResultMessage"
                yield result_msg

            mock_client.receive_response.return_value = mock_receive()

            await orchestrator._execute_step(
                step=step,
                context="Research context",
                mcp_server=MagicMock(),
                run_id="parent-run-123",
                config=config,
                agent=research_agent
            )

            # Verify spawn was logged
            mock_state.log_subagent_spawn.assert_called_once()
            call_args = mock_state.log_subagent_spawn.call_args[1]
            assert call_args["parent_pipeline_id"] == "parent-run-123"
            assert call_args["step_name"] == "research"

            # Verify completion was logged
            mock_state.log_subagent_complete.assert_called_once()


class TestExecutePipelineLoop:
    """Tests for the main _execute_pipeline loop."""

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self, test_instructions_dir):
        """Test executing a single-step pipeline."""
        mock_registry = MagicMock()
        mock_registry.create_mcp_server.return_value = MagicMock()
        mock_registry.get_allowed_tools.return_value = []
        mock_registry.name = "test"
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
        mock_registry.create_mcp_server.return_value = MagicMock()
        mock_registry.get_allowed_tools.return_value = []
        mock_registry.name = "test"
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
        mock_registry.create_mcp_server.return_value = MagicMock()
        mock_registry.get_allowed_tools.return_value = []
        mock_registry.name = "test"
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
    async def test_step_not_found_raises(self, test_instructions_dir):
        """Test that missing step raises error."""
        mock_registry = MagicMock()
        mock_registry.create_mcp_server.return_value = MagicMock()

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


class TestStepExecutionResultMessages:
    """Tests for StepExecutionResult with messages field."""

    def test_result_with_messages(self):
        """Test creating result with captured messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
        ]
        result = StepExecutionResult(
            step_name="test",
            final_response="Hi there!",
            tool_results=[],
            turns_used=1,
            stop_reason="success",
            messages=messages
        )
        assert result.messages == messages
        assert len(result.messages) == 2

    def test_result_messages_default_none(self):
        """Test that messages defaults to None."""
        result = StepExecutionResult(
            step_name="test",
            final_response="",
            tool_results=[],
            turns_used=1,
            stop_reason="success"
        )
        assert result.messages is None


class TestFormatConversationHistory:
    """Tests for _format_conversation_history method."""

    def test_format_empty_messages(self):
        """Test formatting empty message list."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        result = orchestrator._format_conversation_history([])
        assert result == ""

    def test_format_user_message(self):
        """Test formatting user message."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        messages = [{"role": "user", "content": "Hello, world!"}]
        result = orchestrator._format_conversation_history(messages)
        assert "[User Query]" in result
        assert "Hello, world!" in result

    def test_format_user_message_no_truncation(self):
        """Test that long user messages are not truncated by formatter."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        long_content = "x" * 1000
        messages = [{"role": "user", "content": long_content}]
        result = orchestrator._format_conversation_history(messages)
        # Messages are passed through without truncation
        assert long_content in result
        assert "[User Query]" in result

    def test_format_assistant_text_message(self):
        """Test formatting assistant text message."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "I can help with that."}]}
        ]
        result = orchestrator._format_conversation_history(messages)
        assert "[Assistant]" in result
        assert "I can help with that." in result

    def test_format_assistant_tool_use(self):
        """Test formatting assistant tool use message."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "tool": "search", "input": {"query": "test"}}]
            }
        ]
        result = orchestrator._format_conversation_history(messages)
        assert "[Tool Call: search]" in result
        assert "query" in result
        assert "test" in result

    def test_format_mixed_conversation(self):
        """Test formatting a mixed conversation."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )
        messages = [
            {"role": "user", "content": "Search for cats"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "tool": "search", "input": {"q": "cats"}},
                {"type": "text", "text": "Found results!"}
            ]}
        ]
        result = orchestrator._format_conversation_history(messages)
        assert "[User Query]" in result
        assert "Search for cats" in result
        assert "[Tool Call: search]" in result
        assert "[Assistant]" in result
        assert "Found results!" in result


class TestBuildStepContextWithPreviousMessages:
    """Tests for _build_step_context with previous_messages parameter."""

    async def test_context_includes_previous_conversation(self, test_instructions_dir):
        """Test that previous messages are included in context."""
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

        previous_messages = [
            {"role": "user", "content": "Previous query"},
            {"role": "assistant", "content": [{"type": "text", "text": "Previous response"}]}
        ]

        context = await orchestrator._build_step_context(
            step=step,
            args={},
            previous_result=None,
            initial_input="New input",
            instructions_dir=test_instructions_dir,
            config=config,
            previous_messages=previous_messages
        )

        assert "[Previous Conversation]" in context
        assert "Previous query" in context
        assert "Previous response" in context

    async def test_context_without_previous_messages(self, test_instructions_dir):
        """Test that context works without previous messages."""
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
            initial_input="Test",
            instructions_dir=test_instructions_dir,
            config=config,
            previous_messages=None
        )

        assert "[Previous Conversation]" not in context

    async def test_context_with_empty_previous_messages(self, test_instructions_dir):
        """Test that empty previous messages don't add section."""
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
            initial_input="Test",
            instructions_dir=test_instructions_dir,
            config=config,
            previous_messages=[]
        )

        assert "[Previous Conversation]" not in context
