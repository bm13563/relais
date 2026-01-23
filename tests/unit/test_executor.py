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
    SubagentConfig
)
from relais.step import PipelineStep


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating config with minimal fields."""
        steps = {"start": PipelineStep(name="start", instruction="test")}
        config = PipelineConfig(
            name="test",
            steps=steps,
            start_step="start",
            instructions_dir="/path/to/instructions"
        )
        assert config.name == "test"
        assert config.start_step == "start"
        assert config.model == "sonnet"
        assert config.grounded is False
        assert config.cwd is None

    def test_create_full_config(self):
        """Test creating config with all fields."""
        steps = {"s": PipelineStep(name="s", instruction="test")}
        config = PipelineConfig(
            name="full",
            steps=steps,
            start_step="s",
            instructions_dir="/instructions",
            model="opus",
            grounded=True,
            cwd="/workdir"
        )
        assert config.model == "opus"
        assert config.grounded is True
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


class TestSubagentConfig:
    """Tests for SubagentConfig dataclass."""

    def test_create_subagent_config(self):
        """Test creating subagent config."""
        step = PipelineStep(name="research", instruction="research")
        config = SubagentConfig(
            step=step,
            context="Do research on topic X",
            parent_pipeline_id="parent-123"
        )
        assert config.step == step
        assert config.context == "Do research on topic X"
        assert config.parent_pipeline_id == "parent-123"


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
        assert orchestrator.model == "sonnet"
        assert orchestrator.pipelines == {}

    def test_create_orchestrator_with_options(self):
        """Test creating orchestrator with all options."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
            model="opus",
            cwd="/workdir"
        )

        assert orchestrator.model == "opus"
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

        steps = {"s": PipelineStep(name="s", instruction="test")}
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
            steps = {"s": PipelineStep(name="s", instruction="test")}
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

    def test_build_context_basic(self, test_instructions_dir):
        """Test building basic context."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step")
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = orchestrator._build_step_context(
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

    def test_build_context_with_args(self, test_instructions_dir):
        """Test context includes args."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step")
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = orchestrator._build_step_context(
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

    def test_build_context_with_previous_result(self, test_instructions_dir):
        """Test context includes previous result."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="test_step")
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = orchestrator._build_step_context(
            step=step,
            args={},
            previous_result={"output": "previous data"},
            initial_input=None,
            instructions_dir=test_instructions_dir,
            config=config
        )

        assert "[Previous Step Output]" in context
        assert "previous data" in context

    def test_build_context_with_hooks(self, test_instructions_dir):
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
            hooks=[my_hook]
        )
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        context = orchestrator._build_step_context(
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

    def test_build_context_missing_instruction(self, test_instructions_dir):
        """Test handling missing instruction file."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(name="test", instruction="nonexistent")
        config = PipelineConfig(
            name="test_pipeline",
            steps={"test": step},
            start_step="test",
            instructions_dir=str(test_instructions_dir)
        )
        # Should not raise, just log warning
        context = orchestrator._build_step_context(
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


class TestExtractRoutingData:
    """Tests for _extract_routing_data method."""

    def test_extract_from_empty_results(self):
        """Test extracting from empty tool results."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        result = orchestrator._extract_routing_data([])
        assert result is None

    def test_extract_from_json_string_output(self):
        """Test extracting from JSON string output."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {"tool": "classify", "output": '{"category": "question", "confidence": 0.9}'}
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"category": "question", "confidence": 0.9}

    def test_extract_from_dict_output(self):
        """Test extracting from dict output."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {"tool": "process", "output": {"status": "done", "count": 5}}
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"status": "done", "count": 5}

    def test_extract_from_mcp_format(self):
        """Test extracting from MCP tool result format."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {
                "tool": "analyze",
                "output": [
                    {"type": "text", "text": '{"result": "success"}'}
                ]
            }
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"result": "success"}

    def test_extract_from_mcp_wrapper_format(self):
        """Test extracting from MCP wrapper format with 'content' key."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        # This is the format tools return: {"content": [{"type": "text", "text": "..."}]}
        tool_results = [
            {
                "tool": "classify",
                "output": {
                    "content": [
                        {"type": "text", "text": '{"category": "question", "confidence": 0.95}'}
                    ]
                }
            }
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"category": "question", "confidence": 0.95}

    def test_extract_skips_tools_without_output(self):
        """Test that tools without output are skipped when extracting routing data."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        # First tool was called but not executed (no output), second has output
        tool_results = [
            {
                "id": "tool_1",
                "tool": "classify",
                "input": {"category": "question"},
                "output": {
                    "content": [
                        {"type": "text", "text": '{"category": "question"}'}
                    ]
                }
            },
            {
                "id": "tool_2",
                "tool": "unauthorized_tool",
                "input": {"data": "something"}
                # No output - tool was not executed
            }
        ]
        result = orchestrator._extract_routing_data(tool_results)

        # Should use the first tool's output (last one WITH output)
        assert result == {"category": "question"}

    def test_extract_returns_none_when_no_outputs(self):
        """Test that None is returned when no tools have output."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {"tool": "tool1", "input": {"data": "a"}},
            {"tool": "tool2", "input": {"data": "b"}}
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result is None

    def test_extract_non_json_string(self):
        """Test extracting non-JSON string wraps in response."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {"tool": "say", "output": "Just plain text"}
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"response": "Just plain text"}

    def test_extract_uses_last_result(self):
        """Test that last tool result is used."""
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions")
        )

        tool_results = [
            {"tool": "first", "output": '{"order": 1}'},
            {"tool": "second", "output": '{"order": 2}'},
            {"tool": "last", "output": '{"order": 3}'}
        ]
        result = orchestrator._extract_routing_data(tool_results)

        assert result == {"order": 3}


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

        steps = {"s": PipelineStep(name="s", instruction="test")}
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


class TestExecuteMainStep:
    """Tests for _execute_main_step method."""

    @pytest.mark.asyncio
    async def test_execute_main_step_creates_options(self):
        """Test that main step execution creates proper SDK options."""
        mock_registry = MagicMock()
        mock_registry.get_allowed_tools.return_value = ["mcp__test__tool1"]
        mock_registry.name = "test"

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=MagicMock(),
            instructions_dir=Path("/instructions"),
            model="sonnet",
            cwd="/work"
        )

        step = PipelineStep(
            name="main",
            instruction="test",
            max_turns=5,
            tools=["tool1"]
        )

        config = PipelineConfig(
            name="test",
            steps={"main": step},
            start_step="main",
            instructions_dir="/instructions",
            cwd="/work"
        )

        # We need to mock the ClaudeSDKClient
        with patch('relais.executor.ClaudeSDKClient') as mock_sdk_class:
            # Create mock client
            mock_client = MagicMock()
            mock_sdk_class.return_value.__aenter__.return_value = mock_client

            # Mock the query method
            async def mock_query(prompt):
                pass
            mock_client.query = mock_query

            # Mock receive_response to return async generator
            async def mock_receive():
                result_msg = MagicMock()
                result_msg.num_turns = 1
                result_msg.session_id = "session-1"
                result_msg.is_error = False
                type(result_msg).__name__ = "ResultMessage"
                yield result_msg

            mock_client.receive_response.return_value = mock_receive()

            result = await orchestrator._execute_main_step(
                step=step,
                context="Test context",
                mcp_server=MagicMock(),
                config=config
            )

            # Verify ClaudeSDKClient was instantiated with options
            mock_sdk_class.assert_called_once()
            call_kwargs = mock_sdk_class.call_args[1]
            options = call_kwargs["options"]
            assert options.max_turns == 5


class TestExecuteSubagentStep:
    """Tests for _execute_subagent_step method."""

    @pytest.mark.asyncio
    async def test_subagent_logs_to_db(self):
        """Test that subagent execution logs to database."""
        mock_registry = MagicMock()
        mock_registry.get_allowed_tools.return_value = []
        mock_registry.name = "test"

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=Path("/instructions")
        )

        step = PipelineStep(
            name="research",
            instruction="test",
            max_turns=10,
            subagent=True
        )

        config = PipelineConfig(
            name="test",
            steps={"research": step},
            start_step="research",
            instructions_dir="/instructions"
        )

        with patch('relais.executor.ClaudeSDKClient') as mock_sdk_class:
            # Create mock client
            mock_client = MagicMock()
            mock_sdk_class.return_value.__aenter__.return_value = mock_client

            # Mock the query method
            async def mock_query(prompt):
                pass
            mock_client.query = mock_query

            # Mock receive_response to return async generator
            async def mock_receive():
                result_msg = MagicMock()
                result_msg.num_turns = 2
                type(result_msg).__name__ = "ResultMessage"
                yield result_msg

            mock_client.receive_response.return_value = mock_receive()

            await orchestrator._execute_subagent_step(
                step=step,
                context="Research context",
                mcp_server=MagicMock(),
                run_id="parent-run-123",
                config=config
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

        mock_state = MagicMock()

        orchestrator = PipelineOrchestrator(
            tool_registry=mock_registry,
            state_manager=mock_state,
            instructions_dir=test_instructions_dir
        )

        step = PipelineStep(
            name="only",
            instruction="test_step",
            next={"default": None}  # Ends pipeline
        )

        config = PipelineConfig(
            name="single",
            steps={"only": step},
            start_step="only",
            instructions_dir=str(test_instructions_dir)
        )

        with patch.object(orchestrator, '_execute_main_step') as mock_execute:
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
                next={"default": "second"}
            ),
            "second": PipelineStep(
                name="second",
                instruction="test_step",
                next={"default": "third"}
            ),
            "third": PipelineStep(
                name="third",
                instruction="test_step",
                next={"default": None}
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

        with patch.object(orchestrator, '_execute_main_step', side_effect=mock_execute):
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
                next={
                    "field": "route",
                    "routes": [
                        {"equals": "A", "goto": "handle_a"},
                        {"equals": "B", "goto": "handle_b"}
                    ],
                    "default": "handle_default"
                }
            ),
            "handle_a": PipelineStep(name="handle_a", instruction="test_step", next={"default": None}),
            "handle_b": PipelineStep(name="handle_b", instruction="test_step", next={"default": None}),
            "handle_default": PipelineStep(name="handle_default", instruction="test_step", next={"default": None})
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

        with patch.object(orchestrator, '_execute_main_step', side_effect=mock_execute):
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
        orchestrator = PipelineOrchestrator(
            tool_registry=MagicMock(),
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
