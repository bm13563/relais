"""Unit tests for router.py - PipelineRouter class."""

import pytest
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

from relais.router import PipelineRouter


class TestPipelineRouterCreation:
    """Tests for PipelineRouter instantiation."""

    def test_create_router_default_prefix(self):
        """Test creating router with default command prefix."""
        router = PipelineRouter()
        assert router.command_prefix == "#"
        assert router.pipelines == {}

    def test_create_router_custom_prefix(self):
        """Test creating router with custom prefix."""
        router = PipelineRouter(command_prefix="/")
        assert router.command_prefix == "/"

    def test_create_router_special_prefix(self):
        """Test creating router with special character prefix."""
        router = PipelineRouter(command_prefix="@")
        assert router.command_prefix == "@"


class TestRegisterPipeline:
    """Tests for pipeline registration."""

    def test_register_pipeline(self):
        """Test registering a pipeline factory."""
        router = PipelineRouter()
        mock_factory = MagicMock()

        router.register("analyze", mock_factory)

        assert "analyze" in router.pipelines
        assert router.pipelines["analyze"] == mock_factory

    def test_register_multiple_pipelines(self):
        """Test registering multiple pipelines."""
        router = PipelineRouter()

        router.register("cmd1", MagicMock())
        router.register("cmd2", MagicMock())
        router.register("cmd3", MagicMock())

        assert len(router.pipelines) == 3
        assert "cmd1" in router.pipelines
        assert "cmd2" in router.pipelines
        assert "cmd3" in router.pipelines

    def test_register_overwrites_existing(self):
        """Test that registering same name overwrites."""
        router = PipelineRouter()
        factory1 = MagicMock()
        factory2 = MagicMock()

        router.register("cmd", factory1)
        router.register("cmd", factory2)

        assert router.pipelines["cmd"] == factory2


class TestStartPipeline:
    """Tests for starting pipelines."""

    def test_start_existing_pipeline(self):
        """Test starting a registered pipeline."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "run-id-123"
        mock_factory = MagicMock(return_value=mock_pipeline)

        router.register("test", mock_factory)
        run_id = router.start("test", "initial input", {"arg": "value"})

        assert run_id == "run-id-123"
        mock_factory.assert_called_once_with({"arg": "value"})
        mock_pipeline.run.assert_called_once_with("initial input", {"arg": "value"})

    def test_start_nonexistent_pipeline(self):
        """Test starting a pipeline that doesn't exist."""
        router = PipelineRouter()
        result = router.start("nonexistent", "input")

        assert result is None

    def test_start_with_no_args(self):
        """Test starting pipeline without args."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "id-456"
        mock_factory = MagicMock(return_value=mock_pipeline)

        router.register("simple", mock_factory)
        run_id = router.start("simple", "hello")

        mock_factory.assert_called_once_with(None)
        mock_pipeline.run.assert_called_once_with("hello", None)


class TestHandlePrompt:
    """Tests for handle_prompt method."""

    def test_handle_prompt_with_valid_command(self):
        """Test handling prompt with valid command."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "run-123"
        mock_factory = MagicMock(return_value=mock_pipeline)

        router.register("analyze", mock_factory)
        run_id = router.handle_prompt("#analyze some text here")

        assert run_id == "run-123"
        mock_pipeline.run.assert_called_once_with("some text here", None)

    def test_handle_prompt_command_no_args(self):
        """Test handling command without arguments."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "run-456"
        mock_factory = MagicMock(return_value=mock_pipeline)

        router.register("help", mock_factory)
        run_id = router.handle_prompt("#help")

        assert run_id == "run-456"
        mock_pipeline.run.assert_called_once_with("", None)

    def test_handle_prompt_no_command(self):
        """Test handling prompt without command."""
        router = PipelineRouter()
        router.register("test", MagicMock())

        result = router.handle_prompt("just regular text")
        assert result is None

    def test_handle_prompt_unregistered_command(self):
        """Test handling prompt with unregistered command."""
        router = PipelineRouter()
        router.register("other", MagicMock())

        result = router.handle_prompt("#unknown some args")
        assert result is None

    def test_handle_prompt_custom_prefix(self):
        """Test handling prompt with custom prefix."""
        router = PipelineRouter(command_prefix="/")
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "run-789"
        mock_factory = MagicMock(return_value=mock_pipeline)

        router.register("run", mock_factory)
        run_id = router.handle_prompt("/run my script")

        assert run_id == "run-789"

    def test_handle_prompt_empty_string(self):
        """Test handling empty prompt."""
        router = PipelineRouter()
        router.register("test", MagicMock())

        result = router.handle_prompt("")
        assert result is None

    def test_handle_prompt_whitespace_only(self):
        """Test handling whitespace-only prompt."""
        router = PipelineRouter()
        router.register("test", MagicMock())

        result = router.handle_prompt("   ")
        assert result is None


class TestRunCLI:
    """Tests for CLI run method."""

    def test_run_no_args_returns_early(self):
        """Test that run returns early with no CLI args."""
        router = PipelineRouter()
        original_argv = sys.argv

        try:
            sys.argv = ["script.py"]
            router.run()  # Should not raise
        finally:
            sys.argv = original_argv

    def test_run_start_action(self):
        """Test run with start action."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "run-cli-123"
        mock_factory = MagicMock(return_value=mock_pipeline)
        router.register("analyze", mock_factory)

        original_argv = sys.argv
        original_stdin = sys.stdin

        try:
            sys.argv = ["script.py", "start"]
            input_data = json.dumps({
                "prompt": "#analyze this text",
                "args": {"key": "value"}
            })
            sys.stdin = StringIO(input_data)

            # Capture stdout
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                router.run()
                output = mock_stdout.getvalue()

            result = json.loads(output)
            assert result["run_id"] == "run-cli-123"

        finally:
            sys.argv = original_argv
            sys.stdin = original_stdin

    def test_run_start_invalid_command(self):
        """Test run with unregistered command."""
        router = PipelineRouter()
        router.register("other", MagicMock())

        original_argv = sys.argv
        original_stdin = sys.stdin

        try:
            sys.argv = ["script.py", "start"]
            input_data = json.dumps({"prompt": "#nonexistent args"})
            sys.stdin = StringIO(input_data)

            # Should not crash, just return
            router.run()

        finally:
            sys.argv = original_argv
            sys.stdin = original_stdin

    def test_run_status_action(self):
        """Test run with status action."""
        router = PipelineRouter()

        mock_state = MagicMock()
        mock_state.id = "run-status-123"
        mock_state.status = "completed"
        mock_state.current_step = "final"
        mock_state.created_at = "2024-01-15 10:00:00"
        mock_state.updated_at = "2024-01-15 10:05:00"

        mock_pipeline = MagicMock()
        mock_pipeline.get_run.return_value = mock_state
        mock_factory = MagicMock(return_value=mock_pipeline)
        router.register("test", mock_factory)

        original_argv = sys.argv
        original_stdin = sys.stdin

        try:
            sys.argv = ["script.py", "status"]
            input_data = json.dumps({
                "run_id": "run-status-123",
                "pipeline": "test"
            })
            sys.stdin = StringIO(input_data)

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                router.run()
                output = mock_stdout.getvalue()

            result = json.loads(output)
            assert result["run_id"] == "run-status-123"
            assert result["status"] == "completed"

        finally:
            sys.argv = original_argv
            sys.stdin = original_stdin

    def test_run_status_missing_run_id(self):
        """Test status action with missing run_id."""
        router = PipelineRouter()
        router.register("test", MagicMock())

        original_argv = sys.argv
        original_stdin = sys.stdin

        try:
            sys.argv = ["script.py", "status"]
            input_data = json.dumps({"pipeline": "test"})
            sys.stdin = StringIO(input_data)

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                router.run()
                output = mock_stdout.getvalue()

            result = json.loads(output)
            assert "error" in result

        finally:
            sys.argv = original_argv
            sys.stdin = original_stdin

    def test_run_status_run_not_found(self):
        """Test status action when run doesn't exist."""
        router = PipelineRouter()

        mock_pipeline = MagicMock()
        mock_pipeline.get_run.return_value = None
        mock_factory = MagicMock(return_value=mock_pipeline)
        router.register("test", mock_factory)

        original_argv = sys.argv
        original_stdin = sys.stdin

        try:
            sys.argv = ["script.py", "status"]
            input_data = json.dumps({
                "run_id": "nonexistent",
                "pipeline": "test"
            })
            sys.stdin = StringIO(input_data)

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                router.run()
                output = mock_stdout.getvalue()

            result = json.loads(output)
            assert "error" in result

        finally:
            sys.argv = original_argv
            sys.stdin = original_stdin


class TestRouterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_factory_receives_correct_args(self):
        """Test that pipeline factory receives correct arguments."""
        router = PipelineRouter()
        received_args = {}

        def factory(args):
            received_args.update(args or {})
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = "id"
            return mock_pipeline

        router.register("test", factory)
        router.start("test", "input", {"a": 1, "b": 2})

        assert received_args == {"a": 1, "b": 2}

    def test_pipeline_run_receives_correct_input(self):
        """Test that pipeline.run receives correct initial input."""
        router = PipelineRouter()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "id"

        router.register("test", MagicMock(return_value=mock_pipeline))
        router.handle_prompt("#test hello world this is input")

        mock_pipeline.run.assert_called_once()
        call_args = mock_pipeline.run.call_args[0]
        assert call_args[0] == "hello world this is input"
