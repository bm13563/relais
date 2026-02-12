"""Unit tests for pipeline.py - Pipeline class."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from relais.pipeline import Pipeline, cleanup_all_pipeline_states
from relais.step import PipelineStep
from relais.agent import PipelineAgent
from relais.state import PipelineRunState


# Test agent used across all tests
test_agent = PipelineAgent(name="test_agent", steps=None, model="opus")


class TestPipelineCreate:
    """Tests for Pipeline.create factory method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_create_minimal_pipeline(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test creating a minimal pipeline."""
        mock_state = MagicMock()
        mock_state_class.create.return_value = mock_state

        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {
            "start": PipelineStep(name="start", instruction="greet", next={"default": None}, agent=test_agent)
        }

        pipeline = Pipeline.create(
            name="test_pipeline",
            steps=steps,
            start_step="start",
            instructions_dir=test_instructions_dir,
            db_config={"host": "localhost", "database": "test"}
        )

        assert pipeline.name == "test_pipeline"
        assert pipeline.steps == steps
        assert pipeline.start_step == "start"
        assert pipeline.instructions_dir == test_instructions_dir

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_create_with_all_options(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test creating pipeline with all options specified."""
        mock_state = MagicMock()
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {
            "start": PipelineStep(name="start", instruction="greet", next={"default": "end"}, agent=test_agent),
            "end": PipelineStep(name="end", instruction="analyze", next={"default": None}, agent=test_agent)
        }

        pipeline = Pipeline.create(
            name="full_pipeline",
            steps=steps,
            start_step="start",
            instructions_dir=test_instructions_dir,
            db_config={"host": "db.example.com"},
            cwd="/tmp/work"
        )

        # Verify orchestrator was created with correct params
        call_kwargs = mock_orchestrator_class.call_args[1]
        assert call_kwargs["cwd"] == "/tmp/work"

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_create_registers_pipeline_with_orchestrator(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that Pipeline.create registers config with orchestrator."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        Pipeline.create(
            name="registered",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        mock_orchestrator.register_pipeline.assert_called_once()
        config = mock_orchestrator.register_pipeline.call_args[0][0]
        assert config.name == "registered"
        assert config.start_step == "s"


class TestPipelineTool:
    """Tests for Pipeline.tool decorator."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_tool_decorator_registers_tool(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that @pipeline.tool registers with tool registry."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="tooled",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        # Mock the tool registry's tool method
        mock_tool_decorator = MagicMock(return_value=lambda f: f)
        pipeline.tool_registry.tool = MagicMock(return_value=mock_tool_decorator)

        @pipeline.tool("my_tool", "My description")
        async def my_tool(args: dict) -> dict:
            return {"content": []}

        pipeline.tool_registry.tool.assert_called_once_with(
            "my_tool", "My description"
        )


class TestPipelineRun:
    """Tests for Pipeline.run method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_run_calls_orchestrator(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that run delegates to orchestrator."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.start_pipeline.return_value = "run-123"
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="runner",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        run_id = pipeline.run("initial input", {"key": "value"})

        assert run_id == "run-123"
        mock_orchestrator.start_pipeline.assert_called_once_with(
            pipeline_name="runner",
            initial_input="initial input",
            args={"key": "value"},
            session=None
        )

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_run_without_args(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test running pipeline without args."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.start_pipeline.return_value = "run-456"
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="simple",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        run_id = pipeline.run("just input")

        mock_orchestrator.start_pipeline.assert_called_once_with(
            pipeline_name="simple",
            initial_input="just input",
            args=None,
            session=None
        )


class TestPipelineResume:
    """Tests for Pipeline.resume method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_resume_calls_orchestrator(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that resume delegates to orchestrator."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="resumable",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        pipeline.resume("run-to-resume", "new input")

        mock_orchestrator.resume_pipeline.assert_called_once_with(
            "run-to-resume", "new input"
        )


class TestPipelineGetRun:
    """Tests for Pipeline.get_run method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_get_run_returns_state(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test getting run state."""
        now = datetime.now()
        mock_state = MagicMock()
        mock_state.get_pipeline_run.return_value = PipelineRunState(
            id="run-123",
            pipeline_name="test",
            current_step="step1",
            status="running",
            session=None,
            args={},
            conversation_history=[],
            step_results={},
            created_at=now,
            updated_at=now
        )
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        state = pipeline.get_run("run-123")

        assert state.id == "run-123"
        assert state.status == "running"
        mock_state.get_pipeline_run.assert_called_once_with("run-123")

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_get_run_not_found(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test getting non-existent run."""
        mock_state = MagicMock()
        mock_state.get_pipeline_run.return_value = None
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        state = pipeline.get_run("nonexistent")
        assert state is None


class TestPipelineListRuns:
    """Tests for Pipeline.list_runs method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_list_runs_filters_by_pipeline_name(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that list_runs filters by pipeline name."""
        mock_state = MagicMock()
        mock_state.get_pipeline_runs.return_value = []
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="my_pipeline",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        pipeline.list_runs(status="completed", limit=50)

        mock_state.get_pipeline_runs.assert_called_once_with(
            pipeline_name="my_pipeline",
            status="completed",
            limit=50
        )


class TestPipelineInitializeDb:
    """Tests for Pipeline.initialize_db method."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_initialize_db(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test database initialization."""
        mock_state = MagicMock()
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        pipeline.initialize_db()

        mock_state.initialize_schema.assert_called_once()


class TestCleanupAllPipelineStates:
    """Tests for cleanup_all_pipeline_states function."""

    def test_cleanup_deletes_all_runs(self):
        """Test that cleanup deletes all runs."""
        mock_state = MagicMock()
        now = datetime.now()
        mock_state.get_pipeline_runs.return_value = [
            PipelineRunState("run-1", "p", "s", "completed", None, {}, [], {}, now, now),
            PipelineRunState("run-2", "p", "s", "failed", None, {}, [], {}, now, now),
            PipelineRunState("run-3", "p", "s", "running", None, {}, [], {}, now, now),
        ]

        cleanup_all_pipeline_states(mock_state)

        assert mock_state.delete_pipeline_run.call_count == 3
        mock_state.delete_pipeline_run.assert_any_call("run-1")
        mock_state.delete_pipeline_run.assert_any_call("run-2")
        mock_state.delete_pipeline_run.assert_any_call("run-3")

    def test_cleanup_filters_by_pipeline_name(self):
        """Test that cleanup can filter by pipeline name."""
        mock_state = MagicMock()
        mock_state.get_pipeline_runs.return_value = []

        cleanup_all_pipeline_states(mock_state, pipeline_name="specific")

        mock_state.get_pipeline_runs.assert_called_once_with(
            pipeline_name="specific",
            limit=10000
        )

    def test_cleanup_no_runs(self):
        """Test cleanup when no runs exist."""
        mock_state = MagicMock()
        mock_state.get_pipeline_runs.return_value = []

        cleanup_all_pipeline_states(mock_state)

        mock_state.delete_pipeline_run.assert_not_called()


class TestPipelineAttributes:
    """Tests for Pipeline instance attributes."""

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_pipeline_has_tool_registry(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that pipeline has tool registry."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        assert pipeline.tool_registry is not None
        # Tool registry name should include pipeline name
        assert "test" in pipeline.tool_registry.name

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_pipeline_has_state_manager(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that pipeline has state manager."""
        mock_state = MagicMock()
        mock_state_class.create.return_value = mock_state
        mock_orchestrator_class.return_value = MagicMock()

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        assert pipeline.state_manager == mock_state

    @patch('relais.pipeline.SQLiteStateManager')
    @patch('relais.pipeline.PipelineOrchestrator')
    def test_pipeline_has_orchestrator(self, mock_orchestrator_class, mock_state_class, test_instructions_dir):
        """Test that pipeline has orchestrator."""
        mock_state_class.create.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        steps = {"s": PipelineStep(name="s", instruction="greet", agent=test_agent)}
        pipeline = Pipeline.create(
            name="test",
            steps=steps,
            start_step="s",
            instructions_dir=test_instructions_dir,
            db_config={}
        )

        assert pipeline.orchestrator == mock_orchestrator
