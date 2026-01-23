"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from relais.step import PipelineStep
from relais.tools import ToolRegistry
from relais.state import SQLiteStateManager, PipelineRunState


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def test_instructions_dir(tmp_path):
    """Create a temporary instructions directory with test files."""
    instructions_dir = tmp_path / "instructions"
    instructions_dir.mkdir()

    # Create sample instruction files
    (instructions_dir / "test_step.md").write_text("# Test Step\nDo the test task.")
    (instructions_dir / "greet.md").write_text("# Greeting\nGreet the user warmly.")
    (instructions_dir / "analyze.md").write_text("# Analyze\nAnalyze the input.")
    (instructions_dir / "process.md").write_text("# Process\nProcess the data.")

    return instructions_dir


@pytest.fixture
def db_path(tmp_path):
    """SQLite database path for tests."""
    return str(tmp_path / "test_pipeline.db")


# =============================================================================
# Step Fixtures
# =============================================================================

@pytest.fixture
def simple_step():
    """A simple step with default routing (ends pipeline)."""
    return PipelineStep(
        name="simple",
        instruction="test_step",
        max_turns=5,
        tools=["test_tool"],
        next={"default": None}
    )


@pytest.fixture
def routing_step():
    """A step with conditional routing based on category field."""
    return PipelineStep(
        name="router",
        instruction="analyze",
        max_turns=3,
        tools=["classify"],
        next={
            "field": "category",
            "routes": [
                {"equals": "question", "goto": "answer"},
                {"equals": "task", "goto": "execute"},
                {"equals": "chat", "goto": "respond"},
            ],
            "default": "fallback"
        }
    )


@pytest.fixture
def step_with_hooks():
    """A step with hook functions for context injection."""
    def time_hook():
        return {"timestamp": "2024-01-15T10:30:00"}

    def user_hook():
        return {"user": "test_user", "role": "admin"}

    return PipelineStep(
        name="hooked",
        instruction="test_step",
        max_turns=5,
        tools=["context_tool"],
        hooks=[time_hook, user_hook],
        next={"default": None}
    )


@pytest.fixture
def subagent_step():
    """A step that runs as an isolated subagent."""
    return PipelineStep(
        name="subagent_step",
        instruction="process",
        max_turns=10,
        tools=["research_tool"],
        use_subagent=True,
        model="haiku",
        next={"default": "summary"}
    )


# =============================================================================
# Tool Registry Fixtures
# =============================================================================

@pytest.fixture
def tool_registry():
    """A tool registry with test tools registered."""
    registry = ToolRegistry("test_tools")
    return registry


@pytest.fixture
def populated_tool_registry(tool_registry):
    """A tool registry with several test tools registered."""
    @tool_registry.tool("greet", "Greet the user", {"name": str})
    async def greet(args: dict) -> dict:
        return {"content": [{"type": "text", "text": f"Hello {args.get('name', 'user')}!"}]}

    @tool_registry.tool("calculate", "Perform calculation", {"a": int, "b": int, "op": str})
    async def calculate(args: dict) -> dict:
        a, b = args.get("a", 0), args.get("b", 0)
        op = args.get("op", "add")
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        else:
            result = 0
        return {"content": [{"type": "text", "text": str(result)}]}

    return registry


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_state_manager():
    """Mock state manager for unit tests."""
    manager = MagicMock(spec=SQLiteStateManager)
    manager.create_pipeline_run.return_value = "test-run-id-123"
    manager.get_pipeline_run.return_value = PipelineRunState(
        id="test-run-id-123",
        pipeline_name="test_pipeline",
        current_step="test_step",
        status="running",
        args={},
        conversation_history=[],
        step_results={},
        created_at=None,
        updated_at=None
    )
    return manager


@pytest.fixture
def mock_sdk_query():
    """Mock for claude_agent_sdk.query function."""
    async def mock_query(*args, **kwargs):
        # Yield a minimal successful response
        from unittest.mock import MagicMock

        # Create mock assistant message with text block
        text_block = MagicMock()
        text_block.text = "Test response from Claude"

        assistant_msg = MagicMock()
        assistant_msg.content = [text_block]
        type(assistant_msg).__name__ = "AssistantMessage"

        # Create mock result message
        result_msg = MagicMock()
        result_msg.num_turns = 1
        result_msg.session_id = "mock-session-123"
        result_msg.is_error = False
        type(result_msg).__name__ = "ResultMessage"

        yield assistant_msg
        yield result_msg

    return mock_query


# =============================================================================
# Pipeline Fixtures
# =============================================================================

@pytest.fixture
def sample_steps():
    """A dictionary of sample steps for pipeline tests."""
    return {
        "start": PipelineStep(
            name="start",
            instruction="greet",
            max_turns=3,
            tools=["greet"],
            next={"default": "process"}
        ),
        "process": PipelineStep(
            name="process",
            instruction="process",
            max_turns=5,
            tools=["calculate"],
            next={"default": None}
        )
    }


@pytest.fixture
def routing_steps():
    """Steps for testing conditional routing."""
    return {
        "analyze": PipelineStep(
            name="analyze",
            instruction="analyze",
            max_turns=2,
            tools=["classify"],
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                    {"equals": "B", "goto": "handle_b"},
                ],
                "default": "handle_default"
            }
        ),
        "handle_a": PipelineStep(
            name="handle_a",
            instruction="process",
            max_turns=3,
            tools=["process_a"],
            next={"default": None}
        ),
        "handle_b": PipelineStep(
            name="handle_b",
            instruction="process",
            max_turns=3,
            tools=["process_b"],
            next={"default": None}
        ),
        "handle_default": PipelineStep(
            name="handle_default",
            instruction="process",
            max_turns=3,
            tools=["process_default"],
            next={"default": None}
        )
    }
