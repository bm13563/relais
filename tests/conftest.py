"""Pytest configuration and shared fixtures.

All fixtures target the current agent-based API:
- Every PipelineStep carries an explicit `agent=PipelineAgent(...)`.
- `max_turns`, `model`, `thinking` live on the agent, not the step.
- Every step declares a `response_tool`.
- Hooks may be sync or async; `get_hook_data()` is async.

Unit tests are fully offline: no real model calls, no real filesystem writes
from mocked collaborators.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from relais.step import PipelineStep
from relais.agent import PipelineAgent
from relais.tools import ToolRegistry
from relais.state import SQLiteStateManager, PipelineRunState


# =============================================================================
# Helpers
# =============================================================================

def make_agent(name="test_agent", tools=None, **kwargs):
    """Build a PipelineAgent for a step fixture.

    Keeps test step definitions terse while honoring the rule that every step
    must carry an explicit agent.
    """
    return PipelineAgent(name=name, tools=list(tools or []), **kwargs)


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def test_instructions_dir(tmp_path):
    """Create a temporary instructions directory with test files."""
    instructions_dir = tmp_path / "instructions"
    instructions_dir.mkdir()

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
        response_tool="test_tool",
        tools=["test_tool"],
        agent=make_agent("simple_agent", tools=["test_tool"], max_turns=5),
        next={"default": None},
    )


@pytest.fixture
def routing_step():
    """A step with conditional routing based on category field."""
    return PipelineStep(
        name="router",
        instruction="analyze",
        response_tool="classify",
        tools=["classify"],
        agent=make_agent("router_agent", tools=["classify"], max_turns=3),
        next={
            "field": "category",
            "routes": [
                {"equals": "question", "goto": "answer"},
                {"equals": "task", "goto": "execute"},
                {"equals": "chat", "goto": "respond"},
            ],
            "default": "fallback",
        },
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
        response_tool="context_tool",
        tools=["context_tool"],
        hooks=[time_hook, user_hook],
        agent=make_agent("hooked_agent", tools=["context_tool"], max_turns=5),
        next={"default": None},
    )


@pytest.fixture
def limited_agent_step():
    """A step driven by a step-limited (non-persistent) agent."""
    return PipelineStep(
        name="limited_step",
        instruction="process",
        response_tool="research_tool",
        tools=["research_tool"],
        agent=make_agent(
            "limited_agent",
            tools=["research_tool"],
            steps=1,
            max_turns=10,
            model="haiku",
        ),
        next={"default": "summary"},
    )


# =============================================================================
# Tool Registry Fixtures
# =============================================================================

@pytest.fixture
def tool_registry():
    """An empty tool registry."""
    return ToolRegistry("test_tools")


@pytest.fixture
def populated_tool_registry(tool_registry):
    """A tool registry with several test tools registered."""
    from typing import Annotated

    @tool_registry.tool("greet", "Greet the user")
    async def greet(name: Annotated[str, "The name to greet"]) -> dict:
        return {"content": [{"type": "text", "text": f"Hello {name}!"}]}

    @tool_registry.tool("calculate", "Perform calculation")
    async def calculate(
        a: Annotated[int, "First operand"],
        b: Annotated[int, "Second operand"],
        op: Annotated[str, "Operation: add/subtract/multiply"],
    ) -> dict:
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        else:
            result = 0
        return {"content": [{"type": "text", "text": str(result)}]}

    return tool_registry


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_state_manager(tmp_path):
    """Mock state manager for unit tests.

    `db_path` is a real string for the rare collaborator that reads it; the
    orchestrator no longer derives any secondary path from it.
    """
    manager = MagicMock(spec=SQLiteStateManager)
    manager.db_path = str(tmp_path / "mock_pipeline.db")
    manager.create_pipeline_run.return_value = "test-run-id-123"
    manager.get_pipeline_run.return_value = PipelineRunState(
        id="test-run-id-123",
        pipeline_name="test_pipeline",
        current_step="test_step",
        status="running",
        session=None,
        args={},
        conversation_history=[],
        step_results={},
        created_at=None,
        updated_at=None,
    )
    return manager


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
            response_tool="greet",
            tools=["greet"],
            agent=make_agent("start_agent", tools=["greet"], max_turns=3),
            next={"default": "process"},
        ),
        "process": PipelineStep(
            name="process",
            instruction="process",
            response_tool="calculate",
            tools=["calculate"],
            agent=make_agent("process_agent", tools=["calculate"], max_turns=5),
            next={"default": None},
        ),
    }


@pytest.fixture
def routing_steps():
    """Steps for testing conditional routing."""
    return {
        "analyze": PipelineStep(
            name="analyze",
            instruction="analyze",
            response_tool="classify",
            tools=["classify"],
            agent=make_agent("analyze_agent", tools=["classify"], max_turns=2),
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                    {"equals": "B", "goto": "handle_b"},
                ],
                "default": "handle_default",
            },
        ),
        "handle_a": PipelineStep(
            name="handle_a",
            instruction="process",
            response_tool="process_a",
            tools=["process_a"],
            agent=make_agent("handle_a_agent", tools=["process_a"], max_turns=3),
            next={"default": None},
        ),
        "handle_b": PipelineStep(
            name="handle_b",
            instruction="process",
            response_tool="process_b",
            tools=["process_b"],
            agent=make_agent("handle_b_agent", tools=["process_b"], max_turns=3),
            next={"default": None},
        ),
        "handle_default": PipelineStep(
            name="handle_default",
            instruction="process",
            response_tool="process_default",
            tools=["process_default"],
            agent=make_agent("handle_default_agent", tools=["process_default"], max_turns=3),
            next={"default": None},
        ),
    }
