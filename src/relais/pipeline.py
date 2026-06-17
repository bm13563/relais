"""Pipeline system for multi-step AI agent workflows.

This module provides a high-level interface for defining and running pipelines
using the Claude Agent SDK.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

from .step import PipelineStep
from .tools import ToolRegistry, is_tool_function
from .state import SQLiteStateManager
from .executor import PipelineConfig, PipelineOrchestrator, StepExecutionResult
from .logging_config import get_logger

log = get_logger('pipeline')


class Pipeline:
    """High-level interface for creating and running pipelines.

    This class provides a simplified API that combines the orchestrator,
    tool registry, and state manager into a single interface.

    Usage:
        # Define steps
        steps = {
            "start": PipelineStep(
                name="start",
                instruction="start_instruction",
                max_turns=5,
                tools=["my_tool"],
                next={"default": "process"}
            ),
            "process": PipelineStep(
                name="process",
                instruction="process_instruction",
                max_turns=3,
                tools=["other_tool"],
                next={"default": None}
            )
        }

        # Create pipeline
        pipeline = Pipeline.create(
            name="my_pipeline",
            steps=steps,
            start_step="start",
            instructions_dir=Path("./instructions"),
            db_config="./pipeline.db"
        )

        # Register tools (async, with typed parameters)
        @pipeline.tool("my_tool", "Does something useful")
        async def my_tool(arg: Annotated[str, "An input value"]) -> dict:
            return {"content": [{"type": "text", "text": arg}]}

        # Run
        run_id = pipeline.run("User input here", args={"key": "value"})
    """

    def __init__(
        self,
        name: str,
        steps: Dict[str, PipelineStep],
        start_step: str,
        instructions_dir: Path,
        tool_registry: ToolRegistry,
        state_manager: SQLiteStateManager,
        orchestrator: PipelineOrchestrator,
        agents: Dict[str, 'PipelineAgent'] = None
    ):
        """Initialize the pipeline.

        Use Pipeline.create() instead of calling this directly.
        """
        self.name = name
        self.steps = steps
        self.start_step = start_step
        self.instructions_dir = instructions_dir
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.orchestrator = orchestrator
        self.agents = agents or {}

    @classmethod
    def create(
        cls,
        name: str,
        steps: Dict[str, PipelineStep],
        start_step: str,
        # Paths
        instructions_dir: Path,
        db_config: dict,
        cwd: str = None,
        # Debug
        verbose: bool = False,
    ) -> Pipeline:
        """Create a new pipeline.

        Steps can reference PipelineAgent instances directly via the agent parameter.
        The pipeline automatically collects all agents from steps.

        Args:
            name: Unique pipeline identifier
            steps: Dictionary of PipelineStep objects
            start_step: Name of the first step
            instructions_dir: Path to instruction markdown files
            db_config: Path to SQLite database file
            cwd: Working directory for file operations
            verbose: If True, print full step output to console

        Returns:
            Configured Pipeline instance
        """
        log.info("pipeline_create", pipeline=name, steps=len(steps))

        # Collect unique agents from all steps
        agents = {}
        for step in steps.values():
            if step.agent is None:
                raise ValueError(
                    f"Step '{step.name}' is missing required 'agent' parameter. "
                    f"Every step must have an explicit agent assigned."
                )
            agents[step.agent.name] = step.agent

        log.debug("agents_collected", pipeline=name, agents=list(agents.keys()))

        # Initialize components
        tool_registry = ToolRegistry(f"{name}_tools")
        state_manager = SQLiteStateManager.create(db_config)

        # Auto-register @tool decorated functions from agents and normalize to names
        # Agent tools are what get registered with the SDK client
        for agent in agents.values():
            normalized_tools = []
            for t in agent.tools:
                if is_tool_function(t):
                    tool_name = tool_registry.register_tool_function(t)
                    normalized_tools.append(tool_name)
                else:
                    normalized_tools.append(t)
            agent.tools = normalized_tools

        # Normalize step tools and register any that aren't already registered.
        # Step tools define which tools are available per-step (used for allowed_tools).
        for step in steps.values():
            normalized_tools = []
            for t in step.tools:
                if is_tool_function(t):
                    # Register if not already registered, get name
                    tool_name = tool_registry.register_tool_function(t)
                    normalized_tools.append(tool_name)
                else:
                    normalized_tools.append(t)
            step.tools = normalized_tools

        # Create orchestrator (SDK handles authentication via Claude Code)
        orchestrator = PipelineOrchestrator(
            tool_registry=tool_registry,
            state_manager=state_manager,
            instructions_dir=instructions_dir,
            cwd=cwd
        )

        # Register pipeline with orchestrator
        config = PipelineConfig(
            name=name,
            steps=steps,
            start_step=start_step,
            instructions_dir=str(instructions_dir),
            agents=agents,
            cwd=cwd,
            verbose=verbose,
        )
        orchestrator.register_pipeline(config)

        log.info("pipeline_created", pipeline=name)

        return cls(
            name=name,
            steps=steps,
            start_step=start_step,
            instructions_dir=instructions_dir,
            tool_registry=tool_registry,
            state_manager=state_manager,
            orchestrator=orchestrator,
            agents=agents
        )

    def tool(
        self,
        name: str,
        description: str,
    ):
        """Decorator for registering tools.

        Tools must be async functions that return the SDK tool result format.
        Parameter schema is automatically extracted from function signature.
        Use typing.Annotated to add descriptions to parameters.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Decorator function

        Example:
            from typing import Annotated

            @pipeline.tool("greet", "Greet the user")
            async def greet(
                name: Annotated[str, "The name to greet"],
                formal: Annotated[bool, "Use formal greeting"] = False,
            ) -> dict:
                greeting = "Good day" if formal else "Hello"
                return {
                    "content": [{"type": "text", "text": f"{greeting}, {name}!"}]
                }
        """
        return self.tool_registry.tool(name, description)

    def run(
        self,
        initial_input: str,
        args: dict = None,
    ) -> str:
        """Run the pipeline start-to-finish and return the run's UUID.

        Args:
            initial_input: Initial user input/prompt
            args: Pipeline arguments, surfaced to every step as [Pipeline Args]
        """
        return self.orchestrator.start_pipeline(
            pipeline_name=self.name,
            initial_input=initial_input,
            args=args,
        )

    def get_run(self, run_id: str):
        """Get the state of a pipeline run.

        Args:
            run_id: UUID of the run

        Returns:
            PipelineRunState or None
        """
        return self.state_manager.get_pipeline_run(run_id)

    def list_runs(
        self,
        status: str = None,
        limit: int = 100
    ):
        """List pipeline runs.

        Args:
            status: Filter by status (running, completed, failed)
            limit: Maximum results

        Returns:
            List of PipelineRunState
        """
        return self.state_manager.get_pipeline_runs(
            pipeline_name=self.name,
            status=status,
            limit=limit
        )

    def initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        self.state_manager.initialize_schema()


def cleanup_all_pipeline_states(state_manager: SQLiteStateManager, pipeline_name: str = None) -> None:
    """Delete all pipeline runs.

    Args:
        state_manager: State manager instance
        pipeline_name: Optional filter by pipeline name
    """
    runs = state_manager.get_pipeline_runs(pipeline_name=pipeline_name, limit=10000)
    for run in runs:
        state_manager.delete_pipeline_run(run.id)


# Re-export for backwards compatibility
__all__ = [
    'Pipeline',
    'PipelineStep',
    'PipelineConfig',
    'PipelineOrchestrator',
    'StepExecutionResult',
    'ToolRegistry',
    'SQLiteStateManager',
    'cleanup_all_pipeline_states',
]
