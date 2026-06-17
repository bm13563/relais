"""Pipeline step definition with SDK-compatible configuration."""

from __future__ import annotations
import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Any, Union, TYPE_CHECKING

from .utils import read_markdown

if TYPE_CHECKING:
    from .agent import PipelineAgent


@dataclass
class PipelineStep:
    """A single step in a pipeline.

    Attributes:
        name: Unique identifier for this step
        instruction: Name of the instruction file (without .md extension)
        response_tool: Name of the tool that provides this step's structured output.
                       Required. The output of this tool becomes routing_data, which
                       is used for routing decisions and passed to the next step.
                       If the agent doesn't call this tool, the step fails.
        tools: List of tools available for this step. Can be tool names (strings)
               or @tool decorated functions
        hooks: List of callable functions that provide context data
        next: Routing rules for determining the next step. Can be:
              - {"default": "next_step_name"} for simple routing
              - {"default": None} to end the pipeline
              - {"field": "field_name", "routes": [...], "default": "fallback"}
                for conditional routing based on tool result
        agent: PipelineAgent instance to use for this step.
    """
    name: str
    instruction: str
    response_tool: str = ""  # Name of the tool that provides this step's output
    tools: List[Union[str, Callable]] = field(default_factory=list)
    hooks: List[Callable[[], Any]] = field(default_factory=list)
    next: dict = field(default_factory=lambda: {"default": None})
    agent: Optional['PipelineAgent'] = None

    def __post_init__(self):
        if not self.response_tool:
            raise ValueError(
                f"Step '{self.name}' is missing required 'response_tool'. "
                f"Every step must declare which tool provides its output."
            )

    def resolve_next(self, tool_result: dict) -> Optional[str]:
        """Determine the next step based on tool result.

        Args:
            tool_result: The result from the last tool execution

        Returns:
            Name of the next step, or None to end the pipeline
        """
        if "field" in self.next:
            field_value = tool_result.get(self.next["field"])
            for route in self.next.get("routes", []):
                if field_value == route.get("equals"):
                    return route.get("goto")
        return self.next.get("default")

    def get_instruction(self, instructions_dir: Path) -> str:
        """Load the instruction markdown file.

        Args:
            instructions_dir: Directory containing instruction files

        Returns:
            Contents of the instruction file
        """
        return read_markdown(f'{self.instruction}.md', instructions_dir)

    async def get_hook_data(self) -> List[Any]:
        """Execute all hooks and collect their data.

        Supports both sync and async hooks.

        Returns:
            List of results from each hook function
        """
        if not self.hooks:
            return []

        results = []
        for hook in self.hooks:
            if inspect.iscoroutinefunction(hook):
                result = await hook()
            else:
                result = hook()
            results.append(result)
        return results
