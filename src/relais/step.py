"""Pipeline step definition with SDK-compatible configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Any, Union

from .utils import read_markdown


@dataclass
class PipelineStep:
    """A single step in a pipeline.

    Attributes:
        name: Unique identifier for this step
        instruction: Name of the instruction file (without .md extension)
        max_turns: Maximum API round-trips before stopping (default: 10)
        tools: List of tools available for this step. Can be tool names (strings)
               or @tool decorated functions
        hooks: List of callable functions that provide context data
        next: Routing rules for determining the next step. Can be:
              - {"default": "next_step_name"} for simple routing
              - {"default": None} to end the pipeline
              - {"field": "field_name", "routes": [...], "default": "fallback"}
                for conditional routing based on tool result
        subagent: Whether to spawn an isolated subagent (no context sharing)
        subagent_model: Model override for this subagent step
        subagent_grounded: If True, inject grounding prompt for this subagent
        subagent_thinking: Enable/disable extended thinking for this subagent (None inherits)
    """
    name: str
    instruction: str
    max_turns: int = 10
    tools: List[Union[str, Callable]] = field(default_factory=list)
    hooks: List[Callable[[], Any]] = field(default_factory=list)
    next: dict = field(default_factory=lambda: {"default": None})
    subagent: bool = False
    subagent_model: Optional[str] = None
    subagent_grounded: Optional[bool] = None
    subagent_thinking: Optional[bool] = None

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

    def get_hook_data(self) -> List[Any]:
        """Execute all hooks and collect their data.

        Returns:
            List of results from each hook function
        """
        if not self.hooks:
            return []
        return [hook() for hook in self.hooks]
