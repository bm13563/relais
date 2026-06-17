"""Agent definition for pipelines.

A PipelineAgent is who runs a step: it owns a model, a turn budget, and a fixed
tool set. An agent connects one live ClaudeSDKClient when it first runs and keeps
it for the whole pipeline run, so its conversation context lives in the SDK client
(in RAM) rather than being replayed from stored text.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Union, Any


@dataclass
class PipelineAgent:
    """An agent that executes pipeline steps.

    Attributes:
        name: Unique identifier for this agent within a pipeline.
        tools: Tools available to this agent. Tool names (strings) or @tool
               decorated functions. This is the agent's full tool set; per-step
               access is further scoped by each step's own tools list.
        max_turns: Hard ceiling on model round-trips per step (default: 10). One
               turn is one model response; a tool call consumes the next turn to
               feed the result back. A step that calls a tool, reads the result,
               then calls its response tool uses ~3 turns. Set lower to keep a
               step on a tight leash; raise it for agents that iterate.
        model: Model for this agent (opus, sonnet, haiku).
        thinking: Enable extended thinking for this agent.
        client: The live ClaudeSDKClient, set on first run and reused for the
                rest of the pipeline run.
    """

    name: str
    tools: List[Union[str, Callable]] = field(default_factory=list)
    max_turns: int = 10
    model: Optional[str] = "opus"
    thinking: Optional[bool] = False
    client: Any = None  # ClaudeSDKClient; not imported to avoid a circular dep

    def set_client(self, client: Any) -> None:
        """Attach the live ClaudeSDKClient for this agent."""
        self.client = client

    def has_client(self) -> bool:
        """Whether this agent already has a live client."""
        return self.client is not None

    async def disconnect(self) -> None:
        """Disconnect the agent's client if it has one."""
        if self.client is not None:
            await self.client.disconnect()

    def __eq__(self, other: object) -> bool:
        """Agents are equal when their static configuration matches."""
        if not isinstance(other, PipelineAgent):
            return False
        return (
            self.name == other.name
            and self.tools == other.tools
            and self.model == other.model
            and self.thinking == other.thinking
        )
