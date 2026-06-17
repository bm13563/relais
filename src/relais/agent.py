"""Agent definition for pipelines.

A PipelineAgent is who runs a step: it owns a model, a turn budget, a step budget,
and a fixed tool set. A live instance connects one ClaudeSDKClient when it first
runs and keeps it across the steps it participates in, so its conversation context
lives in the SDK client (in RAM). When its step budget is spent it expires; the
next time the pipeline routes to that agent, a fresh instance with clean context
takes over.
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
        steps: How many steps this agent participates in before expiring. None
               means it persists for the whole run (the default). N means a fresh
               instance gets a budget of N steps, carries context across them,
               then expires — the next route to this agent spins up a new instance
               with clean context. This is a stamina budget, like max_turns: the
               route determines when it's spent, the agent owns how much there is.
               Size N to a loop body to reset context on each loop-back; use a
               large N (or None) for refinement loops that should remember.
        max_turns: Hard ceiling on model round-trips per step (default: 10). One
               turn is one model response; a tool call consumes the next turn to
               feed the result back. A step that calls a tool, reads the result,
               then calls its response tool uses ~3 turns. Set lower to keep a
               step on a tight leash; raise it for agents that iterate.
        model: Model for this agent (opus, sonnet, haiku).
        thinking: Enable extended thinking for this agent.
        steps_remaining: Runtime counter for the current live instance (set from
                steps when the instance is created). None for persistent agents.
        client: The live ClaudeSDKClient, set on first run and reused across the
                steps this instance participates in.
    """

    name: str
    tools: List[Union[str, Callable]] = field(default_factory=list)
    steps: Optional[int] = None
    max_turns: int = 10
    model: Optional[str] = "opus"
    thinking: Optional[bool] = False
    steps_remaining: Optional[int] = field(default=None, init=False)
    client: Any = None  # ClaudeSDKClient; not imported to avoid a circular dep

    def __post_init__(self):
        self.steps_remaining = self.steps

    def consume_step(self) -> None:
        """Spend one step of this instance's budget (no-op if persistent)."""
        if self.steps_remaining is not None:
            self.steps_remaining -= 1

    def is_expired(self) -> bool:
        """Whether this instance has spent its step budget."""
        return self.steps_remaining is not None and self.steps_remaining <= 0

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
            and self.steps == other.steps
            and self.model == other.model
            and self.thinking == other.thinking
        )
