"""Agent management for multi-step pipelines.

This module provides the PipelineAgent class which represents an agent instance
that can persist across multiple pipeline steps or be scoped to a specific number
of steps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any, Union


@dataclass
class PipelineAgent:
    """An agent instance that executes pipeline steps.

    Attributes:
        name: Unique identifier for this agent
        tools: List of tools available to this agent. Can be tool names (strings)
               or @tool decorated functions. These are registered with the SDK client.
        steps: Number of steps this agent is available for. None means persistent
               across all steps. Integer N means available for N steps then expires.
        max_turns: Maximum API round-trips per step before stopping (default: 10)
        model: Model override for this agent (opus, sonnet, haiku)
        thinking: Enable/disable extended thinking for this agent (None inherits)
        conversation_history: Message history for this agent
        steps_remaining: Tracks how many steps remain (None for persistent agents)
        client: The ClaudeSDKClient instance (set during execution)
    """

    name: str
    tools: List[Union[str, Callable]] = field(default_factory=list)
    steps: Optional[int] = None
    max_turns: int = 2
    model: Optional[str] = "opus"
    thinking: Optional[bool] = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    steps_remaining: Optional[int] = None
    client: Any = None  # ClaudeSDKClient, but we don't import to avoid circular deps

    def __post_init__(self):
        """Initialize steps_remaining from steps if applicable."""
        if self.steps is not None:
            self.steps_remaining = self.steps

    def is_persistent(self) -> bool:
        """Check if this is a persistent agent (steps=None).

        Returns:
            True if agent persists across all steps, False if it has a step limit
        """
        return self.steps is None

    def consume_step(self) -> None:
        """Consume one step from the agent's lifetime.

        For persistent agents (steps=None), this is a no-op.
        For limited agents, decrements steps_remaining.
        """
        if self.steps_remaining is not None:
            self.steps_remaining -= 1

    def is_expired(self) -> bool:
        """Check if the agent has expired (no steps remaining).

        Returns:
            True if agent has exhausted its steps, False otherwise.
            Persistent agents never expire.
        """
        if self.is_persistent():
            return False
        return self.steps_remaining is not None and self.steps_remaining <= 0

    def reset(self) -> None:
        """Reset the agent's step counter to its original value.

        For persistent agents, this is a no-op.
        """
        if self.steps is not None:
            self.steps_remaining = self.steps

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a single message to conversation history.

        Args:
            message: Message dict with role and content
        """
        self.conversation_history.append(message)

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Add multiple messages to conversation history.

        Args:
            messages: List of message dicts
        """
        self.conversation_history.extend(messages)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the conversation history.

        Returns:
            Copy of conversation history list
        """
        return self.conversation_history.copy()

    def set_client(self, client: Any) -> None:
        """Set the ClaudeSDKClient instance for this agent.

        Args:
            client: ClaudeSDKClient instance
        """
        self.client = client

    def has_client(self) -> bool:
        """Check if this agent has a client set.

        Returns:
            True if client is set, False otherwise
        """
        return self.client is not None

    async def disconnect(self) -> None:
        """Disconnect the agent's client if it exists."""
        if self.client is not None:
            await self.client.disconnect()

    def __eq__(self, other: object) -> bool:
        """Check equality based on name and configuration.

        Args:
            other: Another agent to compare

        Returns:
            True if agents have same name and configuration
        """
        if not isinstance(other, PipelineAgent):
            return False
        return (
            self.name == other.name
            and self.tools == other.tools
            and self.steps == other.steps
            and self.model == other.model
            and self.thinking == other.thinking
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary for persistence.

        Returns:
            Dictionary containing agent state
        """
        # Convert tools to names (handles both strings and @tool decorated functions)
        tool_names = []
        for t in self.tools:
            if callable(t) and hasattr(t, '_tool_name'):
                tool_names.append(t._tool_name)
            elif isinstance(t, str):
                tool_names.append(t)

        return {
            "name": self.name,
            "tools": tool_names,
            "steps": self.steps,
            "steps_remaining": self.steps_remaining,
            "max_turns": self.max_turns,
            "model": self.model,
            "thinking": self.thinking,
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineAgent:
        """Restore agent from dictionary.

        Args:
            data: Dictionary containing agent state

        Returns:
            Restored PipelineAgent instance
        """
        agent = cls(
            name=data["name"],
            tools=data.get("tools", []),
            steps=data.get("steps"),
            max_turns=data.get("max_turns", 10),
            model=data.get("model"),
            thinking=data.get("thinking"),
        )
        # Manually set steps_remaining since it's set in __post_init__
        agent.steps_remaining = data.get("steps_remaining")
        agent.conversation_history = data.get("conversation_history", [])
        return agent
