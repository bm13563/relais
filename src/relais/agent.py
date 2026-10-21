"""Agent definition for pipelines.

A PipelineAgent is who runs a step: it owns a model, a turn budget, a step budget,
a fixed tool set, and the engine its steps run on. A live instance opens its
engine's per-agent state when it first runs — a Claude Code CLI client (sdk) or a
transcript (api) — and keeps it across the steps it participates in, so its
conversation context carries from step to step. When its step budget is spent it
expires; the next time the pipeline routes to that agent, a fresh instance with
clean context takes over.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Union, Any

from .engines import ENGINES


@dataclass
class PipelineAgent:
    """An agent that executes pipeline steps.

    Every field applies on both engines except those marked SDK-ONLY, which
    configure the Claude Code CLI and are ignored by the api engine.

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
        model: Model for this agent (opus, sonnet, haiku — or a full model id).
        thinking: Extended thinking, tri-state (default None):
               None — leave it to the model's default (send nothing). Note
               that the 5.x models think adaptively by default: on the api
               engine None does NOT mean no thinking.
               True — on: sdk, a thinking-token budget; api, adaptive thinking
               ({"type": "adaptive"}).
               False — off: api, the model's own "off" — {"type":
               "between_tools"} on Claude Sonnet / Opus 5.5 (no thinking before
               responding; the short updates between tool calls come back as
               thinking blocks), {"type": "disabled"} on other models
               (engines/api.py THINKING_OFF); sdk, nothing set (as None).
        engine: What this agent's steps run on (default "sdk"):
               "sdk" — a Claude Code CLI subprocess, driven by the Claude Agent SDK;
               "api" — in-process on the Anthropic Messages API (relais.engines).
        permission_mode: SDK-ONLY. How the SDK gates built-in tools for this agent's client
               (default "acceptEdits"). MCP pipeline tools are always gated by the
               registry regardless of this. Use "bypassPermissions" for an agent
               that must run built-in Bash/file tools unattended (a "godmode"
               agent in a headless run, where no human is present to approve a
               Bash call); keep the default for agents that only need edits.
        resume: The continuity handle to resume when this agent first runs (None =
               a fresh start): a Claude Code session id (sdk) or a transcript id
               (api — loaded from the run database). The resumed conversation IS
               the agent's context. A step's session_id is the value to pass back.
        effort: Reasoning-effort level for this agent ("low", "medium", "high",
               "xhigh", "max"). None (default) leaves it to the SDK/model default.
               Higher effort spends more reasoning before answering — raise it for
               agents doing hard analysis, lower it for cheap mechanical steps.
        builtin_tools: SDK-ONLY. Which of Claude Code's BUILT-IN tools (Read, Write, Edit,
               Bash, Glob, Grep, WebSearch, …) this agent's client may have at all —
               the SDK's `tools` option. None (default) leaves the SDK's default set in
               place, gated only by permission_mode; [] removes every built-in, so the
               agent has exactly its pipeline tools and nothing that can read the
               host's files or environment — the setting for an agent inside a server.
        env: Extra environment variables for this agent's Claude Code subprocess,
               merged over the process environment (the SDK's `env` option). The
               host's way to hand a credential in explicitly — e.g.
               {"ANTHROPIC_API_KEY": key} read from its own settings — rather than
               rely on whatever the process inherited. None = inherit only.
               The api engine reads only env["ANTHROPIC_API_KEY"] (its client's
               key; absent, the anthropic SDK reads the process environment) —
               every other variable is SDK-ONLY. Likewise SDK-ONLY: the pipeline's
               cwd and the CLI's setting sources.
        steps_remaining: Runtime counter for the current live instance (set from
                steps when the instance is created). None for persistent agents.
        client: The engine's per-agent state — the live ClaudeSDKClient (sdk) or
                the transcript (api) — set on first run, reused across the steps
                this instance participates in, released by the engine's close().
    """

    name: str
    tools: List[Union[str, Callable]] = field(default_factory=list)
    steps: Optional[int] = None
    max_turns: int = 10
    model: Optional[str] = "opus"
    thinking: Optional[bool] = None
    permission_mode: str = "acceptEdits"
    effort: Optional[str] = None
    resume: Optional[str] = None
    env: Optional[dict] = None
    builtin_tools: Optional[list] = None
    engine: str = "sdk"
    steps_remaining: Optional[int] = field(default=None, init=False)
    client: Any = None  # the engine's per-agent state (see relais.engine)

    def __post_init__(self):
        if self.engine not in ENGINES:
            raise ValueError(
                f"Agent '{self.name}' names engine '{self.engine}'; "
                f"the engines are: {', '.join(ENGINES)}"
            )
        self.steps_remaining = self.steps

    def consume_step(self) -> None:
        """Spend one step of this instance's budget (no-op if persistent)."""
        if self.steps_remaining is not None:
            self.steps_remaining -= 1

    def is_expired(self) -> bool:
        """Whether this instance has spent its step budget."""
        return self.steps_remaining is not None and self.steps_remaining <= 0

    def set_client(self, client: Any) -> None:
        """Attach the engine's per-agent state (a CLI client, a transcript)."""
        self.client = client

    def has_client(self) -> bool:
        """Whether this agent already holds live engine state."""
        return self.client is not None

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
            and self.permission_mode == other.permission_mode
            and self.effort == other.effort
            and self.resume == other.resume
            and self.env == other.env
            and self.builtin_tools == other.builtin_tools
            and self.engine == other.engine
        )
