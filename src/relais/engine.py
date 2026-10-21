"""The engine seam: what runs one pipeline step on a model.

The executor owns everything engine-neutral about a step — its context prompt,
the per-step tool gate, the response-tool capture, StepError, routing, state.
What it hands an engine is the model's part: send the step's prompt, run the
model's turn loop (its tool calls, its streamed prose) and report how it went as
a StepOutcome.

Two engines ship, in relais.engines, picked per agent by PipelineAgent.engine:

  sdk  a Claude Code CLI subprocess per agent, driven by the Claude Agent SDK
       (the default — relais's original behaviour)
  api  in-process on the Anthropic Messages API, via the anthropic SDK

An engine also owns its per-agent state. It keeps it in `agent.client` — the
sdk engine a live ClaudeSDKClient, the api engine the agent's transcript — set
on the agent's first step and reused for every step that agent instance runs,
so the orchestrator's lifecycle (reuse while live, close on expiry and at run
end) is one and the same for every engine. The orchestrator makes one engine
per name, as Engine(cwd=..., state_manager=...), and closes each agent through
the engine that ran it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Protocol

if TYPE_CHECKING:
    from .agent import PipelineAgent
    from .step import PipelineStep
    from .tools import ToolRegistry


# The token counts a StepOutcome's usage carries (and a `usage` event reports).
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

# The orchestrator's async event sink while Pipeline.astream() drives a run, else None.
EventSink = Optional[Callable[[dict], Awaitable[None]]]


@dataclass
class StepOutcome:
    """How a step's run on the model went — what an engine reports back.

    Attributes:
        final_text: The last text block the model produced in the step.
        tool_calls: The tool calls the model made, in order, as
            {"id", "tool", "input"} — tool names as the registry knows them
            (no mcp__ prefix).
        turns_used: Model round-trips the step took.
        is_error: The step ended in error (the executor's answer-first rule then
            decides whether that fails the step: not if the response tool ran).
        error_kind: What the error was — "rate_limit", "max_turns", "max_tokens",
            the CLI's result subtype, or an exception class name. None if none.
        result_text: The engine's error/result text, if it has one.
        session_id: The continuity handle a host passes back as
            PipelineAgent.resume: the CLI session id (sdk) / the transcript id (api).
        usage: Token counts summed over the step, keyed by USAGE_KEYS; {} when
            the engine doesn't know them.
    """
    final_text: str
    tool_calls: list[dict]
    turns_used: int
    is_error: bool
    error_kind: Optional[str]
    result_text: Optional[str]
    session_id: Optional[str]
    usage: dict = field(default_factory=dict)


class Engine(Protocol):
    """Runs one step of a pipeline on a model. See the module docstring."""

    name: str

    async def run_step(
        self,
        *,
        agent: PipelineAgent,
        step: PipelineStep,
        prompt: str,
        images: list[str],
        registry: ToolRegistry,
        emit: EventSink,
        verbose: bool,
    ) -> StepOutcome:
        """Run `step` on `agent`: send `prompt` (with `images`, base64 PNGs, as
        image blocks), let the model work through the step's tools — every call
        goes through the registry's wrapper — and report the outcome. Streams
        text_delta / tool_use events to `emit` as they happen."""
        ...

    async def close(self, agent: PipelineAgent) -> None:
        """Release whatever this engine holds for the agent (its client or its
        transcript) and clear agent.client."""
        ...


class StepEvents:
    """The events an engine streams while a step runs, and its verbose echo.

    One per step run: it counts the step's text blocks, because the model's
    prose arrives in BLOCKS around its tool calls and a host that joins the
    deltas would glue "…the SOC launch.Three of the…" — so a new text block
    after the first is a paragraph break (a "\\n\\n" text_delta).
    """

    def __init__(self, emit: EventSink, step_name: str, verbose: bool):
        self._emit = emit
        self._step = step_name
        self._verbose = verbose
        self._text_blocks = 0

    async def _send(self, event: dict) -> None:
        if self._emit is not None:
            await self._emit(event)

    async def text_block_start(self) -> None:
        """A text content block began streaming."""
        if self._text_blocks:
            await self._send({"type": "text_delta", "step": self._step, "text": "\n\n"})
        self._text_blocks += 1

    async def text_delta(self, text: str) -> None:
        """A piece of the model's prose."""
        await self._send({"type": "text_delta", "step": self._step, "text": text})

    def text(self, text: str) -> None:
        """A complete text block (echoed when verbose; its deltas already streamed)."""
        if self._verbose:
            print(f"\n[{self._step}] {text}")

    async def tool_use(self, name: str, tool_input: dict) -> None:
        """A tool the model called (registry name)."""
        await self._send({"type": "tool_use", "step": self._step, "name": name, "input": tool_input})
        if self._verbose:
            print(f"\n[{self._step}] Tool: {name}  {tool_input}")

    async def usage(self, model: str, usage: dict) -> None:
        """One model turn's token counts."""
        await self._send({"type": "usage", "step": self._step, "model": model, **usage})
