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


class FieldStream:
    """Reads one top-level string field out of a JSON object arriving in chunks.

    feed() takes the next chunk of the object's text (a tool call's
    input_json_delta) and returns the field's decoded text that chunk
    completed — "" until the field's value begins, and after its closing quote.
    It tracks just enough JSON to be sure it is reading THE field: nesting depth,
    strings and their escapes, and whether a top-level string is a key or a
    value — so the same name inside a nested object, or inside another string,
    doesn't count; keys may come in any order; a missing field, or one whose
    value isn't a string, yields nothing. Escapes (\\n \\t \\" \\\\ \\/ \\b \\f
    \\r \\uXXXX, surrogate pairs included) decode correctly when a chunk ends
    mid-escape.
    """

    ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}

    def __init__(self, field: str):
        self.field = field
        self._depth = 0
        self._expect_key = False  # at the top level: the next string is a key
        self._key = None          # the top-level key whose value comes next
        self._in_string = False
        self._is_key = False      # the current string is a top-level key
        self._streaming = False   # the current string is the field's value
        self._done = False        # the field's value has been read
        self._escape = None       # after a backslash: the escape read so far
        self._high = None         # a high surrogate waiting for its low half
        self._chars = []          # the current key's decoded characters

    def feed(self, chunk: str) -> str:
        out = []
        for ch in chunk:
            if self._done:
                break
            if self._in_string:
                self._string_char(ch, out)
            elif ch == '"':
                self._in_string = True
                self._is_key = self._depth == 1 and self._expect_key
                self._streaming = self._depth == 1 and not self._expect_key and self._key == self.field
                self._chars = []
            elif ch in "{[":
                self._depth += 1
                if self._depth == 1:
                    self._expect_key = ch == "{"
            elif ch in "}]":
                self._depth -= 1
            elif self._depth == 1 and ch == ":":
                self._expect_key = False
            elif self._depth == 1 and ch == ",":
                self._expect_key = True
                self._key = None
        return "".join(out)

    def _string_char(self, ch: str, out: list) -> None:
        if self._escape is not None:
            self._escape += ch
            if self._escape[0] != "u":
                self._escape = None
                self._text(self.ESCAPES.get(ch, ch), out)
            elif len(self._escape) == 5:  # u + four hex digits
                unit = int(self._escape[1:], 16)
                self._escape = None
                self._unit(unit, out)
        elif ch == "\\":
            self._escape = ""
        elif ch == '"':
            self._text("", out)  # flushes a lone high surrogate
            self._in_string = False
            if self._is_key:
                self._key = "".join(self._chars)
            if self._streaming:
                self._done = True
        else:
            self._text(ch, out)

    def _unit(self, unit: int, out: list) -> None:
        """One \\uXXXX code unit: a surrogate pair's halves become one character."""
        if 0xD800 <= unit <= 0xDBFF:
            self._text("", out)  # an earlier lone high surrogate
            self._high = unit
        elif 0xDC00 <= unit <= 0xDFFF and self._high is not None:
            pair = chr(0x10000 + ((self._high - 0xD800) << 10) + (unit - 0xDC00))
            self._high = None
            self._text(pair, out)
        else:
            self._text(chr(unit), out)

    def _text(self, text: str, out: list) -> None:
        """Decoded text of the current string, to wherever it belongs."""
        if self._high is not None:  # a high surrogate with no low half
            self._high = None
            text = "\ufffd" + text
        if self._streaming:
            out.append(text)
        elif self._is_key:
            self._chars.append(text)


class StepEvents:
    """The events an engine streams while a step runs, and its verbose echo.

    One per step run: it counts the step's text blocks, because the model's
    prose arrives in BLOCKS around its tool calls and a host that joins the
    deltas would glue "…the SOC launch.Three of the…" — so a new text block
    after the first is a paragraph break (a "\\n\\n" text_delta).

    It also streams the step's `streams` argument (PipelineStep.streams): while
    the model writes a call to the step's response tool, that string argument's
    text goes out as text_delta events, as one more text block.
    """

    def __init__(self, emit: EventSink, step: PipelineStep, verbose: bool):
        self._emit = emit
        self._step = step.name
        self._response_tool = step.response_tool
        self._streams = step.streams
        self._verbose = verbose
        self._text_blocks = 0
        # Content-block index → [FieldStream, whether its text has begun], for a
        # response-tool call being written in the current message.
        self._fields = {}

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

    def tool_block_start(self, index: int, name: str) -> None:
        """A tool_use content block began (`name` as the registry knows it)."""
        if self._streams and name == self._response_tool:
            self._fields[index] = [FieldStream(self._streams), False]
        else:
            self._fields.pop(index, None)  # indexes restart with every message

    async def tool_input_delta(self, index: int, partial_json: str) -> None:
        """A chunk of a tool_use block's input JSON."""
        if index not in self._fields:
            return
        stream, begun = self._fields[index]
        text = stream.feed(partial_json)
        if text:
            if not begun:
                await self.text_block_start()
                self._fields[index][1] = True
            await self.text_delta(text)

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
