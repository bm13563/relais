"""The api engine: a step runs in-process on the Anthropic Messages API.

No subprocess, no CLI: relais runs the model's tool loop itself. A step is one
user message (its context prompt) followed by model turns: each turn is streamed
(text deltas go out as text_delta events), the tool calls it asks for run
through the registry, their results go back as one user message, and the loop
repeats until the model stops calling tools — or the agent's max_turns is spent.

An agent's context is its TRANSCRIPT: the Messages API messages it has
accumulated (its step prompts, its replies, its tool results). It is kept in
agent.client for the steps an agent instance runs in a pipeline run — the api
twin of the sdk engine's live client — and saved to the run database
(SQLiteStateManager.save_transcript) after every step, under the transcript id
that is the step's session_id. A host passes that id back as
PipelineAgent.resume to continue the conversation on a later run.

A transcript resumes only under the PREFIX it was made with — the model, the
system prompt and the offered tools, which its thinking blocks are bound to — so
each is saved with a hash of that prefix. A resume naming an unknown transcript,
or one whose prefix has since changed (a new tool on the agent, another model),
begins a fresh transcript under the same id: the host's handle stays valid, the
model's memory is dropped, and the step runs rather than failing on rejected
thinking.

Every request offers the model the AGENT's tool set — the same tools, in the
same order, on every step, because a thinking block is bound to the system +
tools prefix it was produced under: an agent whose steps offered different tools
would have its earlier thinking rejected on its next step (and lose its prompt
cache). Per-step gating is the registry wrapper's job, as on the sdk engine: a
call to a tool outside the current step gets the wrapper's refusal back as its
tool_result, and the tool body never runs.

From the agent it reads model, max_turns, thinking, effort, resume and
env["ANTHROPIC_API_KEY"]; its other fields configure the Claude Code CLI and are
ignored (see PipelineAgent).
"""

from __future__ import annotations
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from uuid import uuid4

import anthropic
from anthropic import AsyncAnthropic

from ..engine import USAGE_KEYS, EventSink, StepEvents, StepOutcome
from ..logging_config import get_logger
from ..tools import ToolDefinition, ToolRegistry, tool_names

if TYPE_CHECKING:
    from ..agent import PipelineAgent
    from ..state import SQLiteStateManager
    from ..step import PipelineStep

log = get_logger("relais")


# The system prompt of every request. Frozen — it heads the cached prefix, and a
# thinking block is bound to the prefix it was produced under.
API_SYSTEM_PROMPT = """You are one step of a relais pipeline: an automated workflow in which a series of agents each do one focused part of a larger task and hand their result on. The user message describes your step — its input, its instructions, the tools you may use, and the response tool that completes it.

- Work in turns: call a tool, read its result, decide what to do next. Call one tool at a time.
- Verify before you claim: say what the tools actually showed you, and say so when you could not check something.
- When the step's work is done, call the step's response tool exactly once, as your last act. Its output is the only thing passed on.
- When the run is streamed, the prose you write is shown to the person as it streams — the step tells you when it is. Write it for them."""

# PipelineAgent.model aliases; anything else is passed through as a model id.
MODEL_ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

MAX_TOKENS = 16000
MAX_RETRIES = 4  # the SDK retries connection errors, 408/409/429 and 5xx with backoff
CACHE = {"type": "ephemeral"}

# The Messages API needs an object schema; a tool without parameters has {}.
NO_PARAMETERS = {"type": "object", "properties": {}}

# Stop reasons that cut a reply off: a tool call in it may be incomplete. Each
# ends the step in error, named by the stop reason.
CUT_OFF = ("max_tokens", "model_context_window_exceeded", "refusal")


@dataclass
class Transcript:
    """An api-engine agent's conversation: its id (the continuity handle), the
    prefix it is made under (see _prefix), and the Messages API messages
    accumulated under it."""
    id: str
    prefix: str
    messages: list = field(default_factory=list)


class ApiEngine:
    """Runs steps in-process on the Messages API. See the module docstring."""

    name = "api"

    def __init__(self, *, cwd: Optional[str] = None, state_manager: SQLiteStateManager):
        # cwd is the CLI's; this engine touches no files. Transcripts are saved
        # to (and resumed from) the run database.
        self.state_manager = state_manager
        # One client per distinct API key — per event loop, since an async HTTP
        # client's connections belong to the loop that opened them, and each
        # sync pipeline.run() is a fresh asyncio.run() loop.
        self._clients: Dict[Optional[str], Tuple[asyncio.AbstractEventLoop, AsyncAnthropic]] = {}

    def _client(self, agent: PipelineAgent) -> AsyncAnthropic:
        # None lets the anthropic SDK find the key in the process environment.
        key = agent.env.get("ANTHROPIC_API_KEY") if agent.env else None
        loop = asyncio.get_running_loop()
        held = self._clients.get(key)
        if held is None or held[0] is not loop:
            held = (loop, AsyncAnthropic(api_key=key, max_retries=MAX_RETRIES))
            self._clients[key] = held
        return held[1]

    def _open(self, agent: PipelineAgent, prefix: str) -> Transcript:
        """A new transcript, or the one agent.resume names — continued if it was
        made under this prefix, else begun afresh under the same id."""
        if agent.resume is None:
            return Transcript(id=uuid4().hex, prefix=prefix)
        saved = self.state_manager.load_transcript(agent.resume)
        if saved is None or saved[1] != prefix:
            why = "not found" if saved is None else "prefix changed"
            log.info(f"transcript {agent.resume} {why}: beginning afresh", agent=agent.name, transcript=agent.resume)
            return Transcript(id=agent.resume, prefix=prefix)
        return Transcript(id=agent.resume, prefix=prefix, messages=saved[0])

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
        """Run one step: its prompt, then model turns until it stops calling tools."""
        model = MODEL_ALIASES.get(agent.model or "opus", agent.model)
        tools = _agent_tools(agent, registry)
        offered = _offered(tools)
        if not agent.has_client():
            agent.set_client(self._open(agent, _prefix(model, offered)))
        transcript = agent.client
        client = self._client(agent)
        events = StepEvents(emit, step.name, verbose)
        # Built once: every turn sends the same system and tools; `messages` is
        # the transcript itself, so each turn sends it as it has grown.
        request = _request(agent, model, offered, transcript.messages)

        transcript.messages.append({"role": "user", "content": _user_content(prompt, images)})
        final_text = ""
        tool_calls = []
        turns_used = 0
        is_error = False
        error_kind = None
        result_text = None
        usage = dict.fromkeys(USAGE_KEYS, 0)

        try:
            while True:
                message = await _turn(client, request, events)
                turns_used += 1
                turn_usage = {k: getattr(message.usage, k) or 0 for k in USAGE_KEYS}
                for k in USAGE_KEYS:
                    usage[k] += turn_usage[k]
                await events.usage(model, turn_usage)

                # The reply goes into the transcript whole — thinking blocks
                # included, unmodified, as the API requires them back.
                transcript.messages.append({
                    "role": "assistant",
                    "content": [block.model_dump(mode="json", exclude_none=True) for block in message.content],
                })
                for block in message.content:
                    if block.type == "text":
                        final_text = block.text
                        events.text(block.text)
                calls = [block for block in message.content if block.type == "tool_use"]

                if message.stop_reason in CUT_OFF:
                    # Cut off mid-reply: a tool input may be partial, so nothing
                    # runs — but every tool_use needs its tool_result for the
                    # transcript to stay valid for the next step (or a resume).
                    is_error, error_kind = True, message.stop_reason
                    if calls:
                        transcript.messages.append({
                            "role": "user",
                            "content": [_not_run(block.id, message.stop_reason) for block in calls],
                        })
                    break
                if not calls:
                    break  # end_turn: the model is done

                results = []
                for block in calls:
                    tool_calls.append({"id": block.id, "tool": block.name, "input": block.input})
                    await events.tool_use(block.name, block.input)
                    results.append(await _run_tool(block, tools, registry))
                transcript.messages.append({"role": "user", "content": results})

                # The turn cap, checked after the turn's tools ran — so a response
                # tool called in the last allowed turn is still captured.
                if turns_used >= agent.max_turns:
                    is_error, error_kind = True, "max_turns"
                    break
        except anthropic.APIError as e:
            # The API's errors (after the SDK's own retries) are the step's
            # outcome, not an exception: the executor's answer-first rule decides.
            is_error = True
            error_kind = "rate_limit" if isinstance(e, anthropic.RateLimitError) else type(e).__name__
            result_text = str(e)

        self.state_manager.save_transcript(transcript.id, transcript.messages, transcript.prefix)
        return StepOutcome(
            final_text=final_text,
            tool_calls=tool_calls,
            turns_used=turns_used,
            is_error=is_error,
            error_kind=error_kind,
            result_text=result_text,
            session_id=transcript.id,
            usage=usage,
        )

    async def close(self, agent: PipelineAgent) -> None:
        """Let go of the agent's transcript (saved after its last step already)."""
        agent.client = None


def _agent_tools(agent: PipelineAgent, registry: ToolRegistry) -> Dict[str, ToolDefinition]:
    """The agent's tools, by name and in its order — what every request offers.
    (Which of them the current step may call is the registry wrapper's check.)"""
    tools = {}
    for name in tool_names(agent.tools):
        definition = registry.get(name)
        if definition is None:
            raise ValueError(
                f"Agent '{agent.name}' lists '{name}', which is not a registered pipeline tool. "
                f"The api engine runs registered tools only; Claude Code built-ins "
                f"(Read, Bash, …) need the sdk engine."
            )
        tools[name] = definition
    return tools


def _offered(tools: Dict[str, ToolDefinition]) -> list[dict]:
    """The tools as a request offers them: name, description, schema, in order."""
    return [
        {"name": d.name, "description": d.description, "input_schema": d.input_schema or NO_PARAMETERS}
        for d in tools.values()
    ]


def _prefix(model: str, offered: list[dict]) -> str:
    """A hash of what a transcript's thinking is bound to: the model, the system
    prompt and the offered tools. A transcript resumes only under the same one."""
    bound = json.dumps({"model": model, "system": API_SYSTEM_PROMPT, "tools": offered}, sort_keys=True)
    return hashlib.sha256(bound.encode()).hexdigest()


def _request(agent: PipelineAgent, model: str, offered: list[dict], messages: list) -> dict:
    """The keyword arguments of every model turn in a step (system and tools
    are the same on every step the agent runs).

    Prompt caching: explicit breakpoints on the last tool and on the system
    prompt (tools render first, so the system's covers both), plus automatic
    caching — the top-level cache_control — whose breakpoint follows the
    transcript's last block as it grows, so each turn reads the conversation so
    far from cache instead of re-sending it at full price.
    """
    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": API_SYSTEM_PROMPT, "cache_control": CACHE}],
        "messages": messages,
        "cache_control": CACHE,
    }
    if offered:
        request["tools"] = offered[:-1] + [{**offered[-1], "cache_control": CACHE}]
    if agent.thinking:
        request["thinking"] = {"type": "adaptive"}
    if agent.effort:
        request["output_config"] = {"effort": agent.effort}
    return request


def _user_content(prompt: str, images: list[str]):
    """The step's user message content: the prompt, then any images (base64 PNG)."""
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}] + [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}}
        for image in images
    ]


async def _turn(client: AsyncAnthropic, request: dict, events: StepEvents):
    """One model turn, streamed: text deltas go out as they come; returns the
    complete message."""
    async with client.messages.stream(**request) as stream:
        async for event in stream:
            if event.type == "content_block_start" and event.content_block.type == "text":
                await events.text_block_start()
            elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                await events.text_delta(event.delta.text)
        return await stream.get_final_message()


async def _run_tool(block, tools: Dict[str, ToolDefinition], registry: ToolRegistry) -> dict:
    """Run one tool call through the registry wrapper — which refuses a tool
    outside the current step — and return its tool_result block."""
    definition = tools.get(block.name)
    if definition is None:
        # Can't happen — the request offered the agent's tools only — but a name
        # outside them is refused as the registry refuses it, as an error.
        return {"type": "tool_result", "tool_use_id": block.id, "content": registry._refusal(block.name), "is_error": True}
    return _tool_result(block.id, await definition.fn(block.input))


def _tool_result(tool_use_id: str, result: dict) -> dict:
    """An MCP-shaped tool return ({"content": [text / image blocks], "is_error"?})
    as a Messages API tool_result block."""
    content = []
    for item in result["content"]:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": item["mimeType"], "data": item["data"]},
            })
        else:
            raise ValueError(
                f"A tool returned a '{item['type']}' content block; "
                f"the api engine passes text and image blocks"
            )
    tool_result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if result.get("is_error"):
        tool_result["is_error"] = True
    return tool_result


def _not_run(tool_use_id: str, stop_reason: str) -> dict:
    """The tool_result for a call in a cut-off reply: it did not run."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": f"Not run: the reply was cut off ({stop_reason}) before this call was complete.",
        "is_error": True,
    }
