"""The sdk engine: a step runs in a Claude Code CLI subprocess, via the Claude Agent SDK.

Each agent connects one ClaudeSDKClient on its first step and keeps it (in
agent.client) for every step that agent instance runs, so its conversation
context lives in the CLI session; the session id is the continuity handle a host
passes back as PipelineAgent.resume. The pipeline's tools reach the CLI as an
in-process MCP server built from the registry, where the CLI names them
mcp__<server>__<tool> — this module translates both ways, so the rest of relais
only ever sees registry names.

This is the only module that imports claude_agent_sdk.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    StreamEvent,
    SystemMessage,
    create_sdk_mcp_server,
    tool as sdk_tool,
)

from ..engine import USAGE_KEYS, EventSink, StepEvents, StepOutcome
from ..tools import ToolRegistry, tool_names

if TYPE_CHECKING:
    from ..agent import PipelineAgent
    from ..state import SQLiteStateManager
    from ..step import PipelineStep


def mcp_server(registry: ToolRegistry):
    """The registry's tools as an in-process MCP server config for the CLI.

    Each tool is served as its definition's `fn` — the gated, capturing wrapper —
    so the per-step gate holds whatever the CLI's own allowed_tools says.
    """
    definitions = [registry.get(name) for name in registry.list_tools()]
    return create_sdk_mcp_server(
        name=registry.name,
        version="1.0.0",
        tools=[sdk_tool(d.name, d.description, d.input_schema)(d.fn) for d in definitions],
    )


def allowed_tools(registry: ToolRegistry, tools: List) -> List[str]:
    """MCP-formatted tool names for ClaudeAgentOptions.allowed_tools.

    Args:
        registry: The pipeline's tool registry (its name is the MCP server's)
        tools: Tool names (strings) or @tool decorated functions

    Returns:
        mcp__{server}__{tool} for registered tools; any other name (a Claude
        Code built-in: Read, Write, Bash, …) passes through unchanged
    """
    return [
        f"mcp__{registry.name}__{name}" if registry.get(name) is not None else name
        for name in tool_names(tools)
    ]


def registry_name(registry: ToolRegistry, cli_name: str) -> str:
    """A tool name as the CLI reports it, as the registry knows it."""
    return cli_name.removeprefix(f"mcp__{registry.name}__")


class SdkEngine:
    """Runs steps on Claude Code CLI subprocesses. See the module docstring."""

    name = "sdk"

    def __init__(self, *, cwd: Optional[str] = None, state_manager: Optional[SQLiteStateManager] = None):
        # The CLI's working directory. (The CLI keeps its own sessions, so the
        # run database is not needed here.)
        self.cwd = cwd

    def _options(self, agent: PipelineAgent, registry: ToolRegistry, streaming: bool) -> ClaudeAgentOptions:
        """The client's options — fixed at connect(), for every step it runs."""
        options_kwargs = {
            "max_turns": agent.max_turns,
            "model": agent.model or "opus",
            "mcp_servers": {registry.name: mcp_server(registry)},
            # allowed_tools is the SDK's per-client filter, set once at connect().
            # Use the agent's full tool set; per-step scoping is enforced by the
            # registry's wrapper.
            "allowed_tools": allowed_tools(registry, agent.tools),
            "permission_mode": agent.permission_mode,
            "cwd": self.cwd,
            # SDK isolation: do NOT load filesystem settings
            # (~/.claude/settings.json etc). Those layers can silently override
            # the agent's requested model (e.g. an effortLevel/model pin found
            # there was downgrading opus to sonnet). An agent's config is defined
            # here in code, not by the ambient machine's Claude Code settings.
            "setting_sources": [],
        }
        if streaming:
            # The SDK then yields a StreamEvent per API stream event, which is
            # where text deltas come from.
            options_kwargs["include_partial_messages"] = True
        if agent.thinking:
            options_kwargs["max_thinking_tokens"] = 60000
        if agent.effort:
            options_kwargs["effort"] = agent.effort
        if agent.resume:
            options_kwargs["resume"] = agent.resume
        if agent.env:
            options_kwargs["env"] = agent.env
        if agent.builtin_tools is not None:
            # The SDK's `tools`: the built-ins the client has at all ([] = none).
            options_kwargs["tools"] = list(agent.builtin_tools)
        return ClaudeAgentOptions(**options_kwargs)

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
        """Run one step on the agent's live client (connected on its first step)."""
        if not agent.has_client():
            client = ClaudeSDKClient(options=self._options(agent, registry, streaming=emit is not None))
            await client.connect()
            agent.set_client(client)
        client = agent.client

        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img},
                })
            # query() treats a str specially and any non-str as an async iterable
            # of message dicts — so to send content blocks we stream one full user
            # message of our own.
            async def _one_message():
                yield {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}
            await client.query(_one_message())
        else:
            await client.query(prompt)

        events = StepEvents(emit, step, verbose)
        tool_calls = []
        final_text = ""
        turns_used = 0
        is_error = False
        error_kind = None
        result_text = None
        result_subtype = None
        session_id = None
        usage = {}

        async for message in client.receive_response():
            # Every SDK message that knows the session names it; the last word wins.
            session_id = getattr(message, "session_id", None) or session_id
            if isinstance(message, AssistantMessage):
                if message.error:
                    error_kind = message.error
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text
                        events.text(block.text)
                    elif isinstance(block, ToolUseBlock):
                        name = registry_name(registry, block.name)
                        tool_calls.append({"id": block.id, "tool": name, "input": block.input})
                        await events.tool_use(name, block.input)
            elif isinstance(message, StreamEvent):
                # A raw API stream event (only with include_partial_messages):
                # the text deltas — and the step's `streams` argument, as the
                # response-tool call is written — are what a host streams to its user.
                ev = message.event
                if ev.get("type") == "content_block_start":
                    block = ev.get("content_block", {})
                    if block.get("type") == "text":
                        await events.text_block_start()
                    elif block.get("type") == "tool_use":
                        events.tool_block_start(ev["index"], registry_name(registry, block["name"]))
                elif ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        await events.text_delta(delta["text"])
                    elif delta.get("type") == "input_json_delta":
                        await events.tool_input_delta(ev["index"], delta["partial_json"])
            elif isinstance(message, SystemMessage):
                if message.subtype == "init":
                    session_id = message.data.get("session_id") or session_id
            elif isinstance(message, ResultMessage):
                turns_used = message.num_turns
                is_error = message.is_error
                result_text = message.result
                result_subtype = message.subtype
                # The CLI's usage for the whole query (this step).
                usage = {k: message.usage[k] for k in USAGE_KEYS if k in message.usage} if message.usage else {}

        return StepOutcome(
            final_text=final_text,
            tool_calls=tool_calls,
            turns_used=turns_used,
            is_error=is_error,
            # The assistant message's error (e.g. "rate_limit") names it best;
            # failing that, an errored result's subtype (e.g. "error_max_turns").
            error_kind=error_kind or (result_subtype if is_error else None),
            result_text=result_text,
            session_id=session_id,
            usage=usage,
        )

    async def close(self, agent: PipelineAgent) -> None:
        """Disconnect the agent's CLI client (ending its subprocess)."""
        try:
            await agent.client.disconnect()
        finally:
            agent.client = None
