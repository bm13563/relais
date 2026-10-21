"""Unit tests for engines/sdk.py — the Claude Agent SDK engine (no CLI is started).

ClaudeSDKClient is replaced by a scripted fake: it records the options it was
built with and the queries it was sent, and replays FakeClient.script — a list
of real SDK message objects — from receive_response().
"""

from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from relais.agent import PipelineAgent
from relais.engines import sdk as sdk_engine
from relais.engines.sdk import SdkEngine, allowed_tools, mcp_server, registry_name
from relais.step import PipelineStep
from relais.tools import ToolRegistry, tool


class FakeClient:
    """A ClaudeSDKClient stand-in, replaying FakeClient.script for each query."""

    instances = []
    script = []

    def __init__(self, options):
        self.options = options
        self.connected = False
        self.queries = []
        self.disconnect = AsyncMock()
        FakeClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def query(self, prompt):
        # An async iterable (the images path) is drained so the test can see it.
        if isinstance(prompt, str):
            self.queries.append(prompt)
        else:
            self.queries.append([message async for message in prompt])

    async def receive_response(self):
        for message in FakeClient.script:
            yield message


@pytest.fixture
def fake_client():
    FakeClient.instances = []
    with patch.object(sdk_engine, "ClaudeSDKClient", FakeClient):
        yield FakeClient


@pytest.fixture
def registry():
    registry = ToolRegistry("pipe_tools")

    @registry.tool("classify", "Classify the input")
    async def classify(category: Annotated[str, "The category"]) -> dict:
        return {"content": [{"type": "text", "text": f'{{"category": "{category}"}}'}]}

    @registry.tool("lookup", "Look something up")
    async def lookup() -> dict:
        return {"content": [{"type": "text", "text": "found"}]}

    return registry


def result_message(**overrides):
    fields = dict(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=2,
                  session_id="sess-9", result="All done.")
    fields.update(overrides)
    return ResultMessage(**fields)


def assistant(*content, error=None):
    return AssistantMessage(content=list(content), model="claude-opus", error=error)


def stream_event(event: dict):
    return StreamEvent(uuid="u", session_id="sess-9", event=event)


def text_start():
    return stream_event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})


def text_delta(text: str):
    return stream_event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})


def tool_start(index: int, cli_name: str):
    return stream_event({"type": "content_block_start", "index": index,
                         "content_block": {"type": "tool_use", "id": f"tu{index}", "name": cli_name, "input": {}}})


def input_delta(index: int, partial_json: str):
    return stream_event({"type": "content_block_delta", "index": index,
                         "delta": {"type": "input_json_delta", "partial_json": partial_json}})


async def run(engine, agent, registry, *, script, prompt="Do the step.", images=(), emit=None, streams=None):
    """Run one step ("s1", response tool classify) with the fake client replaying `script`."""
    FakeClient.script = script
    step = PipelineStep(name="s1", instruction="x", response_tool="classify", tools=["classify"], agent=agent,
                        streams=streams)
    return await engine.run_step(agent=agent, step=step, prompt=prompt, images=list(images),
                                 registry=registry, emit=emit, verbose=False)


class TestOptions:
    """The ClaudeAgentOptions the agent's client is built with."""

    async def test_defaults(self, fake_client, registry):
        agent = PipelineAgent(name="a", tools=["classify", "Read"])
        await run(SdkEngine(cwd="/work"), agent, registry, script=[result_message()])

        options = fake_client.instances[0].options
        assert options.model == "opus"
        assert options.max_turns == 10
        assert options.cwd == "/work"
        assert options.permission_mode == "acceptEdits"
        assert options.setting_sources == []
        assert options.allowed_tools == ["mcp__pipe_tools__classify", "Read"]
        assert list(options.mcp_servers) == ["pipe_tools"]
        assert options.include_partial_messages is False
        assert options.max_thinking_tokens is None
        assert options.effort is None
        assert options.resume is None
        assert options.env == {}

    async def test_every_agent_field_reaches_the_options(self, fake_client, registry):
        agent = PipelineAgent(
            name="a", tools=["classify"], max_turns=4, model="sonnet", thinking=True,
            permission_mode="bypassPermissions", effort="high", resume="sess-1",
            env={"ANTHROPIC_API_KEY": "k"}, builtin_tools=[],
        )
        await run(SdkEngine(), agent, registry, script=[result_message()], emit=AsyncMock())

        options = fake_client.instances[0].options
        assert options.model == "sonnet"
        assert options.max_turns == 4
        assert options.permission_mode == "bypassPermissions"
        assert options.include_partial_messages is True  # streaming: an event sink is listening
        assert options.max_thinking_tokens == 60000
        assert options.effort == "high"
        assert options.resume == "sess-1"
        assert options.env == {"ANTHROPIC_API_KEY": "k"}
        assert options.tools == []


    @pytest.mark.parametrize("thinking", [None, False])
    async def test_thinking_off_or_default_sets_no_budget(self, fake_client, registry, thinking):
        await run(SdkEngine(), PipelineAgent(name="a", thinking=thinking), registry, script=[result_message()])
        assert fake_client.instances[0].options.max_thinking_tokens is None


class TestClientLifecycle:
    async def test_one_client_per_agent_reused_across_steps(self, fake_client, registry):
        engine = SdkEngine()
        agent = PipelineAgent(name="a")
        await run(engine, agent, registry, script=[result_message()], prompt="first")
        await run(engine, agent, registry, script=[result_message()], prompt="second")

        assert len(fake_client.instances) == 1
        client = fake_client.instances[0]
        assert agent.client is client
        assert client.connected
        assert client.queries == ["first", "second"]

    async def test_images_go_as_one_streamed_user_message(self, fake_client, registry):
        agent = PipelineAgent(name="a")
        await run(SdkEngine(), agent, registry, script=[result_message()], prompt="Look.", images=["AAA", "BBB"])

        [messages] = fake_client.instances[0].queries
        assert messages == [{
            "type": "user",
            "message": {"role": "user", "content": [
                {"type": "text", "text": "Look."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBB"}},
            ]},
            "parent_tool_use_id": None,
        }]

    async def test_close_disconnects_and_clears(self, fake_client, registry):
        agent = PipelineAgent(name="a")
        await run(SdkEngine(), agent, registry, script=[result_message()])
        client = agent.client

        await SdkEngine().close(agent)

        client.disconnect.assert_awaited_once()
        assert agent.client is None

    async def test_close_clears_even_when_disconnect_fails(self):
        agent = PipelineAgent(name="a")
        agent.set_client(MagicMock(disconnect=AsyncMock(side_effect=RuntimeError("cli gone"))))
        with pytest.raises(RuntimeError):
            await SdkEngine().close(agent)
        assert agent.client is None


class TestOutcome:
    """The message loop → StepOutcome."""

    async def test_a_clean_step(self, fake_client, registry):
        agent = PipelineAgent(name="a")
        script = [
            SystemMessage(subtype="init", data={"session_id": "sess-init"}),
            assistant(TextBlock(text="Let me classify."),
                      ToolUseBlock(id="tu1", name="mcp__pipe_tools__classify", input={"category": "q"})),
            assistant(TextBlock(text="It's a question.")),
            result_message(usage={"input_tokens": 50, "output_tokens": 9, "cache_read_input_tokens": 40,
                                  "cache_creation_input_tokens": 0, "service_tier": "standard"}),
        ]
        outcome = await run(SdkEngine(), agent, registry, script=script)

        assert outcome.final_text == "It's a question."
        # Registry names — the mcp__<server>__ prefix is the CLI's business.
        assert outcome.tool_calls == [{"id": "tu1", "tool": "classify", "input": {"category": "q"}}]
        assert outcome.turns_used == 2
        assert outcome.is_error is False
        assert outcome.error_kind is None
        assert outcome.result_text == "All done."
        assert outcome.session_id == "sess-9"  # the last word wins
        assert outcome.usage == {"input_tokens": 50, "output_tokens": 9, "cache_read_input_tokens": 40,
                                 "cache_creation_input_tokens": 0}

    async def test_no_usage_is_empty(self, fake_client, registry):
        outcome = await run(SdkEngine(), PipelineAgent(name="a"), registry, script=[result_message(usage=None)])
        assert outcome.usage == {}

    async def test_an_errored_result_is_named_by_its_subtype(self, fake_client, registry):
        outcome = await run(SdkEngine(), PipelineAgent(name="a"), registry,
                            script=[result_message(is_error=True, subtype="error_max_turns", result=None)])
        assert outcome.is_error is True
        assert outcome.error_kind == "error_max_turns"

    async def test_an_assistant_error_names_it_best(self, fake_client, registry):
        script = [
            assistant(TextBlock(text="API Error: rate limited"), error="rate_limit"),
            result_message(is_error=True, subtype="success", result="API Error: rate limited"),
        ]
        outcome = await run(SdkEngine(), PipelineAgent(name="a"), registry, script=script)
        assert outcome.error_kind == "rate_limit"
        assert outcome.result_text == "API Error: rate limited"


class TestEvents:
    async def test_text_deltas_with_a_paragraph_break_and_tool_use_by_registry_name(self, fake_client, registry):
        events = []

        async def sink(event):
            events.append(event)

        script = [
            text_start(), text_delta("Checking"), text_delta("…"),
            assistant(TextBlock(text="Checking…"),
                      ToolUseBlock(id="tu1", name="mcp__pipe_tools__lookup", input={})),
            text_start(), text_delta("Found it."),
            assistant(TextBlock(text="Found it.")),
            result_message(),
        ]
        await run(SdkEngine(), PipelineAgent(name="a"), registry, script=script, emit=sink)

        assert events == [
            {"type": "text_delta", "step": "s1", "text": "Checking"},
            {"type": "text_delta", "step": "s1", "text": "…"},
            {"type": "tool_use", "step": "s1", "name": "lookup", "input": {}},
            {"type": "text_delta", "step": "s1", "text": "\n\n"},
            {"type": "text_delta", "step": "s1", "text": "Found it."},
        ]


class TestStreamedAnswer:
    """PipelineStep.streams on the sdk engine: the same extraction, from the
    CLI's raw stream events (the tool named mcp__<server>__<tool>)."""

    async def test_the_argument_streams_after_a_paragraph_break(self, fake_client, registry):
        events = []

        async def sink(event):
            events.append(event)

        tool_input = {"category": 'Line "a"\nb — é'}
        raw = '{"category": "Line \\"a\\"\\nb \\u2014 \\u00e9"}'
        script = [
            text_start(), text_delta("Checking."),
            tool_start(1, "mcp__pipe_tools__classify"),
            *[input_delta(1, raw[at:at + 6]) for at in range(0, len(raw), 6)],
            assistant(TextBlock(text="Checking."), ToolUseBlock(id="tu1", name="mcp__pipe_tools__classify", input=tool_input)),
            result_message(),
        ]
        await run(SdkEngine(), PipelineAgent(name="a"), registry, script=script, emit=sink, streams="category")

        texts = [e["text"] for e in events if e["type"] == "text_delta"]
        assert texts[:2] == ["Checking.", "\n\n"]
        assert "".join(texts[2:]) == tool_input["category"]
        assert len(texts) > 4
        assert events[-1] == {"type": "tool_use", "step": "s1", "name": "classify", "input": tool_input}

    async def test_other_tools_do_not_stream(self, fake_client, registry):
        events = []

        async def sink(event):
            events.append(event)

        script = [
            tool_start(0, "mcp__pipe_tools__lookup"), input_delta(0, '{"category": "not this"}'),
            result_message(),
        ]
        await run(SdkEngine(), PipelineAgent(name="a"), registry, script=script, emit=sink, streams="category")
        assert events == []


class TestMcpNames:
    """The registry's tools as the CLI sees them (moved from ToolRegistry)."""

    def test_allowed_tools_registered(self, registry):
        assert allowed_tools(registry, ["classify", "lookup"]) == [
            "mcp__pipe_tools__classify", "mcp__pipe_tools__lookup",
        ]

    def test_allowed_tools_builtins_and_unregistered_pass_through(self, registry):
        assert allowed_tools(registry, ["Read", "Bash", "unregistered"]) == ["Read", "Bash", "unregistered"]

    def test_allowed_tools_empty(self, registry):
        assert allowed_tools(registry, []) == []

    def test_allowed_tools_with_function_references(self, registry):
        @tool("classify", "Classify the input")
        async def standalone(category: str) -> dict:
            return {"content": []}

        assert allowed_tools(registry, [standalone, "Read"]) == ["mcp__pipe_tools__classify", "Read"]

    def test_registry_name(self, registry):
        assert registry_name(registry, "mcp__pipe_tools__classify") == "classify"
        assert registry_name(registry, "Read") == "Read"

    def test_mcp_server_serves_each_definitions_fn(self, registry):
        with patch.object(sdk_engine, "create_sdk_mcp_server") as create:
            mcp_server(registry)

        kwargs = create.call_args.kwargs
        assert kwargs["name"] == "pipe_tools"
        assert kwargs["version"] == "1.0.0"
        served = {t.name: t for t in kwargs["tools"]}
        assert set(served) == {"classify", "lookup"}
        assert served["classify"].handler is registry.get("classify").fn
        assert served["classify"].input_schema == registry.get("classify").input_schema
        assert served["lookup"].description == "Look something up"

    def test_mcp_server_with_no_tools(self):
        with patch.object(sdk_engine, "create_sdk_mcp_server") as create:
            mcp_server(ToolRegistry("empty"))
        assert create.call_args.kwargs["tools"] == []

    async def test_the_served_tool_is_gated_per_step(self, registry):
        """The MCP handler IS the registry wrapper, so the per-step gate holds."""
        with patch.object(sdk_engine, "create_sdk_mcp_server") as create:
            mcp_server(registry)
        handler = {t.name: t for t in create.call_args.kwargs["tools"]}["lookup"].handler

        registry.set_current_step("s1", ["classify"])
        result = await handler({})

        assert "is not available in step 's1'" in result["content"][0]["text"]
