"""Unit tests for engines/api.py — the in-process Messages API engine (offline).

AsyncAnthropic is replaced by a stub whose messages.stream(...) is an async
context manager over scripted stream events, with get_final_message() returning
a real anthropic Message. Each scripted turn is either (events, message) or an
exception the stream raises on entry. Every request is recorded (deep-copied, as
the transcript keeps growing after it is sent).
"""

import asyncio
import copy
from pathlib import Path
from typing import Annotated

import anthropic
import httpx2
import pytest
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)

from relais.agent import PipelineAgent
from relais.engines import api as api_engine
from relais.engines.api import API_SYSTEM_PROMPT, MAX_TOKENS, ApiEngine, Transcript
from relais.executor import PipelineConfig, PipelineOrchestrator
from relais.state import SQLiteStateManager
from relais.step import PipelineStep
from relais.tools import ToolRegistry, ToolResponse


# =============================================================================
# The stub client
# =============================================================================

class StubStream:
    def __init__(self, events, message):
        self._events = events
        self._message = message

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._message


class StubStreamManager:
    def __init__(self, turn):
        self._turn = turn

    async def __aenter__(self):
        if isinstance(self._turn, Exception):
            raise self._turn
        return StubStream(*self._turn)

    async def __aexit__(self, *exc):
        return False


class StubMessages:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def stream(self, **request):
        self.requests.append(copy.deepcopy(request))
        return StubStreamManager(self.turns.pop(0))


class StubAnthropic:
    def __init__(self, turns=()):
        self.messages = StubMessages(turns)


@pytest.fixture
def stub(monkeypatch):
    """The stub client every AsyncAnthropic(...) returns; `stub.made` records
    the constructor's kwargs. Script it with stub.messages.turns = [...]."""
    client = StubAnthropic()
    client.made = []

    def factory(**kwargs):
        client.made.append(kwargs)
        return client

    monkeypatch.setattr(api_engine, "AsyncAnthropic", factory)
    return client


def turn(*blocks, stop_reason=None, usage=(10, 5, 0, 0)):
    """A scripted model turn: the stream events for `blocks` and the final Message.
    stop_reason defaults to tool_use when a block is a tool call, else end_turn."""
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    events = []
    for index, block in enumerate(blocks):
        if block.type == "text":
            events.append(RawContentBlockStartEvent(
                type="content_block_start", index=index, content_block=TextBlock(type="text", text="")))
            for piece in (block.text[: len(block.text) // 2], block.text[len(block.text) // 2:]):
                events.append(RawContentBlockDeltaEvent(
                    type="content_block_delta", index=index, delta=TextDelta(type="text_delta", text=piece)))
        else:
            events.append(RawContentBlockStartEvent(type="content_block_start", index=index, content_block=block))
    message = Message(
        id="msg", type="message", role="assistant", model="claude-test", content=list(blocks),
        stop_reason=stop_reason, stop_sequence=None,
        usage=Usage(input_tokens=usage[0], output_tokens=usage[1],
                    cache_read_input_tokens=usage[2], cache_creation_input_tokens=usage[3]),
    )
    return events, message


def text(t):
    return TextBlock(type="text", text=t)


def call(tool_id, name, tool_input=None):
    return ToolUseBlock(type="tool_use", id=tool_id, name=name, input=tool_input or {})


def api_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(f"{status} from the API", response=httpx2.Response(status, request=request), body=None)


# =============================================================================
# Fixtures: a real registry, state database, step
# =============================================================================

@pytest.fixture
def registry():
    registry = ToolRegistry("pipe_tools")

    @registry.tool("classify", "Record the category — the step's response tool")
    async def classify(category: Annotated[str, "The category"]) -> dict:
        return ToolResponse(data={"category": category}).to_mcp()

    @registry.tool("lookup", "Look something up")
    async def lookup(query: Annotated[str, "What to look up"]) -> dict:
        return {"content": [{"type": "text", "text": f"found {query}"}]}

    @registry.tool("screenshot", "Take a screenshot")
    async def screenshot() -> dict:
        return ToolResponse(data={"taken": True}, images=[("iVBORw0", "image/png")]).to_mcp()

    @registry.tool("secret", "Not for this step")
    async def secret() -> dict:
        return {"content": [{"type": "text", "text": "the secret"}]}

    return registry


@pytest.fixture
def state(tmp_path):
    manager = SQLiteStateManager.create(str(tmp_path / "run.db"))
    manager.initialize_schema()
    return manager


def make_step(agent, tools=("lookup", "classify"), name="s1"):
    return PipelineStep(name=name, instruction="x", response_tool="classify", tools=list(tools), agent=agent)


async def run(engine, agent, registry, *, step=None, prompt="Classify this.", images=(), emit=None):
    """Run one step as the executor does: gate the registry, then the engine."""
    step = step or make_step(agent)
    registry.set_current_step(step.name, step.tools)
    return await engine.run_step(agent=agent, step=step, prompt=prompt, images=list(images),
                                 registry=registry, emit=emit, verbose=False)


# =============================================================================
# Tests
# =============================================================================

class TestTextAndEvents:
    async def test_text_deltas_with_a_paragraph_break_and_tool_use_events(self, stub, state, registry):
        stub.messages.turns = [
            turn(text("Let me look."), call("t1", "lookup", {"query": "x"})),
            turn(text("Found it."), call("t2", "classify", {"category": "q"})),
            turn(text("Done.")),
        ]
        events = []

        async def sink(event):
            events.append(event)

        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry, emit=sink)

        prose = [e for e in events if e["type"] in ("text_delta", "tool_use")]
        assert prose == [
            {"type": "text_delta", "step": "s1", "text": "Let me"},
            {"type": "text_delta", "step": "s1", "text": " look."},
            {"type": "tool_use", "step": "s1", "name": "lookup", "input": {"query": "x"}},
            {"type": "text_delta", "step": "s1", "text": "\n\n"},
            {"type": "text_delta", "step": "s1", "text": "Foun"},
            {"type": "text_delta", "step": "s1", "text": "d it."},
            {"type": "tool_use", "step": "s1", "name": "classify", "input": {"category": "q"}},
            {"type": "text_delta", "step": "s1", "text": "\n\n"},
            {"type": "text_delta", "step": "s1", "text": "Do"},
            {"type": "text_delta", "step": "s1", "text": "ne."},
        ]
        assert outcome.final_text == "Done."
        assert outcome.turns_used == 3
        assert outcome.is_error is False and outcome.error_kind is None

    async def test_no_sink_no_events(self, stub, state, registry):
        stub.messages.turns = [turn(text("Hi."))]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)
        assert outcome.final_text == "Hi."


class TestTools:
    async def test_a_tool_call_runs_through_the_registry(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "the weather"})),
            turn(text("Sunny.")),
        ]
        agent = PipelineAgent(name="a")
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert outcome.tool_calls == [{"id": "t1", "tool": "lookup", "input": {"query": "the weather"}}]
        # The second turn was sent the tool's result, as one user message.
        second = stub.messages.requests[1]["messages"]
        assert second[-1] == {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "found the weather"}]},
        ]}
        assert registry.get_tool_result("lookup") == ("lookup", {"content": [{"type": "text", "text": "found the weather"}]})

    async def test_the_wrappers_own_check_still_refuses(self, stub, state, registry):
        """The belt: a tool the request offered but the registry's gate doesn't
        allow is refused by the wrapper, body unrun."""
        stub.messages.turns = [turn(call("t1", "lookup", {"query": "x"})), turn(text("ok"))]
        agent = PipelineAgent(name="a")
        step = make_step(agent)
        engine = ApiEngine(state_manager=state)

        registry.set_current_step("other_step", ["classify"])  # the gate disagrees with the step
        await engine.run_step(agent=agent, step=step, prompt="p", images=[], registry=registry,
                              emit=None, verbose=False)

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert "is not available in step 'other_step'" in result["content"][0]["text"]
        assert registry.get_tool_result("lookup") is None

    async def test_a_tool_outside_the_step_is_refused_as_an_error(self, stub, state, registry):
        """The model can't name a tool the request didn't offer — but if it does,
        the registry's refusal goes back as an error result and nothing runs."""
        stub.messages.turns = [turn(call("t1", "secret")), turn(text("ok"))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert result["is_error"] is True
        assert result["content"] == (
            "Tool 'secret' is not available in step 's1'. Available tools: classify, lookup"
        )
        assert registry.get_tool_result("secret") is None

    async def test_the_response_tool_is_captured(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "question"})), turn(text("Done."))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)

        name, result = registry.get_tool_result("classify")
        assert name == "classify"
        assert result == {"content": [{"type": "text", "text": '{"category": "question"}'}]}

    async def test_image_results_become_image_blocks(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "screenshot")), turn(text("I see it."))]
        agent = PipelineAgent(name="a")
        await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["screenshot", "classify"]))

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert result["content"] == [
            {"type": "text", "text": '{"taken": true}'},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0"}},
        ]

    async def test_is_error_passes_through(self, stub, state):
        registry = ToolRegistry("t")

        @registry.tool("flaky", "Fails politely")
        async def flaky() -> dict:
            return {"content": [{"type": "text", "text": "upstream down"}], "is_error": True}

        stub.messages.turns = [turn(call("t1", "flaky")), turn(text("It failed."))]
        agent = PipelineAgent(name="a")
        await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["flaky"]))

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert result == {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                          "content": [{"type": "text", "text": "upstream down"}]}

    async def test_a_step_tool_the_registry_lacks_is_refused_up_front(self, stub, state, registry):
        agent = PipelineAgent(name="a")
        with pytest.raises(ValueError, match="'Read', which is not a registered pipeline tool"):
            await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["Read", "classify"]))


class TestRequest:
    async def test_only_the_steps_tools_are_offered(self, stub, state, registry):
        stub.messages.turns = [turn(text("ok"))]
        agent = PipelineAgent(name="a", tools=["lookup", "classify", "secret", "screenshot"])
        await run(ApiEngine(state_manager=state), agent, registry)

        tools = stub.messages.requests[0]["tools"]
        assert [t["name"] for t in tools] == ["lookup", "classify"]
        assert tools[0] == {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": registry.get("lookup").input_schema,
        }
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_a_tool_without_parameters_gets_an_object_schema(self, stub, state, registry):
        stub.messages.turns = [turn(text("ok"))]
        agent = PipelineAgent(name="a")
        await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["screenshot"]))
        assert stub.messages.requests[0]["tools"][0]["input_schema"] == {"type": "object", "properties": {}}

    async def test_the_request_shape(self, stub, state, registry):
        stub.messages.turns = [turn(text("ok"))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry, prompt="The step's prompt.")

        request = stub.messages.requests[0]
        assert request["model"] == "claude-opus-5-5"
        assert request["max_tokens"] == MAX_TOKENS == 16000
        assert request["system"] == [{"type": "text", "text": API_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        assert request["cache_control"] == {"type": "ephemeral"}  # automatic caching of the transcript tail
        assert request["messages"] == [{"role": "user", "content": "The step's prompt."}]
        assert "thinking" not in request
        assert "output_config" not in request

    async def test_thinking_and_effort(self, stub, state, registry):
        stub.messages.turns = [turn(text("ok"))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a", thinking=True, effort="high"), registry)

        request = stub.messages.requests[0]
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": "high"}

    @pytest.mark.parametrize("model,resolved", [
        ("opus", "claude-opus-5-5"),
        ("sonnet", "claude-sonnet-5-5"),
        ("haiku", "claude-haiku-5-5"),
        (None, "claude-opus-5-5"),
        ("claude-fable-5-1", "claude-fable-5-1"),
    ])
    async def test_model_aliases(self, stub, state, registry, model, resolved):
        stub.messages.turns = [turn(text("ok"))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a", model=model), registry)
        assert stub.messages.requests[0]["model"] == resolved

    async def test_images_ride_in_the_user_message(self, stub, state, registry):
        stub.messages.turns = [turn(text("A cat."))]
        await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry, prompt="What is this?",
                  images=["AAA", "BBB"])

        assert stub.messages.requests[0]["messages"] == [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBB"}},
        ]}]


class TestStops:
    async def test_max_turns(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "a"})),
            turn(call("t2", "lookup", {"query": "b"})),
            turn(text("never reached")),
        ]
        agent = PipelineAgent(name="a", max_turns=2)
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert outcome.is_error is True
        assert outcome.error_kind == "max_turns"
        assert outcome.turns_used == 2
        assert len(stub.messages.requests) == 2
        # The last turn's tools still ran (a response tool there would be captured).
        assert [c["input"]["query"] for c in outcome.tool_calls] == ["a", "b"]
        assert agent.client.messages[-1]["content"][0]["tool_use_id"] == "t2"

    async def test_a_response_tool_in_the_last_allowed_turn_is_captured(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "q"}))]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a", max_turns=1), registry)
        assert outcome.error_kind == "max_turns"
        assert registry.get_tool_result("classify") is not None

    async def test_max_tokens(self, stub, state, registry):
        """Cut off mid-reply: the (possibly partial) tool call is not run, and is
        answered as not-run so the transcript stays valid."""
        stub.messages.turns = [turn(text("Let me"), call("t1", "lookup", {"query": "par"}), stop_reason="max_tokens")]
        agent = PipelineAgent(name="a")
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert outcome.is_error is True
        assert outcome.error_kind == "max_tokens"
        assert outcome.final_text == "Let me"
        assert outcome.tool_calls == []
        assert registry.get_tool_result("lookup") is None
        [not_run] = agent.client.messages[-1]["content"]
        assert not_run["tool_use_id"] == "t1" and not_run["is_error"] is True

    async def test_refusal(self, stub, state, registry):
        stub.messages.turns = [turn(text("I can't help with that."), stop_reason="refusal")]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)
        assert (outcome.is_error, outcome.error_kind) == (True, "refusal")

    async def test_a_rate_limit_is_an_outcome(self, stub, state, registry):
        stub.messages.turns = [api_error(anthropic.RateLimitError, 429)]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)

        assert outcome.is_error is True
        assert outcome.error_kind == "rate_limit"
        assert outcome.result_text == "429 from the API"
        assert outcome.turns_used == 0

    async def test_other_api_errors_are_named_by_class(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "x"})),
            api_error(anthropic.InternalServerError, 500),
        ]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry)

        assert outcome.error_kind == "InternalServerError"
        assert outcome.turns_used == 1
        assert outcome.tool_calls == [{"id": "t1", "tool": "lookup", "input": {"query": "x"}}]


class TestUsage:
    async def test_usage_is_summed_and_reported_per_turn(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "x"}), usage=(100, 20, 0, 900)),
            turn(text("Done."), usage=(30, 10, 900, 40)),
        ]
        events = []

        async def sink(event):
            events.append(event)

        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a", model="sonnet"), registry, emit=sink)

        assert outcome.usage == {"input_tokens": 130, "output_tokens": 30,
                                 "cache_read_input_tokens": 900, "cache_creation_input_tokens": 940}
        assert [e for e in events if e["type"] == "usage"] == [
            {"type": "usage", "step": "s1", "model": "claude-sonnet-5-5", "input_tokens": 100,
             "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 900},
            {"type": "usage", "step": "s1", "model": "claude-sonnet-5-5", "input_tokens": 30,
             "output_tokens": 10, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 40},
        ]


class TestTranscript:
    async def test_the_transcript_is_reused_across_an_agents_steps(self, stub, state, registry):
        stub.messages.turns = [turn(text("One.")), turn(text("Two."))]
        engine = ApiEngine(state_manager=state)
        agent = PipelineAgent(name="a")

        first = await run(engine, agent, registry, prompt="Step one.")
        second = await run(engine, agent, registry, step=make_step(agent, name="s2"), prompt="Step two.")

        assert first.session_id == second.session_id == agent.client.id
        assert stub.messages.requests[1]["messages"] == [
            {"role": "user", "content": "Step one."},
            {"role": "assistant", "content": [{"type": "text", "text": "One."}]},
            {"role": "user", "content": "Step two."},
        ]

    async def test_saved_after_every_step_and_resumed(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "q"})), turn(text("Classified."))]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry, prompt="Go.")

        saved = state.load_transcript(outcome.session_id)
        assert [m["role"] for m in saved] == ["user", "assistant", "user", "assistant"]
        assert saved[1]["content"] == [{"type": "tool_use", "id": "t1", "name": "classify", "input": {"category": "q"}}]

        # A later run: a fresh engine and agent resume the transcript by its id.
        stub.messages.turns = [turn(text("You asked me to classify."))]
        resumed = PipelineAgent(name="a", resume=outcome.session_id)
        later = await run(ApiEngine(state_manager=state), resumed, registry, prompt="What did I ask?")

        assert later.session_id == outcome.session_id
        assert stub.messages.requests[-1]["messages"] == saved + [{"role": "user", "content": "What did I ask?"}]
        assert len(state.load_transcript(outcome.session_id)) == 6

    async def test_saved_when_the_step_errors(self, stub, state, registry):
        stub.messages.turns = [api_error(anthropic.RateLimitError, 429)]
        outcome = await run(ApiEngine(state_manager=state), PipelineAgent(name="a"), registry, prompt="Go.")
        assert state.load_transcript(outcome.session_id) == [{"role": "user", "content": "Go."}]

    async def test_resuming_an_unknown_transcript_fails(self, stub, state, registry):
        with pytest.raises(ValueError, match="transcript 'gone'"):
            await run(ApiEngine(state_manager=state), PipelineAgent(name="a", resume="gone"), registry)

    async def test_close_lets_go_of_the_transcript(self, state):
        agent = PipelineAgent(name="a")
        agent.set_client(Transcript(id="t"))
        await ApiEngine(state_manager=state).close(agent)
        assert agent.client is None


class TestClients:
    async def test_one_client_per_key(self, stub, state, registry):
        stub.messages.turns = [turn(text("1")), turn(text("2")), turn(text("3"))]
        engine = ApiEngine(state_manager=state)
        await run(engine, PipelineAgent(name="a", env={"ANTHROPIC_API_KEY": "k1"}), registry)
        await run(engine, PipelineAgent(name="b", env={"ANTHROPIC_API_KEY": "k1"}), registry)
        await run(engine, PipelineAgent(name="c"), registry)

        assert stub.made == [{"api_key": "k1", "max_retries": 4}, {"api_key": None, "max_retries": 4}]

    def test_a_new_event_loop_gets_a_new_client(self, stub, state, registry):
        """pipeline.run() is a fresh asyncio.run() per call; an HTTP client's
        connections can't outlive the loop that opened them."""
        stub.messages.turns = [turn(text("1")), turn(text("2"))]
        engine = ApiEngine(state_manager=state)
        asyncio.run(run(engine, PipelineAgent(name="a"), registry))
        asyncio.run(run(engine, PipelineAgent(name="a"), registry))
        assert len(stub.made) == 2


class TestThroughTheOrchestrator:
    async def test_an_api_agent_runs_a_pipeline(self, stub, state, registry):
        """The executor's half is engine-neutral: the response tool's capture
        routes, step_done carries the transcript id and usage."""
        stub.messages.turns = [
            turn(text("It's a question."), call("t1", "classify", {"category": "question"})),
            turn(text("Done.")),
        ]
        agent = PipelineAgent(name="a", engine="api", tools=["classify"])
        step = PipelineStep(name="s1", instruction="x", instruction_text="Classify it.", response_tool="classify",
                            tools=["classify"], agent=agent, next={"default": None})
        orchestrator = PipelineOrchestrator(tool_registry=registry, state_manager=state, instructions_dir=Path("/nowhere"))
        orchestrator.register_pipeline(PipelineConfig(name="p", steps={"s1": step}, start_step="s1",
                                                      instructions_dir="/nowhere", agents={"a": agent}))
        events = []

        async def sink(event):
            events.append(event)

        orchestrator._on_event = sink
        state_run = state.create_pipeline_run("p", "s1")
        await orchestrator._execute_pipeline(state_run, orchestrator.pipelines["p"], "Is this a question?", {})

        done = [e for e in events if e["type"] == "step_done"][0]
        assert done["routing_data"] == {"category": "question"}
        assert done["usage"]["input_tokens"] == 20
        assert state.load_transcript(done["session_id"])[0]["content"].startswith("[User Input]\nIs this a question?")
        assert state.get_pipeline_run(state_run).status == "completed"
