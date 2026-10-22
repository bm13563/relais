"""Unit tests for engines/api.py — the in-process Messages API engine (offline).

AsyncAnthropic is replaced by a stub whose messages.stream(...) is an async
context manager over scripted stream events, with get_final_message() returning
a real anthropic Message. Each scripted turn is either (events, message) or an
exception the stream raises on entry. Every request is recorded (deep-copied, as
the transcript keeps growing after it is sent).
"""

import asyncio
import copy
import json
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import anthropic
import httpx2
import pytest
from anthropic.types import (
    InputJSONDelta,
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


def turn(*blocks, stop_reason=None, usage=(10, 5, 0, 0), json_chunk=7):
    """A scripted model turn: the stream events for `blocks` and the final Message.
    stop_reason defaults to tool_use when a block is a tool call, else end_turn.
    A tool call streams as the API streams it: a block start with empty input,
    then its input JSON in `json_chunk`-character input_json_delta pieces."""
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
            events.append(RawContentBlockStartEvent(
                type="content_block_start", index=index, content_block=block.model_copy(update={"input": {}})))
            raw = json.dumps(block.input)
            for at in range(0, len(raw), json_chunk):
                events.append(RawContentBlockDeltaEvent(
                    type="content_block_delta", index=index,
                    delta=InputJSONDelta(type="input_json_delta", partial_json=raw[at:at + json_chunk])))
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


def answer(category="q", tool_id="tr"):
    """The model calling the step's response tool (classify)."""
    return call(tool_id, "classify", {"category": category})


def answered(final="Done."):
    """A well-behaved step's script: one turn that answers and calls the
    response tool — the step ends there."""
    return [turn(text(final), answer())]


def assert_alternates(messages):
    """A valid transcript: user first, then strictly alternating roles."""
    assert [m["role"] for m in messages] == ["user", "assistant"] * (len(messages) // 2) + ["user"] * (len(messages) % 2)


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

    @registry.tool("finish", "Hand over the answer — a response tool whose answer streams")
    async def finish(answer: Annotated[str, "The answer, for the person"],
                     sources: Annotated[list, "Pages used"] = None) -> dict:
        return ToolResponse(data={"answer": answer, "sources": sources or []}).to_mcp()

    return registry


@pytest.fixture
def state(tmp_path):
    manager = SQLiteStateManager.create(str(tmp_path / "run.db"))
    manager.initialize_schema()
    return manager


def make_agent(name="a", tools=("lookup", "classify"), **kwargs):
    return PipelineAgent(name=name, tools=list(tools), **kwargs)


def make_step(agent, tools=("lookup", "classify"), name="s1", response_tool="classify", streams=None):
    return PipelineStep(name=name, instruction="x", response_tool=response_tool, tools=list(tools), agent=agent,
                        streams=streams)


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
            turn(text("never requested — the step ended on its response tool")),
        ]
        events = []

        async def sink(event):
            events.append(event)

        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry, emit=sink)

        prose = [e for e in events if e["type"] in ("text_delta", "tool_use")]
        assert prose == [
            {"type": "text_delta", "step": "s1", "text": "Let me"},
            {"type": "text_delta", "step": "s1", "text": " look."},
            {"type": "tool_use", "step": "s1", "name": "lookup", "input": {"query": "x"}},
            {"type": "text_delta", "step": "s1", "text": "\n\n"},
            {"type": "text_delta", "step": "s1", "text": "Foun"},
            {"type": "text_delta", "step": "s1", "text": "d it."},
            {"type": "tool_use", "step": "s1", "name": "classify", "input": {"category": "q"}},
        ]
        assert outcome.final_text == "Found it."
        assert outcome.turns_used == 2
        assert outcome.is_error is False and outcome.error_kind is None

    async def test_no_sink_no_events(self, stub, state, registry):
        stub.messages.turns = answered("Hi.")
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry)
        assert outcome.final_text == "Hi."


class TestTools:
    async def test_a_tool_call_runs_through_the_registry(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "the weather"})),
            *answered("Sunny."),
        ]
        agent = make_agent("a")
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert outcome.tool_calls == [
            {"id": "t1", "tool": "lookup", "input": {"query": "the weather"}},
            {"id": "tr", "tool": "classify", "input": {"category": "q"}},
        ]
        # The second turn was sent the tool's result, as one user message.
        second = stub.messages.requests[1]["messages"]
        assert second[-1] == {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "found the weather"}]},
        ]}
        assert registry.get_tool_result("lookup") == ("lookup", {"content": [{"type": "text", "text": "found the weather"}]})

    async def test_a_tool_outside_the_agents_set_is_refused_as_an_error(self, stub, state, registry):
        """The model can't name a tool the request didn't offer — but if it does,
        the registry's refusal goes back as an error result and nothing runs."""
        stub.messages.turns = [turn(call("t1", "secret")), *answered("ok")]
        await run(ApiEngine(state_manager=state), make_agent("a"), registry)

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert result["is_error"] is True
        assert result["content"] == (
            "Tool 'secret' is not available in step 's1'. Available tools: classify, lookup"
        )
        assert registry.get_tool_result("secret") is None

    async def test_the_response_tool_is_captured(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "question"})), turn(text("Done."))]
        await run(ApiEngine(state_manager=state), make_agent("a"), registry)

        name, result = registry.get_tool_result("classify")
        assert name == "classify"
        assert result == {"content": [{"type": "text", "text": '{"category": "question"}'}]}

    async def test_image_results_become_image_blocks(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "screenshot")), *answered("I see it.")]
        agent = make_agent("a", tools=["screenshot", "classify"])
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

        stub.messages.turns = [turn(call("t1", "flaky"))]
        agent = make_agent("a", tools=["flaky"])
        await run(ApiEngine(state_manager=state), agent, registry,
                  step=make_step(agent, tools=["flaky"], response_tool="flaky"))

        [result] = agent.client.messages[-1]["content"]
        assert result == {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                          "content": [{"type": "text", "text": "upstream down"}]}

    async def test_an_agent_tool_the_registry_lacks_is_refused_up_front(self, stub, state, registry):
        agent = make_agent("a", tools=["Read", "classify"])
        with pytest.raises(ValueError, match="'Read', which is not a registered pipeline tool"):
            await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["classify"]))


class TestRequest:
    async def test_the_agents_tools_are_offered_and_the_step_gates_them(self, stub, state, registry):
        """Every request offers the agent's whole tool set; a call to one outside
        the current step gets the wrapper's refusal and its body never runs."""
        stub.messages.turns = [turn(call("t1", "secret")), *answered("ok")]
        agent = make_agent("a", tools=["lookup", "classify", "secret"])
        await run(ApiEngine(state_manager=state), agent, registry, step=make_step(agent, tools=["lookup", "classify"]))

        tools = stub.messages.requests[0]["tools"]
        assert [t["name"] for t in tools] == ["lookup", "classify", "secret"]
        assert tools[0] == {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": registry.get("lookup").input_schema,
        }
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

        [result] = stub.messages.requests[1]["messages"][-1]["content"]
        assert result == {"type": "tool_result", "tool_use_id": "t1", "content": [{
            "type": "text", "text": "Tool 'secret' is not available in step 's1'. Available tools: classify, lookup",
        }]}
        assert registry.get_tool_result("secret") is None

    async def test_the_offered_tools_are_the_same_on_every_step(self, stub, state, registry):
        """Steps that list different tools still send one tools prefix — the one
        the agent's thinking blocks are bound to."""
        stub.messages.turns = [*answered("one"), *answered("two")]
        engine = ApiEngine(state_manager=state)
        agent = make_agent("a", tools=["lookup", "classify", "screenshot"])
        await run(engine, agent, registry, step=make_step(agent, tools=["lookup", "classify"]))
        await run(engine, agent, registry, step=make_step(agent, tools=["screenshot", "classify"], name="s2"))

        requests = stub.messages.requests
        assert len(requests) == 2
        assert all(r["tools"] == requests[0]["tools"] and r["system"] == requests[0]["system"] for r in requests)

    async def test_a_tool_without_parameters_gets_an_object_schema(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "screenshot")), turn(text("ok"))]
        agent = make_agent("a", tools=["screenshot"])
        await run(ApiEngine(state_manager=state), agent, registry,
                  step=make_step(agent, tools=["screenshot"], response_tool="screenshot"))
        assert stub.messages.requests[0]["tools"][0]["input_schema"] == {"type": "object", "properties": {}}

    async def test_the_request_shape(self, stub, state, registry):
        stub.messages.turns = answered("ok")
        await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="The step's prompt.")

        request = stub.messages.requests[0]
        assert request["model"] == "claude-opus-5-5"
        assert request["max_tokens"] == MAX_TOKENS == 32000
        assert request["system"] == [{"type": "text", "text": API_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        assert request["cache_control"] == {"type": "ephemeral"}  # automatic caching of the transcript tail
        assert request["messages"] == [{"role": "user", "content": "The step's prompt."}]
        assert "thinking" not in request
        assert "output_config" not in request

    async def test_effort(self, stub, state, registry):
        stub.messages.turns = answered("ok")
        await run(ApiEngine(state_manager=state), make_agent("a", effort="high"), registry)
        assert stub.messages.requests[0]["output_config"] == {"effort": "high"}

    @pytest.mark.parametrize("model,thinking,sent", [
        ("sonnet", None, None),                         # the model's default: no field at all
        ("haiku", None, None),
        ("sonnet", True, {"type": "adaptive"}),
        ("haiku", True, {"type": "adaptive"}),
        ("haiku", False, {"type": "disabled"}),         # off, as Haiku 5.5 spells it
        ("sonnet", False, {"type": "between_tools"}),   # off, as Sonnet / Opus 5.5 spell it
        ("opus", False, {"type": "between_tools"}),
        ("claude-sonnet-5-5-20261001", False, {"type": "between_tools"}),  # a prefix match
        ("claude-fable-5-1", False, {"type": "disabled"}),                 # any other model
    ])
    async def test_thinking_is_a_tri_state(self, stub, state, registry, model, thinking, sent):
        stub.messages.turns = answered("ok")
        await run(ApiEngine(state_manager=state), make_agent("a", model=model, thinking=thinking), registry)

        request = stub.messages.requests[0]
        if sent is None:
            assert "thinking" not in request
        else:
            assert request["thinking"] == sent

    async def test_the_system_prompt_says_thinking_is_private(self):
        assert "Your thinking is private: nobody sees it. Anything the person should read, write as text." in API_SYSTEM_PROMPT

    @pytest.mark.parametrize("model,resolved", [
        ("opus", "claude-opus-5-5"),
        ("sonnet", "claude-sonnet-5-5"),
        ("haiku", "claude-haiku-5-5"),
        (None, "claude-opus-5-5"),
        ("claude-fable-5-1", "claude-fable-5-1"),
    ])
    async def test_model_aliases(self, stub, state, registry, model, resolved):
        stub.messages.turns = answered("ok")
        await run(ApiEngine(state_manager=state), make_agent("a", model=model), registry)
        assert stub.messages.requests[0]["model"] == resolved

    async def test_images_ride_in_the_user_message(self, stub, state, registry):
        stub.messages.turns = answered("A cat.")
        await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="What is this?",
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
        agent = make_agent("a", max_turns=2)
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert outcome.is_error is True
        assert outcome.error_kind == "max_turns"
        assert outcome.turns_used == 2
        assert len(stub.messages.requests) == 2
        # The last turn's tools still ran (a response tool there would be captured).
        assert [c["input"]["query"] for c in outcome.tool_calls] == ["a", "b"]
        assert agent.client.messages[-1]["content"][0]["tool_use_id"] == "t2"

    async def test_a_response_tool_in_the_last_allowed_turn_ends_the_step_cleanly(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "q"}))]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a", max_turns=1), registry)
        assert (outcome.is_error, outcome.error_kind) == (False, None)
        assert registry.get_tool_result("classify") is not None

    async def test_max_tokens(self, stub, state, registry):
        """Cut off mid-reply: the (possibly partial) tool call is not run, and is
        answered as not-run so the transcript stays valid."""
        stub.messages.turns = [turn(text("Let me"), call("t1", "lookup", {"query": "par"}), stop_reason="max_tokens")]
        agent = make_agent("a")
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
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry)
        assert (outcome.is_error, outcome.error_kind) == (True, "refusal")

    async def test_a_rate_limit_is_an_outcome(self, stub, state, registry):
        stub.messages.turns = [api_error(anthropic.RateLimitError, 429)]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry)

        assert outcome.is_error is True
        assert outcome.error_kind == "rate_limit"
        assert outcome.result_text == "429 from the API"
        assert outcome.turns_used == 0

    async def test_other_api_errors_are_named_by_class(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "x"})),
            api_error(anthropic.InternalServerError, 500),
        ]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry)

        assert outcome.error_kind == "InternalServerError"
        assert outcome.turns_used == 1
        assert outcome.tool_calls == [{"id": "t1", "tool": "lookup", "input": {"query": "x"}}]


NUDGE = ("You ended your turn without calling classify, which completes this step. "
         "Call it now with your result; nothing else is passed on.")


class TestNudge:
    """A reply that ends the turn without the response tool gets ONE reminder."""

    async def test_an_end_turn_without_the_response_tool_is_nudged(self, stub, state, registry):
        stub.messages.turns = [turn(text("It's a question.")), turn(answer("question"))]
        agent = make_agent("a")
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert len(stub.messages.requests) == 2
        assert stub.messages.requests[1]["messages"][-1] == {"role": "user", "content": NUDGE}
        assert registry.get_tool_result("classify")[1] == {"content": [{"type": "text", "text": '{"category": "question"}'}]}
        assert (outcome.is_error, outcome.turns_used, outcome.final_text) == (False, 2, "It's a question.")
        # Kept in the transcript as its own user message, after the reply it answers.
        saved, _ = state.load_transcript(outcome.session_id)
        assert saved.count({"role": "user", "content": NUDGE}) == 1
        assert_alternates(saved)

    async def test_a_second_end_turn_without_it_ends_the_step(self, stub, state, registry):
        stub.messages.turns = [turn(text("A.")), turn(text("B.")), turn(text("never sent"))]
        agent = make_agent("a")
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert len(stub.messages.requests) == 2
        assert (outcome.is_error, outcome.turns_used, outcome.final_text) == (False, 2, "B.")
        assert registry.get_tool_result("classify") is None  # the executor fails the step, as before
        assert agent.client.messages.count({"role": "user", "content": NUDGE}) == 1
        assert agent.client.messages[-1]["role"] == "assistant"

    async def test_a_step_whose_response_tool_ran_is_not_nudged(self, stub, state, registry):
        stub.messages.turns = answered("Done.")
        agent = make_agent("a")
        await run(ApiEngine(state_manager=state), agent, registry)

        assert len(stub.messages.requests) == 1
        assert {"role": "user", "content": NUDGE} not in agent.client.messages

    async def test_no_nudge_once_the_turns_are_spent(self, stub, state, registry):
        stub.messages.turns = [turn(text("A.")), turn(text("never sent"))]
        agent = make_agent("a", max_turns=1)
        outcome = await run(ApiEngine(state_manager=state), agent, registry)

        assert len(stub.messages.requests) == 1
        assert outcome.turns_used == 1
        assert {"role": "user", "content": NUDGE} not in agent.client.messages

    async def test_the_nudged_turn_counts_toward_max_turns(self, stub, state, registry):
        stub.messages.turns = [turn(text("A.")), turn(call("t1", "lookup", {"query": "x"})), turn(text("never sent"))]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a", max_turns=2), registry)

        assert len(stub.messages.requests) == 2
        assert outcome.error_kind == "max_turns"
        assert registry.get_tool_result("classify") is None


class TestConsecutiveUserMessages:
    """A transcript can end with a user message (tool results); the next user
    message joins it — tool_result blocks first — rather than following it."""

    async def test_a_step_after_a_cut_off_step(self, stub, state, registry):
        stub.messages.turns = [
            turn(text("Let me"), call("t1", "lookup", {"query": "par"}), stop_reason="max_tokens"),
            *answered("ok"),
        ]
        engine = ApiEngine(state_manager=state)
        agent = make_agent("a")
        await run(engine, agent, registry, prompt="Step one.")
        await run(engine, agent, registry, step=make_step(agent, name="s2"), prompt="Step two.", images=["AAA"])

        sent = stub.messages.requests[1]["messages"]
        assert_alternates(sent)
        assert [b["type"] for b in sent[-1]["content"]] == ["tool_result", "text", "image"]
        assert sent[-1]["content"][0]["tool_use_id"] == "t1" and sent[-1]["content"][0]["is_error"] is True
        assert sent[-1]["content"][1:] == [
            {"type": "text", "text": "Step two."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        ]

    async def test_after_the_turn_cap(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "lookup", {"query": "x"})), *answered("ok")]
        engine = ApiEngine(state_manager=state)
        agent = make_agent("a", max_turns=1)
        first = await run(engine, agent, registry, prompt="Step one.")
        await run(engine, agent, registry, step=make_step(agent, name="s2"), prompt="Step two.")

        assert first.error_kind == "max_turns"
        sent = stub.messages.requests[1]["messages"]
        assert_alternates(sent)
        assert sent[-1]["content"][-1] == {"type": "text", "text": "Step two."}

    async def test_the_plain_case_appends_a_new_message(self, stub, state, registry):
        """A step that ended on an assistant reply (here: nudged, and still no
        response tool) leaves the next prompt a message of its own, as given."""
        stub.messages.turns = [turn(text("A.")), turn(text("B.")), *answered("ok")]
        engine = ApiEngine(state_manager=state)
        agent = make_agent("a")
        await run(engine, agent, registry, prompt="Step one.")
        await run(engine, agent, registry, step=make_step(agent, name="s2"), prompt="Step two.")

        sent = stub.messages.requests[2]["messages"]
        assert_alternates(sent)
        assert sent[-2] == {"role": "assistant", "content": [{"type": "text", "text": "B."}]}
        assert sent[-1] == {"role": "user", "content": "Step two."}


ANSWER = 'Line one.\nShe said "hi" — café 😀 \\ done.'


class TestStreamedAnswer:
    """PipelineStep.streams: the response tool's string argument streams as text
    while the model writes the call."""

    def _agent_and_step(self, streams="answer"):
        agent = make_agent("a", tools=["lookup", "finish"])
        return agent, make_step(agent, tools=["lookup", "finish"], response_tool="finish", streams=streams)

    async def _prose(self, stub, state, registry, turns, streams="answer"):
        stub.messages.turns = turns
        events = []

        async def sink(event):
            events.append(event)

        agent, step = self._agent_and_step(streams)
        outcome = await run(ApiEngine(state_manager=state), agent, registry, step=step, emit=sink)
        return outcome, [e for e in events if e["type"] in ("text_delta", "tool_use")]

    async def test_the_answer_streams_in_order_as_it_is_written(self, stub, state, registry):
        outcome, events = await self._prose(stub, state, registry, [
            turn(call("t1", "finish", {"sources": ["p1"], "answer": ANSWER}), json_chunk=5),
        ])

        deltas = [e["text"] for e in events if e["type"] == "text_delta"]
        assert "".join(deltas) == ANSWER
        assert len(deltas) > 3  # piece by piece, not at the end
        assert events[-1] == {"type": "tool_use", "step": "s1", "name": "finish",
                              "input": {"sources": ["p1"], "answer": ANSWER}}
        # Nothing else changes: the full input is still the captured output.
        _, result = registry.get_tool_result("finish")
        assert json.loads(result["content"][0]["text"]) == {"answer": ANSWER, "sources": ["p1"]}
        assert outcome.final_text == ""  # the answer was never a text block

    async def test_a_paragraph_break_when_prose_came_first(self, stub, state, registry):
        _, events = await self._prose(stub, state, registry, [
            turn(text("Let me check."), call("t1", "lookup", {"query": "x"})),
            turn(call("t2", "finish", {"answer": "It's x."}), json_chunk=1000),  # one delta
        ])

        assert [e.get("text", e.get("name")) for e in events] == [
            "Let me", " check.", "lookup", "\n\n", "It's x.", "finish",
        ]

    async def test_without_streams_nothing_streams(self, stub, state, registry):
        _, events = await self._prose(stub, state, registry, [turn(call("t1", "finish", {"answer": ANSWER}))],
                                      streams=None)
        assert [e["type"] for e in events] == ["tool_use"]

    async def test_another_tool_with_the_same_argument_does_not_stream(self, stub, state, registry):
        _, events = await self._prose(stub, state, registry, [
            turn(call("t1", "lookup", {"query": "x", "answer": "a lookup argument"})),
            turn(call("t2", "finish", {"answer": "Yes."}), json_chunk=1000),
        ])
        assert [e["text"] for e in events if e["type"] == "text_delta"] == ["Yes."]

    async def test_an_absent_argument_streams_nothing(self, stub, state, registry):
        agent = make_agent("a", tools=["finish"])
        step = make_step(agent, tools=["finish"], response_tool="finish", streams="summary")
        stub.messages.turns = [turn(call("t1", "finish", {"answer": "Yes."}))]
        events = []

        async def sink(event):
            events.append(event)

        await run(ApiEngine(state_manager=state), agent, registry, step=step, emit=sink)
        assert [e["type"] for e in events if e["type"] in ("text_delta", "tool_use")] == ["tool_use"]


class TestUsage:
    async def test_usage_is_summed_and_reported_per_turn(self, stub, state, registry):
        stub.messages.turns = [
            turn(call("t1", "lookup", {"query": "x"}), usage=(100, 20, 0, 900)),
            turn(text("Done."), answer(), usage=(30, 10, 900, 40)),
        ]
        events = []

        async def sink(event):
            events.append(event)

        outcome = await run(ApiEngine(state_manager=state), make_agent("a", model="sonnet"), registry, emit=sink)

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
        stub.messages.turns = [*answered("One."), *answered("Two.")]
        engine = ApiEngine(state_manager=state)
        agent = make_agent("a")

        first = await run(engine, agent, registry, prompt="Step one.")
        second = await run(engine, agent, registry, step=make_step(agent, name="s2"), prompt="Step two.")

        assert first.session_id == second.session_id == agent.client.id
        # Step one ended on its response tool, so its tool result is the last
        # message; step two's prompt joins it rather than making two user
        # messages in a row.
        assert stub.messages.requests[1]["messages"] == [
            {"role": "user", "content": "Step one."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "One."},
                {"type": "tool_use", "id": "tr", "name": "classify", "input": {"category": "q"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tr",
                 "content": [{"type": "text", "text": '{"category": "q"}'}]},
                {"type": "text", "text": "Step two."},
            ]},
        ]

    async def test_saved_after_every_step_and_resumed_under_the_same_prefix(self, stub, state, registry):
        stub.messages.turns = [turn(call("t1", "classify", {"category": "q"}))]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="Go.")

        saved, prefix = state.load_transcript(outcome.session_id)
        assert len(prefix) == 64  # a sha256 hex digest
        assert [m["role"] for m in saved] == ["user", "assistant", "user"]  # ends in its tool result
        assert saved[1]["content"] == [{"type": "tool_use", "id": "t1", "name": "classify", "input": {"category": "q"}}]

        # A later run: a fresh engine and agent resume the transcript by its id.
        stub.messages.turns = answered("You asked me to classify.")
        resumed = make_agent("a", resume=outcome.session_id)
        later = await run(ApiEngine(state_manager=state), resumed, registry, prompt="What did I ask?")

        assert later.session_id == outcome.session_id
        # The resumed prompt joins the saved tool result: no two user messages in a row.
        sent = stub.messages.requests[1]["messages"]
        assert sent == saved[:2] + [{"role": "user", "content": saved[2]["content"] + [
            {"type": "text", "text": "What did I ask?"},
        ]}]
        messages, still = state.load_transcript(outcome.session_id)
        assert still == prefix
        assert messages[:3] == sent
        assert messages[3]["content"][0] == {"type": "text", "text": "You asked me to classify."}
        assert_alternates(messages)

    async def test_saved_when_the_step_errors(self, stub, state, registry):
        stub.messages.turns = [api_error(anthropic.RateLimitError, 429)]
        outcome = await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="Go.")
        assert state.load_transcript(outcome.session_id)[0] == [{"role": "user", "content": "Go."}]

    async def test_an_unknown_transcript_begins_afresh_under_its_id(self, stub, state, registry):
        stub.messages.turns = answered("Hello.")
        with patch.object(api_engine, "log") as log:
            outcome = await run(ApiEngine(state_manager=state), make_agent("a", resume="gone"), registry, prompt="Hi.")

        assert outcome.session_id == "gone"
        assert outcome.is_error is False
        assert stub.messages.requests[0]["messages"] == [{"role": "user", "content": "Hi."}]
        assert state.load_transcript("gone")[0][0] == {"role": "user", "content": "Hi."}
        assert log.info.call_args.args == ("transcript_afresh",)
        assert log.info.call_args.kwargs["transcript"] == "gone" and log.info.call_args.kwargs["why"] == "not found"

    @pytest.mark.parametrize("changed", [
        {"tools": ["lookup", "classify", "screenshot"]},  # a tool added to the agent
        {"model": "sonnet"},                               # another model
    ])
    async def test_a_changed_prefix_begins_afresh_under_the_same_id(self, stub, state, registry, changed):
        stub.messages.turns = [*answered("One."), *answered("Fresh.")]
        first = await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="Before.")
        _, old_prefix = state.load_transcript(first.session_id)

        with patch.object(api_engine, "log") as log:
            later = await run(ApiEngine(state_manager=state), make_agent("a", resume=first.session_id, **changed),
                              registry, prompt="After.")

        assert later.session_id == first.session_id
        assert stub.messages.requests[1]["messages"] == [{"role": "user", "content": "After."}]
        messages, new_prefix = state.load_transcript(first.session_id)
        assert messages[0] == {"role": "user", "content": "After."}
        assert new_prefix != old_prefix
        assert log.info.call_args.args == ("transcript_afresh",)
        assert log.info.call_args.kwargs["transcript"] == first.session_id
        assert log.info.call_args.kwargs["why"] == "prefix changed"

    async def test_the_prefix_ignores_thinking_and_effort(self, stub, state, registry):
        """Request settings outside system / tools / model don't bind thinking."""
        stub.messages.turns = [*answered("One."), *answered("Two.")]
        first = await run(ApiEngine(state_manager=state), make_agent("a"), registry, prompt="Before.")
        await run(ApiEngine(state_manager=state), make_agent("a", resume=first.session_id, effort="high", thinking=True),
                  registry, prompt="After.")
        assert len(stub.messages.requests[1]["messages"]) == 3  # continued: "After." joins step one's tool result

    async def test_close_lets_go_of_the_transcript(self, state):
        agent = make_agent("a")
        agent.set_client(Transcript(id="t", prefix="p"))
        await ApiEngine(state_manager=state).close(agent)
        assert agent.client is None


class TestClients:
    async def test_one_client_per_key(self, stub, state, registry):
        stub.messages.turns = [*answered("1"), *answered("2"), *answered("3")]
        engine = ApiEngine(state_manager=state)
        await run(engine, make_agent("a", env={"ANTHROPIC_API_KEY": "k1"}), registry)
        await run(engine, make_agent("b", env={"ANTHROPIC_API_KEY": "k1"}), registry)
        await run(engine, make_agent("c"), registry)

        assert stub.made == [{"api_key": "k1", "max_retries": 4}, {"api_key": None, "max_retries": 4}]

    def test_a_new_event_loop_gets_a_new_client(self, stub, state, registry):
        """pipeline.run() is a fresh asyncio.run() per call; an HTTP client's
        connections can't outlive the loop that opened them."""
        stub.messages.turns = [*answered("1"), *answered("2")]
        engine = ApiEngine(state_manager=state)
        asyncio.run(run(engine, make_agent("a"), registry))
        asyncio.run(run(engine, make_agent("a"), registry))
        assert len(stub.made) == 2


class TestThroughTheOrchestrator:
    async def test_an_api_agent_runs_a_pipeline(self, stub, state, registry):
        """The executor's half is engine-neutral: the response tool's capture
        routes, step_done carries the transcript id and usage."""
        stub.messages.turns = [
            turn(text("It's a question."), call("t1", "classify", {"category": "question"})),
            turn(text("Done.")),
        ]
        agent = make_agent("a", engine="api", tools=["classify"])
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
        assert done["turns"] == 1  # it ended on its response tool
        assert done["usage"]["input_tokens"] == 10
        messages, _ = state.load_transcript(done["session_id"])
        assert messages[0]["content"].startswith("[User Input]\nIs this a question?")
        assert state.get_pipeline_run(state_run).status == "completed"
