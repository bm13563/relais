"""Unit tests for engine.py — FieldStream (the incremental field decoder) and
StepEvents' streaming of a response-tool argument."""

import json

import pytest

from relais.agent import PipelineAgent
from relais.engine import FieldStream, StepEvents
from relais.step import PipelineStep


def feed_all(stream: FieldStream, chunks) -> str:
    return "".join(stream.feed(chunk) for chunk in chunks)


# A tool input with escapes of every kind, a surrogate pair, raw unicode, a
# nested object and a decoy before the field, and a key after it.
SAMPLE = (
    '{"meta": {"answer": "nested decoy", "list": [1, {"answer": "deeper"}]}, '
    '"note": "\\"answer\\": not this one", '
    '"answer": "Line one.\\nTab:\\there \\"quoted\\" back\\\\slash \\/slash '
    'caf\\u00e9 raw-é \\ud83d\\ude00 raw-😀 \\b\\f\\r end", '
    '"after": "not streamed"}'
)
EXPECTED = json.loads(SAMPLE)["answer"]


class TestFieldStream:
    def test_the_whole_input_at_once(self):
        assert FieldStream("answer").feed(SAMPLE) == EXPECTED

    @pytest.mark.parametrize("cut", range(len(SAMPLE) + 1))
    def test_split_at_every_boundary(self, cut):
        """A chunk may end anywhere — mid-key, mid-escape, mid-\\uXXXX, between
        a surrogate pair's halves."""
        assert feed_all(FieldStream("answer"), [SAMPLE[:cut], SAMPLE[cut:]]) == EXPECTED

    def test_one_character_at_a_time(self):
        assert feed_all(FieldStream("answer"), SAMPLE) == EXPECTED

    def test_text_arrives_as_it_is_written(self):
        stream = FieldStream("answer")
        assert stream.feed('{"answer": "Hel') == "Hel"
        assert stream.feed("lo, wor") == "lo, wor"
        assert stream.feed('ld", "x": "y"}') == "ld"

    def test_a_key_after_other_keys(self):
        src = '{"sources": ["a", "b"], "count": 2, "ok": true, "empty": null, "answer": "last"}'
        assert FieldStream("answer").feed(src) == "last"

    def test_a_missing_key_yields_nothing(self):
        assert feed_all(FieldStream("answer"), ['{"sources": ["a"],', ' "text": "elsewhere"}']) == ""

    def test_a_nested_object_before_the_key(self):
        src = '{"meta": {"answer": "no", "deep": {"answer": "no"}}, "answer": "yes"}'
        assert FieldStream("answer").feed(src) == "yes"

    def test_escaped_quotes_inside_the_value(self):
        src = json.dumps({"answer": 'She said "hi" and "bye".'})
        assert FieldStream("answer").feed(src) == 'She said "hi" and "bye".'

    def test_the_name_inside_another_string_is_not_the_key(self):
        src = '{"note": "answer", "x": "\\"answer\\": \\"decoy\\"", "answer": "real"}'
        assert FieldStream("answer").feed(src) == "real"

    def test_a_non_string_value_yields_nothing(self):
        assert FieldStream("answer").feed('{"answer": null, "other": "text"}') == ""
        assert FieldStream("answer").feed('{"answer": ["a", "b"]}') == ""

    def test_an_escaped_key_still_matches(self):
        assert FieldStream("answer").feed('{"\\u0061nswer": "yes"}') == "yes"

    def test_a_lone_high_surrogate_becomes_a_replacement_character(self):
        assert FieldStream("answer").feed('{"answer": "a\\ud83dz"}') == "a�z"


class TestStepEventsStreaming:
    """StepEvents streams the response tool's `streams` argument as text."""

    def _events(self, streams="answer", sink=None):
        step = PipelineStep(name="s", instruction="x", response_tool="finish", tools=["finish"],
                            agent=PipelineAgent(name="a"), streams=streams)
        return StepEvents(sink, step, verbose=False)

    async def test_the_argument_streams_as_a_text_block(self):
        sent = []

        async def sink(event):
            sent.append(event)

        events = self._events(sink=sink)
        await events.text_block_start()
        await events.text_delta("Prose first.")
        events.tool_block_start(1, "finish")
        for chunk in ['{"answer": "Hi', ' there"', ', "n": 1}']:
            await events.tool_input_delta(1, chunk)

        assert [e["text"] for e in sent] == ["Prose first.", "\n\n", "Hi", " there"]

    async def test_other_tools_and_unset_streams_stream_nothing(self):
        sent = []

        async def sink(event):
            sent.append(event)

        events = self._events(sink=sink)
        events.tool_block_start(0, "lookup")
        await events.tool_input_delta(0, '{"answer": "not the response tool"}')

        unset = self._events(streams=None, sink=sink)
        unset.tool_block_start(0, "finish")
        await unset.tool_input_delta(0, '{"answer": "streams is unset"}')

        assert sent == []

    async def test_a_later_block_at_the_same_index_is_not_the_response_tool(self):
        """Indexes restart with every message: a new tool block at an index
        replaces whatever was there."""
        sent = []

        async def sink(event):
            sent.append(event)

        events = self._events(sink=sink)
        events.tool_block_start(0, "finish")
        events.tool_block_start(0, "lookup")
        await events.tool_input_delta(0, '{"answer": "lookup input"}')
        assert sent == []
