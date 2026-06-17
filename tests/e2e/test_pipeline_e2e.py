"""End-to-end tests for pipeline execution.

These tests make real Claude model calls through the Claude Agent SDK and assert
on observable behavior (which tools were called, with what arguments, how routing
resolved), not just completion status.

They target the current agent-based API: every step carries an explicit
PipelineAgent, max_turns/model live on the agent, and every step declares a
response_tool.

Run with:  pytest tests/e2e -m e2e
The whole module skips automatically when no model credentials are available.
"""

import json
import shutil

import pytest
from typing import Annotated, List, Dict, Any
from dataclasses import dataclass, field

from relais import Pipeline, PipelineStep, PipelineAgent
from relais.state import SQLiteStateManager

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# Model used for e2e runs. haiku keeps them fast and cheap while still exercising
# the full agent loop, tool-gating, and routing machinery.
E2E_MODEL = "haiku"


def _has_credentials() -> bool:
    """Best-effort check that a model backend is reachable.

    The SDK can authenticate either via ANTHROPIC_API_KEY or a logged-in
    `claude` CLI. We accept either.
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    return shutil.which("claude") is not None


pytestmark.append(
    pytest.mark.skipif(
        not _has_credentials(),
        reason="No model credentials (set ANTHROPIC_API_KEY or log in with the claude CLI)",
    )
)


# =============================================================================
# Tool-call tracking
# =============================================================================

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


@dataclass
class ToolTracker:
    calls: List[ToolCall] = field(default_factory=list)

    def record(self, name: str, args: dict):
        self.calls.append(ToolCall(name=name, args=args))

    def get_calls(self, name: str) -> List[ToolCall]:
        return [c for c in self.calls if c.name == name]

    def was_called(self, name: str) -> bool:
        return bool(self.get_calls(name))

    def call_count(self, name: str) -> int:
        return len(self.get_calls(name))

    def last_call(self, name: str):
        calls = self.get_calls(name)
        return calls[-1] if calls else None


@pytest.fixture
def tracker():
    return ToolTracker()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "e2e.db")


@pytest.fixture
def instructions_dir(tmp_path):
    """Instruction files with explicit, deterministic rules."""
    d = tmp_path / "instructions"
    d.mkdir()

    (d / "greet.md").write_text(
        "Use the `send_greeting` tool to greet the user.\n"
        "Extract the user's name from the input; if none is given, use \"friend\".\n"
        "Set `message` to a greeting that includes the word \"Hello\".\n"
        "Call the tool exactly once."
    )
    (d / "classify.md").write_text(
        "Classify the input with these EXACT rules:\n"
        "1. Contains a question mark -> \"question\"\n"
        "2. Contains calculate/compute/add/multiply -> \"task\"\n"
        "3. Otherwise -> \"chat\"\n"
        "Call `classify_input` with category (one of question/task/chat) and confidence=0.95. "
        "Call it exactly once."
    )
    (d / "answer.md").write_text(
        "Use `send_answer` to answer the question. The answer must contain the word \"answer\". "
        "Call it once."
    )
    (d / "execute.md").write_text(
        "Use `execute_task`. Set result to include the word \"completed\". Call it once."
    )
    (d / "chat.md").write_text(
        "Use `send_chat` to respond casually. The message must include \"hey\". Call it once."
    )
    (d / "produce.md").write_text(
        "Call `produce_data` with output_value set to \"TOKEN_XYZ789\"."
    )
    (d / "consume.md").write_text(
        "Look at [Previous Step Output]. Call `consume_data` with received_value set to the "
        "\"produced\" value you see there."
    )
    (d / "use_args.md").write_text(
        "Look at the [Pipeline Args] section. Call `echo_arg` with mode set to the EXACT value "
        "of the \"mode\" arg you find there."
    )
    return d


def _make_pipeline(name, steps, start, instructions_dir, db_path):
    return Pipeline.create(
        name=name,
        steps=steps,
        start_step=start,
        instructions_dir=instructions_dir,
        db_config=db_path,
    )


# =============================================================================
# Greeting: argument extraction
# =============================================================================

class TestGreetingBehavior:
    def test_greeting_extracts_name_from_input(self, db_path, instructions_dir, tracker):
        agent = PipelineAgent(name="greeter", tools=["send_greeting"], max_turns=3, model=E2E_MODEL)
        steps = {
            "greet": PipelineStep(
                name="greet", instruction="greet", response_tool="send_greeting",
                tools=["send_greeting"], agent=agent, next={"default": None},
            )
        }
        pipeline = _make_pipeline("e2e_greeting_name", steps, "greet", instructions_dir, db_path)

        @pipeline.tool("send_greeting", "Send a greeting")
        async def send_greeting(
            message: Annotated[str, "The greeting message"],
            name: Annotated[str, "The name to greet"],
        ) -> dict:
            tracker.record("send_greeting", {"message": message, "name": name})
            return {"content": [{"type": "text", "text": json.dumps({"sent": True, "name": name})}]}

        pipeline.initialize_db()
        state = pipeline.get_run(pipeline.run("Please greet Alice!"))

        assert state.status == "completed"
        assert tracker.was_called("send_greeting")
        call = tracker.last_call("send_greeting")
        assert call.args.get("name") == "Alice"
        assert "hello" in call.args.get("message", "").lower()


# =============================================================================
# Classification routing
# =============================================================================

class TestClassificationRouting:
    def _build(self, name, db_path, instructions_dir, tracker):
        def agent(n, tools):
            return PipelineAgent(name=n, tools=tools, max_turns=2, model=E2E_MODEL)

        steps = {
            "classify": PipelineStep(
                name="classify", instruction="classify", response_tool="classify_input",
                tools=["classify_input"], agent=agent("classifier", ["classify_input"]),
                next={
                    "field": "category",
                    "routes": [
                        {"equals": "question", "goto": "answer"},
                        {"equals": "task", "goto": "execute"},
                    ],
                    "default": "chat",
                },
            ),
            "answer": PipelineStep(
                name="answer", instruction="answer", response_tool="send_answer",
                tools=["send_answer"], agent=agent("answerer", ["send_answer"]),
                next={"default": None},
            ),
            "execute": PipelineStep(
                name="execute", instruction="execute", response_tool="execute_task",
                tools=["execute_task"], agent=agent("executor", ["execute_task"]),
                next={"default": None},
            ),
            "chat": PipelineStep(
                name="chat", instruction="chat", response_tool="send_chat",
                tools=["send_chat"], agent=agent("chatter", ["send_chat"]),
                next={"default": None},
            ),
        }
        pipeline = _make_pipeline(name, steps, "classify", instructions_dir, db_path)

        @pipeline.tool("classify_input", "Classify the input")
        async def classify_input(
            category: Annotated[str, "question|task|chat"],
            confidence: Annotated[float, "confidence 0-1"],
        ) -> dict:
            tracker.record("classify_input", {"category": category, "confidence": confidence})
            return {"content": [{"type": "text", "text": json.dumps({"category": category})}]}

        @pipeline.tool("send_answer", "Answer a question")
        async def send_answer(answer: Annotated[str, "the answer"]) -> dict:
            tracker.record("send_answer", {"answer": answer})
            return {"content": [{"type": "text", "text": json.dumps({"answered": True})}]}

        @pipeline.tool("execute_task", "Execute a task")
        async def execute_task(result: Annotated[str, "task result"]) -> dict:
            tracker.record("execute_task", {"result": result})
            return {"content": [{"type": "text", "text": json.dumps({"executed": True})}]}

        @pipeline.tool("send_chat", "Chat casually")
        async def send_chat(message: Annotated[str, "chat message"]) -> dict:
            tracker.record("send_chat", {"message": message})
            return {"content": [{"type": "text", "text": json.dumps({"chatted": True})}]}

        pipeline.initialize_db()
        return pipeline

    def test_question_routes_to_answer(self, db_path, instructions_dir, tracker):
        pipeline = self._build("e2e_route_q", db_path, instructions_dir, tracker)
        state = pipeline.get_run(pipeline.run("What is the capital of France?"))
        assert state.status == "completed"
        assert tracker.last_call("classify_input").args.get("category") == "question"
        assert tracker.was_called("send_answer")
        assert not tracker.was_called("execute_task")
        assert not tracker.was_called("send_chat")

    def test_task_routes_to_execute(self, db_path, instructions_dir, tracker):
        pipeline = self._build("e2e_route_t", db_path, instructions_dir, tracker)
        state = pipeline.get_run(pipeline.run("Please calculate 25 times 4"))
        assert state.status == "completed"
        assert tracker.last_call("classify_input").args.get("category") == "task"
        assert tracker.was_called("execute_task")
        assert not tracker.was_called("send_answer")

    def test_default_routes_to_chat(self, db_path, instructions_dir, tracker):
        pipeline = self._build("e2e_route_c", db_path, instructions_dir, tracker)
        state = pipeline.get_run(pipeline.run("Hey, how's it going today"))
        assert state.status == "completed"
        assert tracker.was_called("send_chat")
        assert not tracker.was_called("execute_task")


# =============================================================================
# Multi-step data passing
# =============================================================================

class TestMultiStepDataPassing:
    def test_step_output_visible_to_next_step(self, db_path, instructions_dir, tracker):
        produce_agent = PipelineAgent(name="producer", tools=["produce_data"], max_turns=2, model=E2E_MODEL)
        consume_agent = PipelineAgent(name="consumer", tools=["consume_data"], max_turns=2, model=E2E_MODEL)
        steps = {
            "produce": PipelineStep(
                name="produce", instruction="produce", response_tool="produce_data",
                tools=["produce_data"], agent=produce_agent, next={"default": "consume"},
            ),
            "consume": PipelineStep(
                name="consume", instruction="consume", response_tool="consume_data",
                tools=["consume_data"], agent=consume_agent, next={"default": None},
            ),
        }
        pipeline = _make_pipeline("e2e_data_passing", steps, "produce", instructions_dir, db_path)

        @pipeline.tool("produce_data", "Produce data")
        async def produce_data(output_value: Annotated[str, "value to output"]) -> dict:
            tracker.record("produce_data", {"output_value": output_value})
            return {"content": [{"type": "text", "text": json.dumps({"produced": "TOKEN_XYZ789"})}]}

        @pipeline.tool("consume_data", "Consume data")
        async def consume_data(received_value: Annotated[str, "value received"]) -> dict:
            tracker.record("consume_data", {"received_value": received_value})
            return {"content": [{"type": "text", "text": json.dumps({"consumed": received_value})}]}

        pipeline.initialize_db()
        state = pipeline.get_run(pipeline.run("Run both steps"))

        assert state.status == "completed"
        assert tracker.was_called("produce_data")
        assert tracker.was_called("consume_data")
        assert "TOKEN_XYZ789" in tracker.last_call("consume_data").args.get("received_value", "")


# =============================================================================
# Pipeline args are visible to steps
# =============================================================================

class TestPipelineArgs:
    def test_args_visible_in_step_context(self, db_path, instructions_dir, tracker):
        agent = PipelineAgent(name="arg_reader", tools=["echo_arg"], max_turns=2, model=E2E_MODEL)
        steps = {
            "use_args": PipelineStep(
                name="use_args", instruction="use_args", response_tool="echo_arg",
                tools=["echo_arg"], agent=agent, next={"default": None},
            )
        }
        pipeline = _make_pipeline("e2e_args", steps, "use_args", instructions_dir, db_path)

        @pipeline.tool("echo_arg", "Echo an arg value")
        async def echo_arg(mode: Annotated[str, "the mode arg"]) -> dict:
            tracker.record("echo_arg", {"mode": mode})
            return {"content": [{"type": "text", "text": json.dumps({"mode": mode})}]}

        pipeline.initialize_db()
        run_id = pipeline.run("Process this", args={"mode": "production", "user_id": "u-42"})
        state = pipeline.get_run(run_id)

        assert state.status == "completed"
        # args stored on the run...
        assert state.args.get("mode") == "production"
        # ...and visible to the agent, which echoed the value back
        assert tracker.last_call("echo_arg").args.get("mode") == "production"


# =============================================================================
# Hooks inject runtime context
# =============================================================================

class TestHooks:
    def test_hook_data_reaches_step(self, db_path, instructions_dir, tracker):
        def ctx_hook():
            return {"user_id": "user-12345", "tier": "premium"}

        (instructions_dir / "contextual.md").write_text(
            "Read [Hook Data]. Call `contextual_response` with user_id set to the user_id "
            "from the hook data, and context_used=true."
        )
        agent = PipelineAgent(name="ctx", tools=["contextual_response"], max_turns=3, model=E2E_MODEL)
        steps = {
            "contextual": PipelineStep(
                name="contextual", instruction="contextual", response_tool="contextual_response",
                tools=["contextual_response"], hooks=[ctx_hook], agent=agent, next={"default": None},
            )
        }
        pipeline = _make_pipeline("e2e_hooks", steps, "contextual", instructions_dir, db_path)

        @pipeline.tool("contextual_response", "Respond using hook context")
        async def contextual_response(
            user_id: Annotated[str, "user id from hook data"],
            context_used: Annotated[bool, "whether hook data was used"],
        ) -> dict:
            tracker.record("contextual_response", {"user_id": user_id, "context_used": context_used})
            return {"content": [{"type": "text", "text": json.dumps({"ok": True})}]}

        pipeline.initialize_db()
        state = pipeline.get_run(pipeline.run("Give me a contextual response"))

        assert state.status == "completed"
        call = tracker.last_call("contextual_response")
        assert call is not None
        assert "12345" in call.args.get("user_id", "")


# =============================================================================
# Tool gating: an agent cannot use a tool it was not granted
# =============================================================================

class TestToolGating:
    def test_unauthorized_tool_is_blocked(self, db_path, instructions_dir, tracker):
        """The step grants only `allowed_tool`; the model is told about a
        `secret_tool` in the instruction but must not be able to use it."""
        (instructions_dir / "gated.md").write_text(
            "First try to call `secret_tool` with payload=\"x\". "
            "If that fails or is unavailable, call `allowed_tool` with note=\"done\" to finish."
        )
        agent = PipelineAgent(name="gated_agent", tools=["allowed_tool"], max_turns=4, model=E2E_MODEL)
        steps = {
            "gated": PipelineStep(
                name="gated", instruction="gated", response_tool="allowed_tool",
                tools=["allowed_tool"], agent=agent, next={"default": None},
            )
        }
        pipeline = _make_pipeline("e2e_gating", steps, "gated", instructions_dir, db_path)

        # secret_tool is registered on the pipeline but NOT granted to the step.
        @pipeline.tool("secret_tool", "A tool the step is not allowed to use")
        async def secret_tool(payload: Annotated[str, "payload"]) -> dict:
            tracker.record("secret_tool", {"payload": payload})
            return {"content": [{"type": "text", "text": json.dumps({"leaked": True})}]}

        @pipeline.tool("allowed_tool", "The only tool this step may use")
        async def allowed_tool(note: Annotated[str, "a note"]) -> dict:
            tracker.record("allowed_tool", {"note": note})
            return {"content": [{"type": "text", "text": json.dumps({"done": True})}]}

        pipeline.initialize_db()
        state = pipeline.get_run(pipeline.run("Do the task"))

        assert state.status == "completed"
        # The hard constraint: secret_tool's body never executed, even if the
        # model attempted it.
        assert not tracker.was_called("secret_tool"), "secret_tool must never execute"
        assert tracker.was_called("allowed_tool")
