"""End-to-end tests for pipeline execution.

These tests verify specific behaviors and outcomes, not just completion status.

Run with: pytest tests/e2e/ -m e2e
"""

import pytest
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Annotated, List, Dict, Any

from relais import Pipeline, PipelineStep
from relais.state import SQLiteStateManager

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    name: str
    args: Dict[str, Any]


@dataclass
class ToolTracker:
    """Tracks tool invocations for assertions."""
    calls: List[ToolCall] = field(default_factory=list)

    def record(self, name: str, args: dict):
        self.calls.append(ToolCall(name=name, args=args))

    def get_calls(self, tool_name: str) -> List[ToolCall]:
        return [c for c in self.calls if c.name == tool_name]

    def was_called(self, tool_name: str) -> bool:
        return len(self.get_calls(tool_name)) > 0

    def call_count(self, tool_name: str) -> int:
        return len(self.get_calls(tool_name))

    def last_call(self, tool_name: str) -> ToolCall | None:
        calls = self.get_calls(tool_name)
        return calls[-1] if calls else None

    def clear(self):
        self.calls.clear()


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """SQLite database path for tests."""
    return str(tmp_path_factory.mktemp("data") / "e2e_test.db")


@pytest.fixture(scope="module")
def instructions_dir(tmp_path_factory):
    """Create instruction files with explicit, deterministic instructions."""
    dir_path = tmp_path_factory.mktemp("instructions")

    # Greeting instruction - very specific about what to do
    (dir_path / "greet.md").write_text("""# Greeting Step

You MUST use the `send_greeting` tool to greet the user.

Extract the user's name from the input. If no name is given, use "friend".

Call the tool with:
- message: A greeting that includes "Hello"
- name: The extracted name

You MUST call the tool exactly once and include the word "Hello" in your greeting.
""")

    # Classification instruction - deterministic rules
    (dir_path / "classify.md").write_text("""# Classification Step

Classify the user's input using these EXACT rules:

1. If the input contains a question mark (?), classify as "question"
2. If the input contains words like "calculate", "compute", "add", "multiply", classify as "task"
3. Otherwise, classify as "chat"

You MUST call the `classify_input` tool with:
- category: One of "question", "task", or "chat" based on the rules above
- confidence: Always use 0.95

Call the tool exactly once.
""")

    # Answer instruction
    (dir_path / "answer.md").write_text("""# Answer Step

You received a question. Use the `send_answer` tool to respond.

The answer field MUST contain the word "answer" in it.

Call the tool exactly once.
""")

    # Task execution instruction
    (dir_path / "execute.md").write_text("""# Execute Task Step

You received a task request. Use the `execute_task` tool.

Set the result field to include the word "completed".

Call the tool exactly once.
""")

    # Chat instruction
    (dir_path / "chat.md").write_text("""# Chat Step

Use the `send_chat` tool to respond casually.

The message MUST include the word "hey" (lowercase).

Call the tool exactly once.
""")

    # Process instruction
    (dir_path / "process.md").write_text("""# Process Step

You MUST call one of your available tools to process this step.

Look at your available tools and call the most appropriate one.
If you have a tool that matches your step name (e.g., main_tool for main step), use it.

You MUST call a tool - do not just provide a text response.
""")

    # Research instruction for subagent
    (dir_path / "research.md").write_text("""# Research Step

You are a research subagent. Use the `search` tool to find information.

Search for exactly what was requested in the input.

Call the search tool at least once.
""")

    # Summarize instruction
    (dir_path / "summarize.md").write_text("""# Summarize Step

The previous step provided research findings in the [Previous Step Output] section.

Use the `create_summary` tool to summarize. You MUST:
- Extract findings from the previous output
- Include "summary" in the title
- List at least one key point

Call the tool exactly once.
""")

    # Priority analysis instruction
    (dir_path / "analyze_priority.md").write_text("""# Priority Analysis

Analyze the input and determine priority using these rules:

- If input contains "urgent" or "emergency" -> priority: "high"
- If input contains "when you can" or "no rush" -> priority: "low"
- Otherwise -> priority: "medium"

Call the `set_priority` tool with the determined priority.
""")

    # Hook-aware instruction
    (dir_path / "contextual.md").write_text("""# Contextual Greeting

Check the [Hook Data] section for context information.

Use the `contextual_response` tool and you MUST:
- Include the timestamp from hook data if available
- Include the user_id from hook data if available
- Set context_used to true if you found hook data

Call the tool exactly once.
""")

    return dir_path


@pytest.fixture
def state_manager(db_path):
    """Create state manager."""
    manager = SQLiteStateManager.create(db_path)
    manager.initialize_schema()
    return manager


@pytest.fixture
def tracker():
    """Fresh tool tracker for each test."""
    return ToolTracker()


@pytest.fixture(autouse=True)
def cleanup(state_manager):
    """Clean up pipeline runs after each test."""
    yield
    runs = state_manager.get_pipeline_runs(limit=1000)
    for run in runs:
        if run.pipeline_name.startswith("e2e_"):
            state_manager.delete_pipeline_run(run.id)


class TestGreetingBehavior:
    """Tests that verify specific greeting behavior."""

    @pytest.mark.slow
    def test_greeting_extracts_name_from_input(self, db_path, instructions_dir, tracker):
        """Test that greeting tool receives the correct name from input."""
        steps = {
            "greet": PipelineStep(
                name="greet",
                instruction="greet",
                response_tool="test_tool",
                max_turns=3,
                tools=["send_greeting"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_greeting_name",
            steps=steps,
            start_step="greet",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("send_greeting", "Send greeting")
        async def send_greeting(
            message: Annotated[str, "The greeting message"],
            name: Annotated[str, "The name to greet"],
        ) -> dict:
            tracker.record("send_greeting", {"message": message, "name": name})
            return {"content": [{"type": "text", "text": json.dumps({
                "sent": True,
                "message": message,
                "name": name
            })}]}

        run_id = pipeline.run("Please greet Alice!")
        state = pipeline.get_run(run_id)

        # Verify completion
        assert state.status == "completed", f"Pipeline failed: {state}"

        # Verify tool was called
        assert tracker.was_called("send_greeting"), "send_greeting tool was never called"

        # Verify specific behavior: name should be "Alice"
        call = tracker.last_call("send_greeting")
        assert call.args.get("name") == "Alice", f"Expected name 'Alice', got '{call.args.get('name')}'"

        # Verify greeting contains "Hello"
        message = call.args.get("message", "")
        assert "Hello" in message or "hello" in message.lower(), f"Greeting should contain 'Hello': {message}"

    @pytest.mark.slow
    def test_greeting_uses_default_when_no_name(self, db_path, instructions_dir, tracker):
        """Test that greeting uses 'friend' when no name provided."""
        steps = {
            "greet": PipelineStep(
                name="greet",
                instruction="greet",
                response_tool="test_tool",
                max_turns=3,
                tools=["send_greeting"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_greeting_default",
            steps=steps,
            start_step="greet",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("send_greeting", "Send greeting")
        async def send_greeting(
            message: Annotated[str, "The greeting message"],
            name: Annotated[str, "The name to greet"],
        ) -> dict:
            tracker.record("send_greeting", {"message": message, "name": name})
            return {"content": [{"type": "text", "text": json.dumps({"sent": True})}]}

        run_id = pipeline.run("Hi there!")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"
        assert tracker.was_called("send_greeting")

        call = tracker.last_call("send_greeting")
        # Should use "friend" as default
        assert call.args.get("name") == "friend", f"Expected default name 'friend', got '{call.args.get('name')}'"


class TestClassificationRouting:
    """Tests that verify classification routes to correct steps."""

    @pytest.mark.slow
    def test_question_routes_to_answer_step(self, db_path, instructions_dir, tracker):
        """Test that questions (with ?) route to answer step."""
        steps = {
            "classify": PipelineStep(
                name="classify",
                instruction="classify",
                response_tool="test_tool",
                max_turns=2,
                tools=["classify_input"],
                next={
                    "field": "category",
                    "routes": [
                        {"equals": "question", "goto": "answer"},
                        {"equals": "task", "goto": "execute"},
                    ],
                    "default": "chat"
                }
            ),
            "answer": PipelineStep(
                name="answer",
                instruction="answer",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_answer"],
                next={"default": None}
            ),
            "execute": PipelineStep(
                name="execute",
                instruction="execute",
                response_tool="test_tool",
                max_turns=2,
                tools=["execute_task"],
                next={"default": None}
            ),
            "chat": PipelineStep(
                name="chat",
                instruction="chat",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_chat"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_routing_question",
            steps=steps,
            start_step="classify",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("classify_input", "Classify input")
        async def classify_input(
            category: Annotated[str, "The classification category"],
            confidence: Annotated[float, "Confidence score"],
        ) -> dict:
            tracker.record("classify_input", {"category": category, "confidence": confidence})
            return {"content": [{"type": "text", "text": json.dumps({
                "category": category,
                "confidence": confidence
            })}]}

        @pipeline.tool("send_answer", "Send answer")
        async def send_answer(answer: Annotated[str, "The answer"]) -> dict:
            tracker.record("send_answer", {"answer": answer})
            return {"content": [{"type": "text", "text": json.dumps({"answered": True})}]}

        @pipeline.tool("execute_task", "Execute task")
        async def execute_task(result: Annotated[str, "The task result"]) -> dict:
            tracker.record("execute_task", {"result": result})
            return {"content": [{"type": "text", "text": json.dumps({"executed": True})}]}

        @pipeline.tool("send_chat", "Send chat")
        async def send_chat(message: Annotated[str, "The chat message"]) -> dict:
            tracker.record("send_chat", {"message": message})
            return {"content": [{"type": "text", "text": json.dumps({"chatted": True})}]}

        # Input with question mark should route to "answer"
        run_id = pipeline.run("What is the capital of France?")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify classification was "question"
        classify_call = tracker.last_call("classify_input")
        assert classify_call is not None, "classify_input was not called"
        assert classify_call.args.get("category") == "question", \
            f"Expected category 'question', got '{classify_call.args.get('category')}'"

        # Verify we went to answer step, not execute or chat
        assert tracker.was_called("send_answer"), "Should have routed to answer step"
        assert not tracker.was_called("execute_task"), "Should NOT have routed to execute step"
        assert not tracker.was_called("send_chat"), "Should NOT have routed to chat step"

        # Verify answer step results
        assert "answer" in state.step_results, "answer step should be in results"
        assert "classify" in state.step_results, "classify step should be in results"

    @pytest.mark.slow
    def test_task_routes_to_execute_step(self, db_path, instructions_dir, tracker):
        """Test that task requests route to execute step."""
        steps = {
            "classify": PipelineStep(
                name="classify",
                instruction="classify",
                response_tool="test_tool",
                max_turns=2,
                tools=["classify_input"],
                next={
                    "field": "category",
                    "routes": [
                        {"equals": "question", "goto": "answer"},
                        {"equals": "task", "goto": "execute"},
                    ],
                    "default": "chat"
                }
            ),
            "answer": PipelineStep(
                name="answer",
                instruction="answer",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_answer"],
                next={"default": None}
            ),
            "execute": PipelineStep(
                name="execute",
                instruction="execute",
                response_tool="test_tool",
                max_turns=2,
                tools=["execute_task"],
                next={"default": None}
            ),
            "chat": PipelineStep(
                name="chat",
                instruction="chat",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_chat"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_routing_task",
            steps=steps,
            start_step="classify",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("classify_input", "Classify")
        async def classify_input(
            category: Annotated[str, "The classification category"],
            confidence: Annotated[float, "Confidence score"],
        ) -> dict:
            tracker.record("classify_input", {"category": category, "confidence": confidence})
            return {"content": [{"type": "text", "text": json.dumps({
                "category": category,
                "confidence": confidence
            })}]}

        @pipeline.tool("send_answer", "Answer")
        async def send_answer(answer: Annotated[str, "The answer"]) -> dict:
            tracker.record("send_answer", {"answer": answer})
            return {"content": [{"type": "text", "text": json.dumps({"answered": True})}]}

        @pipeline.tool("execute_task", "Execute")
        async def execute_task(result: Annotated[str, "The task result"]) -> dict:
            tracker.record("execute_task", {"result": result})
            return {"content": [{"type": "text", "text": json.dumps({"executed": True})}]}

        @pipeline.tool("send_chat", "Chat")
        async def send_chat(message: Annotated[str, "The chat message"]) -> dict:
            tracker.record("send_chat", {"message": message})
            return {"content": [{"type": "text", "text": json.dumps({"chatted": True})}]}

        # Input with "calculate" should route to execute
        run_id = pipeline.run("Please calculate 25 times 4")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify classification was "task"
        classify_call = tracker.last_call("classify_input")
        assert classify_call.args.get("category") == "task", \
            f"Expected category 'task', got '{classify_call.args.get('category')}'"

        # Verify correct routing
        assert tracker.was_called("execute_task"), "Should have routed to execute step"
        assert not tracker.was_called("send_answer"), "Should NOT have routed to answer step"
        assert not tracker.was_called("send_chat"), "Should NOT have routed to chat step"

    @pytest.mark.slow
    def test_chat_is_default_route(self, db_path, instructions_dir, tracker):
        """Test that generic input routes to chat (default)."""
        steps = {
            "classify": PipelineStep(
                name="classify",
                instruction="classify",
                response_tool="test_tool",
                max_turns=2,
                tools=["classify_input"],
                next={
                    "field": "category",
                    "routes": [
                        {"equals": "question", "goto": "answer"},
                        {"equals": "task", "goto": "execute"},
                    ],
                    "default": "chat"
                }
            ),
            "answer": PipelineStep(
                name="answer",
                instruction="answer",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_answer"],
                next={"default": None}
            ),
            "execute": PipelineStep(
                name="execute",
                instruction="execute",
                response_tool="test_tool",
                max_turns=2,
                tools=["execute_task"],
                next={"default": None}
            ),
            "chat": PipelineStep(
                name="chat",
                instruction="chat",
                response_tool="test_tool",
                max_turns=2,
                tools=["send_chat"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_routing_chat",
            steps=steps,
            start_step="classify",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("classify_input", "Classify")
        async def classify_input(
            category: Annotated[str, "The classification category"],
            confidence: Annotated[float, "Confidence score"],
        ) -> dict:
            tracker.record("classify_input", {"category": category, "confidence": confidence})
            return {"content": [{"type": "text", "text": json.dumps({
                "category": category,
                "confidence": confidence
            })}]}

        @pipeline.tool("send_answer", "Answer")
        async def send_answer(answer: Annotated[str, "The answer"]) -> dict:
            tracker.record("send_answer", {"answer": answer})
            return {"content": [{"type": "text", "text": json.dumps({"answered": True})}]}

        @pipeline.tool("execute_task", "Execute")
        async def execute_task(result: Annotated[str, "The task result"]) -> dict:
            tracker.record("execute_task", {"result": result})
            return {"content": [{"type": "text", "text": json.dumps({"executed": True})}]}

        @pipeline.tool("send_chat", "Chat")
        async def send_chat(message: Annotated[str, "The chat message"]) -> dict:
            tracker.record("send_chat", {"message": message})
            return {"content": [{"type": "text", "text": json.dumps({"chatted": True})}]}

        # Generic greeting should route to chat
        run_id = pipeline.run("Hey, how's it going today")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify classification was "chat"
        classify_call = tracker.last_call("classify_input")
        assert classify_call.args.get("category") == "chat", \
            f"Expected category 'chat', got '{classify_call.args.get('category')}'"

        # Verify correct routing
        assert tracker.was_called("send_chat"), "Should have routed to chat step"
        assert not tracker.was_called("send_answer"), "Should NOT have routed to answer step"
        assert not tracker.was_called("execute_task"), "Should NOT have routed to execute step"


class TestSubagentDataFlow:
    """Tests that verify data flows correctly through subagents."""

    @pytest.mark.slow
    def test_subagent_results_passed_to_next_step(self, db_path, instructions_dir, tracker):
        """Test that subagent results are available to the following step."""
        steps = {
            "research": PipelineStep(
                name="research",
                instruction="research",
                response_tool="test_tool",
                max_turns=3,
                tools=["search"],
                subagent=True,
                next={"default": "summarize"}
            ),
            "summarize": PipelineStep(
                name="summarize",
                instruction="summarize",
                response_tool="test_tool",
                max_turns=3,
                tools=["create_summary"],
                subagent=False,
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_subagent_flow",
            steps=steps,
            start_step="research",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        # Search returns specific findings that summarize should reference
        @pipeline.tool("search", "Search for info")
        async def search(query: Annotated[str, "The search query"]) -> dict:
            tracker.record("search", {"query": query})
            # Return specific, identifiable findings
            findings = [
                "Python was created by Guido van Rossum",
                "Python was first released in 1991",
                "Python emphasizes code readability"
            ]
            return {"content": [{"type": "text", "text": json.dumps({
                "query": query,
                "findings": findings,
                "count": len(findings)
            })}]}

        @pipeline.tool("create_summary", "Create summary")
        async def create_summary(
            title: Annotated[str, "The summary title"],
            key_points: Annotated[list, "List of key points"],
        ) -> dict:
            tracker.record("create_summary", {"title": title, "key_points": key_points})
            return {"content": [{"type": "text", "text": json.dumps({
                "title": title,
                "key_points": key_points
            })}]}

        run_id = pipeline.run("Research the history of Python programming language")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify search was called (in subagent)
        assert tracker.was_called("search"), "search tool should have been called"
        search_call = tracker.last_call("search")
        assert "python" in search_call.args.get("query", "").lower(), \
            f"Search query should mention Python: {search_call.args.get('query')}"

        # Verify summary was created
        assert tracker.was_called("create_summary"), "create_summary should have been called"
        summary_call = tracker.last_call("create_summary")

        # Summary title should reference the research
        title = summary_call.args.get("title", "").lower()
        assert "summary" in title or "python" in title, \
            f"Summary title should be relevant: {summary_call.args.get('title')}"

        # Key points should have been extracted
        key_points = summary_call.args.get("key_points", [])
        assert len(key_points) >= 1, "Summary should have at least one key point"

        # Verify both steps completed
        assert "research" in state.step_results
        assert "summarize" in state.step_results


class TestHookDataInjection:
    """Tests that verify hook data is properly injected into context."""

    @pytest.mark.slow
    def test_hook_data_available_to_step(self, db_path, instructions_dir, tracker):
        """Test that hook data is accessible within the step."""
        test_timestamp = "2024-01-15T10:30:00Z"
        test_user_id = "user-12345"

        def time_hook():
            return {"timestamp": test_timestamp, "timezone": "UTC"}

        def user_hook():
            return {"user_id": test_user_id, "role": "tester"}

        steps = {
            "contextual": PipelineStep(
                name="contextual",
                instruction="contextual",
                response_tool="test_tool",
                max_turns=3,
                tools=["contextual_response"],
                hooks=[time_hook, user_hook],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_hooks_injection",
            steps=steps,
            start_step="contextual",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("contextual_response", "Respond with context")
        async def contextual_response(
            timestamp_used: Annotated[str, "The timestamp from hook data"],
            user_id_used: Annotated[str, "The user ID from hook data"],
            context_used: Annotated[bool, "Whether context was used"],
        ) -> dict:
            tracker.record("contextual_response", {
                "timestamp_used": timestamp_used,
                "user_id_used": user_id_used,
                "context_used": context_used
            })
            return {"content": [{"type": "text", "text": json.dumps({
                "timestamp": timestamp_used,
                "user": user_id_used,
                "used_context": context_used
            })}]}

        run_id = pipeline.run("Give me a contextual greeting")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"
        assert tracker.was_called("contextual_response")

        call = tracker.last_call("contextual_response")

        # Verify hook data was used
        assert call.args.get("context_used") is True, "Tool should indicate context was used"

        # Verify specific hook values were passed through
        timestamp_used = call.args.get("timestamp_used", "")
        assert test_timestamp in timestamp_used or "2024" in timestamp_used, \
            f"Timestamp from hook should be used: {timestamp_used}"

        user_id_used = call.args.get("user_id_used", "")
        assert test_user_id in user_id_used or "12345" in user_id_used, \
            f"User ID from hook should be used: {user_id_used}"


class TestMultiStepDataPassing:
    """Tests that verify data passes correctly between steps."""

    @pytest.mark.slow
    def test_step_output_available_to_next_step(self, db_path, instructions_dir, tracker):
        """Test that a step's output is available in the next step's context."""
        steps = {
            "step1": PipelineStep(
                name="step1",
                instruction="process",
                response_tool="test_tool",
                max_turns=2,
                tools=["produce_data"],
                next={"default": "step2"}
            ),
            "step2": PipelineStep(
                name="step2",
                instruction="process",
                response_tool="test_tool",
                max_turns=2,
                tools=["consume_data"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_data_passing",
            steps=steps,
            start_step="step1",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        # Step 1 produces a unique identifier
        unique_value = "UNIQUE_TOKEN_XYZ789"

        @pipeline.tool("produce_data", "Produce data")
        async def produce_data(output_value: Annotated[str, "The output value"]) -> dict:
            tracker.record("produce_data", {"output_value": output_value})
            return {"content": [{"type": "text", "text": json.dumps({
                "produced": unique_value,
                "step": "step1"
            })}]}

        @pipeline.tool("consume_data", "Consume data")
        async def consume_data(received_value: Annotated[str, "The received value"]) -> dict:
            tracker.record("consume_data", {"received_value": received_value})
            return {"content": [{"type": "text", "text": json.dumps({
                "consumed": received_value,
                "step": "step2"
            })}]}

        # Instructions tell step2 to look for and use the value from step1
        (instructions_dir / "process.md").write_text(f"""# Process Step

If this is step1: Call produce_data with output_value set to "{unique_value}"

If this is step2 (you can see [Previous Step Output]):
Call consume_data with received_value set to the "produced" value from the previous output.
You MUST pass the exact value "{unique_value}" if you see it in the previous output.
""")

        run_id = pipeline.run("Process this data through both steps")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify step1 produced the value
        assert tracker.was_called("produce_data")

        # Verify step2 consumed and referenced the value
        assert tracker.was_called("consume_data")
        consume_call = tracker.last_call("consume_data")

        # The consumed value should match or reference what was produced
        received = consume_call.args.get("received_value", "")
        assert unique_value in received or "XYZ789" in received, \
            f"Step2 should have received the value from Step1. Got: {received}"

    @pytest.mark.slow
    def test_subsequent_steps_can_access_original_input(self, db_path, instructions_dir, tracker):
        """Test that subsequent steps can see the original [User Input].

        This is critical for pipelines where step 2 needs to reference
        the original user request, not just the routing data from step 1.
        """
        # The original input contains a unique identifier
        unique_input = "ORIGINAL_REQUEST_ABC123"

        # Create instruction files
        (instructions_dir / "step1_pass.md").write_text("""# Step 1
Call the `step1_tool` tool with acknowledged=true.
""")
        (instructions_dir / "step2_echo.md").write_text(f"""# Step 2

Look at the [User Input] section. It should contain "{unique_input}".

You MUST call the `echo_input` tool with:
- original_input: The EXACT text from the [User Input] section

This tests that subsequent steps can see the original user request.
""")

        steps = {
            "step1": PipelineStep(
                name="step1",
                instruction="step1_pass",
                response_tool="test_tool",
                max_turns=2,
                tools=["step1_tool"],
                next={"default": "step2"}
            ),
            "step2": PipelineStep(
                name="step2",
                instruction="step2_echo",
                response_tool="test_tool",
                max_turns=2,
                tools=["echo_input"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_original_input",
            steps=steps,
            start_step="step1",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("step1_tool", "Step 1 tool")
        async def step1_tool(acknowledged: Annotated[bool, "Acknowledgement flag"]) -> dict:
            tracker.record("step1_tool", {"acknowledged": acknowledged})
            return {"content": [{"type": "text", "text": json.dumps({"step": 1})}]}

        @pipeline.tool("echo_input", "Echo the original input")
        async def echo_input(original_input: Annotated[str, "The original input"]) -> dict:
            tracker.record("echo_input", {"original_input": original_input})
            return {"content": [{"type": "text", "text": json.dumps({"echoed": True})}]}

        run_id = pipeline.run(unique_input)
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify step 1 completed
        assert tracker.was_called("step1_tool")

        # Verify step 2 was able to see and echo the original input
        assert tracker.was_called("echo_input"), "echo_input should have been called"
        echo_call = tracker.last_call("echo_input")

        original = echo_call.args.get("original_input", "")
        assert unique_input in original, \
            f"Step 2 should see original input '{unique_input}'. Got: '{original}'"


class TestModelConfiguration:
    """Tests that verify model configuration is respected."""

    @pytest.mark.slow
    def test_subagent_uses_configured_model(self, db_path, instructions_dir, tracker):
        """Test that subagent model configuration is applied."""
        steps = {
            "main": PipelineStep(
                name="main",
                instruction="process",
                response_tool="test_tool",
                max_turns=2,
                tools=["main_tool"],
                subagent=False,
                next={"default": "sub"}
            ),
            "sub": PipelineStep(
                name="sub",
                instruction="process",
                response_tool="test_tool",
                max_turns=2,
                tools=["sub_tool"],
                subagent=True,
                subagent_model="haiku",  # Explicit model for subagent
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_model_config",
            steps=steps,
            start_step="main",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("main_tool", "Main tool")
        async def main_tool(data: Annotated[str, "The data"]) -> dict:
            tracker.record("main_tool", {"data": data})
            return {"content": [{"type": "text", "text": json.dumps({"main": True})}]}

        @pipeline.tool("sub_tool", "Sub tool")
        async def sub_tool(data: Annotated[str, "The data"]) -> dict:
            tracker.record("sub_tool", {"data": data})
            return {"content": [{"type": "text", "text": json.dumps({"sub": True})}]}

        run_id = pipeline.run("Test model configuration")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Both tools should be called
        assert tracker.was_called("main_tool"), "Main step tool should be called"
        assert tracker.was_called("sub_tool"), "Subagent step tool should be called"

        # Both steps should complete
        assert "main" in state.step_results
        assert "sub" in state.step_results


class TestPipelineArgs:
    """Tests that verify pipeline arguments are accessible."""

    @pytest.mark.slow
    def test_args_stored_and_retrievable(self, db_path, instructions_dir, tracker):
        """Test that pipeline args are stored and can be retrieved."""
        steps = {
            "process": PipelineStep(
                name="process",
                instruction="process",
                response_tool="test_tool",
                max_turns=2,
                tools=["use_args"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_args_test",
            steps=steps,
            start_step="process",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("use_args", "Use args")
        async def use_args(acknowledged: Annotated[bool, "Acknowledgement flag"]) -> dict:
            tracker.record("use_args", {"acknowledged": acknowledged})
            return {"content": [{"type": "text", "text": json.dumps({"done": True})}]}

        # Pass specific args
        test_args = {
            "user_id": "test-user-999",
            "mode": "testing",
            "config": {"feature_a": True, "limit": 100}
        }

        run_id = pipeline.run("Process with args", args=test_args)
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Verify args were stored correctly
        assert state.args is not None, "Args should be stored"
        assert state.args.get("user_id") == "test-user-999"
        assert state.args.get("mode") == "testing"
        assert state.args.get("config", {}).get("feature_a") is True
        assert state.args.get("config", {}).get("limit") == 100


class TestToolCallCounts:
    """Tests that verify tools are called the expected number of times."""

    @pytest.mark.slow
    def test_tool_called_exactly_once_per_step(self, db_path, instructions_dir, tracker):
        """Test that each step calls its tool exactly once."""
        # Create step-specific instruction files
        (instructions_dir / "step1_tool.md").write_text("""# Step 1

You MUST call the `tool1` tool exactly once.
- message: A greeting message
- name: Extract from user input, or use "friend"
""")
        (instructions_dir / "step2_tool.md").write_text("""# Step 2

You MUST call the `tool2` tool exactly once.
- message: A response message
- name: Use the name from previous step or "friend"

IMPORTANT: Do NOT call tool1. Only call tool2.
""")

        steps = {
            "step1": PipelineStep(
                name="step1",
                instruction="step1_tool",
                response_tool="test_tool",
                max_turns=2,
                tools=["tool1"],
                next={"default": "step2"}
            ),
            "step2": PipelineStep(
                name="step2",
                instruction="step2_tool",
                response_tool="test_tool",
                max_turns=2,
                tools=["tool2"],
                next={"default": None}
            )
        }

        pipeline = Pipeline.create(
            name="e2e_call_count",
            steps=steps,
            start_step="step1",
            instructions_dir=instructions_dir,
            db_config=db_path,
            model="haiku"
        )

        @pipeline.tool("tool1", "Tool 1 for step 1")
        async def tool1(
            message: Annotated[str, "A greeting message"],
            name: Annotated[str, "The name"],
        ) -> dict:
            tracker.record("tool1", {"message": message, "name": name})
            return {"content": [{"type": "text", "text": json.dumps({"step": 1})}]}

        @pipeline.tool("tool2", "Tool 2 for step 2")
        async def tool2(
            message: Annotated[str, "A response message"],
            name: Annotated[str, "The name"],
        ) -> dict:
            tracker.record("tool2", {"message": message, "name": name})
            return {"content": [{"type": "text", "text": json.dumps({"step": 2})}]}

        run_id = pipeline.run("Hello Bob!")
        state = pipeline.get_run(run_id)

        assert state.status == "completed"

        # Each tool should be called exactly once
        assert tracker.call_count("tool1") == 1, f"tool1 should be called once, was called {tracker.call_count('tool1')} times"
        assert tracker.call_count("tool2") == 1, f"tool2 should be called once, was called {tracker.call_count('tool2')} times"
