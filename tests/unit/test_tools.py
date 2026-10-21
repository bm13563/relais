"""Unit tests for tools.py - ToolRegistry class."""

import pytest
from typing import Annotated
from unittest.mock import AsyncMock
import asyncio

from relais.tools import ToolRegistry, ToolDefinition, tool, tool_names


class TestToolRegistryCreation:
    """Tests for ToolRegistry instantiation."""

    def test_create_registry_with_default_name(self):
        """Test creating registry with default name."""
        registry = ToolRegistry()
        assert registry.name == "pipeline_tools"

    def test_create_registry_with_custom_name(self):
        """Test creating registry with custom name."""
        registry = ToolRegistry("my_custom_tools")
        assert registry.name == "my_custom_tools"

    def test_new_registry_is_empty(self):
        """Test that new registry has no tools."""
        registry = ToolRegistry("empty")
        assert registry.list_tools() == []


class TestToolRegistration:
    """Tests for tool registration via decorator."""

    def test_register_simple_tool(self):
        """Test registering a simple tool."""
        registry = ToolRegistry("test")

        @registry.tool("greet", "Greet the user")
        async def greet(name: Annotated[str, "The name to greet"]) -> dict:
            return {"content": [{"type": "text", "text": f"Hello {name}"}]}

        # Verify tool was registered
        assert "greet" in registry.list_tools()
        tool_def = registry.get("greet")
        assert tool_def is not None
        assert tool_def.name == "greet"
        assert tool_def.description == "Greet the user"
        assert tool_def.input_schema == {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The name to greet"}},
            "required": ["name"],
        }

    def test_register_tool_without_schema(self):
        """Test registering a tool without input schema."""
        registry = ToolRegistry("test")

        @registry.tool("simple", "A simple tool")
        async def simple() -> dict:
            return {"content": [{"type": "text", "text": "done"}]}

        tool_def = registry.get("simple")
        assert tool_def.input_schema == {}

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry("multi")

        @registry.tool("tool1", "First tool")
        async def tool1() -> dict:
            return {"content": [{"type": "text", "text": "1"}]}

        @registry.tool("tool2", "Second tool")
        async def tool2() -> dict:
            return {"content": [{"type": "text", "text": "2"}]}

        @registry.tool("tool3", "Third tool")
        async def tool3() -> dict:
            return {"content": [{"type": "text", "text": "3"}]}

        assert set(registry.list_tools()) == {"tool1", "tool2", "tool3"}

    def test_register_tool_complex_schema(self):
        """Test registering a tool with complex input schema."""
        registry = ToolRegistry("complex")

        @registry.tool("search", "Search with filters")
        async def search(
            query: Annotated[str, "Search query"],
            limit: Annotated[int, "Max results"],
            include_archived: Annotated[bool, "Include archived items"] = False,
        ) -> dict:
            return {"content": [{"type": "text", "text": "results"}]}

        tool_def = registry.get("search")
        assert tool_def.input_schema == {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
                "include_archived": {"type": "boolean", "description": "Include archived items"},
            },
            "required": ["query", "limit"],
        }


class TestToolRetrieval:
    """Tests for retrieving tools from registry."""

    def test_get_existing_tool(self):
        """Test getting an existing tool."""
        registry = ToolRegistry("test")

        @registry.tool("exists", "Exists")
        async def exists() -> dict:
            return {"content": [{"type": "text", "text": "here"}]}

        tool = registry.get("exists")
        assert tool is not None
        assert tool.name == "exists"

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry("test")
        tool = registry.get("nonexistent")
        assert tool is None

    def test_list_tools_order(self):
        """Test that list_tools returns all registered tools."""
        registry = ToolRegistry("test")

        @registry.tool("alpha", "Alpha")
        async def alpha() -> dict:
            return {"content": []}

        @registry.tool("beta", "Beta")
        async def beta() -> dict:
            return {"content": []}

        tools = registry.list_tools()
        assert "alpha" in tools
        assert "beta" in tools
        assert len(tools) == 2


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition."""
        fn = AsyncMock()
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            fn=fn,
        )
        assert definition.name == "test_tool"
        assert definition.description == "A test tool"
        assert definition.input_schema == {"type": "object", "properties": {}}
        assert definition.fn is fn

    def test_tool_definition_equality(self):
        """Test ToolDefinition equality."""
        fn = AsyncMock()
        def1 = ToolDefinition("tool", "desc", {"a": int}, fn)
        def2 = ToolDefinition("tool", "desc", {"a": int}, fn)
        assert def1 == def2

    def test_registered_fn_is_what_the_decorator_returns(self):
        """The decorated name is bound to the definition's fn: the wrapped tool."""
        registry = ToolRegistry("test")

        @registry.tool("t", "T")
        async def t() -> dict:
            return {"content": []}

        assert registry.get("t").fn is t

    def test_register_tool_function_wraps_it(self):

        @tool("standalone", "Standalone")
        async def standalone(x: Annotated[int, "x"]) -> dict:
            return {"content": [{"type": "text", "text": str(x * 2)}]}

        registry = ToolRegistry("test")
        assert registry.register_tool_function(standalone) == "standalone"
        result = asyncio.run(registry.get("standalone").fn({"x": 21}))
        assert result == {"content": [{"type": "text", "text": "42"}]}
        assert registry.get_tool_result("standalone") == ("standalone", result)


class TestToolExecution:
    """Tests for actual tool execution (async)."""

    def test_tool_returns_correct_format(self):
        """Test that a registered tool returns the MCP content format."""
        registry = ToolRegistry("test")

        @registry.tool("format_test", "Test format")
        async def format_test() -> dict:
            return {
                "content": [
                    {"type": "text", "text": "result"}
                ]
            }

        # Run the async function
        result = asyncio.run(format_test({}))

        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"

    def test_tool_receives_args_as_kwargs(self):
        """Test that the args dict is unpacked into the tool's typed parameters."""
        registry = ToolRegistry("test")
        received = {}

        @registry.tool("capture", "Capture args")
        async def capture(
            key: Annotated[str, "a string"],
            num: Annotated[int, "a number"],
        ) -> dict:
            received["key"] = key
            received["num"] = num
            return {"content": [{"type": "text", "text": "ok"}]}

        asyncio.run(capture({"key": "value", "num": 42}))

        assert received == {"key": "value", "num": 42}


class TestStepValidation:
    """Tests for step-based tool access validation."""

    def test_set_current_step_with_string_tools(self):
        """Test setting current step with string tool names."""
        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a() -> dict:
            return {"content": []}

        @registry.tool("tool_b", "Tool B")
        async def tool_b() -> dict:
            return {"content": []}

        registry.set_current_step("step1", ["tool_a"])

        assert registry._current_step_name == "step1"
        assert registry._current_allowed_tools == {"tool_a"}

    def test_set_current_step_with_function_references(self):
        """Test setting current step with @tool decorated function references."""


        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a() -> dict:
            return {"content": []}

        # Create standalone @tool decorated function
        @tool("tool_a", "Tool A")
        async def standalone_a() -> dict:
            return {"content": []}

        registry.set_current_step("step1", [standalone_a])

        assert registry._current_step_name == "step1"
        assert registry._current_allowed_tools == {"tool_a"}

    def test_is_tool_allowed(self):
        """Test checking if a tool is allowed in current step."""
        registry = ToolRegistry("test")

        @registry.tool("allowed", "Allowed")
        async def allowed() -> dict:
            return {"content": []}

        @registry.tool("not_allowed", "Not allowed")
        async def not_allowed() -> dict:
            return {"content": []}

        registry.set_current_step("step1", ["allowed"])

        assert registry.is_tool_allowed("allowed") is True
        assert registry.is_tool_allowed("not_allowed") is False

    def test_tool_validation_blocks_unauthorized_call(self):
        """Test that tool validation blocks unauthorized tool calls."""
        registry = ToolRegistry("test")

        execution_count = {"count": 0}

        @registry.tool("protected", "Protected tool")
        async def protected(name: Annotated[str, "Name"]) -> dict:
            execution_count["count"] += 1
            return {"content": [{"type": "text", "text": f"Hello {name}"}]}

        # Set current step without this tool
        registry.set_current_step("other_step", [])

        # Try to call the tool - should be blocked
        result = asyncio.run(protected({"name": "Alice"}))

        # Verify tool was NOT executed
        assert execution_count["count"] == 0

        # Verify error message format
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "is not available in step" in result["content"][0]["text"]
        assert "other_step" in result["content"][0]["text"]

    def test_tool_validation_allows_authorized_call(self):
        """Test that tool validation allows authorized tool calls."""
        registry = ToolRegistry("test")

        execution_count = {"count": 0}

        @registry.tool("allowed", "Allowed tool")
        async def allowed(name: Annotated[str, "Name"]) -> dict:
            execution_count["count"] += 1
            return {"content": [{"type": "text", "text": f"Hello {name}"}]}

        # Set current step WITH this tool
        registry.set_current_step("correct_step", ["allowed"])

        # Call the tool - should succeed
        result = asyncio.run(allowed({"name": "Bob"}))

        # Verify tool WAS executed
        assert execution_count["count"] == 1

        # Verify normal result
        assert result["content"][0]["text"] == "Hello Bob"

    def test_tool_validation_with_multiple_allowed_tools(self):
        """Test tool validation with multiple tools allowed in a step."""
        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a() -> dict:
            return {"content": [{"type": "text", "text": "A"}]}

        @registry.tool("tool_b", "Tool B")
        async def tool_b() -> dict:
            return {"content": [{"type": "text", "text": "B"}]}

        @registry.tool("tool_c", "Tool C")
        async def tool_c() -> dict:
            return {"content": [{"type": "text", "text": "C"}]}

        # Allow only tool_a and tool_b
        registry.set_current_step("step1", ["tool_a", "tool_b"])

        # tool_a and tool_b should work
        result_a = asyncio.run(tool_a({}))
        assert result_a["content"][0]["text"] == "A"

        result_b = asyncio.run(tool_b({}))
        assert result_b["content"][0]["text"] == "B"

        # tool_c should be blocked
        result_c = asyncio.run(tool_c({}))
        assert "is not available in step" in result_c["content"][0]["text"]
        assert "tool_c" in result_c["content"][0]["text"]

    def test_set_current_step_with_empty_tools(self):
        """Test setting current step with no allowed tools."""
        registry = ToolRegistry("test")

        @registry.tool("any_tool", "Any tool")
        async def any_tool() -> dict:
            return {"content": [{"type": "text", "text": "result"}]}

        # Set step with no allowed tools
        registry.set_current_step("restricted_step", [])

        # No tools should be allowed
        assert registry.is_tool_allowed("any_tool") is False

        # Tool call should be blocked
        result = asyncio.run(any_tool({}))
        assert "is not available in step" in result["content"][0]["text"]
        assert "restricted_step" in result["content"][0]["text"]


class TestToolNames:
    def test_names_and_tool_functions_normalize_to_names(self):

        @tool("decorated", "D")
        async def decorated() -> dict:
            return {"content": []}

        assert tool_names(["plain", decorated, "Read"]) == ["plain", "decorated", "Read"]
        assert tool_names(None) == []


class TestRefusal:
    def test_refusal_names_the_tool_the_step_and_what_it_has(self):
        registry = ToolRegistry("test")
        registry.set_current_step("intro", ["b_tool", "a_tool"])
        assert registry._refusal("secret") == (
            "Tool 'secret' is not available in step 'intro'. Available tools: a_tool, b_tool"
        )
