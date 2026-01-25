"""Unit tests for tools.py - ToolRegistry class."""

import pytest
from typing import Annotated
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from relais.tools import ToolRegistry, ToolDefinition


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

    @patch('relais.tools.sdk_tool')
    def test_register_simple_tool(self, mock_sdk_tool):
        """Test registering a simple tool."""
        mock_decorated = MagicMock()
        mock_sdk_tool.return_value = lambda f: mock_decorated

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

    @patch('relais.tools.sdk_tool')
    def test_register_tool_without_schema(self, mock_sdk_tool):
        """Test registering a tool without input schema."""
        mock_decorated = MagicMock()
        mock_sdk_tool.return_value = lambda f: mock_decorated

        registry = ToolRegistry("test")

        @registry.tool("simple", "A simple tool")
        async def simple(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "done"}]}

        tool_def = registry.get("simple")
        assert tool_def.input_schema == {}

    @patch('relais.tools.sdk_tool')
    def test_register_multiple_tools(self, mock_sdk_tool):
        """Test registering multiple tools."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("multi")

        @registry.tool("tool1", "First tool")
        async def tool1(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "1"}]}

        @registry.tool("tool2", "Second tool")
        async def tool2(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "2"}]}

        @registry.tool("tool3", "Third tool")
        async def tool3(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "3"}]}

        assert set(registry.list_tools()) == {"tool1", "tool2", "tool3"}

    @patch('relais.tools.sdk_tool')
    def test_register_tool_complex_schema(self, mock_sdk_tool):
        """Test registering a tool with complex input schema."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

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

    @patch('relais.tools.sdk_tool')
    def test_get_existing_tool(self, mock_sdk_tool):
        """Test getting an existing tool."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("test")

        @registry.tool("exists", "Exists")
        async def exists(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "here"}]}

        tool = registry.get("exists")
        assert tool is not None
        assert tool.name == "exists"

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry("test")
        tool = registry.get("nonexistent")
        assert tool is None

    @patch('relais.tools.sdk_tool')
    def test_list_tools_order(self, mock_sdk_tool):
        """Test that list_tools returns all registered tools."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("test")

        @registry.tool("alpha", "Alpha")
        async def alpha(args: dict) -> dict:
            return {"content": []}

        @registry.tool("beta", "Beta")
        async def beta(args: dict) -> dict:
            return {"content": []}

        tools = registry.list_tools()
        assert "alpha" in tools
        assert "beta" in tools
        assert len(tools) == 2


class TestGetAllowedTools:
    """Tests for get_allowed_tools method."""

    @patch('relais.tools.sdk_tool')
    def test_get_allowed_tools_registered(self, mock_sdk_tool):
        """Test getting allowed tools for registered tools."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("my_server")

        @registry.tool("tool_a", "Tool A")
        async def tool_a(args: dict) -> dict:
            return {"content": []}

        @registry.tool("tool_b", "Tool B")
        async def tool_b(args: dict) -> dict:
            return {"content": []}

        allowed = registry.get_allowed_tools(["tool_a", "tool_b"])
        assert allowed == ["mcp__my_server__tool_a", "mcp__my_server__tool_b"]

    @patch('relais.tools.sdk_tool')
    def test_get_allowed_tools_subset(self, mock_sdk_tool):
        """Test getting allowed tools for a subset."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("server")

        @registry.tool("a", "A")
        async def a(args: dict) -> dict:
            return {"content": []}

        @registry.tool("b", "B")
        async def b(args: dict) -> dict:
            return {"content": []}

        @registry.tool("c", "C")
        async def c(args: dict) -> dict:
            return {"content": []}

        # Only request a subset
        allowed = registry.get_allowed_tools(["a", "c"])
        assert allowed == ["mcp__server__a", "mcp__server__c"]

    def test_get_allowed_tools_builtin_passthrough(self):
        """Test that built-in tools pass through unchanged."""
        registry = ToolRegistry("test")

        # Built-in tools like Read, Write, Bash should pass through
        allowed = registry.get_allowed_tools(["Read", "Write", "Bash"])
        assert allowed == ["Read", "Write", "Bash"]

    @patch('relais.tools.sdk_tool')
    def test_get_allowed_tools_mixed(self, mock_sdk_tool):
        """Test mixed registered and built-in tools."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("tools")

        @registry.tool("custom", "Custom tool")
        async def custom(args: dict) -> dict:
            return {"content": []}

        allowed = registry.get_allowed_tools(["custom", "Read", "Bash"])
        assert "mcp__tools__custom" in allowed
        assert "Read" in allowed
        assert "Bash" in allowed

    def test_get_allowed_tools_empty_list(self):
        """Test with empty tool list."""
        registry = ToolRegistry("test")
        allowed = registry.get_allowed_tools([])
        assert allowed == []

    def test_get_allowed_tools_unregistered(self):
        """Test with unregistered tool names."""
        registry = ToolRegistry("test")
        # Unregistered tools pass through (treated as built-ins)
        allowed = registry.get_allowed_tools(["unregistered"])
        assert allowed == ["unregistered"]

    @patch('relais.tools.sdk_tool')
    def test_get_allowed_tools_with_function_reference(self, mock_sdk_tool):
        """Test get_allowed_tools with @tool decorated function references."""
        from relais.tools import tool

        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("server")

        # Register tools via registry
        @registry.tool("tool_a", "Tool A")
        async def tool_a(args: dict) -> dict:
            return {"content": []}

        @registry.tool("tool_b", "Tool B")
        async def tool_b(args: dict) -> dict:
            return {"content": []}

        # Create standalone @tool decorated functions (simulating what users do)
        @tool("tool_a", "Tool A")
        async def standalone_a(args: dict) -> dict:
            return {"content": []}

        @tool("tool_b", "Tool B")
        async def standalone_b(args: dict) -> dict:
            return {"content": []}

        # Pass function references instead of string names
        allowed = registry.get_allowed_tools([standalone_a, standalone_b])
        assert allowed == ["mcp__server__tool_a", "mcp__server__tool_b"]

    @patch('relais.tools.sdk_tool')
    def test_get_allowed_tools_mixed_functions_and_strings(self, mock_sdk_tool):
        """Test get_allowed_tools with mix of functions and string names."""
        from relais.tools import tool

        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("tools")

        @registry.tool("custom", "Custom tool")
        async def custom(args: dict) -> dict:
            return {"content": []}

        # Create standalone @tool decorated function
        @tool("custom", "Custom tool")
        async def custom_func(args: dict) -> dict:
            return {"content": []}

        # Mix function reference with built-in string names
        allowed = registry.get_allowed_tools([custom_func, "Read", "Bash"])
        assert "mcp__tools__custom" in allowed
        assert "Read" in allowed
        assert "Bash" in allowed


class TestCreateMcpServer:
    """Tests for create_mcp_server method."""

    @patch('relais.tools.create_sdk_mcp_server')
    @patch('relais.tools.sdk_tool')
    def test_create_mcp_server_basic(self, mock_sdk_tool, mock_create_server):
        """Test creating MCP server."""
        mock_sdk_tool.return_value = lambda f: MagicMock()
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        registry = ToolRegistry("my_tools")

        @registry.tool("test", "Test tool")
        async def test(args: dict) -> dict:
            return {"content": []}

        server = registry.create_mcp_server()

        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args
        assert call_kwargs[1]["name"] == "my_tools"
        assert call_kwargs[1]["version"] == "1.0.0"
        assert server == mock_server

    @patch('relais.tools.create_sdk_mcp_server')
    @patch('relais.tools.sdk_tool')
    def test_create_mcp_server_custom_version(self, mock_sdk_tool, mock_create_server):
        """Test creating MCP server with custom version."""
        mock_sdk_tool.return_value = lambda f: MagicMock()
        mock_create_server.return_value = MagicMock()

        registry = ToolRegistry("versioned")

        @registry.tool("t", "T")
        async def t(args: dict) -> dict:
            return {"content": []}

        registry.create_mcp_server(version="2.5.0")

        call_kwargs = mock_create_server.call_args
        assert call_kwargs[1]["version"] == "2.5.0"

    @patch('relais.tools.create_sdk_mcp_server')
    def test_create_mcp_server_no_tools(self, mock_create_server):
        """Test creating MCP server with no tools registered."""
        mock_create_server.return_value = MagicMock()

        registry = ToolRegistry("empty")
        registry.create_mcp_server()

        call_kwargs = mock_create_server.call_args
        assert call_kwargs[1]["tools"] == []


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition."""
        mock_sdk_tool = MagicMock()
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"arg": str},
            sdk_tool=mock_sdk_tool
        )
        assert definition.name == "test_tool"
        assert definition.description == "A test tool"
        assert definition.input_schema == {"arg": str}
        assert definition.sdk_tool == mock_sdk_tool

    def test_tool_definition_equality(self):
        """Test ToolDefinition equality."""
        mock_sdk = MagicMock()
        def1 = ToolDefinition("tool", "desc", {"a": int}, mock_sdk)
        def2 = ToolDefinition("tool", "desc", {"a": int}, mock_sdk)
        assert def1 == def2


class TestToolExecution:
    """Tests for actual tool execution (async)."""

    @patch('relais.tools.sdk_tool')
    def test_tool_returns_correct_format(self, mock_sdk_tool):
        """Test that registered tool returns SDK-compatible format."""
        # Make sdk_tool return the function itself for testing
        mock_sdk_tool.return_value = lambda f: f

        registry = ToolRegistry("test")

        @registry.tool("format_test", "Test format")
        async def format_test(args: dict) -> dict:
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

    @patch('relais.tools.sdk_tool')
    def test_tool_receives_args(self, mock_sdk_tool):
        """Test that tool receives args correctly."""
        mock_sdk_tool.return_value = lambda f: f

        registry = ToolRegistry("test")
        received_args = {}

        @registry.tool("capture", "Capture args")
        async def capture(args: dict) -> dict:
            received_args.update(args)
            return {"content": [{"type": "text", "text": "ok"}]}

        asyncio.run(capture({"key": "value", "num": 42}))

        assert received_args == {"key": "value", "num": 42}


class TestStepValidation:
    """Tests for step-based tool access validation."""

    @patch('relais.tools.sdk_tool')
    def test_set_current_step_with_string_tools(self, mock_sdk_tool):
        """Test setting current step with string tool names."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a(args: dict) -> dict:
            return {"content": []}

        @registry.tool("tool_b", "Tool B")
        async def tool_b(args: dict) -> dict:
            return {"content": []}

        registry.set_current_step("step1", ["tool_a"])

        assert registry._current_step_name == "step1"
        assert registry._current_allowed_tools == {"tool_a"}

    @patch('relais.tools.sdk_tool')
    def test_set_current_step_with_function_references(self, mock_sdk_tool):
        """Test setting current step with @tool decorated function references."""
        from relais.tools import tool

        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a(args: dict) -> dict:
            return {"content": []}

        # Create standalone @tool decorated function
        @tool("tool_a", "Tool A")
        async def standalone_a(args: dict) -> dict:
            return {"content": []}

        registry.set_current_step("step1", [standalone_a])

        assert registry._current_step_name == "step1"
        assert registry._current_allowed_tools == {"tool_a"}

    @patch('relais.tools.sdk_tool')
    def test_is_tool_allowed(self, mock_sdk_tool):
        """Test checking if a tool is allowed in current step."""
        mock_sdk_tool.return_value = lambda f: MagicMock()

        registry = ToolRegistry("test")

        @registry.tool("allowed", "Allowed")
        async def allowed(args: dict) -> dict:
            return {"content": []}

        @registry.tool("not_allowed", "Not allowed")
        async def not_allowed(args: dict) -> dict:
            return {"content": []}

        registry.set_current_step("step1", ["allowed"])

        assert registry.is_tool_allowed("allowed") is True
        assert registry.is_tool_allowed("not_allowed") is False

    @patch('relais.tools.sdk_tool')
    def test_tool_validation_blocks_unauthorized_call(self, mock_sdk_tool):
        """Test that tool validation blocks unauthorized tool calls."""
        # Make sdk_tool return the function itself for testing
        mock_sdk_tool.return_value = lambda f: f

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
        assert "only available in specific pipeline steps" in result["content"][0]["text"]
        assert "other_step" in result["content"][0]["text"]

    @patch('relais.tools.sdk_tool')
    def test_tool_validation_allows_authorized_call(self, mock_sdk_tool):
        """Test that tool validation allows authorized tool calls."""
        mock_sdk_tool.return_value = lambda f: f

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

    @patch('relais.tools.sdk_tool')
    def test_tool_validation_with_multiple_allowed_tools(self, mock_sdk_tool):
        """Test tool validation with multiple tools allowed in a step."""
        mock_sdk_tool.return_value = lambda f: f

        registry = ToolRegistry("test")

        @registry.tool("tool_a", "Tool A")
        async def tool_a(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "A"}]}

        @registry.tool("tool_b", "Tool B")
        async def tool_b(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "B"}]}

        @registry.tool("tool_c", "Tool C")
        async def tool_c(args: dict) -> dict:
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
        assert "only available in specific pipeline steps" in result_c["content"][0]["text"]

    @patch('relais.tools.sdk_tool')
    def test_set_current_step_with_empty_tools(self, mock_sdk_tool):
        """Test setting current step with no allowed tools."""
        mock_sdk_tool.return_value = lambda f: f

        registry = ToolRegistry("test")

        @registry.tool("any_tool", "Any tool")
        async def any_tool(args: dict) -> dict:
            return {"content": [{"type": "text", "text": "result"}]}

        # Set step with no allowed tools
        registry.set_current_step("restricted_step", [])

        # No tools should be allowed
        assert registry.is_tool_allowed("any_tool") is False

        # Tool call should be blocked
        result = asyncio.run(any_tool({}))
        assert "only available in specific pipeline steps" in result["content"][0]["text"]
