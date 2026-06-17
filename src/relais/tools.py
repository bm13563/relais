"""Tool registry wrapping Claude Agent SDK's @tool decorator."""

from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

from claude_agent_sdk import tool as sdk_tool, create_sdk_mcp_server

from .logging_config import get_logger

log = get_logger('tools')


@dataclass
class ToolResponse:
    """Standardized response format for pipeline tools.

    All tools should return a ToolResponse to ensure consistent MCP formatting
    and proper routing data extraction.

    Attributes:
        data: The response data (dict, list, or any JSON-serializable value).
              This is what gets passed to the next step and used for routing.
        message: Optional human-readable message for the model to see.
        limit: Optional character limit for the response. If set, truncates the
               JSON output to this many characters with a truncation notice.
        images: Optional list of images to include in the response. Each image
                is a tuple of (base64_data, mime_type). Images are sent to the
                model but not preserved in routing data between steps.

    Example:
        @tool("run_query", "Execute a SQL query")
        async def run_query(sql: str) -> dict:
            try:
                result = db.execute(sql)
                return ToolResponse(
                    data={"success": True, "rows": result},
                    limit=5000  # Limit response to 5k chars
                ).to_mcp()
            except Exception as e:
                return ToolResponse(
                    data={"success": False, "error": str(e)},
                    message=f"Query failed: {e}"
                ).to_mcp()

    Example with image:
        @tool("screenshot", "Take a screenshot")
        async def screenshot() -> dict:
            img_bytes = take_screenshot()
            img_b64 = base64.b64encode(img_bytes).decode()
            return ToolResponse(
                data={"status": "screenshot_taken"},
                images=[(img_b64, "image/png")]
            ).to_mcp()
    """
    data: dict
    message: str = ""
    limit: int = None
    images: List[tuple] = None  # List of (base64_data, mime_type)

    def to_mcp(self) -> dict:
        """Convert to MCP tool response format.

        Returns the data as JSON inside the MCP content structure,
        ensuring routing fields are preserved through the MCP layer.
        If limit is set, truncates the output with a notice.
        Images are included as separate content blocks.
        """
        import json
        content = []

        # Add text content (routing data as JSON)
        text = json.dumps(self.data, default=str)
        if self.limit and len(text) > self.limit:
            text = text[:self.limit] + f"\n\n[TRUNCATED - output exceeded {self.limit} chars]"
        content.append({"type": "text", "text": text})

        # Add image content blocks
        if self.images:
            for img_data, mime_type in self.images:
                content.append({
                    "type": "image",
                    "data": img_data,
                    "mimeType": mime_type
                })

        return {"content": content}


# Type mapping from Python types to JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _extract_schema_from_signature(func: Callable) -> dict:
    """Extract JSON Schema from function signature using type hints.

    Supports typing.Annotated for parameter descriptions:
        def func(query: Annotated[str, "The search query"]) -> dict:

    Parameters with default values are marked as optional.

    Args:
        func: The function to extract schema from

    Returns:
        JSON Schema dict with 'type', 'properties', and 'required' fields
    """
    sig = inspect.signature(func)

    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip *args, **kwargs, and 'args' dict parameter (old style)
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param_name == "args" and hints.get(param_name) == dict:
            continue

        hint = hints.get(param_name)
        if hint is None:
            continue

        # Extract type and description from Annotated
        param_type = hint
        description = None

        if get_origin(hint) is Annotated:
            args = get_args(hint)
            param_type = args[0]  # First arg is the actual type
            # Look for string description in metadata
            for meta in args[1:]:
                if isinstance(meta, str):
                    description = meta
                    break

        # Convert Python type to JSON Schema type
        json_type = _TYPE_MAP.get(param_type, "string")

        prop: dict = {"type": json_type}
        if description:
            prop["description"] = description

        properties[param_name] = prop

        # Parameters without defaults are required
        if param.default is param.empty:
            required.append(param_name)

    if not properties:
        return {}

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _create_args_wrapper(func: Callable, registry: 'ToolRegistry' = None, tool_name: str = None) -> Callable:
    """Create a wrapper that converts dict args to keyword arguments.

    The SDK expects tools to have signature: async def tool(args: dict) -> dict
    But we want users to define tools with proper parameters.
    This wrapper bridges the gap and validates tool access.

    Args:
        func: The function with proper parameter signature
        registry: Optional ToolRegistry for step validation
        tool_name: Optional tool name for validation

    Returns:
        A wrapper function with (args: dict) -> dict signature
    """
    import functools

    sig = inspect.signature(func)
    param_names = [
        name for name, param in sig.parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]

    # Check if function already uses old-style args: dict signature
    is_old_style = len(param_names) == 1 and param_names[0] == "args"

    # If old-style and no validation needed, return unwrapped
    if is_old_style and not (registry and tool_name):
        return func

    @functools.wraps(func)
    async def wrapper(args: dict) -> dict:
        import traceback

        # Validate tool access if registry provided
        if registry and tool_name:
            if not registry.is_tool_allowed(tool_name):
                allowed = ", ".join(sorted(registry._current_allowed_tools))
                error_msg = f"Tool '{tool_name}' is not available in step '{registry._current_step_name}'. Available tools: {allowed}"
                log.warning(f"Blocked unauthorized tool call: {tool_name} in step {registry._current_step_name}")
                return {
                    "content": [{
                        "type": "text",
                        "text": error_msg
                    }]
                }

        try:
            # For old-style functions, just pass args through
            if is_old_style:
                result = await func(args)
            else:
                # Extract values from args dict and pass as keyword arguments
                kwargs = {name: args.get(name) for name in param_names if name in args}
                result = await func(**kwargs)

            # Capture result for routing
            if registry:
                registry._tool_results.append((tool_name, result))

            return result

        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {e}\n{traceback.format_exc()}"
            log.error(error_msg)

            error_result = {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "error": str(e),
            }

            # Still capture error result for routing
            if registry:
                registry._tool_results.append((tool_name, error_result))

            return error_result

    return wrapper


def tool(
    name: str,
    description: str,
) -> Callable:
    """Decorator to define a pipeline tool.

    Marks a function as a tool with metadata. The function can then be
    passed directly to PipelineStep.tools (like hooks) and will be
    auto-registered when the pipeline is created.

    Parameter schema is automatically extracted from function signature.
    Use typing.Annotated to add descriptions to parameters.
    Parameters with default values are treated as optional.

    Args:
        name: Unique tool identifier
        description: Human-readable description for Claude

    Returns:
        Decorator function

    Example:
        from typing import Annotated
        from relais import tool

        @tool("search", "Search for information")
        async def search(
            query: Annotated[str, "The search query"],
            limit: Annotated[int, "Max results to return"] = 10,
        ) -> dict:
            return {"content": [{"type": "text", "text": f"Results for {query}"}]}

        # Then use directly in step:
        PipelineStep(
            name="research",
            tools=[search],  # Pass function directly, like hooks
            ...
        )
    """
    def decorator(func: Callable[..., Awaitable[dict]]) -> Callable:
        # Extract schema from function signature
        schema = _extract_schema_from_signature(func)

        # Attach metadata to the function
        func._tool_name = name
        func._tool_description = description
        func._tool_schema = schema
        return func
    return decorator


def is_tool_function(obj: Any) -> bool:
    """Check if an object is a @tool decorated function."""
    return callable(obj) and hasattr(obj, '_tool_name')


@dataclass
class ToolDefinition:
    """Definition of a tool available to pipeline steps.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for Claude
        input_schema: Schema for the tool's input parameters
        sdk_tool: The SDK tool instance created by @tool decorator
    """
    name: str
    description: str
    input_schema: dict
    sdk_tool: Any  # SdkMcpTool


class ToolRegistry:
    """Registry for pipeline tools using Claude Agent SDK.

    Wraps the SDK's @tool decorator to provide a registry pattern
    that integrates with pipelines.

    Usage:
        from typing import Annotated

        registry = ToolRegistry("my_tools")

        @registry.tool("greet", "Greet the user")
        async def greet(
            name: Annotated[str, "The name to greet"],
            formal: Annotated[bool, "Use formal greeting"] = False,
        ) -> dict:
            greeting = "Good day" if formal else "Hello"
            return {
                "content": [{"type": "text", "text": f"{greeting}, {name}!"}]
            }

        # Get MCP server config for SDK options
        server_config = registry.create_mcp_server()

        # Get allowed_tools list for step
        allowed = registry.get_allowed_tools(["greet"])
    """

    def __init__(self, name: str = "pipeline_tools"):
        """Initialize the registry.

        Args:
            name: Name for the MCP server that will host these tools
        """
        self._name = name
        self._tools: Dict[str, ToolDefinition] = {}
        self._sdk_tools: List[Any] = []
        self._current_step_name: Optional[str] = None
        self._current_allowed_tools: set = set()
        self._tool_results: List[tuple] = []  # [(tool_name, result), ...]
        log.info(f"Created tool registry: {name}")

    def tool(
        self,
        name: str,
        description: str,
    ) -> Callable:
        """Decorator for registering tools.

        Uses the SDK's @tool decorator internally. Tools must be async
        and return the SDK-expected format.

        Parameter schema is automatically extracted from function signature.
        Use typing.Annotated to add descriptions to parameters.
        Parameters with default values are treated as optional.

        Args:
            name: Unique tool identifier
            description: Human-readable description

        Returns:
            Decorator function

        Example:
            @registry.tool("search", "Search for info")
            async def search(
                query: Annotated[str, "The search query"],
                limit: Annotated[int, "Max results"] = 10,
            ) -> dict:
                result = do_search(query)
                return {
                    "content": [{"type": "text", "text": result}]
                }
        """
        def decorator(func: Callable[..., Awaitable[dict]]) -> Any:
            # Extract schema from function signature
            schema = _extract_schema_from_signature(func)

            log.info(f"Registering tool: {name}")
            log.debug(f"Tool schema: {schema}")

            # Wrap function to convert dict args to keyword arguments and validate access
            wrapped = _create_args_wrapper(func, registry=self, tool_name=name)

            # Create SDK tool using the @tool decorator
            decorated = sdk_tool(name, description, schema)(wrapped)

            # Store in our registry
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=schema,
                sdk_tool=decorated
            )
            self._sdk_tools.append(decorated)

            return decorated

        return decorator

    def register_tool_function(self, func: Callable) -> str:
        """Register a @tool decorated function.

        Args:
            func: A function decorated with @tool

        Returns:
            The tool name

        Raises:
            ValueError: If function is not decorated with @tool
        """
        if not is_tool_function(func):
            raise ValueError(f"Function {func} is not decorated with @tool")

        name = func._tool_name
        description = func._tool_description
        schema = func._tool_schema

        # Skip if already registered
        if name in self._tools:
            log.debug(f"Tool '{name}' already registered, skipping")
            return name

        log.info(f"Registering tool: {name}")
        log.debug(f"Tool schema: {schema}")

        # Wrap function to convert dict args to keyword arguments and validate access
        wrapped = _create_args_wrapper(func, registry=self, tool_name=name)

        # Create SDK tool using the @tool decorator
        decorated = sdk_tool(name, description, schema)(wrapped)

        # Store in our registry
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=schema,
            sdk_tool=decorated
        )
        self._sdk_tools.append(decorated)

        return name

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def get_allowed_tools(self, tools: List) -> List[str]:
        """Get MCP-formatted tool names for ClaudeAgentOptions.allowed_tools.

        Args:
            tools: List of tool names (strings) or @tool decorated functions

        Returns:
            List formatted for allowed_tools (mcp__{server}__{tool})
        """
        allowed = []
        for item in tools:
            # Handle @tool decorated functions
            if callable(item) and hasattr(item, '_tool_name'):
                name = item._tool_name
            else:
                name = item

            if name in self._tools:
                # SDK MCP tools are named: mcp__{server_name}__{tool_name}
                allowed.append(f"mcp__{self._name}__{name}")
            else:
                # Allow built-in tools to pass through (Read, Write, Bash, etc.)
                allowed.append(name)
        return allowed

    def create_mcp_server(self, version: str = "1.0.0"):
        """Create an SDK MCP server config with all registered tools.

        Args:
            version: Server version string

        Returns:
            McpSdkServerConfig for ClaudeAgentOptions.mcp_servers
        """
        log.info(f"Creating MCP server '{self._name}' v{version} with {len(self._sdk_tools)} tools")
        return create_sdk_mcp_server(
            name=self._name,
            version=version,
            tools=self._sdk_tools
        )

    def set_current_step(self, step_name: str, allowed_tools: List) -> None:
        """Set the current executing step and its allowed tools.

        Args:
            step_name: Name of the step being executed
            allowed_tools: List of tool names or @tool functions allowed in this step
        """
        self._current_step_name = step_name

        # Extract tool names from the allowed_tools list
        tool_names = set()
        for item in allowed_tools:
            # Handle @tool decorated functions
            if callable(item) and hasattr(item, '_tool_name'):
                tool_names.add(item._tool_name)
            elif isinstance(item, str):
                tool_names.add(item)

        self._current_allowed_tools = tool_names
        self._tool_results = []
        log.debug(f"Set current step: {step_name}, allowed tools: {tool_names}")

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current step.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if allowed, False otherwise. If no step has been set, all tools are allowed.
        """
        # If no step has been set, allow all tools (not in pipeline context)
        if self._current_step_name is None:
            return True
        return tool_name in self._current_allowed_tools

    @property
    def name(self) -> str:
        """Get the registry/server name."""
        return self._name

    def get_tool_result(self, tool_name: str) -> Optional[tuple]:
        """Get the result of a specific tool by name.

        If the tool was called multiple times, returns the last result.

        Args:
            tool_name: Name of the tool to find

        Returns:
            Tuple of (tool_name, result) or None if tool wasn't called.
        """
        for name, result in reversed(self._tool_results):
            if name == tool_name:
                return (name, result)
        return None
