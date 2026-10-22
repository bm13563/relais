"""Pipeline tools: the @tool decorator, the registry, per-step gating, capture.

Engine-neutral. A registered tool is a ToolDefinition whose `fn` is the tool as
every engine calls it: an async (args: dict) -> dict that refuses the call when
the tool is not in the current step's tools, runs the tool, captures its result
for routing, and returns MCP-shaped content ({"content": [text / image blocks]}).
The sdk engine serves those definitions to the CLI as an in-process MCP server;
the api engine offers them to the Messages API and calls `fn` itself.
"""

from __future__ import annotations
import functools
import inspect
import json
import traceback
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

from .logging_config import get_logger

log = get_logger("relais.tools")


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
    """Wrap a tool function so an engine can call it with a single args dict.

    Engines invoke tools as ``async def tool(args: dict) -> dict`` (the model's
    tool input). Tools in Relais are defined with explicit, typed parameters;
    this wrapper unpacks the args dict into keyword arguments, enforces per-step
    tool access, and captures the result for routing.

    Args:
        func: The tool function (defined with typed parameters)
        registry: ToolRegistry for per-step access checks and result capture
        tool_name: Tool name used for access checks and capture

    Returns:
        A wrapper with the engines' ``(args: dict) -> dict`` signature
    """
    sig = inspect.signature(func)
    param_names = [
        name for name, param in sig.parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]

    @functools.wraps(func)
    async def wrapper(args: dict) -> dict:
        # Validate tool access if registry provided
        if registry and tool_name:
            if not registry.is_tool_allowed(tool_name):
                log.warning(
                    "tool_blocked", tool=tool_name, step=registry._current_step_name,
                    allowed=", ".join(sorted(registry._current_allowed_tools)),
                )
                return {
                    "content": [{
                        "type": "text",
                        "text": registry._refusal(tool_name)
                    }]
                }

        try:
            # Unpack the args dict into the tool's keyword parameters
            kwargs = {name: args.get(name) for name in param_names if name in args}
            result = await func(**kwargs)

            # Capture result for routing
            if registry:
                registry._tool_results.append((tool_name, result))

            return result

        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {e}\n{traceback.format_exc()}"
            log.error("tool_error", tool=tool_name, error=str(e))

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


def tool_names(tools) -> List[str]:
    """Normalize a tools list (tool names, or @tool decorated functions) to names."""
    return [t._tool_name if is_tool_function(t) else t for t in tools or []]


@dataclass
class ToolDefinition:
    """Definition of a tool available to pipeline steps.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for Claude
        input_schema: JSON Schema for the tool's input parameters ({} = none)
        fn: The tool as engines call it — async (args: dict) -> dict, the
            output of _create_args_wrapper: gated on the current step, result
            captured for routing, MCP-shaped return
    """
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], Awaitable[dict]]


class ToolRegistry:
    """Registry for pipeline tools.

    Holds each tool's definition, the current step's allow-list (the per-step
    gate every call goes through) and the results captured during the step.

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

        # Gate calls to the "intro" step's tools
        registry.set_current_step("intro", ["greet"])
    """

    def __init__(self, name: str = "pipeline_tools"):
        """Initialize the registry.

        Args:
            name: The registry's name (the sdk engine's MCP server is named after it)
        """
        self._name = name
        self._tools: Dict[str, ToolDefinition] = {}
        self._current_step_name: Optional[str] = None
        self._current_allowed_tools: set = set()
        self._tool_results: List[tuple] = []  # [(tool_name, result), ...]
        log.debug("registry_created", registry=name)

    def tool(
        self,
        name: str,
        description: str,
    ) -> Callable:
        """Decorator for registering tools.

        Tools must be async and return the MCP content format. The decorated
        name is bound to the wrapped tool (the definition's `fn`).

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
        def decorator(func: Callable[..., Awaitable[dict]]) -> Callable:
            return self._register(name, description, _extract_schema_from_signature(func), func).fn

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
            log.debug("tool_already_registered", tool=name)
            return name

        self._register(name, description, schema, func)
        return name

    def _register(self, name: str, description: str, schema: dict, func: Callable) -> ToolDefinition:
        """Wrap a tool function (dict args → keyword arguments, per-step access
        check, result capture) and store its definition."""
        log.debug("tool_registered", tool=name, schema=schema)
        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=schema,
            fn=_create_args_wrapper(func, registry=self, tool_name=name),
        )
        self._tools[name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def set_current_step(self, step_name: str, allowed_tools: List) -> None:
        """Set the current executing step and its allowed tools.

        Args:
            step_name: Name of the step being executed
            allowed_tools: List of tool names or @tool functions allowed in this step
        """
        self._current_step_name = step_name
        self._current_allowed_tools = set(tool_names(allowed_tools))
        self._tool_results = []
        log.debug("step_tools", step=step_name, allowed=sorted(self._current_allowed_tools))

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

    def _refusal(self, tool_name: str) -> str:
        """What the model is told when it calls a tool the current step doesn't have."""
        allowed = ", ".join(sorted(self._current_allowed_tools))
        return (
            f"Tool '{tool_name}' is not available in step '{self._current_step_name}'. "
            f"Available tools: {allowed}"
        )

    @property
    def name(self) -> str:
        """Get the registry name."""
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
