"""Contextual greeting tool that uses hook-injected context."""

from relais import tool, Annotated


@tool("contextual_greeting", "Send a greeting that incorporates contextual information")
async def contextual_greeting(
    name: Annotated[str, "The name of the person to greet"],
    context: Annotated[str, "Additional context to include in the greeting"] = "",
) -> dict:
    """Send a greeting with context."""
    return {
        "content": [{
            "type": "text",
            "text": f"Hello, {name}! {context}"
        }]
    }
