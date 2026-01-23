"""Greeting tool for simple pipelines."""

from relais import tool, Annotated


@tool("send_greeting", "Send a friendly greeting to a user")
async def send_greeting(
    name: Annotated[str, "The name of the person to greet"] = "friend",
) -> dict:
    """Send a friendly greeting."""
    return {
        "content": [{"type": "text", "text": f"Hello, {name}! Welcome!"}]
    }
