"""Chat tool for casual conversation in routing pipelines."""

from relais import tool, Annotated


@tool("chat_response", "Respond to casual conversation")
async def chat_response(
    message: Annotated[str, "The user's message"],
    response: Annotated[str, "The response to send"],
) -> dict:
    """Respond to chat."""
    return {
        "content": [{
            "type": "text",
            "text": response
        }]
    }
