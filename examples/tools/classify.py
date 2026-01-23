"""Classification tool for routing pipelines."""

from relais import tool, Annotated


@tool("classify_request", "Classify a user request into a category")
async def classify_request(
    input_text: Annotated[str, "The user input text to classify"],
    category: Annotated[str, "The determined category (question, task, or chat)"],
) -> dict:
    """Classify a request into a category."""
    return {
        "content": [{
            "type": "text",
            "text": f"Classified as: {category}"
        }],
        "category": category
    }
