"""Search tool for research pipelines."""

from relais import tool, Annotated


@tool("search", "Search for information on a topic")
async def search(
    query: Annotated[str, "The search query to execute"],
) -> dict:
    """Simulated search - returns placeholder results."""
    return {
        "content": [{
            "type": "text",
            "text": f"Search results for '{query}':\n"
                    f"1. Overview of {query}\n"
                    f"2. History and background of {query}\n"
                    f"3. Key facts about {query}\n"
                    f"4. Recent developments in {query}"
        }]
    }
