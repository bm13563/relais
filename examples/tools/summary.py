"""Summary tool for research pipelines."""

from relais import tool, Annotated


@tool("create_summary", "Create a summary of research findings")
async def create_summary(
    findings: Annotated[str, "The research findings to summarize"],
    format: Annotated[str, "Output format: 'brief' or 'detailed'"] = "brief",
) -> dict:
    """Create a formatted summary of findings."""
    if format == "detailed":
        prefix = "Detailed Summary:\n"
    else:
        prefix = "Summary:\n"

    return {
        "content": [{
            "type": "text",
            "text": f"{prefix}{findings}"
        }]
    }
