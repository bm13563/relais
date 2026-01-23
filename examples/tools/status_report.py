"""Status report tool for hooks pipeline."""

from relais import tool, Annotated


@tool("report_status", "Generate a system status report")
async def report_status(
    include_details: Annotated[bool, "Whether to include detailed information"] = True,
) -> dict:
    """Generate a status report."""
    if include_details:
        status = "System Status: All systems operational. No issues detected."
    else:
        status = "Status: OK"
    return {
        "content": [{"type": "text", "text": status}]
    }
