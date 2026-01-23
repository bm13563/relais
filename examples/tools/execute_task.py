"""Task execution tool for routing pipelines."""

from relais import tool, Annotated


@tool("execute_task", "Execute a task and return the result")
async def execute_task(
    task: Annotated[str, "Description of the task to execute"],
    result: Annotated[str, "The result of executing the task"],
) -> dict:
    """Execute a task."""
    return {
        "content": [{
            "type": "text",
            "text": f"Task: {task}\nResult: {result}"
        }]
    }
