"""Hook to inject current time into pipeline context."""

from datetime import datetime


def get_current_time(args: dict) -> str:
    """Return the current time as a formatted string.

    Args:
        args: Pipeline arguments (unused but required by hook interface)

    Returns:
        Current timestamp string
    """
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
