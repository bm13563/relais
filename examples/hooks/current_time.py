"""Hook to inject current time into pipeline context."""

from datetime import datetime


def get_current_time() -> str:
    """Return the current time as a formatted string.

    Hooks are called with no arguments; their return value is injected into the
    step context as [Hook Data].
    """
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
