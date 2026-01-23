"""Hook functions for injecting context into pipeline steps."""

from .current_time import get_current_time
from .user_preferences import get_user_preferences
from .system_status import get_system_status

__all__ = [
    "get_current_time",
    "get_user_preferences",
    "get_system_status",
]
