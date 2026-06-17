"""Hook to inject user preferences into pipeline context."""


def get_user_preferences() -> str:
    """Return user preferences.

    Hooks are called with no arguments. In a real application this would fetch
    per-user settings from a database; here it returns static defaults.
    """
    return "User preferences: language=en, theme=dark, notifications=enabled"
