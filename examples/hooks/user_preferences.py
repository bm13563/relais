"""Hook to inject user preferences into pipeline context."""


def get_user_preferences(args: dict) -> str:
    """Return user preferences based on user_id.

    Args:
        args: Pipeline arguments, may contain 'user_id'

    Returns:
        User preferences string
    """
    user_id = args.get("user_id", "anonymous")
    # In a real application, this would fetch from a database
    return f"User preferences for {user_id}: language=en, theme=dark, notifications=enabled"
