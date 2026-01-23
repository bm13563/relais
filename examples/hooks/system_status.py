"""Hook to inject system status into pipeline context."""


def get_system_status(args: dict) -> str:
    """Return current system status.

    Args:
        args: Pipeline arguments (unused but required by hook interface)

    Returns:
        System status string
    """
    # In a real application, this would check actual system metrics
    return "System status: CPU=45%, Memory=62%, Disk=38%, All services healthy"
