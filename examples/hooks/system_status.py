"""Hook to inject system status into pipeline context."""


def get_system_status() -> str:
    """Return current system status.

    Hooks are called with no arguments. In a real application this would read
    actual system metrics.
    """
    return "System status: CPU=45%, Memory=62%, Disk=38%, All services healthy"
