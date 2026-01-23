"""Utility functions for Claude Code MCP pipelines."""

import re
from pathlib import Path
from typing import Optional


def read_markdown(file_path: str, base_dir: Path = None) -> str:
    """Read a markdown file and return its contents.

    Args:
        file_path: Path to the markdown file (can be relative or absolute)
        base_dir: Base directory for relative paths. If None, uses current working directory.
    """
    if base_dir:
        full_path = base_dir / file_path
    else:
        full_path = Path(file_path)

    with open(full_path, 'r', encoding='utf-8') as file:
        return file.read()


def parse_command(prompt: str, prefix: str = "#") -> Optional[dict]:
    """Parse a command from user prompt.

    Args:
        prompt: The user's input prompt
        prefix: The command prefix (default: "#")

    Returns:
        {'command': str, 'args': str|None} or None if no command found

    Example:
        parse_command("#learn ordering coffee")
        -> {'command': 'learn', 'args': 'ordering coffee'}
    """
    pattern = rf'{re.escape(prefix)}(\w+)(?:\s+(.+))?$'
    match = re.search(pattern, prompt.strip())
    if not match:
        return None
    return {
        'command': match.group(1),
        'args': match.group(2)
    }
