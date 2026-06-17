"""Utility functions for Relais pipelines."""

from pathlib import Path


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
