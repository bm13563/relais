"""Logging configuration for the pipeline system."""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the pipeline system.

    Args:
        log_file: Path to log file (defaults to LOG_FILE env var or 'pipeline.log')
        log_level: Log level (defaults to LOG_LEVEL env var or 'DEBUG')
        log_format: Log format string

    Returns:
        Configured root logger
    """
    log_file = log_file or os.getenv('LOG_FILE', 'pipeline.log')
    log_level = log_level or os.getenv('LOG_LEVEL', 'DEBUG')
    log_format = log_format or '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'

    # Create logs directory if needed
    log_path = Path(log_file)
    if log_path.parent != Path('.'):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger('relais')
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Console handler - less verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g., 'executor', 'state')

    Returns:
        Logger instance
    """
    return logging.getLogger(f'relais.{name}')


# Initialize logging on module import
_root_logger = setup_logging()
