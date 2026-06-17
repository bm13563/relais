"""Logging for relais, backed by spool.

relais logs structured events to spool (process-safe JSONL you can search and
navigate). Per-pipeline loggers write to their own stream so a single pipeline's
history can be queried in isolation.

    from relais.logging_config import get_logger
    log = get_logger("relais.my_pipeline")
    log.info("step_done", run=run_id, step="analyze", turns=3)

Query the resulting JSONL with the `spool` TUI or reader API.
"""

from __future__ import annotations
import re

from spool.writer import get_logger as _spool_get_logger


def _sanitize(name: str) -> str:
    """Make a logger name safe to use as a spool filename."""
    # spool turns the name into {name}.jsonl; keep it path-free and tidy.
    name = name.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "relais"


def get_logger(name: str):
    """Return a spool logger for the given name.

    Args:
        name: Logger name, e.g. "relais.<pipeline>". Becomes a JSONL filename.
    """
    return _spool_get_logger(_sanitize(name))


def setup_logging(*args, **kwargs):
    """Compatibility shim.

    Logging is configured per-call via spool's get_logger; there is no global
    setup to perform. Returns the default relais logger.
    """
    return get_logger("relais")
