"""Logging for relais: Python's standard `logging`, nothing else.

relais is a library, so it never decides where its lines go. Every logger is a
child of "relais" ("relais.pipeline", "relais.tools", "relais.<pipeline name>"),
"relais" carries a NullHandler, and the host's own logging configuration — its
handlers, format and level — decides what is written and where (stdout, a file,
CloudWatch). A host that configures nothing sees relais's warnings and errors on
stderr (Python's last-resort handler) and nothing else; one that wants the run's
story sets "relais" to INFO.

Each line is an EVENT plus fields, as relais has always logged:

    log = get_logger("relais.my_pipeline")
    log.info("step_done", run=run_id, step="analyze", turns=3)

The message reads `step_done run=… step=analyze turns=3`, so any formatter shows
everything; the fields also ride the record as `record.relais` (a dict), for a
handler that wants them structured.

Keep the fields CONTENT-FREE at INFO and above — names, ids, counts, usage —
never a prompt, a model's output or a credential: a host ships INFO to its log
store. The DEBUG events that carry content (step_context, tool_call) say so
where they are logged.
"""

from __future__ import annotations

import logging

logging.getLogger("relais").addHandler(logging.NullHandler())


class EventLogger:
    """A standard logger that takes an event and its fields (the module docstring)."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _log(self, level: int, event: str, fields: dict) -> None:
        # The check first: a disabled DEBUG line never renders its fields.
        if not self.logger.isEnabledFor(level):
            return
        message = " ".join([event, *(f"{key}={value}" for key, value in fields.items())])
        # stacklevel: the record names relais's caller, not this adapter.
        self.logger.log(level, message, extra={"relais": fields}, stacklevel=3)

    def debug(self, event: str, **fields) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields) -> None:
        self._log(logging.ERROR, event, fields)


def get_logger(name: str) -> EventLogger:
    """The event logger for `name` — a child of "relais", e.g. "relais.my_pipeline"."""
    return EventLogger(logging.getLogger(name))
