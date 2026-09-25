"""Compact terminal logging with context isolated to each asynchronous workflow task."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import re


_context: ContextVar[dict[str, str]] = ContextVar("workflow_log_context", default={})


def stage_label(step_id: str) -> str:
    match = re.match(r"agent(\d{2})(?:_|$)", step_id)
    if match:
        return f"agent {match.group(1)}"
    return "python" if step_id.startswith("python_") else "-"


@contextmanager
def log_context(*, source: str | None = None, figure: str | None = None, step: str | None = None):
    """Carry source/figure/stage through nested tools without changing their APIs."""
    values = dict(_context.get())
    if source is not None:
        values["source"] = str(source).replace("\\", "/").rsplit("/", 1)[-1] or "-"
    if figure is not None:
        values["figure"] = str(figure) or "-"
    if step is not None:
        values["stage"] = stage_label(step)
    token = _context.set(values)
    try:
        yield
    finally:
        _context.reset(token)


class ProgressFilter(logging.Filter):
    """Keep workflow progress and every warning/error; omit routine SDK/HTTP chatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or (
            record.levelno >= logging.INFO
            and (record.name == "chart_extract" or record.name.startswith("chart_extract."))
        )


class ProgressFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = _context.get()
        message = " ".join(record.getMessage().split())
        if record.levelno >= logging.WARNING:
            message = f"{record.levelname}: {message}"
        # Panel failures persist their complete traceback in error.txt. Keep the
        # console to one line, without mutating the record seen by file handlers.
        if record.exc_info and record.exc_info[1]:
            error = " ".join(str(record.exc_info[1]).split())
            if error and error not in message:
                message += f": {error}"
        fields = (self.formatTime(record, "%H:%M:%S"), context.get("source", "-"),
                  context.get("figure", "-"), context.get("stage", "-"), message)
        return " | ".join(fields)


def configure_console_logging(stream=None) -> logging.Handler:
    """Install the CLI console handler, preserving any existing file log handlers."""
    root = logging.getLogger()
    logging.getLogger("bokeh").setLevel(logging.WARNING)
    for handler in tuple(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.addFilter(ProgressFilter())
    handler.setFormatter(ProgressFormatter())
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler
