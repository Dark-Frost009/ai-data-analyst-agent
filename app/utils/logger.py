"""
Minimal, centralized logging setup.

Streamlit re-executes the whole script on every user interaction, so this
module guards against attaching duplicate handlers to the same logger on
repeated calls within one process.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import logging
from typing import Iterator
from uuid import uuid4

from app.config import config

_CONFIGURED_LOGGERS: set[str] = set()
_analysis_id: ContextVar[str] = ContextVar("analysis_id", default="-")


class _AnalysisContextFilter(logging.Filter):
    """Attach the current safe analysis identifier to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.analysis_id = _analysis_id.get()
        return True


def get_analysis_id() -> str:
    """Return the current analysis identifier, or ``-`` outside an analysis."""
    return _analysis_id.get()


@contextmanager
def analysis_log_context() -> Iterator[str]:
    """Set a short, random correlation ID for one analysis run."""
    analysis_id = uuid4().hex[:12]
    token = _analysis_id.set(analysis_id)

    try:
        yield analysis_id
    finally:
        _analysis_id.reset(token)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured `logging.Logger`.

    Safe to call multiple times with the same name (e.g. on every
    Streamlit rerun) — handlers are only attached once per process.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        logger.setLevel(config.log_level)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | analysis_id=%(analysis_id)s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(_AnalysisContextFilter())

        logger.addHandler(handler)
        logger.propagate = False  # avoid duplicate lines via the root logger

        _CONFIGURED_LOGGERS.add(name)

    return logger
