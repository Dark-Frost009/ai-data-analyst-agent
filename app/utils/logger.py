"""
Minimal, centralized logging setup.

Streamlit re-executes the whole script on every user interaction, so this
module guards against attaching duplicate handlers to the same logger on
repeated calls within one process.
"""

import logging

from app.config import config

_CONFIGURED_LOGGERS: set[str] = set()


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
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # avoid duplicate lines via the root logger

        _CONFIGURED_LOGGERS.add(name)

    return logger
