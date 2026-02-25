"""Structured logging for QuickRAG."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a QuickRAG module.

    Args:
        name: Module name (e.g. "quickrag.pipeline").

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
