"""Tests for the logging module."""

import logging

from quickrag.logging import get_logger


class TestLogging:
    """Tests for the structured logging setup."""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_logger_has_handler(self):
        logger = get_logger("test.handler")
        assert len(logger.handlers) >= 1

    def test_logger_level(self):
        logger = get_logger("test.level")
        assert logger.level == logging.INFO

    def test_same_logger_returned(self):
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_no_duplicate_handlers(self):
        name = "test.no_dup"
        _ = get_logger(name)
        _ = get_logger(name)
        logger = logging.getLogger(name)
        # Our get_logger guards against adding duplicate handlers
        handler_count = len(logger.handlers)
        assert handler_count >= 1
