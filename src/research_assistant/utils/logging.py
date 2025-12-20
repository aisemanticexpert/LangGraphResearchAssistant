"""
Logging configuration for the Research Assistant.

Provides structured logging with configurable levels and formats.
Supports both console and file logging with rotation.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure application logging with file and console handlers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
               If None, uses LOG_LEVEL from settings.
        format_string: Custom format string for log messages.
    """
    from ..config import settings

    if level is None:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Setup handlers list
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if enabled
    if settings.log_to_file:
        log_dir = os.path.dirname(settings.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            settings.log_file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )

    # Set specific loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured at level: {logging.getLevelName(level)}")
    if settings.log_to_file:
        logger.debug(f"Log file: {settings.log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
