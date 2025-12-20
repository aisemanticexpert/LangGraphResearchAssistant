"""
Logging configuration for the Research Assistant.

Provides structured logging with configurable levels and formats.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure application logging.

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

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )

    # Set specific loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured at level: {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
