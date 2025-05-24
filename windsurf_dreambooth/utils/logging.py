"""Logging configuration for the application."""

import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from loguru import Logger

from diffusers.utils import logging as diffusers_logging
from loguru import logger

# Configure logger to stdout
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Enable detailed download logs and progress bars for diffusers
diffusers_logging.set_verbosity_info()
diffusers_logging.enable_progress_bar()

# Initialize logger
logger.info("DreamBooth App initialization complete, ready to serve.")


def get_logger(name: Optional[str] = None) -> "Logger":
    """Get a logger instance."""
    return logger


__all__ = ["get_logger", "logger"]
