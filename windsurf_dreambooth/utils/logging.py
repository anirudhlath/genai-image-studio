"""
Logging configuration for the application
"""
import sys
from loguru import logger
from diffusers import logging as diffusers_logging

# Configure logger to stdout
logger.remove()
logger.add(
    sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Enable detailed download logs and progress bars for diffusers
diffusers_logging.set_verbosity_info()
diffusers_logging.enable_progress_bar()

# Initialize logger
logger.info("DreamBooth App initialization complete, ready to serve.")

__all__ = ["logger"]
