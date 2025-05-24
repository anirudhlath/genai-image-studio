"""
Main entry point for the DreamBooth application
"""
from .ui.app import launch_app
from .utils.logging import logger


def main():
    """
    Main entry point for the application
    """
    logger.info("Starting DreamBooth Studio")
    launch_app(server_name="0.0.0.0", server_port=8000, queue=True)


if __name__ == "__main__":
    main()
