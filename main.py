"""
DreamBooth Studio - Main Entry Point

This is the main entry point for the DreamBooth Studio application.
Run this file to start the application.
"""
from windsurf_dreambooth.ui.app import launch_app


if __name__ == "__main__":
    # Launch the application
    launch_app(server_name="0.0.0.0", server_port=8000, queue=True)
