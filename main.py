"""
DreamBooth Studio - Main Entry Point

This is the main entry point for the DreamBooth Studio application.
It can run in different modes:
- gradio: Gradio UI only (default)
- api: FastAPI REST API only  
- both: Both Gradio and API simultaneously
"""
import argparse
import asyncio
import sys
import threading
from pathlib import Path

import uvicorn

from windsurf_dreambooth.config.settings import settings
from windsurf_dreambooth.utils.logging import get_logger

logger = get_logger(__name__)


def run_gradio():
    """Run Gradio interface."""
    from windsurf_dreambooth.ui.app import launch_app
    
    logger.info(f"Starting Gradio interface on port {settings.gradio_server_port}")
    launch_app(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
        queue=True
    )


def run_api():
    """Run FastAPI REST API."""
    from windsurf_dreambooth.api.app import app
    from windsurf_dreambooth.api.auth import RateLimitMiddleware, SecurityHeadersMiddleware
    
    # Add middleware
    app.add_middleware(RateLimitMiddleware, calls=100, window=60)
    app.add_middleware(SecurityHeadersMiddleware)
    
    logger.info(f"Starting FastAPI on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )


def run_both():
    """Run both Gradio and FastAPI simultaneously."""
    # Run API in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Run Gradio in main thread
    run_gradio()


def create_env_file():
    """Create a template .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        template = """# DreamBooth Studio Configuration

# Security
SECRET_KEY=your-secret-key-here
REQUIRE_API_KEY=false
API_KEY_HEADER=X-API-Key

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
CORS_ORIGINS=["http://localhost:3000"]

# Gradio Settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Model Settings
DEFAULT_MODEL=stabilityai/stable-diffusion-xl-base-1.0
MODEL_CACHE_DIR=./model_cache
MAX_CACHED_MODELS=3

# Training Settings
DEFAULT_LEARNING_RATE=5e-6
DEFAULT_TRAIN_STEPS=1000
DEFAULT_BATCH_SIZE=1
MAX_TRAIN_STEPS=10000
GRADIENT_CLIP_VAL=1.0

# Storage Settings
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
FINETUNED_MODELS_DIR=./finetuned_models
MAX_UPLOAD_SIZE_MB=50

# Performance Settings
ENABLE_MEMORY_EFFICIENT_ATTENTION=true
ENABLE_VAE_SLICING=true
ENABLE_VAE_TILING=false
DATALOADER_NUM_WORKERS=4

# Logging
LOG_LEVEL=INFO
"""
        env_file.write_text(template)
        logger.info("Created .env template file. Please update with your settings.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DreamBooth Studio")
    parser.add_argument(
        "--mode",
        choices=["gradio", "api", "both"],
        default="gradio",
        help="Run mode: gradio (UI only), api (REST API only), or both"
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create a template .env file"
    )
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_file()
        return
    
    # Setup directories
    settings.setup_directories()
    
    # Log startup info
    logger.info(f"Starting DreamBooth Studio in {args.mode} mode")
    logger.info(f"Device: {'CUDA' if settings.device == 'cuda' else 'CPU'}")
    
    # Run based on mode
    if args.mode == "gradio":
        run_gradio()
    elif args.mode == "api":
        run_api()
    elif args.mode == "both":
        run_both()


if __name__ == "__main__":
    main()