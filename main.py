"""DreamBooth Studio - Main Entry Point.

This is the main entry point for the DreamBooth Studio application.
It can run in different modes:
- gradio: Gradio UI only (default)
- api: FastAPI REST API only
- both: Both Gradio and API simultaneously
"""

import argparse
from pathlib import Path
import threading
from typing import Optional

import uvicorn

from windsurf_dreambooth.config.settings import settings
from windsurf_dreambooth.utils.logging import get_logger

logger = get_logger(__name__)


def run_gradio() -> None:
    """Run Gradio interface."""
    from windsurf_dreambooth.ui.app import launch_app

    logger.info(f"Starting Gradio interface on port {settings.gradio_server_port}")
    launch_app(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        queue=True,
    )


def run_api() -> None:
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


def run_both() -> None:
    """Run both Gradio and FastAPI simultaneously."""
    # Run API in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Run Gradio in main thread
    run_gradio()


def create_env_file() -> None:
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


def test_model_loading(
    model_id: str,
    pipeline_type: str = "Generic",
    precision: str = "f16",
    cpu_offload: bool = False,
    sequential_cpu_offload: bool = False,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> None:
    """Test loading a model to verify it works.

    Args:
        model_id: HuggingFace model ID or path to load
        pipeline_type: Type of pipeline to use
        precision: Model precision (bf16, f16, f32)
        cpu_offload: Enable CPU offloading for memory management
        sequential_cpu_offload: Enable sequential CPU offloading (more aggressive memory saving)
        device_map: Device mapping strategy (e.g., 'balanced')
        low_cpu_mem_usage: Load model weights sequentially to reduce peak memory usage
    """
    import time

    from windsurf_dreambooth.config.constants import PIPELINE_MAPPING, PRECISION_OPTIONS
    from windsurf_dreambooth.models.manager import DreamBoothManager

    # Validate pipeline type
    if pipeline_type not in PIPELINE_MAPPING:
        logger.error(f"❌ Invalid pipeline type: '{pipeline_type}'")
        logger.info(f"Valid pipeline types: {', '.join(PIPELINE_MAPPING.keys())}")
        logger.info("Examples:")
        logger.info("  - Generic: Auto-detect pipeline type (default)")
        logger.info("  - StableDiffusion: For SD 1.x/2.x models")
        logger.info("  - StableDiffusionXL: For SDXL models")
        logger.info("  - StableDiffusion3: For SD3 models")
        logger.info("  - Flux: For Flux models")
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    # Validate precision
    if precision not in PRECISION_OPTIONS:
        logger.error(f"❌ Invalid precision: '{precision}'")
        logger.info(f"Valid precision options: {', '.join(PRECISION_OPTIONS)}")
        logger.info("Examples:")
        logger.info("  - f16: Half precision (default, fastest)")
        logger.info("  - bf16: Brain float 16 (good balance)")
        logger.info("  - f32: Full precision (best quality)")
        logger.info("  - int8: 8-bit quantization (~50% memory reduction)")
        logger.info("  - int4: 4-bit quantization (~75% memory reduction)")
        logger.info("  - nf4: 4-bit NormalFloat (better quality than int4)")
        logger.info("  - fp4: 4-bit FloatingPoint (experimental)")
        raise ValueError(f"Invalid precision: {precision}")

    logger.info(f"Testing model loading for: {model_id}")
    logger.info(f"Pipeline type: {pipeline_type}")
    logger.info(f"Precision: {precision}")

    try:
        start_time = time.time()
        manager = DreamBoothManager()

        logger.info("Loading model...")
        if cpu_offload:
            logger.info("CPU offloading enabled")
        if sequential_cpu_offload:
            logger.info("Sequential CPU offloading enabled")
        if device_map:
            logger.info(f"Device mapping strategy: {device_map}")
        if low_cpu_mem_usage:
            logger.info("Low CPU memory usage enabled")
        pipeline = manager.load(
            model_id,
            pipeline_type,
            precision,
            use_cpu_offload=cpu_offload,
            use_sequential_cpu_offload=sequential_cpu_offload,
            device_map_strategy=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")

        # Log model info
        if hasattr(pipeline, "unet"):
            param_count = sum(p.numel() for p in pipeline.unet.parameters()) / 1e6
            logger.info(f"UNet parameters: {param_count:.1f}M")

        if hasattr(pipeline, "text_encoder"):
            param_count = sum(p.numel() for p in pipeline.text_encoder.parameters()) / 1e6
            logger.info(f"Text encoder parameters: {param_count:.1f}M")

        if hasattr(pipeline, "vae"):
            param_count = sum(p.numel() for p in pipeline.vae.parameters()) / 1e6
            logger.info(f"VAE parameters: {param_count:.1f}M")

        # Test a simple generation to ensure model works
        logger.info("Testing generation...")
        start_time = time.time()

        import torch

        with torch.no_grad():
            _ = pipeline(  # type: ignore[operator]
                "a test image",
                num_inference_steps=1,
                guidance_scale=7.5,
                height=512,
                width=512,
            )

        gen_time = time.time() - start_time
        logger.info(f"✅ Test generation completed in {gen_time:.2f} seconds")

        # Clear memory
        import gc

        import torch

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("✅ All tests passed! Model is working correctly.")

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")

        # If it's a model not found error, show available models
        error_str = str(e).lower()
        if any(
            phrase in error_str
            for phrase in ["not found", "does not exist", "404", "cannot find", "repository"]
        ):
            logger.info("\n📋 Here are some available models you can try:")

            try:
                # Import and use the same fetch function from UI
                from windsurf_dreambooth.ui.app import fetch_hf_models

                available_models = fetch_hf_models(limit=20)

                # Group models by type for better organization
                sd_models = [
                    m
                    for m in available_models
                    if "stable-diffusion" in m.lower() and "xl" not in m.lower()
                ]
                sdxl_models = [m for m in available_models if "xl" in m.lower()]
                other_models = [
                    m for m in available_models if m not in sd_models and m not in sdxl_models
                ]

                if sd_models:
                    logger.info("\n🎨 Stable Diffusion 1.x/2.x Models:")
                    for model in sd_models[:5]:
                        logger.info(f"  - {model}")

                if sdxl_models:
                    logger.info("\n🎨 Stable Diffusion XL Models:")
                    for model in sdxl_models[:5]:
                        logger.info(f"  - {model}")

                if other_models:
                    logger.info("\n🎨 Other Models:")
                    for model in other_models[:5]:
                        logger.info(f"  - {model}")

                # Try to suggest similar models based on what user typed
                if "/" in model_id:
                    search_term = model_id.split("/")[-1].lower()
                    similar_models = [m for m in available_models if search_term in m.lower()]
                    if similar_models and similar_models[0] not in available_models[:15]:
                        logger.info("\n💡 Did you mean one of these?")
                        for model in similar_models[:3]:
                            logger.info(f"  - {model}")

                logger.info("\n📝 Usage examples:")
                logger.info("  python main.py --load runwayml/stable-diffusion-v1-5")
                logger.info(
                    "  python main.py --load stabilityai/stable-diffusion-xl-base-1.0 --pipeline StableDiffusionXL"
                )
                logger.info(
                    "  python main.py --load prompthero/openjourney --pipeline StableDiffusion"
                )

            except Exception as fetch_error:
                logger.debug(f"Could not fetch model list: {fetch_error}")
                # Fallback to hardcoded popular models
                logger.info("\n📋 Popular models you can try:")
                logger.info("  - runwayml/stable-diffusion-v1-5")
                logger.info("  - stabilityai/stable-diffusion-2-1")
                logger.info("  - stabilityai/stable-diffusion-xl-base-1.0")
                logger.info("  - prompthero/openjourney")
                logger.info("  - CompVis/stable-diffusion-v1-4")

        raise


def main() -> None:
    """Main entry point."""
    # Import here to avoid circular imports
    from windsurf_dreambooth.config.constants import (
        DEVICE_MAP_STRATEGIES,
        PIPELINE_MAPPING,
        PRECISION_OPTIONS,
    )

    # Custom error handler for better error messages
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:  # type: ignore[override]
            if "invalid choice:" in message:
                # Extract the argument name and invalid value
                parts = message.split("'")
                if len(parts) >= 3:
                    invalid_value = parts[1]
                    if "argument --pipeline" in message:
                        self.print_usage()
                        print(f"\n❌ Error: Invalid pipeline type '{invalid_value}'")
                        print("\nValid pipeline types:")
                        for name, cls in PIPELINE_MAPPING.items():
                            print(f"  - {name}: {cls.__name__}")
                        print(
                            "\nExample: python main.py --load model_id --pipeline StableDiffusionXL"
                        )
                    elif "argument --precision" in message:
                        self.print_usage()
                        print(f"\n❌ Error: Invalid precision '{invalid_value}'")
                        print("\nValid precision options:")
                        print("  Float precisions: f16, f32")
                        print("  Brain float: bf16")
                        print("  Quantization: int8, int4, nf4, fp4")
                        print("\nExample: python main.py --load model_id --precision f16")
                    elif "argument --mode" in message:
                        self.print_usage()
                        print(f"\n❌ Error: Invalid mode '{invalid_value}'")
                        print("\nValid modes:")
                        print("  - gradio: Run Gradio UI only (default)")
                        print("  - api: Run REST API only")
                        print("  - both: Run both Gradio and API")
                    else:
                        super().error(message)
                else:
                    super().error(message)
                self.exit(2)
            else:
                super().error(message)

    parser = CustomArgumentParser(
        description="DreamBooth Studio - Fine-tune and generate images with Diffusion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Gradio UI (default)
  python main.py

  # Test model loading
  python main.py --load runwayml/stable-diffusion-v1-5
  python main.py --load stabilityai/stable-diffusion-xl-base-1.0 --pipeline StableDiffusionXL

  # Run API server
  python main.py --mode api

  # Create environment template
  python main.py --create-env
""",
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "api", "both"],
        default="gradio",
        help="Run mode: gradio (UI only), api (REST API only), or both",
    )
    parser.add_argument("--create-env", action="store_true", help="Create a template .env file")
    parser.add_argument(
        "--load",
        type=str,
        help="Test loading a model (e.g., --load runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=list(PIPELINE_MAPPING.keys()),
        default="Generic",
        help=f"Pipeline type for --load. Choices: {', '.join(PIPELINE_MAPPING.keys())} (default: Generic)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=PRECISION_OPTIONS,
        default="f16",
        help=f"Model precision for --load. Choices: {', '.join(PRECISION_OPTIONS)} (default: f16)",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offloading for --load (helps with large models)",
    )
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help="Enable sequential CPU offloading for --load (more aggressive memory saving, may be slower)",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Enable low CPU memory usage for --load (loads weights sequentially, reduces peak memory)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        choices=DEVICE_MAP_STRATEGIES,
        help=f"Device mapping strategy for --load. Options: {', '.join(DEVICE_MAP_STRATEGIES)}",
    )

    args = parser.parse_args()

    if args.create_env:
        create_env_file()
        return

    # Setup directories
    settings.setup_directories()

    # Test model loading if requested
    if args.load:
        test_model_loading(
            args.load,
            args.pipeline,
            args.precision,
            args.cpu_offload,
            args.sequential_cpu_offload,
            args.device_map,
            args.low_cpu_mem_usage,
        )
        return

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
