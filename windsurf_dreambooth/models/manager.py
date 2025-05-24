"""Model management functionality for loading, saving, and deleting models with improved caching."""

from collections import OrderedDict
import gc
from pathlib import Path
import shutil
import time
from typing import Any, Optional

import torch

from ..config.constants import FINETUNED_MODELS_DIR, PIPELINE_MAPPING
from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelCache:
    """LRU cache for model pipelines with memory-aware eviction."""

    def __init__(self, max_size: int = 3, memory_threshold: float = 0.9):
        """Initialize model cache.

        Args:
            max_size: Maximum number of models to cache
            memory_threshold: GPU memory usage threshold for eviction (0.0-1.0)
        """
        self.max_size = max_size
        self.memory_threshold = memory_threshold
        self.cache: OrderedDict = OrderedDict()
        self.access_times: dict = {}
        self.model_sizes: dict = {}

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as a fraction."""
        if not torch.cuda.is_available():
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        except Exception:
            return 0.0

    def _estimate_model_size(self, pipeline: object) -> float:
        """Estimate model size in GB."""
        total_size = 0

        # Count parameters in all model components
        for attr_name in ["unet", "text_encoder", "text_encoder_2", "vae"]:
            if hasattr(pipeline, attr_name):
                model = getattr(pipeline, attr_name)
                if model is not None:
                    total_size += sum(p.numel() * p.element_size() for p in model.parameters())

        return total_size / (1024**3)  # Convert to GB

    def get(self, key: str) -> Optional[object]:
        """Get model from cache, updating access order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, pipeline: object) -> None:
        """Add model to cache with memory-aware eviction."""
        # Check if we need to evict based on memory usage
        memory_usage = self._get_gpu_memory_usage()

        if memory_usage > self.memory_threshold:
            logger.warning(f"GPU memory usage high ({memory_usage:.1%}), evicting models")
            self._evict_until_memory_available()

        # Check if we need to evict based on cache size
        while len(self.cache) >= self.max_size:
            self._evict_oldest()

        # Add to cache
        self.cache[key] = pipeline
        self.access_times[key] = time.time()
        self.model_sizes[key] = self._estimate_model_size(pipeline)

        logger.info(f"Cached model {key} (size: {self.model_sizes[key]:.2f} GB)")

    def _evict_oldest(self) -> None:
        """Evict the least recently used model."""
        if not self.cache:
            return

        # Get oldest key (first in OrderedDict)
        oldest_key = next(iter(self.cache))
        self._evict(oldest_key)

    def _evict_until_memory_available(self) -> None:
        """Evict models until GPU memory usage is below threshold."""
        while self.cache and self._get_gpu_memory_usage() > self.memory_threshold * 0.9:
            # Evict oldest model
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)

    def _evict(self, key: str) -> None:
        """Evict a specific model from cache."""
        if key not in self.cache:
            return

        logger.info(f"Evicting model {key} from cache")

        # Delete the pipeline
        del self.cache[key]
        del self.access_times[key]
        if key in self.model_sizes:
            del self.model_sizes[key]

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self) -> None:
        """Clear entire cache."""
        logger.info("Clearing model cache")
        self.cache.clear()
        self.access_times.clear()
        self.model_sizes.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_info(self) -> dict:
        """Get cache information."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "models": list(self.cache.keys()),
            "total_size_gb": sum(self.model_sizes.values()),
            "gpu_memory_usage": self._get_gpu_memory_usage(),
        }


class DreamBoothManager:
    """Manager class for DreamBooth models with improved caching and memory management."""

    def __init__(self, max_cache_size: Optional[int] = None):
        """Initialize the model manager with device detection and model cache."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize cache
        max_cache_size = max_cache_size or settings.max_cached_models
        self.cache = ModelCache(max_size=max_cache_size)

        # Legacy compatibility
        self._models: dict = {}  # For backward compatibility

        logger.info(f"DreamBoothManager initialized with device {self.device}")
        logger.info(f"Model cache size: {max_cache_size}")

    def load_model(
        self,
        model_name: str,
        pipeline_type: str = "Generic",
        precision: str = "f16",
        use_cpu_offload: bool = False,
        use_sequential_cpu_offload: bool = False,
        device_map_strategy: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
    ) -> object:
        """Load a model with enhanced memory management and caching.

        Args:
            model_name: Model name or HuggingFace model ID
            pipeline_type: Type of pipeline to use
            precision: Model precision (bf16, bf32, f16, f32)
            use_cpu_offload: Enable CPU offloading
            use_sequential_cpu_offload: Enable sequential CPU offloading (more aggressive memory saving)
            device_map_strategy: Device mapping strategy (e.g., 'balanced', or None for no device mapping)
            low_cpu_mem_usage: Load model weights sequentially to reduce peak memory usage
            enable_attention_slicing: Enable attention slicing for memory efficiency
            enable_vae_slicing: Enable VAE slicing
            enable_vae_tiling: Enable VAE tiling for very large images

        Returns:
            Loaded pipeline
        """
        # Create cache key
        cache_key = f"{model_name}_{pipeline_type}_{precision}_{use_cpu_offload}_{use_sequential_cpu_offload}_{device_map_strategy}_{low_cpu_mem_usage}"

        # Check cache
        cached_pipeline = self.cache.get(cache_key)
        if cached_pipeline is not None:
            logger.info(f"Using cached pipeline for {model_name}")
            return cached_pipeline

        # Log all model loading settings
        logger.info("=" * 60)
        logger.info("MODEL LOADING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Pipeline Type: {pipeline_type}")
        logger.info(f"Precision: {precision}")
        logger.info("Memory Management Options:")
        logger.info(f"  - CPU Offload: {'✓' if use_cpu_offload else '✗'}")
        logger.info(f"  - Sequential CPU Offload: {'✓' if use_sequential_cpu_offload else '✗'}")
        logger.info(f"  - Device Map Strategy: {device_map_strategy or 'None'}")
        logger.info(f"  - Low CPU Memory Usage: {'✓' if low_cpu_mem_usage else '✗'}")
        logger.info("Optimization Options:")
        logger.info(f"  - Attention Slicing: {'✓' if enable_attention_slicing else '✗'}")
        logger.info(f"  - VAE Slicing: {'✓' if enable_vae_slicing else '✗'}")
        logger.info(f"  - VAE Tiling: {'✓' if enable_vae_tiling else '✗'}")
        logger.info("=" * 60)

        # Determine torch dtype and quantization config
        dtype_map = {
            "bf16": torch.bfloat16,
            "f16": torch.float16,
            "f32": torch.float32,
            "int8": torch.float16,  # Will use 8-bit quantization
            "int4": torch.float16,  # Will use 4-bit quantization
            "nf4": torch.float16,  # Will use 4-bit NormalFloat
            "fp4": torch.float16,  # Will use 4-bit FloatingPoint
        }
        torch_dtype = dtype_map.get(precision, torch.float16)

        # Prepare quantization config if needed
        quantization_config = None
        load_in_8bit = False
        load_in_4bit = False

        if precision == "int8":
            load_in_8bit = True
            logger.info("Quantization: 8-bit integer quantization enabled")
            logger.info("  - Expected memory reduction: ~50%")
        elif precision in ["int4", "nf4", "fp4"]:
            load_in_4bit = True
            try:
                from transformers import BitsAndBytesConfig

                # Configure 4-bit quantization
                bnb_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch_dtype,
                }

                if precision == "nf4":
                    bnb_config["bnb_4bit_quant_type"] = "nf4"
                    logger.info("Quantization: 4-bit NormalFloat (nf4) quantization enabled")
                elif precision == "fp4":
                    bnb_config["bnb_4bit_quant_type"] = "fp4"
                    logger.info("Quantization: 4-bit FloatingPoint (fp4) quantization enabled")
                else:
                    bnb_config["bnb_4bit_quant_type"] = "int4"
                    logger.info("Quantization: 4-bit integer quantization enabled")

                logger.info("  - Expected memory reduction: ~75%")
                logger.info(f"  - Compute dtype: {torch_dtype}")

                quantization_config = BitsAndBytesConfig(**bnb_config)
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, falling back to fp16")
                load_in_4bit = False

        # Get pipeline class
        pipeline_class = PIPELINE_MAPPING.get(pipeline_type, PIPELINE_MAPPING["Generic"])

        # Check if it's a local fine-tuned model
        local_path = Path(FINETUNED_MODELS_DIR) / model_name
        model_id = str(local_path) if local_path.exists() else model_name

        try:
            # Clear CUDA cache before loading to ensure maximum available memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Report memory status before loading
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                )
                total_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(
                    f"CUDA memory before loading: {free_memory / 1e9:.2f}GB free / {total_memory / 1e9:.2f}GB total"
                )

            # Load pipeline
            use_safetensors = settings.use_safetensors
            logger.info("\nStarting model loading process...")
            logger.info(f"Source: {model_id}")
            logger.info(f"Format: {'Safetensors' if use_safetensors else 'PyTorch'}")

            # Don't specify variant for bf16/bf32 as they don't have variant files
            variant = "fp16" if precision == "f16" else None
            if variant:
                logger.info(f"Variant: {variant}")

            # Prepare loading kwargs
            load_kwargs: dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "safety_checker": None,
                "requires_safety_checker": False,
                "use_safetensors": use_safetensors,
                "variant": variant,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }

            # Add device mapping if requested
            if device_map_strategy:
                load_kwargs["device_map"] = device_map_strategy
                # Device mapping requires low_cpu_mem_usage=True
                load_kwargs["low_cpu_mem_usage"] = True
                logger.info(f"\nDevice Mapping Strategy: {device_map_strategy}")
                if device_map_strategy == "balanced":
                    logger.info("  - Distributes model layers across devices")
                    logger.info("  - Uses both VRAM and system RAM")
                    logger.info("  - Balances memory usage between devices")
                elif device_map_strategy == "auto":
                    logger.info("  - Fills VRAM first, then overflows to RAM")
                    logger.info("  - Optimizes for performance")
                else:
                    logger.info(f"  - Custom strategy: {device_map_strategy}")

            # Add quantization parameters if needed
            if load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = (
                    "balanced"  # Balanced is the only supported option in diffusers
                )
                # Quantization requires low_cpu_mem_usage=True
                load_kwargs["low_cpu_mem_usage"] = True
            elif load_in_4bit and quantization_config:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = (
                    "balanced"  # Balanced is the only supported option in diffusers
                )
                # Quantization requires low_cpu_mem_usage=True
                load_kwargs["low_cpu_mem_usage"] = True

            pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)

            # Apply memory optimizations
            if hasattr(pipeline, "vae") and pipeline.vae is not None:
                if enable_vae_slicing:
                    pipeline.vae.enable_slicing()
                if enable_vae_tiling:
                    pipeline.vae.enable_tiling()

            if hasattr(pipeline, "unet") and pipeline.unet is not None:
                if enable_attention_slicing:
                    pipeline.enable_attention_slicing("auto")

                # Enable xformers if available
                if hasattr(pipeline.unet, "enable_xformers_memory_efficient_attention"):
                    try:
                        pipeline.unet.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers memory efficient attention")
                    except Exception as e:
                        logger.debug(f"Could not enable xformers: {e}")

            # Apply CPU offloading or move to device
            # Skip device movement for models with device_map (quantized or device mapped)
            if load_in_8bit or load_in_4bit or device_map_strategy:
                logger.info("\n✓ Model distributed across devices with automatic mapping")
            elif use_sequential_cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
                logger.info("\n✓ Sequential CPU offloading enabled")
                logger.info("  - Minimal VRAM usage")
                logger.info("  - Layers processed sequentially through CPU")
            elif use_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                logger.info("\n✓ Standard CPU offloading enabled")
                logger.info("  - Inactive components moved to CPU")
            else:
                pipeline = pipeline.to(self.device)
                logger.info(f"\n✓ Model loaded to {self.device.upper()}")

            # Cache the pipeline
            self.cache.put(cache_key, pipeline)

            # Update legacy cache for compatibility
            self._models[model_name] = pipeline

            # Report memory status after loading
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated() / 1e9
                reserved_memory = torch.cuda.memory_reserved() / 1e9
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                )
                logger.info("\nCUDA memory after loading:")
                logger.info(f"  - Allocated: {allocated_memory:.2f}GB")
                logger.info(f"  - Reserved: {reserved_memory:.2f}GB")
                logger.info(f"  - Free: {free_memory / 1e9:.2f}GB")

            logger.info("\n" + "=" * 60)
            logger.info("MODEL LOADING COMPLETE")
            logger.info("=" * 60)

            return pipeline

        except Exception as e:
            error_msg = str(e)

            # Provide more helpful error messages
            if "404" in error_msg or "not found" in error_msg.lower():
                logger.error(f"Model '{model_name}' not found on HuggingFace Hub")
                logger.error("Please check:")
                logger.error("  1. The model ID is spelled correctly")
                logger.error("  2. The model exists and is public")
                logger.error("  3. You have internet connectivity")
                raise ValueError(
                    f"Model '{model_name}' not found. Please check the model ID."
                ) from e
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error(f"Access denied for model '{model_name}'")
                logger.error("This model may be private or require authentication")
                raise ValueError(
                    f"Access denied for model '{model_name}'. It may be private."
                ) from e
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.error("Network connection error")
                logger.error("Please check your internet connection")
                raise ConnectionError("Failed to connect to HuggingFace Hub") from e
            else:
                logger.error(f"Error loading model {model_name}: {e!s}")
                raise

    def load(
        self,
        model_name: str,
        pipeline_type: str = "Generic",
        precision: str = "f16",
        use_cpu_offload: bool = False,
        use_sequential_cpu_offload: bool = False,
        device_map_strategy: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
    ) -> object:
        """Legacy method for backward compatibility."""
        return self.load_model(
            model_name=model_name,
            pipeline_type=pipeline_type,
            precision=precision,
            use_cpu_offload=use_cpu_offload,
            use_sequential_cpu_offload=use_sequential_cpu_offload,
            device_map_strategy=device_map_strategy,
            low_cpu_mem_usage=low_cpu_mem_usage,
            enable_attention_slicing=enable_attention_slicing,
            enable_vae_slicing=enable_vae_slicing,
            enable_vae_tiling=enable_vae_tiling,
        )

    def unload_model(self, model_name: str) -> None:
        """Unload a specific model from cache."""
        # Remove from cache
        keys_to_remove = [k for k in self.cache.cache if k.startswith(model_name)]
        for key in keys_to_remove:
            self.cache._evict(key)

        # Remove from legacy cache
        if model_name in self._models:
            del self._models[model_name]

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self._models.clear()

    def get_cache_info(self) -> dict:
        """Get information about cached models."""
        return self.cache.get_info()

    def list_loaded_models(self) -> list:
        """List currently loaded models."""
        return list(self.cache.cache.keys())


def list_models() -> list[str]:
    """List all available fine-tuned models.

    Returns:
        List of model names
    """
    models_dir = Path(FINETUNED_MODELS_DIR)
    if not models_dir.exists():
        logger.warning(f"Fine-tuned models directory {models_dir} does not exist")
        return []

    try:
        # Filter for directories that contain model files
        model_dirs = [
            item.name
            for item in models_dir.iterdir()
            if item.is_dir() and (any(item.glob("*.safetensors")) or any(item.glob("*.bin")))
        ]

        return sorted(model_dirs)
    except Exception as e:
        logger.error(f"Failed to list models: {e!s}")
        return []


def delete_model(model_name: str) -> tuple[str, list]:
    """Delete a fine-tuned model.

    Args:
        model_name: Name of the model to delete

    Returns:
        Tuple of (status message, updated model list)
    """
    from ..config.constants import SUPPORTED_MODELS

    if model_name in SUPPORTED_MODELS:
        return f"Cannot delete base model {model_name}", list_models()

    model_path = Path(FINETUNED_MODELS_DIR) / model_name

    if not model_path.exists():
        logger.warning(f"Model {model_name} not found at {model_path}")
        return f"Model {model_name} not found", list_models()

    try:
        shutil.rmtree(model_path)
        logger.info(f"Deleted model {model_name}")
        return f"Deleted {model_name}", list_models()
    except Exception as e:
        logger.error(f"Failed to delete model {model_name}: {e!s}")
        return f"Error deleting {model_name}: {e!s}", list_models()


def get_model_info(model_name: str) -> Optional[dict]:
    """Get information about a fine-tuned model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information or None
    """
    model_path = Path(FINETUNED_MODELS_DIR) / model_name

    if not model_path.exists():
        return None

    info = {
        "name": model_name,
        "path": str(model_path),
        "size_gb": sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024**3),
        "created": model_path.stat().st_ctime,
    }

    # Check for training info
    training_info_path = model_path / "training_info.json"
    if training_info_path.exists():
        import json

        with training_info_path.open() as f:
            info["training_info"] = json.load(f)

    return info
