"""Model management functionality for loading, saving, and deleting models with improved caching."""

from collections import OrderedDict
import gc
from pathlib import Path
import shutil
import time
from typing import Optional

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
            enable_attention_slicing: Enable attention slicing for memory efficiency
            enable_vae_slicing: Enable VAE slicing
            enable_vae_tiling: Enable VAE tiling for very large images

        Returns:
            Loaded pipeline
        """
        # Create cache key
        cache_key = f"{model_name}_{pipeline_type}_{precision}_{use_cpu_offload}"

        # Check cache
        cached_pipeline = self.cache.get(cache_key)
        if cached_pipeline is not None:
            logger.info(f"Using cached pipeline for {model_name}")
            return cached_pipeline

        logger.info(
            f"Loading model {model_name} with {pipeline_type} pipeline at {precision} precision"
        )

        # Determine torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "bf32": torch.float32,
            "f16": torch.float16,
            "f32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.float16)

        # Get pipeline class
        pipeline_class = PIPELINE_MAPPING.get(pipeline_type, PIPELINE_MAPPING["Generic"])

        # Check if it's a local fine-tuned model
        local_path = Path(FINETUNED_MODELS_DIR) / model_name
        model_id = str(local_path) if local_path.exists() else model_name

        try:
            # Load pipeline
            logger.info(f"Loading from: {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                variant="fp16" if precision == "f16" else None,
            )

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
            if use_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                logger.info("Enabled CPU offloading")
            else:
                pipeline = pipeline.to(self.device)

            # Cache the pipeline
            self.cache.put(cache_key, pipeline)

            # Update legacy cache for compatibility
            self._models[model_name] = pipeline

            return pipeline

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e!s}")
            raise

    def load(
        self,
        model_name: str,
        pipeline_type: str = "Generic",
        precision: str = "f16",
        use_cpu_offload: bool = False,
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
