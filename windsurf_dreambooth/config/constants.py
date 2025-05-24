"""Application constants and configuration settings."""

from pathlib import Path

# All pipelines now available in diffusers>=0.33.0
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FluxPipeline,
    PNDMScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)

# Now that we have newer diffusers, both SD3 and Flux are available!

# Directory for storing fine-tuned models
FINETUNED_MODELS_DIR = "finetuned_models"
Path(FINETUNED_MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Pipeline mapping for different model types

PIPELINE_MAPPING = {
    "StableDiffusion": StableDiffusionPipeline,
    "StableDiffusionXL": StableDiffusionXLPipeline,
    "StableDiffusion3": StableDiffusion3Pipeline,
    "Flux": FluxPipeline,
    "Generic": DiffusionPipeline,
}

# Scheduler options

SCHEDULER_MAPPING = {
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
}

# Precision options
PRECISION_OPTIONS = [
    "bf16",  # BFloat16 - good balance
    "f16",  # Float16 - fastest, some quality loss
    "f32",  # Float32 - best quality, uses most memory
    "int8",  # 8-bit quantization - ~50% memory reduction
    "int4",  # 4-bit quantization - ~75% memory reduction
    "nf4",  # 4-bit NormalFloat - better than int4
    "fp4",  # 4-bit FloatingPoint - experimental
]

# Device mapping strategies (from diffusers)
try:
    from diffusers.pipelines.pipeline_utils import SUPPORTED_DEVICE_MAP

    DEVICE_MAP_STRATEGIES = SUPPORTED_DEVICE_MAP
except ImportError:
    DEVICE_MAP_STRATEGIES = ["balanced"]

# For API compatibility
SUPPORTED_MODELS = ["custom"]  # All models are now custom HuggingFace models
PIPELINE_TYPES = list(PIPELINE_MAPPING.keys())
SCHEDULERS = list(SCHEDULER_MAPPING.keys())
