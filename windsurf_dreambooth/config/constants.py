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

# Base model configurations
AVAILABLE_MODELS = {
    "sd1": "runwayml/stable-diffusion-v1-5",
    "sd2": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

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
PRECISION_OPTIONS = ["bf16", "bf32", "f16", "f32"]

# For API compatibility
SUPPORTED_MODELS = [*list(AVAILABLE_MODELS.keys()), "custom"]
PIPELINE_TYPES = list(PIPELINE_MAPPING.keys())
SCHEDULERS = list(SCHEDULER_MAPPING.keys())
