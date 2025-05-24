"""
Application constants and configuration settings
"""
import os
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

# Try to import newer pipelines if available
try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    StableDiffusion3Pipeline = None
    
try:
    from diffusers import FluxPipeline
except ImportError:
    FluxPipeline = None

# Base model configurations
AVAILABLE_MODELS = {
    "sd1": "runwayml/stable-diffusion-v1-5",
    "sd2": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

# Directory for storing fine-tuned models
FINETUNED_MODELS_DIR = "finetuned_models"
os.makedirs(FINETUNED_MODELS_DIR, exist_ok=True)

# Pipeline mapping for different model types

PIPELINE_MAPPING = {
    "StableDiffusion": StableDiffusionPipeline,
    "StableDiffusionXL": StableDiffusionXLPipeline,
    "Generic": DiffusionPipeline
}

# Add newer pipelines if available
if StableDiffusion3Pipeline is not None:
    PIPELINE_MAPPING["StableDiffusion3"] = StableDiffusion3Pipeline
    
if FluxPipeline is not None:
    PIPELINE_MAPPING["Flux"] = FluxPipeline

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
    "bf16", "bf32", 
    "f16", "f32"
]

# For API compatibility
SUPPORTED_MODELS = list(AVAILABLE_MODELS.keys()) + ["custom"]
PIPELINE_TYPES = list(PIPELINE_MAPPING.keys())
SCHEDULERS = list(SCHEDULER_MAPPING.keys())
