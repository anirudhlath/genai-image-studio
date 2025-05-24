"""
Application constants and configuration settings
"""
import os
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

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
    "StableDiffusion3": StableDiffusion3Pipeline,
    "Flux": FluxPipeline,
    "Generic": DiffusionPipeline
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
    "bf16", "bf32", 
    "f16", "f32"
]
