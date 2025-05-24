"""Image generation functionality using diffusion models."""

from pathlib import Path
from typing import Optional

import torch

from ..config.constants import SCHEDULER_MAPPING
from ..utils.logging import logger


def estimate_time(steps: int, batch_size: int, width: int, height: int, precision: str) -> str:
    """Estimate the time required for image generation.

    Args:
        steps: Number of inference steps
        batch_size: Batch size for generation
        width: Image width
        height: Image height
        precision: Precision to use for generation

    Returns:
        Formatted string with time estimate
    """
    # Base time in seconds for a standard 512x512 image at f16 precision with 50 steps
    base_time_per_step = 0.12

    # Calculate multipliers based on parameters
    size_multiplier = (width * height) / (512 * 512)
    precision_multiplier = 1.0
    if precision in ("f32", "bf32"):
        precision_multiplier = 1.5

    # Estimate total time in seconds
    estimated_time = (
        steps * batch_size * base_time_per_step * size_multiplier * precision_multiplier
    )

    # Format the output
    if estimated_time < 60:
        return f"Estimated time: {estimated_time:.1f} seconds"
    elif estimated_time < 3600:
        return f"Estimated time: {estimated_time / 60:.1f} minutes"
    else:
        return f"Estimated time: {estimated_time / 3600:.1f} hours"


def get_max_steps(model_key: str, precision: str, current_steps: int) -> int:
    """Determine maximum recommended steps based on model and precision.

    Args:
        model_key: Model key or ID
        precision: Precision setting
        current_steps: Current step setting

    Returns:
        Updated step value if needed
    """
    # Default max steps for most models
    max_steps = 1000

    # Adjust based on model type
    if "xl" in model_key.lower():
        max_steps = 100  # SDXL models usually need fewer steps
    elif "sd3" in model_key.lower() or "stable-diffusion-3" in model_key.lower():
        max_steps = 50  # SD3 is even faster

    # Adjust for high precision which might run out of memory with many steps
    if precision in ["f32", "bf32"]:
        max_steps = min(max_steps, 500)

    # Return the minimum of current steps and max steps
    return min(current_steps, max_steps)


def generate_image(  # type: ignore
    model_key,
    custom_model,
    subject_name,
    prompt,
    steps,
    guidance,
    batch_size,
    precision,
    pipeline_type,
    scheduler_name,
    width,
    height,
    seed,
    cpu_offload,
    sequential_cpu_offload=False,
    device_map_strategy=None,
    low_cpu_mem_usage=False,
):
    """Generate images using the specified model and parameters.

    Args:
        model_key: Key of the model to use or "custom" for custom_model
        custom_model: HuggingFace model ID for custom models
        subject_name: Subject token to include in the prompt
        prompt: Text prompt for image generation
        steps: Number of inference steps
        guidance: Guidance scale for prompt adherence
        batch_size: Number of images to generate
        precision: Precision to use for generation
        pipeline_type: Type of pipeline to use
        scheduler_name: Name of the scheduler to use
        width: Image width
        height: Image height
        seed: Random seed for reproducibility
        cpu_offload: Whether to enable CPU offloading
        sequential_cpu_offload: Whether to enable sequential CPU offloading
        device_map_strategy: Device mapping strategy for model loading
        low_cpu_mem_usage: Whether to enable low CPU memory usage mode

    Returns:
        List of generated image paths
    """
    import uuid

    from ..models.manager import DreamBoothManager

    manager = DreamBoothManager()

    try:
        # Validate inputs
        if not prompt or prompt.strip() == "":
            logger.error("No prompt provided")
            return []

        # Use model_key directly as it now contains the final model ID
        model_id = model_key

        if not model_id:
            logger.error("No model selected")
            return []

        # Validate numerical parameters
        if steps <= 0:
            logger.error(f"Invalid steps value: {steps}")
            return []

        if width <= 0 or height <= 0:
            logger.error(f"Invalid image dimensions: {width}x{height}")
            return []

        # Load the model
        pipeline = manager.load(
            model_id,
            pipeline_type,
            precision,
            cpu_offload,
            sequential_cpu_offload,
            device_map_strategy,
            low_cpu_mem_usage,
        )

        # Apply the selected scheduler if it exists
        if scheduler_name in SCHEDULER_MAPPING:
            logger.info(f"Using {scheduler_name} scheduler")
            pipeline.scheduler = SCHEDULER_MAPPING[scheduler_name].from_config(
                pipeline.scheduler.config
            )

        # Set the seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Prepare the prompt
        if subject_name:
            full_prompt = prompt.replace(subject_name, subject_name)
            if subject_name not in prompt:
                full_prompt = f"a photo of {subject_name}, {prompt}"
        else:
            full_prompt = prompt

        logger.info(f"Generating with prompt: {full_prompt}")

        # Generate the images
        output = pipeline(
            full_prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )

        # Save the images
        Path("outputs").mkdir(parents=True, exist_ok=True)
        image_paths = []

        for i, image in enumerate(output.images):
            output_path = f"outputs/generated_{uuid.uuid4().hex[:8]}_{i}.png"
            image.save(output_path)
            image_paths.append(output_path)
            logger.info(f"Saved image to {output_path}")

        return image_paths

    except Exception as e:
        logger.error(f"Generation failed: {e!s}")
        return []


class ImageGenerator:
    """Image generation class for the API."""

    def __init__(self) -> None:
        """Initialize the image generator."""
        from ..models.manager import DreamBoothManager

        self.manager = DreamBoothManager()

    def generate(
        self,
        model_id: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        precision: str = "f16",
        cpu_offload: bool = False,
    ) -> list[str]:
        """Generate images using the specified model and parameters.

        Args:
            model_id: Model ID to use for generation
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            num_images: Number of images to generate
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            scheduler: Scheduler name
            precision: Precision to use
            cpu_offload: Whether to enable CPU offloading

        Returns:
            List of generated image paths
        """
        return generate_image(
            model_key=model_id,
            custom_model=None,
            subject_name="",
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance_scale,
            batch_size=num_images,
            precision=precision,
            pipeline_type="StableDiffusionPipeline",
            scheduler_name=scheduler or "default",
            width=width,
            height=height,
            seed=seed,
            cpu_offload=cpu_offload,
        )
