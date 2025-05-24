"""
Image generation functionality using diffusion models
"""
import torch
from ..utils.logging import logger
from ..config.constants import SCHEDULER_MAPPING


def estimate_time(steps, batch_size, width, height, precision):
    """
    Estimate the time required for image generation
    
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
    if precision == "f32" or precision == "bf32":
        precision_multiplier = 1.5
    
    # Estimate total time in seconds
    estimated_time = steps * batch_size * base_time_per_step * size_multiplier * precision_multiplier
    
    # Format the output
    if estimated_time < 60:
        return f"Estimated time: {estimated_time:.1f} seconds"
    elif estimated_time < 3600:
        return f"Estimated time: {estimated_time/60:.1f} minutes"
    else:
        return f"Estimated time: {estimated_time/3600:.1f} hours"


def get_max_steps(model_key, precision, current_steps):
    """
    Determine maximum recommended steps based on model and precision
    
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


def generate_image(
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
):
    """
    Generate images using the specified model and parameters
    
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
        
    Returns:
        List of generated image paths
    """
    from ..models.manager import DreamBoothManager
    import uuid
    import os
    
    manager = DreamBoothManager()
    
    try:
        # Validate inputs
        if not prompt or prompt.strip() == "":
            logger.error("No prompt provided")
            return []
        
        # Determine the model to use
        if model_key == "custom" and custom_model:
            model_id = custom_model
        else:
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
        pipeline = manager.load(model_id, pipeline_type, precision, cpu_offload)
        
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
        os.makedirs("outputs", exist_ok=True)
        image_paths = []
        
        for i, image in enumerate(output.images):
            output_path = f"outputs/generated_{uuid.uuid4().hex[:8]}_{i}.png"
            image.save(output_path)
            image_paths.append(output_path)
            logger.info(f"Saved image to {output_path}")
            
        return image_paths
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return []
