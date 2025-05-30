"""Main application interface for the DreamBooth app."""

import time

import gradio as gr
from PIL import Image

from ..models.generator import estimate_time, generate_image as _generate_image, get_max_steps
from ..models.manager import DreamBoothManager, delete_model
from ..training.trainer import train_model as _train_model
from ..utils.logging import logger
from .components import create_generate_tab, create_models_tab, create_train_tab


# Create wrapper functions
def train_model(*args):
    """Wrapper for backward compatibility."""
    return _train_model(*args)


def generate_image(*args):
    """Wrapper for backward compatibility."""
    return _generate_image(*args)


def fetch_hf_models(limit=30):
    """Fetch models from HuggingFace Hub.

    Args:
        limit: Maximum number of models to return

    Returns:
        List of model IDs
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(task="text-to-image", sort="downloads", direction=-1, limit=limit)
        model_ids = [m.modelId for m in models]
        logger.info(f"Found {len(model_ids)} models from HuggingFace Hub")

        return model_ids
    except Exception as e:
        logger.warning(f"Failed to fetch models from HuggingFace Hub: {e!s}")
        # Return some popular HuggingFace models as fallback
        return [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "CompVis/stable-diffusion-v1-4",
            "prompthero/openjourney",
        ]


def load_finetuned_model(model_name, progress=gr.Progress()):
    """Load a fine-tuned model into memory with progress.

    Args:
        model_name: Name of the model to load
        progress: Gradio progress indicator

    Returns:
        Status message
    """
    try:
        progress(0, desc="Loading model...")
        manager = DreamBoothManager()
        manager.load(model_name)
        progress(1.0, desc="Model loaded!")
        logger.info(f"Loaded model {model_name}")
        return f"✅ Loaded model {model_name}"
    except Exception as e:
        logger.error(f"Failed to load model: {e!s}")
        return f"❌ Error loading model: {e!s}"


def create_app():
    """Create and configure the Gradio application.

    Returns:
        Configured Gradio interface
    """
    # Create the interface with a title
    with gr.Blocks(title="DreamBooth Studio") as demo:
        gr.Markdown(
            """
            # DreamBooth Studio

            Fine-tune and generate images with Diffusion models. Configure your global settings below, then select a tab to get started.
            """
        )

        # Global model configuration section
        gr.Markdown("### 🎛️ Model Configuration")

        # Update model lists
        models = fetch_hf_models()
        # Ensure models is not empty
        if not models:
            logger.warning("No models found, using default placeholder")
            models = ["No models available - please check your connection"]

        from ..config.constants import (
            DEVICE_MAP_STRATEGIES,
            PIPELINE_MAPPING,
            PRECISION_OPTIONS,
            SCHEDULER_MAPPING,
        )

        # Model selection row
        with gr.Row():
            with gr.Column(scale=2):
                global_model = gr.Dropdown(
                    choices=models,
                    label="HuggingFace Model",
                    allow_custom_value=True,
                    info="Select from popular models",
                    interactive=True,
                )
            with gr.Column(scale=2):
                global_custom_model = gr.Textbox(
                    label="Custom Model ID (Optional)",
                    placeholder="e.g. prompthero/openjourney-v4",
                    info="Override with any HuggingFace model ID",
                )
            with gr.Column(scale=3), gr.Group():
                gr.Markdown("**Advanced Settings**")
                with gr.Row():
                    global_pipeline = gr.Dropdown(
                        choices=list(PIPELINE_MAPPING.keys()),
                        value="Generic",
                        label="Pipeline",
                        info="Architecture type",
                        interactive=True,
                    )
                    global_precision = gr.Dropdown(
                        choices=PRECISION_OPTIONS,
                        value="f16",
                        label="Precision",
                        info="f16=fast, f32=quality, int8/int4=memory saving",
                        interactive=True,
                    )
                with gr.Row():
                    global_scheduler = gr.Dropdown(
                        choices=list(SCHEDULER_MAPPING.keys()),
                        value="UniPC",
                        label="Scheduler",
                        info="Sampling method",
                        interactive=True,
                    )
                    global_cpu_offload = gr.Checkbox(
                        value=False, label="CPU Offload", info="Save VRAM (slower)"
                    )
                with gr.Row():
                    global_sequential_cpu_offload = gr.Checkbox(
                        value=False,
                        label="Sequential CPU Offload",
                        info="More aggressive VRAM saving (much slower)",
                    )
                    global_low_cpu_mem_usage = gr.Checkbox(
                        value=False,
                        label="Low CPU Memory Usage",
                        info="Load weights sequentially (reduces peak memory)",
                    )
                with gr.Row():
                    global_device_map = gr.Dropdown(
                        choices=["None", *DEVICE_MAP_STRATEGIES],
                        value="None",
                        label="Device Map Strategy",
                        info="How to distribute model across devices",
                        interactive=True,
                    )

        # Create tabs
        train_tab, train_inputs, train_outputs = create_train_tab()
        generate_tab, generate_inputs, generate_outputs = create_generate_tab()
        models_tab, models_inputs, models_outputs = create_models_tab()

        # Training with progress
        def train_with_progress(
            name,
            imgs,
            train_steps,
            batch_size,
            model,
            custom_model,
            precision,
            pipeline_type,
            cpu_offload,
            sequential_cpu_offload,
            progress=gr.Progress(),
        ):
            """Train model with progress updates."""
            try:
                # Validate inputs
                if not imgs or len(imgs) == 0:
                    return (
                        "Error: No training images provided",
                        None,
                        "### Training Failed\n\nNo images uploaded.",
                    )

                if not name or name.strip() == "":
                    return (
                        "Error: Please provide a subject token",
                        None,
                        "### Training Failed\n\nNo subject token provided.",
                    )

                # Update UI
                progress(0, desc="Initializing training...")

                # Show image previews
                image_paths = []
                for img_file in imgs[:8]:  # Show up to 8 images
                    try:
                        img = Image.open(img_file.name)
                        image_paths.append(img)
                    except:  # nosec B110
                        pass

                # Simulate progress updates for demo
                # In real implementation, you'd get progress from the training function
                total_steps = train_steps
                for step in range(0, total_steps, 10):
                    progress(step / total_steps, desc=f"Training step {step}/{total_steps}")
                    time.sleep(0.1)  # Simulate training time

                # Use custom model if provided, otherwise use dropdown selection
                final_model = (
                    custom_model.strip() if custom_model and custom_model.strip() else model
                )

                result = _train_model(
                    final_model,
                    "",  # Pass empty string for custom_model (legacy parameter)
                    name,
                    imgs,
                    train_steps,
                    batch_size,
                    precision,
                    pipeline_type,
                    cpu_offload,
                    sequential_cpu_offload,
                )

                progress(1.0, desc="Training complete!")

                return result, image_paths

            except Exception as e:
                logger.error(f"Training error: {e!s}")
                return f"Error: {e!s}", None

        # Set up training tab events
        train_inputs["train_button"].click(
            train_with_progress,
            [
                train_inputs["name"],
                train_inputs["imgs"],
                train_inputs["train_steps"],
                train_inputs["batch_size"],
                global_model,
                global_custom_model,
                global_precision,
                global_pipeline,
                global_cpu_offload,
                global_sequential_cpu_offload,
            ],
            [
                train_outputs["output"],
                train_outputs["image_preview"],
            ],
            queue=True,
        )

        # Generation with progress
        def generate_with_progress(
            name,
            prompt,
            steps,
            guidance,
            batch_size,
            width,
            height,
            seed,
            model,
            custom_model,
            precision,
            pipeline_type,
            cpu_offload,
            sequential_cpu_offload,
            device_map,
            low_cpu_mem_usage,
            scheduler,
            progress=gr.Progress(),
        ):
            """Generate images with progress updates."""
            try:
                progress(0, desc="Loading model...")

                # Simulate progress for each step
                for step in range(0, steps, 5):
                    progress(step / steps, desc=f"Generating... Step {step}/{steps}")
                    time.sleep(0.05)  # Simulate generation time

                # Use custom model if provided, otherwise use dropdown selection
                final_model = (
                    custom_model.strip() if custom_model and custom_model.strip() else model
                )

                # Convert "None" to None for device_map
                device_map_strategy = None if device_map == "None" else device_map

                images = _generate_image(
                    final_model,
                    "",  # Pass empty string for custom_model (legacy parameter)
                    name,
                    prompt,
                    steps,
                    guidance,
                    batch_size,
                    precision,
                    pipeline_type,
                    scheduler,
                    width,
                    height,
                    seed,
                    cpu_offload,
                    sequential_cpu_offload,
                    device_map_strategy,
                    low_cpu_mem_usage,
                )

                progress(1.0, desc="Generation complete!")

                return images

            except Exception as e:
                logger.error(f"Generation error: {e!s}")
                return None

        # Set up generation tab events
        generate_inputs["generate_button"].click(
            generate_with_progress,
            [
                generate_inputs["name"],
                generate_inputs["prompt"],
                generate_inputs["steps"],
                generate_inputs["guidance"],
                generate_inputs["batch_size"],
                generate_inputs["width"],
                generate_inputs["height"],
                generate_inputs["seed"],
                global_model,
                global_custom_model,
                global_precision,
                global_pipeline,
                global_cpu_offload,
                global_sequential_cpu_offload,
                global_device_map,
                global_low_cpu_mem_usage,
                global_scheduler,
            ],
            [generate_outputs["gallery"]],
            queue=True,
        )

        # Set up dynamic updates for generation tab
        for comp in [
            generate_inputs["steps"],
            generate_inputs["batch_size"],
            generate_inputs["width"],
            generate_inputs["height"],
        ]:
            comp.change(
                lambda steps, batch_size, width, height: estimate_time(
                    steps, batch_size, width, height, global_precision.value
                ),
                [
                    generate_inputs["steps"],
                    generate_inputs["batch_size"],
                    generate_inputs["width"],
                    generate_inputs["height"],
                ],
                generate_outputs["estimate"],
            )

        global_precision.change(
            lambda steps, batch_size, width, height, precision: estimate_time(
                steps, batch_size, width, height, precision
            ),
            [
                generate_inputs["steps"],
                generate_inputs["batch_size"],
                generate_inputs["width"],
                generate_inputs["height"],
                global_precision,
            ],
            generate_outputs["estimate"],
        )

        # Dynamically adjust max inference steps based on model and precision
        global_model.change(
            lambda model, precision, steps: get_max_steps(model, precision, steps),
            [global_model, global_precision, generate_inputs["steps"]],
            generate_inputs["steps"],
        )
        global_precision.change(
            lambda model, precision, steps: get_max_steps(model, precision, steps),
            [global_model, global_precision, generate_inputs["steps"]],
            generate_inputs["steps"],
        )
        global_pipeline.change(
            lambda model, precision, steps: get_max_steps(model, precision, steps),
            [global_model, global_precision, generate_inputs["steps"]],
            generate_inputs["steps"],
        )

        # Set up models tab events
        models_inputs["load_button"].click(
            load_finetuned_model, [models_inputs["model_list"]], models_outputs["load_output"]
        )

        models_inputs["delete_button"].click(
            delete_model,
            [models_inputs["model_list"]],
            [models_outputs["delete_output"], models_inputs["model_list"]],
        )

    return demo


def launch_app(server_name="0.0.0.0", server_port=8000, queue=True):  # nosec B104
    """Launch the DreamBooth application.

    Args:
        server_name: Server hostname
        server_port: Server port
        queue: Whether to enable request queuing
    """
    demo = create_app()

    if queue:
        demo.queue()  # Enable request queuing and streaming of training progress

    logger.info(f"Launching Gradio interface on http://{server_name}:{server_port}")
    demo.launch(server_name=server_name, server_port=server_port)
