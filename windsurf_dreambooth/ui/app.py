"""
Main application interface for the DreamBooth app
"""
import gradio as gr
import time
import threading
from pathlib import Path
from PIL import Image

from .components import create_train_tab, create_generate_tab, create_models_tab
from ..utils.logging import logger
from ..models.manager import DreamBoothManager, delete_model
from ..training.trainer import train_model as _train_model
from ..models.generator import generate_image as _generate_image

# Create wrapper functions
def train_model(*args):
    """Wrapper for backward compatibility."""
    return _train_model(*args)

def generate_image(*args):
    """Wrapper for backward compatibility."""
    return _generate_image(*args)


def fetch_hf_models(filter="stable-diffusion", limit=20):
    """
    Fetch models from HuggingFace Hub
    
    Args:
        filter: String to filter model types
        limit: Maximum number of models to return
        
    Returns:
        List of model IDs
    """
    from ..config.constants import AVAILABLE_MODELS
    
    # Always include the base models
    base_models = list(AVAILABLE_MODELS.keys())
    logger.info(f"Including base models: {base_models}")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        models = api.list_models(
            task="text-to-image", sort="downloads", direction=-1, limit=100
        )
        model_ids = [m.modelId for m in models if filter.lower() in m.modelId.lower()]
        logger.info(f"Found {len(model_ids)} models from HuggingFace Hub")
        return base_models + model_ids[:limit]
    except Exception as e:
        logger.warning(f"Failed to fetch models from HuggingFace Hub: {str(e)}")
        return base_models


def load_finetuned_model(model_name, progress=gr.Progress()):
    """
    Load a fine-tuned model into memory with progress
    
    Args:
        model_name: Name of the model to load
        
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
        logger.error(f"Failed to load model: {str(e)}")
        return f"❌ Error loading model: {str(e)}"


def create_app():
    """
    Create and configure the Gradio application
    
    Returns:
        Configured Gradio interface
    """
    # Create the interface with a title
    with gr.Blocks(title="DreamBooth Studio") as demo:
        gr.Markdown(
            """
            # DreamBooth Studio
            
            Fine-tune and generate images with Diffusion models. Select a tab below to get started.
            """
        )
        
        # Create tabs
        train_tab, train_inputs, train_outputs = create_train_tab()
        generate_tab, generate_inputs, generate_outputs = create_generate_tab()
        models_tab, models_inputs, models_outputs = create_models_tab()
        
        # Update model lists
        models = fetch_hf_models()
        # Ensure models is not empty
        if not models:
            logger.warning("No models found, using default placeholder")
            models = ["No models available - please check your connection"]
        
        train_inputs["model"].choices = models
        generate_inputs["model"].choices = models
        
        # Training with progress
        def train_with_progress(
            model, custom_model, name, imgs, train_steps, batch_size, 
            precision, pipeline_type, cpu_offload, progress=gr.Progress()
        ):
            """Train model with progress updates."""
            try:
                # Validate inputs
                if not imgs or len(imgs) == 0:
                    return "Error: No training images provided", None, "### Training Failed\n\nNo images uploaded."
                
                if not name or name.strip() == "":
                    return "Error: Please provide a subject token", None, "### Training Failed\n\nNo subject token provided."
                
                # Update UI
                progress(0, desc="Initializing training...")
                training_info = f"### Training Started\n\n- Model: {model or custom_model}\n- Steps: {train_steps}\n- Batch Size: {batch_size}"
                
                # Show image previews
                image_paths = []
                for img_file in imgs[:8]:  # Show up to 8 images
                    try:
                        img = Image.open(img_file.name)
                        image_paths.append(img)
                    except:
                        pass
                
                # Simulate progress updates for demo
                # In real implementation, you'd get progress from the training function
                total_steps = train_steps
                for step in range(0, total_steps, 10):
                    progress(step / total_steps, desc=f"Training step {step}/{total_steps}")
                    time.sleep(0.1)  # Simulate training time
                
                # Call actual training function
                result = _train_model(
                    model, custom_model, name, imgs, train_steps,
                    batch_size, precision, pipeline_type, cpu_offload
                )
                
                progress(1.0, desc="Training complete!")
                final_info = f"### Training Complete!\n\n- Model saved successfully\n- Total steps: {train_steps}\n- Ready for generation"
                
                return result, image_paths, final_info
                
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                return f"Error: {str(e)}", None, f"### Training Failed\n\n{str(e)}"
        
        # Set up training tab events
        train_inputs["train_button"].click(
            train_with_progress,
            [
                train_inputs["model"],
                train_inputs["custom_model"],
                train_inputs["name"],
                train_inputs["imgs"],
                train_inputs["train_steps"],
                train_inputs["batch_size"],
                train_inputs["precision"],
                train_inputs["pipeline_type"],
                train_inputs["cpu_offload"],
            ],
            [
                train_outputs["output"],
                train_outputs["image_preview"],
                train_outputs["training_info"]
            ],
            queue=True,
        )
        
        # Generation with progress
        def generate_with_progress(
            model, custom_model, name, prompt, steps, guidance, batch_size,
            precision, pipeline_type, scheduler, width, height, seed, cpu_offload,
            progress=gr.Progress()
        ):
            """Generate images with progress updates."""
            try:
                progress(0, desc="Loading model...")
                status_text = f"### Generating Images\n\n- Model: {model or custom_model}\n- Steps: {steps}\n- Batch: {batch_size}"
                
                # Simulate progress for each step
                for step in range(0, steps, 5):
                    progress(step / steps, desc=f"Generating... Step {step}/{steps}")
                    time.sleep(0.05)  # Simulate generation time
                
                # Call actual generation function
                images = _generate_image(
                    model, custom_model, name, prompt, steps, guidance,
                    batch_size, precision, pipeline_type, scheduler,
                    width, height, seed, cpu_offload
                )
                
                progress(1.0, desc="Generation complete!")
                final_status = f"### Generation Complete!\n\n- Generated {len(images) if images else 0} images\n- Ready to download"
                
                return images, final_status
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return None, f"### Generation Failed\n\n{str(e)}"
        
        # Set up generation tab events
        generate_inputs["generate_button"].click(
            generate_with_progress,
            [
                generate_inputs["model"],
                generate_inputs["custom_model"],
                generate_inputs["name"],
                generate_inputs["prompt"],
                generate_inputs["steps"],
                generate_inputs["guidance"],
                generate_inputs["batch_size"],
                generate_inputs["precision"],
                generate_inputs["pipeline_type"],
                generate_inputs["scheduler"],
                generate_inputs["width"],
                generate_inputs["height"],
                generate_inputs["seed"],
                generate_inputs["cpu_offload"],
            ],
            [
                generate_outputs["gallery"],
                generate_outputs["generation_status"]
            ],
            queue=True,
        )
        
        # Set up models tab events
        models_inputs["load_button"].click(
            load_finetuned_model, 
            [models_inputs["model_list"]], 
            models_outputs["load_output"]
        )
        
        models_inputs["delete_button"].click(
            delete_model, 
            [models_inputs["model_list"]], 
            [models_outputs["delete_output"], models_inputs["model_list"]]
        )
        
    return demo


def launch_app(server_name="0.0.0.0", server_port=8000, queue=True):
    """
    Launch the DreamBooth application
    
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
