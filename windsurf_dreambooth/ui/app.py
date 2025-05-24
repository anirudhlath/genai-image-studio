"""
Main application interface for the DreamBooth app
"""
import gradio as gr
from .components import create_train_tab, create_generate_tab, create_models_tab
from ..utils.logging import logger
from ..models.manager import DreamBoothManager, delete_model
from ..training.trainer import train_model
from ..models.generator import generate_image


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


def load_finetuned_model(model_name):
    """
    Load a fine-tuned model into memory
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Status message
    """
    manager = DreamBoothManager()
    manager.load(model_name)
    logger.info(f"Loaded model {model_name}")
    return f"Loaded model {model_name}"


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
        
        # Set up training tab events
        train_inputs["train_button"].click(
            train_model,
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
            train_outputs["output"],
            queue=True,
        )
        
        # Set up generation tab events
        generate_inputs["generate_button"].click(
            generate_image,
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
            generate_outputs["gallery"],
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
