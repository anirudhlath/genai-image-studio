"""
UI components for the DreamBooth application
"""
import gradio as gr
from ..config.constants import PIPELINE_MAPPING, SCHEDULER_MAPPING, PRECISION_OPTIONS
from ..models.generator import estimate_time, get_max_steps
from ..models.manager import list_models


def create_train_tab():
    """
    Create the training tab UI with progress indicators
    
    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("Train") as tab:
        gr.Markdown(
            "**Training Tab**: Select a base model, provide example images of your subject, and customize training parameters. A unique identifier will be created for your custom model."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components - Using an HTML dropdown to ensure full text display
                model = gr.Dropdown([], label="Model", allow_custom_value=True)
                custom_model = gr.Textbox(
                    label="Custom HF Model ID",
                    placeholder="e.g. runwayml/stable-diffusion-v1-5",
                )
                name = gr.Textbox(label="Subject Token", 
                                 placeholder="Type a unique token for your subject, e.g. 'my_cat'")
                imgs = gr.File(
                    file_count="multiple", label="Training Images", file_types=["image"]
                )
                train_steps_slider = gr.Slider(
                    1, 1000, value=150, step=1, label="Training Steps"
                )
                batch_size_slider = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                pipeline_dropdown = gr.Dropdown(
                    list(PIPELINE_MAPPING.keys()), value="Generic", label="Pipeline Type"
                )
                precision_dropdown = gr.Dropdown(
                    PRECISION_OPTIONS, value="f16", label="Precision"
                )
                cpu_offload = gr.Checkbox(value=False, label="Enable CPU Offload")
                
                # Training button
                btn = gr.Button("Train", variant="primary")
            
            with gr.Column(scale=1):
                # Output components with progress indicators
                out = gr.Textbox(label="Status", lines=3)
                progress = gr.Progress()
                training_info = gr.Markdown("### Training Info\n\nReady to train...")
                
                # Image preview of uploaded training images
                image_preview = gr.Gallery(
                    label="Training Images Preview",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
        
        return tab, {
            "model": model,
            "custom_model": custom_model,
            "name": name,
            "imgs": imgs,
            "train_steps": train_steps_slider,
            "batch_size": batch_size_slider,
            "precision": precision_dropdown,
            "pipeline_type": pipeline_dropdown,
            "cpu_offload": cpu_offload,
            "train_button": btn
        }, {
            "output": out,
            "progress": progress,
            "training_info": training_info,
            "image_preview": image_preview
        }


def create_generate_tab():
    """
    Create the generation tab UI with progress indicators
    
    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("Generate") as tab:
        gr.Markdown(
            "**Generation Tab**: Select a model, enter subject token & prompt. Adjust inference steps (quality vs speed), guidance scale (prompt adherence), width/height, and seed for reproducibility."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                model = gr.Dropdown([], label="Model", allow_custom_value=True)
                custom_model = gr.Textbox(
                    label="Custom HF Model ID",
                    placeholder="e.g. runwayml/stable-diffusion-v1-5",
                )
                name = gr.Textbox(label="Subject Token")
                pipeline = gr.Dropdown(
                    list(PIPELINE_MAPPING.keys()), value="Generic", label="Pipeline Type"
                )
                prompt_in = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here...")
                
                with gr.Accordion("Advanced Settings", open=False):
                    cpu_offload = gr.Checkbox(value=True, label="Enable CPU Offload")
                    scheduler = gr.Dropdown(
                        list(SCHEDULER_MAPPING.keys()), value="UniPC", label="Scheduler"
                    )
                    steps = gr.Slider(1, 1000, value=150, step=1, label="Inference Steps")
                    guidance = gr.Slider(1.0, 50.0, value=7.5, step=0.5, label="Guidance Scale")
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                    precision = gr.Dropdown(PRECISION_OPTIONS, value="f16", label="Precision")
                    width = gr.Slider(64, 2048, value=512, step=64, label="Width")
                    height = gr.Slider(64, 2048, value=512, step=64, label="Height")
                    seed = gr.Number(value=42, label="Seed")
                
                estimate_output = gr.Markdown(label="Time Estimate", value="⏱️ Estimated time: calculating...")
        
                # Set up dynamic estimate updates
                for comp in [steps, batch_size, width, height, precision]:
                    comp.change(
                        estimate_time, [steps, batch_size, width, height, precision], estimate_output
                    )
                
                # Dynamically adjust max inference steps based on model scheduler
                model.change(get_max_steps, [model, precision, steps], steps)
                precision.change(get_max_steps, [model, precision, steps], steps)
                pipeline.change(get_max_steps, [model, precision, steps], steps)
        
            with gr.Column(scale=1):
                # Generate button with progress
                btn = gr.Button("Generate", variant="primary", size="lg")
                generation_progress = gr.Progress()
                generation_status = gr.Markdown("### Generation Status\n\nReady to generate...")
                
                # Output components
                img_out = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    preview=True,
                    container=True
                )
        
        return tab, {
            "model": model,
            "custom_model": custom_model,
            "name": name,
            "prompt": prompt_in,
            "steps": steps,
            "guidance": guidance,
            "batch_size": batch_size,
            "precision": precision,
            "pipeline_type": pipeline,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "seed": seed,
            "cpu_offload": cpu_offload,
            "generate_button": btn
        }, {
            "gallery": img_out,
            "estimate": estimate_output,
            "generation_progress": generation_progress,
            "generation_status": generation_status
        }


def create_models_tab():
    """
    Create the models management tab UI
    
    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("Models") as tab:
        gr.Markdown(
            "**Models Tab**: Manage your fine-tuned models. Select one to load into memory or delete from disk. Dropdown updates after deletion."
        )
        
        # Input components
        model_list = gr.Dropdown(list_models(), label="Fine-tuned Models")
        load_btn = gr.Button("Load Model")
        delete_btn = gr.Button("Delete Model")
        
        # Output components
        load_output = gr.Textbox(label="Load Status")
        delete_output = gr.Textbox(label="Delete Status")
        
        return tab, {
            "model_list": model_list,
            "load_button": load_btn,
            "delete_button": delete_btn
        }, {
            "load_output": load_output,
            "delete_output": delete_output
        }
