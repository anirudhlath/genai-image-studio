"""UI components for the DreamBooth application."""

import gradio as gr

from ..models.manager import list_models


def create_train_tab():
    """Create the training tab UI with progress indicators.

    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("🎯 Train") as tab:
        gr.Markdown(
            "**Fine-tune a model** with your custom images. Upload training images and configure parameters to create a personalized model."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                # Primary training inputs
                with gr.Group():
                    name = gr.Textbox(
                        label="Subject Token",
                        placeholder="e.g. 'my_dog', 'alice_person'",
                        info="Unique identifier for your subject",
                    )
                    imgs = gr.File(
                        file_count="multiple",
                        label="Training Images (3-20 recommended)",
                        file_types=["image"],
                    )

                # Training parameters in organized section
                with gr.Accordion("⚙️ Training Parameters", open=True), gr.Row():
                    train_steps_slider = gr.Slider(
                        1,
                        1000,
                        value=150,
                        step=1,
                        label="Training Steps",
                        info="More steps = better quality, longer training",
                    )
                    batch_size_slider = gr.Slider(
                        1,
                        8,
                        value=1,
                        step=1,
                        label="Batch Size",
                        info="Higher = faster training, more VRAM",
                    )

                # Training button
                btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Image preview of uploaded training images
                image_preview = gr.Gallery(
                    label="📸 Training Images Preview",
                    show_label=True,
                    elem_id="train_gallery",
                    columns=4,
                    rows=2,
                    height="auto",
                    preview=True,
                )

                # Training status and progress
                with gr.Group():
                    out = gr.Textbox(label="Training Log", lines=8, interactive=True, max_lines=20)
                    progress = gr.Progress()

        return (
            tab,
            {
                "name": name,
                "imgs": imgs,
                "train_steps": train_steps_slider,
                "batch_size": batch_size_slider,
                "train_button": btn,
            },
            {
                "output": out,
                "progress": progress,
                "image_preview": image_preview,
            },
        )


def create_generate_tab():
    """Create the generation tab UI with progress indicators.

    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("✨ Generate") as tab:
        gr.Markdown(
            "**Generate images** using your trained model. Enter your subject token and creative prompt to create new images."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                # Primary generation inputs
                with gr.Group():
                    name = gr.Textbox(
                        label="Subject Token",
                        placeholder="e.g. 'my_dog', 'alice_person'",
                        info="Token from your trained model",
                    )
                    prompt_in = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="a photo of [subject_token] sitting in a park...",
                        info="Describe the image you want to generate",
                    )

                # Generation settings organized in accordion
                with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                    with gr.Row():
                        steps = gr.Slider(
                            1, 1000, value=150, step=1, label="Steps", info="Quality vs speed"
                        )
                        guidance = gr.Slider(
                            1.0,
                            50.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance",
                            info="Prompt adherence",
                        )
                    with gr.Row():
                        width = gr.Slider(64, 2048, value=512, step=64, label="Width")
                        height = gr.Slider(64, 2048, value=512, step=64, label="Height")
                    with gr.Row():
                        batch_size = gr.Slider(
                            1, 8, value=1, step=1, label="Batch Size", info="Number of images"
                        )
                        seed = gr.Number(value=42, label="Seed", info="-1 for random")

                estimate_output = gr.Markdown(value="⏱️ **Estimated time**: calculating...")

                # Generate button
                btn = gr.Button("✨ Generate Images", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output gallery
                img_out = gr.Gallery(
                    label="🖼️ Generated Images",
                    show_label=True,
                    elem_id="generation_gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    preview=True,
                    container=True,
                    allow_preview=True,
                )

                # Generation status and progress
                with gr.Group():
                    generation_progress = gr.Progress()

        return (
            tab,
            {
                "name": name,
                "prompt": prompt_in,
                "steps": steps,
                "guidance": guidance,
                "batch_size": batch_size,
                "width": width,
                "height": height,
                "seed": seed,
                "generate_button": btn,
            },
            {
                "gallery": img_out,
                "estimate": estimate_output,
                "generation_progress": generation_progress,
            },
        )


def create_models_tab():
    """Create the models management tab UI.

    Returns:
        Tuple of (tab, input components, output components)
    """
    with gr.Tab("📚 Models") as tab:
        gr.Markdown(
            "**Manage your trained models**. Load models into memory for faster generation or delete models you no longer need."
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_list = gr.Dropdown(
                    list_models(),
                    label="Fine-tuned Models",
                    info="Select a model from your collection",
                )
            with gr.Column(scale=1), gr.Group():
                gr.Markdown("**Actions**")
                load_btn = gr.Button("📥 Load Model", variant="primary")
                delete_btn = gr.Button("🗑️ Delete Model", variant="stop")

        # Status outputs
        with gr.Row():
            load_output = gr.Textbox(label="Load Status", interactive=False)
            delete_output = gr.Textbox(label="Delete Status", interactive=False)

        return (
            tab,
            {"model_list": model_list, "load_button": load_btn, "delete_button": delete_btn},
            {"load_output": load_output, "delete_output": delete_output},
        )
