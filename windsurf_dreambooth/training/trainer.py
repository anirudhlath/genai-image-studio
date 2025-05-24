"""
Training functionality for DreamBooth fine-tuning
"""
import os
import uuid
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
from ..utils.logging import logger
from ..config.constants import FINETUNED_MODELS_DIR
from .dataset import DreamBoothDataset


def train_model(
    model_key, 
    custom_model, 
    subject_name, 
    img_paths, 
    train_steps, 
    batch_size, 
    precision, 
    pipeline_type, 
    cpu_offload
):
    """
    Train a DreamBooth model with the provided parameters
    
    Args:
        model_key: Key of the base model or "custom" to use custom_model
        custom_model: HuggingFace model ID for custom models
        subject_name: Token name for the subject to train on
        img_paths: List of paths to training images
        train_steps: Number of training steps
        batch_size: Training batch size
        precision: Precision to use for training
        pipeline_type: Type of pipeline to use
        cpu_offload: Whether to enable CPU offloading
        
    Returns:
        Status message indicating success or failure
    """
    from ..models.manager import DreamBoothManager
    manager = DreamBoothManager()
    
    try:
        # Validate inputs
        if not img_paths or len(img_paths) == 0:
            logger.error("No training images provided")
            return "Error: Please upload at least one training image"
        
        if not subject_name or subject_name.strip() == "":
            logger.error("No subject token provided")
            return "Error: Please provide a subject token name"
        
        # Determine the model to use
        if model_key == "custom" and custom_model:
            model_id = custom_model
        else:
            model_id = model_key
        
        if not model_id:
            logger.error("No model selected")
            return "Error: Please select a base model"
        
        # Generate a unique model name
        output_dir = os.path.join(FINETUNED_MODELS_DIR, f"{subject_name}_{uuid.uuid4().hex[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Will save fine-tuned model to {output_dir}")
        
        # Load the base model
        logger.info(f"Loading base model {model_id} for fine-tuning")
        pipeline = manager.load(model_id, pipeline_type, precision, cpu_offload)
        
        # Set up tokenizer, unet, text encoder, and vae
        tokenizer = pipeline.tokenizer
        
        # Create the training dataset
        logger.info(f"Creating dataset with {len(img_paths)} images")
        train_dataset = DreamBoothDataset(
            image_paths=img_paths,
            tokenizer=tokenizer,
            prompt=f"a photo of {subject_name}",
            size=512 if "xl" not in model_id.lower() else 1024,
            device=manager.device,
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Set up optimization
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(),
            lr=5e-6,
        )
        
        # Create learning rate scheduler
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps,
        )
        
        # Training loop
        progress_bar = tqdm(range(train_steps))
        
        # Set models to training mode
        pipeline.unet.train()
        if hasattr(pipeline, "text_encoder"):
            pipeline.text_encoder.train()
        
        for step in range(train_steps):
            # Get a batch from the dataloader, cycling through it
            batch_idx = step % len(train_dataloader)
            batch = list(train_dataloader)[batch_idx]
            
            # Generate latents
            latents = torch.randn(
                (batch["pixel_values"].shape[0], pipeline.unet.config.in_channels, 64, 64)
            ).to(manager.device)
            
            # Set timesteps
            pipeline.scheduler.set_timesteps(1000)
            timesteps = torch.randint(
                0, 1000, (batch["pixel_values"].shape[0],), device=manager.device
            ).long()
            
            # Encode images to latent space if VAE is available
            if hasattr(pipeline, "vae"):
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(manager.device)
                    latents = pipeline.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            with torch.no_grad():
                if hasattr(pipeline, "text_encoder"):
                    # Ensure input_ids are on the correct device
                    input_ids = batch["input_ids"].to(manager.device)
                    text_embeddings = pipeline.text_encoder(input_ids)[0]
                else:
                    # Fallback for models without text_encoder
                    text_embeddings = batch["input_ids"].to(manager.device)
            
            # Predict noise
            noise_pred = pipeline.unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            progress_bar.update(1)
            
            # Report progress every 10%
            if step % max(1, train_steps // 10) == 0 or step == train_steps - 1:
                logger.info(f"Step {step}/{train_steps}, Loss: {loss.item():.4f}")
                
        # Save the model
        logger.info(f"Saving fine-tuned model to {output_dir}")
        pipeline.save_pretrained(output_dir)
        
        return f"Training complete! Model saved as: {os.path.basename(output_dir)}"
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return f"Error during training: {str(e)}"
