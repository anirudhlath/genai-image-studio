"""
Training functionality for DreamBooth fine-tuning with improved memory efficiency and error handling.
"""

import gc
import os
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from ..config.settings import settings
from ..utils.logging import get_logger
from .dataset import DreamBoothDataset

logger = get_logger(__name__)


class DreamBoothTrainer:
    """DreamBooth trainer with memory-efficient training and progress tracking."""

    def __init__(
        self,
        model_name: str,
        instance_prompt: str,
        class_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_train_steps: int = 1000,
        learning_rate: float = 5e-6,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "no",
        use_cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        dataloader_num_workers: int = 4,
        save_steps: int = 500,
        save_total_limit: int = 2,
        resume_from_checkpoint: Optional[str] = None,
        validation_prompt: Optional[str] = None,
        validation_steps: int = 100,
        enable_xformers: bool = True,
    ):
        """Initialize DreamBooth trainer with configuration."""
        self.model_name = model_name
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.output_dir = Path(output_dir or f"./finetuned_models/{uuid.uuid4().hex[:8]}")
        self.num_train_steps = min(num_train_steps, settings.max_train_steps)
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.use_cpu_offload = use_cpu_offload
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.dataloader_num_workers = dataloader_num_workers
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.resume_from_checkpoint = resume_from_checkpoint
        self.validation_prompt = validation_prompt
        self.validation_steps = validation_steps
        self.enable_xformers = enable_xformers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir=str(self.output_dir),
        )

        # Set seed for reproducibility
        set_seed(seed)

        self.pipeline = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None

    def _load_pipeline(self):
        """Load the base model pipeline."""
        from ..models.manager import DreamBoothManager

        manager = DreamBoothManager()
        self.pipeline = manager.load_model(
            self.model_name,
            precision="f16" if self.mixed_precision == "fp16" else "f32",
            use_cpu_offload=self.use_cpu_offload,
        )

        # Enable memory efficient attention
        if self.enable_xformers and hasattr(
            self.pipeline.unet, "enable_xformers_memory_efficient_attention"
        ):
            try:
                self.pipeline.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Failed to enable xformers: {e}")

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self.pipeline.unet.enable_gradient_checkpointing()
            if hasattr(self.pipeline, "text_encoder"):
                self.pipeline.text_encoder.gradient_checkpointing_enable()

        # Enable VAE slicing for memory efficiency
        if hasattr(self.pipeline, "vae"):
            self.pipeline.vae.enable_slicing()
            if settings.enable_vae_tiling:
                self.pipeline.vae.enable_tiling()

    def _create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        # Prepare parameters for training
        params_to_optimize = list(self.pipeline.unet.parameters())
        if hasattr(self.pipeline, "text_encoder") and self.class_prompt:
            params_to_optimize.extend(self.pipeline.text_encoder.parameters())

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08,
        )

        # Create scheduler
        self.lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=self.num_train_steps,
        )

        # Create gradient scaler for mixed precision
        if self.mixed_precision == "fp16":
            self.scaler = GradScaler()

    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save models
        self.pipeline.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler states
        torch.save(
            {
                "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            },
            checkpoint_dir / "training_state.pt",
        )

        # Clean up old checkpoints
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1])
        )
        if len(checkpoints) > self.save_total_limit:
            for checkpoint in checkpoints[: -self.save_total_limit]:
                import shutil

                shutil.rmtree(checkpoint)

    def _validate(self, step: int):
        """Run validation and generate sample images."""
        if not self.validation_prompt:
            return

        self.pipeline.unet.eval()

        with torch.no_grad():
            # Generate validation images
            images = self.pipeline(
                self.validation_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                num_images_per_prompt=4,
            ).images

            # Save validation images
            val_dir = self.output_dir / f"validation-{step}"
            val_dir.mkdir(exist_ok=True)
            for i, image in enumerate(images):
                image.save(val_dir / f"image_{i}.png")

        self.pipeline.unet.train()

    def train(
        self,
        instance_images: List[Union[str, Path]],
        class_images: Optional[List[Union[str, Path]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Train the model with provided images."""
        try:
            # Load pipeline
            logger.info(f"Loading base model: {self.model_name}")
            self._load_pipeline()

            # Create dataset
            logger.info(f"Creating dataset with {len(instance_images)} instance images")
            train_dataset = DreamBoothDataset(
                instance_images=instance_images,
                instance_prompt=self.instance_prompt,
                class_images=class_images,
                class_prompt=self.class_prompt,
                tokenizer=self.pipeline.tokenizer,
                size=512 if "xl" not in self.model_name.lower() else 1024,
                center_crop=True,
            )

            # Create dataloader with memory-efficient settings
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
                drop_last=True,
            )

            # Create optimizer and scheduler
            self._create_optimizer_and_scheduler()

            # Prepare for training with accelerator
            if hasattr(self.pipeline, "text_encoder") and self.pipeline.text_encoder is not None:
                (
                    self.pipeline.unet,
                    self.pipeline.text_encoder,
                    self.optimizer,
                    train_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.pipeline.unet,
                    self.pipeline.text_encoder,
                    self.optimizer,
                    train_dataloader,
                    self.lr_scheduler,
                )
            else:
                (
                    self.pipeline.unet,
                    self.optimizer,
                    train_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.pipeline.unet,
                    self.optimizer,
                    train_dataloader,
                    self.lr_scheduler,
                )

            # Resume from checkpoint if specified
            starting_step = 0
            if self.resume_from_checkpoint:
                checkpoint_path = Path(self.resume_from_checkpoint)
                if checkpoint_path.exists():
                    state = torch.load(checkpoint_path / "training_state.pt")
                    starting_step = state["step"]
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                    self.lr_scheduler.load_state_dict(state["scheduler_state_dict"])
                    if self.scaler and state["scaler_state_dict"]:
                        self.scaler.load_state_dict(state["scaler_state_dict"])
                    logger.info(f"Resumed from checkpoint at step {starting_step}")

            # Training loop
            logger.info("Starting training...")
            progress_bar = tqdm(
                range(starting_step, self.num_train_steps),
                desc="Training",
                disable=not self.accelerator.is_local_main_process,
            )

            global_step = starting_step
            train_loss = 0.0

            # Set models to training mode
            self.pipeline.unet.train()
            if hasattr(self.pipeline, "text_encoder") and self.class_prompt:
                self.pipeline.text_encoder.train()

            while global_step < self.num_train_steps:
                for batch in train_dataloader:
                    with self.accelerator.accumulate(self.pipeline.unet):
                        # Convert images to latent space
                        latents = self.pipeline.vae.encode(
                            batch["pixel_values"].to(self.accelerator.device)
                        ).latent_dist.sample()
                        latents = latents * self.pipeline.vae.config.scaling_factor

                        # Sample noise
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        # Sample timesteps
                        timesteps = torch.randint(
                            0,
                            self.pipeline.scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        timesteps = timesteps.long()

                        # Add noise to latents
                        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

                        # Get text embeddings
                        encoder_hidden_states = self.pipeline.text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        if self.mixed_precision == "fp16" and self.scaler:
                            with autocast():
                                model_pred = self.pipeline.unet(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states,
                                ).sample

                                # Calculate loss
                                loss = F.mse_loss(
                                    model_pred.float(), noise.float(), reduction="mean"
                                )

                            # Backward pass with mixed precision
                            self.scaler.scale(loss).backward()

                            if self.accelerator.sync_gradients:
                                self.scaler.unscale_(self.optimizer)
                                self.accelerator.clip_grad_norm_(
                                    self.pipeline.unet.parameters(), self.max_grad_norm
                                )

                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            model_pred = self.pipeline.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states,
                            ).sample

                            # Calculate loss
                            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                            # Backward pass
                            self.accelerator.backward(loss)

                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(
                                    self.pipeline.unet.parameters(), self.max_grad_norm
                                )

                            self.optimizer.step()

                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    # Update progress
                    if self.accelerator.sync_gradients:
                        global_step += 1
                        train_loss += loss.detach().item()

                        # Update progress bar
                        progress_bar.update(1)
                        avg_loss = train_loss / (global_step - starting_step)
                        progress_bar.set_postfix(loss=avg_loss)

                        # Call progress callback
                        if progress_callback:
                            progress_callback(global_step, self.num_train_steps)

                        # Save checkpoint
                        if global_step % self.save_steps == 0:
                            if self.accelerator.is_main_process:
                                self._save_checkpoint(global_step)
                                logger.info(f"Saved checkpoint at step {global_step}")

                        # Run validation
                        if self.validation_steps > 0 and global_step % self.validation_steps == 0:
                            if self.accelerator.is_main_process:
                                self._validate(global_step)

                        # Clear cache periodically
                        if global_step % 100 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    if global_step >= self.num_train_steps:
                        break

            # Save final model
            if self.accelerator.is_main_process:
                logger.info(f"Saving final model to {self.output_dir}")
                self.pipeline.save_pretrained(self.output_dir)

                # Save training info
                training_info = {
                    "model_name": self.model_name,
                    "instance_prompt": self.instance_prompt,
                    "class_prompt": self.class_prompt,
                    "num_train_steps": self.num_train_steps,
                    "learning_rate": self.learning_rate,
                    "train_batch_size": self.train_batch_size,
                    "final_loss": train_loss / (global_step - starting_step),
                }

                import json

                with open(self.output_dir / "training_info.json", "w") as f:
                    json.dump(training_info, f, indent=2)

            logger.info("Training completed successfully!")

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory! Try reducing batch size or enabling CPU offload.")
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_model(
    model_key,
    custom_model,
    subject_name,
    img_paths,
    train_steps,
    batch_size,
    precision,
    pipeline_type,
    cpu_offload,
):
    """
    Legacy training function for backward compatibility.
    """
    # Determine model name
    model_name = custom_model if model_key == "custom" else model_key

    # Map precision to mixed_precision format
    mixed_precision = "fp16" if precision in ["f16", "bf16"] else "no"

    # Create trainer
    trainer = DreamBoothTrainer(
        model_name=model_name,
        instance_prompt=f"a photo of {subject_name}",
        num_train_steps=train_steps,
        train_batch_size=batch_size,
        mixed_precision=mixed_precision,
        use_cpu_offload=cpu_offload,
        learning_rate=settings.default_learning_rate,
    )

    try:
        # Train model
        trainer.train(instance_images=img_paths)
        return f"Training complete! Model saved to: {trainer.output_dir}"
    except Exception as e:
        return f"Error during training: {str(e)}"
