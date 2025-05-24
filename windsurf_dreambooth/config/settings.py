"""Application settings and configuration."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")  # nosec B104
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")  # Optimized for 5800X3D (8 cores)
    cors_origins: list[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")

    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    api_key_header: str = Field("X-API-Key", env="API_KEY_HEADER")
    require_api_key: bool = Field(False, env="REQUIRE_API_KEY")

    # Model Settings
    default_model: str = Field("stabilityai/stable-diffusion-xl-base-1.0", env="DEFAULT_MODEL")
    model_cache_dir: Path = Field(Path("./model_cache"), env="MODEL_CACHE_DIR")
    max_cached_models: int = Field(6, env="MAX_CACHED_MODELS")  # RTX 4090 has 24GB VRAM

    # Training Settings
    default_learning_rate: float = Field(5e-6, env="DEFAULT_LEARNING_RATE")
    default_train_steps: int = Field(1000, env="DEFAULT_TRAIN_STEPS")
    default_batch_size: int = Field(
        4, env="DEFAULT_BATCH_SIZE"
    )  # RTX 4090 can handle larger batches
    max_train_steps: int = Field(10000, env="MAX_TRAIN_STEPS")
    gradient_clip_val: float = Field(1.0, env="GRADIENT_CLIP_VAL")

    # Storage Settings
    upload_dir: Path = Field(Path("./uploads"), env="UPLOAD_DIR")
    output_dir: Path = Field(Path("./outputs"), env="OUTPUT_DIR")
    finetuned_models_dir: Path = Field(Path("./finetuned_models"), env="FINETUNED_MODELS_DIR")
    max_upload_size_mb: int = Field(50, env="MAX_UPLOAD_SIZE_MB")

    # Performance Settings
    enable_memory_efficient_attention: bool = Field(True, env="ENABLE_MEMORY_EFFICIENT_ATTENTION")
    enable_vae_slicing: bool = Field(False, env="ENABLE_VAE_SLICING")  # RTX 4090 has enough VRAM
    enable_vae_tiling: bool = Field(False, env="ENABLE_VAE_TILING")
    dataloader_num_workers: int = Field(
        8, env="DATALOADER_NUM_WORKERS"
    )  # 5800X3D has 8 cores/16 threads
    use_safetensors: bool = Field(True, env="USE_SAFETENSORS")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )

    # Gradio Settings
    gradio_server_name: str = Field("0.0.0.0", env="GRADIO_SERVER_NAME")  # nosec B104
    gradio_server_port: int = Field(7860, env="GRADIO_SERVER_PORT")
    gradio_share: bool = Field(False, env="GRADIO_SHARE")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def setup_directories(self):
        """Create required directories."""
        for dir_path in [
            self.upload_dir,
            self.output_dir,
            self.finetuned_models_dir,
            self.model_cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Create settings instance
try:
    settings = Settings()
except Exception:
    # Fallback for missing SECRET_KEY
    settings = Settings(secret_key=os.urandom(32).hex())


# Add device property using a wrapper to avoid pydantic validation
class SettingsWrapper:
    def __init__(self, settings):
        self._settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getattr__(self, name):
        return getattr(self._settings, name)


# Wrap settings
settings = SettingsWrapper(settings)

# Setup directories
settings.setup_directories()
