"""FastAPI application for REST API endpoints."""

import asyncio
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config.constants import PIPELINE_TYPES, SCHEDULERS, SUPPORTED_MODELS
from ..models.generator import ImageGenerator
from ..models.manager import DreamBoothManager
from ..training.trainer import DreamBoothTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Windsurf DreamBooth API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_manager = DreamBoothManager()
active_training_tasks = {}
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Pydantic models
class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_model: str = Field(..., description="Base model to fine-tune")
    instance_prompt: str = Field(..., description="Prompt with subject identifier")
    class_prompt: Optional[str] = Field(None, description="Class prompt for preservation")
    num_train_steps: int = Field(1000, ge=100, le=10000)
    learning_rate: float = Field(5e-6, gt=0, le=1e-3)
    train_batch_size: int = Field(1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(1, ge=1, le=16)
    mixed_precision: str = Field("no", pattern="^(no|fp16|bf16)$")
    use_cpu_offload: bool = Field(False)
    output_dir: Optional[str] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prompt: str
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = Field(50, ge=1, le=200)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_images: int = Field(1, ge=1, le=10)
    width: Optional[int] = Field(None, ge=64, le=2048)
    height: Optional[int] = Field(None, ge=64, le=2048)
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    precision: str = Field("bf16", pattern="^(bf16|bf32|f16|f32)$")
    use_cpu_offload: bool = False


class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    pipeline_type: str
    loaded: bool
    size_gb: Optional[float] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}


@app.get("/models")
async def list_models():
    """List available models."""
    models = []
    for model_name in SUPPORTED_MODELS:
        # Determine appropriate pipeline type based on model name
        if model_name == "custom":
            pipeline_type = "Generic"
        elif "xl" in model_name:
            pipeline_type = "StableDiffusionXL"
        else:
            pipeline_type = "StableDiffusion"

        loaded = model_name in model_manager._models
        models.append(ModelInfo(name=model_name, pipeline_type=pipeline_type, loaded=loaded))
    return {"models": models}


@app.post("/upload-images/")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload training images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Validate files
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} has unsupported extension. Allowed: {allowed_extensions}",
            )

    # Create unique session directory
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save files
    saved_files = []
    try:
        for file in files:
            file_path = session_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(str(file_path))

    except Exception as e:
        # Cleanup on error
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")

    return {"session_id": session_id, "files": saved_files, "count": len(saved_files)}


async def train_model_task(task_id: str, session_id: str, request: TrainRequest):
    """Background task for model training."""
    training_status = active_training_tasks[task_id]
    training_status["status"] = "running"

    try:
        # Get training images
        session_dir = UPLOAD_DIR / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")

        image_paths = list(session_dir.glob("*"))
        if not image_paths:
            raise ValueError("No images found in session")

        # Initialize trainer
        trainer = DreamBoothTrainer(
            model_name=request.base_model,
            instance_prompt=request.instance_prompt,
            class_prompt=request.class_prompt,
            output_dir=request.output_dir or f"./finetuned_models/{task_id}",
            num_train_steps=request.num_train_steps,
            learning_rate=request.learning_rate,
            train_batch_size=request.train_batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            mixed_precision=request.mixed_precision,
            use_cpu_offload=request.use_cpu_offload,
        )

        # Train with progress callback
        def progress_callback(step, total_steps):
            progress = step / total_steps
            training_status["progress"] = progress
            training_status["message"] = f"Training step {step}/{total_steps}"

        trainer.train(instance_images=image_paths, progress_callback=progress_callback)

        # Success
        training_status["status"] = "completed"
        training_status["completed_at"] = datetime.now()
        training_status["message"] = "Training completed successfully"
        training_status["progress"] = 1.0

    except Exception as e:
        logger.error(f"Training failed for task {task_id}: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
        training_status["completed_at"] = datetime.now()

    finally:
        # Cleanup uploaded files
        shutil.rmtree(UPLOAD_DIR / session_id, ignore_errors=True)


@app.post("/train-model/")
async def train_model(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    base_model: str = Form(...),
    instance_prompt: str = Form(...),
    class_prompt: Optional[str] = Form(None),
    num_train_steps: int = Form(1000),
    learning_rate: float = Form(5e-6),
    train_batch_size: int = Form(1),
    gradient_accumulation_steps: int = Form(1),
    mixed_precision: str = Form("no"),
    use_cpu_offload: bool = Form(False),
    output_dir: Optional[str] = Form(None),
):
    """Start model training."""
    # Validate session
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Create training request
    request = TrainRequest(
        model_name=base_model,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        use_cpu_offload=use_cpu_offload,
        output_dir=output_dir,
    )

    # Create task
    task_id = str(uuid.uuid4())
    training_status = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Training queued",
        "started_at": datetime.now(),
        "completed_at": None,
        "error": None,
    }
    active_training_tasks[task_id] = training_status

    # Start background task
    background_tasks.add_task(train_model_task, task_id, session_id, request)

    return TrainingStatus(**training_status)


@app.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    """Get training task status."""
    if task_id not in active_training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return TrainingStatus(**active_training_tasks[task_id])


@app.post("/generate/")
async def generate_images(request: GenerateRequest):
    """Generate images."""
    try:
        # Load model if needed
        model_manager.load_model(
            request.base_model, precision=request.precision, use_cpu_offload=request.use_cpu_offload
        )

        # Initialize generator
        generator = ImageGenerator(model_manager)

        # Generate images
        images = generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            width=request.width,
            height=request.height,
            seed=request.seed,
            scheduler=request.scheduler,
        )

        # Save images
        output_paths = []
        batch_id = str(uuid.uuid4())
        batch_dir = OUTPUT_DIR / batch_id
        batch_dir.mkdir(exist_ok=True)

        for i, image in enumerate(images):
            filename = f"image_{i}.png"
            filepath = batch_dir / filename
            image.save(filepath)
            output_paths.append(f"/outputs/{batch_id}/{filename}")

        return {"batch_id": batch_id, "images": output_paths, "count": len(images)}

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outputs/{batch_id}/{filename}")
async def get_output_image(batch_id: str, filename: str):
    """Serve generated images."""
    # Validate path to prevent traversal
    if ".." in batch_id or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    filepath = OUTPUT_DIR / batch_id / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filepath, media_type="image/png")


@app.delete("/outputs/{batch_id}")
async def delete_output_batch(batch_id: str):
    """Delete generated image batch."""
    if ".." in batch_id:
        raise HTTPException(status_code=400, detail="Invalid batch ID")

    batch_dir = OUTPUT_DIR / batch_id
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
        return {"message": "Batch deleted"}

    raise HTTPException(status_code=404, detail="Batch not found")


@app.post("/clear-cache")
async def clear_model_cache():
    """Clear model cache."""
    try:
        model_manager.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
