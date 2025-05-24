# DreamBooth Studio

A modular web application that allows users to train custom Diffusion models using DreamBooth and generate images based on their personal style.

## Features

- **Model Training**
  - Upload multiple images for DreamBooth training
  - Support for multiple pipeline types (SD, SDXL, SD3, Flux)
  - Memory-efficient training with gradient checkpointing
  - Mixed precision training (fp16/bf16)
  - Checkpoint saving and resume capabilities
  - Prior preservation loss support

- **Image Generation**
  - REST API and Gradio UI interfaces
  - Batch generation support
  - Multiple scheduler options
  - Flexible precision settings
  - CPU offload for low-memory systems

- **Performance & Security**
  - LRU model caching with memory-aware eviction
  - Automatic GPU memory management
  - Rate limiting and API key authentication
  - Input validation and sanitization
  - Security headers and CORS support

- **Developer Experience**
  - Comprehensive configuration via environment variables
  - Extensive logging and error handling
  - Unit tests with pytest
  - Docker support with multi-stage builds
  - CI/CD with GitHub Actions

## Prerequisites

- Python 3.8-3.11
- NVIDIA GPU with CUDA support (recommended)
- CPU-only mode supported but very slow
- Docker (optional, for containerized deployment)

## Installation

### Using UV (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd windsurf-project

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .env file from template
cp .env.example .env
# Edit .env and set SECRET_KEY

# Install dependencies
uv sync

# Run the application
uv run python main.py --mode both  # Runs both Gradio UI and REST API
```

### Using pip
```bash
pip install -e .
python main.py
```

### Using Docker
```bash
docker build -t windsurf-dreambooth .
docker run -p 8000:8000 --gpus all windsurf-dreambooth
```

## Running Modes

```bash
# Gradio UI only (default)
python main.py

# REST API only
python main.py --mode api

# Both Gradio and API
python main.py --mode both

# Create .env template
python main.py --create-env
```

## API Endpoints

When running in API mode, the following endpoints are available:

- `GET /health` - Health check
- `GET /models` - List available models
- `POST /upload-images/` - Upload training images
- `POST /train-model/` - Start model training
- `GET /training-status/{task_id}` - Check training status
- `POST /generate/` - Generate images
- `GET /outputs/{batch_id}/{filename}` - Get generated images
- `DELETE /outputs/{batch_id}` - Delete image batch
- `POST /clear-cache` - Clear model cache

## Project Structure

```
windsurf_dreambooth/            # Main package
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point when running as module
├── api/                        # REST API
│   ├── app.py                  # FastAPI application
│   └── auth.py                 # Authentication and security
├── config/                     # Configuration settings
│   ├── constants.py            # Constants and settings
│   └── settings.py             # Application settings with env support
├── models/                     # Model management
│   ├── generator.py            # Image generation functionality
│   └── manager.py              # Model loading with LRU cache
├── training/                   # Training functionality
│   ├── dataset.py              # Dataset with validation
│   └── trainer.py              # Memory-efficient training
├── ui/                         # User interface
│   ├── app.py                  # Gradio interface
│   └── components.py           # UI components
└── utils/                      # Utilities
    └── logging.py              # Logging configuration

frontend/                       # React frontend
├── src/
│   └── App.js                  # Main React app
└── package.json                # Frontend dependencies

tests/                          # Unit tests
├── test_dataset.py             # Dataset tests
├── test_model_cache.py         # Cache tests
└── test_auth.py                # Security tests
```

## Running the Application

### Local Development

Use the main entry point:

```bash
python main.py
```

Or use UV package manager:

```bash
uv run main.py
```

Or run it as a module:

```bash
python -m windsurf_dreambooth
```

The application will be available at http://localhost:8000

### Using Docker

Build and run the Docker container:
```bash
docker build -t dreambooth-studio .
docker run -d --gpus all -p 8000:8000 dreambooth-studio
```

## Usage

1. **Train Tab**: Upload multiple images of the same subject, select a base model, and click "Train" to start the DreamBooth training process
2. **Generate Tab**: Select a model (either base or fine-tuned), enter a prompt, customize generation parameters, and click "Generate"
3. **Models Tab**: View, load, or delete your fine-tuned models

## Key Components

- **DreamBoothManager**: Handles model loading, precision settings, and caching
- **DreamBoothDataset**: Prepares training images with necessary transformations
- **UI Components**: Modular UI structure for easier customization

## Configuration

All settings can be configured via environment variables. See `.env.example` for available options:

- `SECRET_KEY` - Required for security
- `DEFAULT_MODEL` - Default model to load
- `MAX_CACHED_MODELS` - Number of models to keep in memory
- `DEFAULT_LEARNING_RATE` - Training learning rate
- `ENABLE_MEMORY_EFFICIENT_ATTENTION` - Enable xformers
- And many more...

## Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=windsurf_dreambooth

# Run linting
uv run flake8 windsurf_dreambooth
uv run black windsurf_dreambooth
uv run mypy windsurf_dreambooth
```

## Recent Improvements

### Security Enhancements
- Added authentication middleware with API key support
- Implemented rate limiting to prevent abuse
- Added security headers (CSP, X-Frame-Options, etc.)
- Input validation and path traversal protection
- Non-root Docker user for better container security

### Performance Optimizations
- Implemented LRU cache with memory-aware eviction
- Fixed memory leak in training dataloader
- Added gradient checkpointing and mixed precision training
- Enabled xformers memory-efficient attention
- Optimized Docker image with multi-stage build

### API & Integration
- Created comprehensive REST API for frontend integration
- Added WebSocket support for real-time training updates
- Implemented proper CORS handling
- Added health check endpoints
- Frontend proxy configuration for seamless development

### Code Quality
- Added dependency version constraints for security
- Implemented comprehensive error handling
- Created unit tests for core functionality
- Added CI/CD pipeline with GitHub Actions
- Improved logging throughout the application

## Note

The DreamBooth training process can be memory-intensive. The application now includes:
- Automatic memory management with cache eviction
- Gradient checkpointing to reduce memory usage
- CPU offload options for systems with limited GPU memory
- Configurable batch sizes and gradient accumulation
