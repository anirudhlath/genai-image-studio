# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DreamBooth fine-tuning application that provides a web interface for training and generating images with custom diffusion models. It consists of a Python backend using Gradio/FastAPI and a React frontend.

## Commands

### Backend Development
```bash
# Install dependencies
uv sync  # Preferred
# or
pip install -e .

# Run the application
uv run main.py  # Starts Gradio interface on port 8000
uv run main.py --mode api  # REST API only
uv run main.py --mode both  # Both Gradio and API
# or
python -m windsurf_dreambooth

# Run tests
uv run pytest
uv run pytest --cov=windsurf_dreambooth
make test-fast  # Run tests excluding slow ones
make test-parallel  # Run tests in parallel

# Linting and formatting
make lint  # Run flake8, pylint, and mypy
make format  # Format with black and isort
uv run ruff check .  # Fast linting
uv run ruff check --fix .  # Auto-fix linting issues

# Demo progress indicators
uv run python demo_progress.py

# Run with Docker
docker build -t windsurf-dreambooth .
docker run -p 8000:8000 --gpus all windsurf-dreambooth
```

### Frontend Development
```bash
cd frontend
npm install
npm start    # Development server on port 3000
npm build    # Production build
npm test     # Run tests (no test files currently)
```

## Architecture

### Core Modules
- **windsurf_dreambooth/models/manager.py**: Model management with LRU cache and memory-aware eviction
- **windsurf_dreambooth/models/generator.py**: Image generation with batch support and progress tracking
- **windsurf_dreambooth/training/trainer.py**: Memory-efficient training with gradient checkpointing and progress callbacks
- **windsurf_dreambooth/ui/app.py**: Gradio interface with real-time progress indicators
- **windsurf_dreambooth/api/app.py**: FastAPI REST endpoints with WebSocket support for progress updates
- **windsurf_dreambooth/api/auth.py**: Security middleware with rate limiting and API key authentication

### Model Pipeline Support
The application supports multiple diffusion model types defined in `config/constants.py`:
- Stable Diffusion (SD)
- Stable Diffusion XL (SDXL)
- Stable Diffusion 3 (SD3)
- Flux

### Key Design Patterns
1. **Model Caching**: DreamBoothManager implements singleton pattern for efficient model reuse
2. **Pipeline Factory**: Dynamic pipeline creation based on model type
3. **Memory Optimization**: Supports CPU offloading and different precision modes (bf16, f16, f32)
4. **Component-Based UI**: Gradio components organized in `ui/components.py`

### Configuration
- Model mappings and supported pipelines: `config/constants.py`
- Default models and scheduler options are predefined
- Supports both HuggingFace model IDs and local model paths

## Development Notes

- The application requires CUDA-capable GPU for training
- Models are cached with LRU eviction based on memory usage
- Frontend communicates with backend via REST API on port 8000
- Gradio interface includes progress indicators for all long-running operations
- Use UV package manager for faster dependency resolution
- All file uploads are validated and sanitized for security
- Training includes checkpoint saving and resume capabilities
- API endpoints support real-time progress updates via polling

## UI Design Guidelines

For UI development in this project, refer to:
- `ui-design-guidelines.md` - General UI principles and patterns
- `gradio-ui-best-practices.md` - Gradio-specific implementation patterns

Key principles followed in this project:
- Global settings outside tabs to eliminate duplication
- Balanced column layouts with proper scaling (scale=2, scale=2, scale=3)
- Advanced settings grouped in secondary sections with `gr.Group()`
- Clear, concise labels and info text
- Logical information hierarchy: Global → Specific functionality

## Testing Infrastructure

The project includes comprehensive testing with 40+ dev dependencies:
- **pytest** with markers for slow, gpu, integration, unit, flaky tests
- **hypothesis** for property-based testing
- **pytest-mock** and **factory-boy** for mocking and fixtures
- **pytest-xdist** for parallel test execution
- **pytest-benchmark** for performance testing
- **freezegun** and **faker** for test data generation

## Security Features

- JWT-based authentication for API endpoints
- Rate limiting and request validation
- Input sanitization in dataset handling
- Non-root Docker container execution
- Environment-based configuration with pydantic-settings
- Security scanning with bandit and safety

## Progress Indicators

The application includes comprehensive progress tracking:

1. **Upload Progress**: Shows file-by-file upload status with percentage
2. **Training Progress**: Real-time training steps, loss values, and ETA
3. **Generation Progress**: Step-by-step denoising progress
4. **Loading States**: Model loading stages and cache status

Both Gradio and React frontends display these indicators with:
- Progress bars (linear and circular)
- Status messages and ETAs
- Skeleton loaders for pending content
- Animation effects (fade, grow) for smooth transitions
