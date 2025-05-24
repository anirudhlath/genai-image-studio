# DreamBooth Studio

A modular web application that allows users to train custom Diffusion models using DreamBooth and generate images based on their personal style.

## Features

- Upload multiple images for training
- Train custom models with multiple pipeline options (StableDiffusion, StableDiffusionXL, StableDiffusion3, Flux)
- Flexible precision options for different memory configurations
- CPU offload support for lower memory requirements
- Customizable scheduler selection
- GPU acceleration support (NVIDIA CUDA)
- Modern, responsive UI

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- CPU-only mode supported but very slow

## Installation

1. Clone the repository
2. Install Python dependencies:
```bash
pip install -e .
```

## Project Structure

The project has been restructured for better maintainability and clarity:

```
windsurf_dreambooth/            # Main package
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point when running as module
├── config/                     # Configuration settings
│   └── constants.py            # Constants and settings
├── models/                     # Model management
│   ├── generator.py            # Image generation functionality
│   └── manager.py              # Model loading and caching
├── training/                   # Training functionality
│   ├── dataset.py              # Dataset implementation
│   └── trainer.py              # Training loop implementation
├── ui/                         # User interface
│   ├── app.py                  # Main app interface
│   └── components.py           # UI components
└── utils/                      # Utilities
    └── logging.py              # Logging configuration
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

## Note

The DreamBooth training process can take several hours depending on the number of images and GPU performance. Make sure you have enough GPU memory available or enable the CPU offload option.
