FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    snapd \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# RUN uv venv

# Install PyTorch GPU build via UV
# RUN uv pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch
# RUN uv pip install setuptools

# Copy project.toml and application code
# COPY pyproject.toml .
COPY . .

# Install dependencies via UV
RUN uv sync

# Ensure Gradio cache directory exists to handle file uploads
RUN mkdir -p /tmp/gradio

# Expose port
EXPOSE 8000

# Start the application via UV
CMD ["uv", "run", "main.py"]
