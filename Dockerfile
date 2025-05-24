# Multi-stage build for smaller, more secure image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy only necessary files
COPY pyproject.toml .
COPY windsurf_dreambooth/ ./windsurf_dreambooth/
COPY main.py .

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/outputs /app/finetuned_models /app/model_cache /tmp/gradio \
    && chown -R appuser:appuser /app /tmp/gradio

# Install dependencies
RUN uv sync --no-dev && \
    rm -rf /root/.cache/uv

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8000 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Security: Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use exec form for better signal handling
ENTRYPOINT ["uv", "run"]
CMD ["python", "main.py"]