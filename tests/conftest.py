"""Pytest configuration and shared fixtures."""

import asyncio
from pathlib import Path
import tempfile
from unittest.mock import Mock

from faker import Faker
from hypothesis import strategies as st
import pytest

# Initialize faker
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Create a mock diffusion model."""
    model = Mock()
    model.device = "cpu"
    model.unet = Mock()
    model.vae = Mock()
    model.text_encoder = Mock()
    model.tokenizer = Mock()
    return model


@pytest.fixture
def sample_image_paths(tmp_dir):
    """Create sample image paths for testing."""
    from PIL import Image

    paths = []
    for i in range(5):
        path = tmp_dir / f"test_image_{i}.jpg"
        # Create a simple test image
        img = Image.new(
            "RGB",
            (512, 512),
            color=(fake.random_int(0, 255), fake.random_int(0, 255), fake.random_int(0, 255)),
        )
        img.save(path)
        paths.append(path)

    return paths


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing endpoints."""
    from fastapi.testclient import TestClient

    from windsurf_dreambooth.api.app import app

    return TestClient(app)


# Hypothesis strategies for property-based testing
@st.composite
def image_dimensions(draw):
    """Generate valid image dimensions."""
    # Use sampled_from for more reliable generation
    valid_sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048]
    width = draw(st.sampled_from(valid_sizes))
    height = draw(st.sampled_from(valid_sizes))
    return width, height


@st.composite
def training_params(draw):
    """Generate valid training parameters."""
    return {
        "num_train_steps": draw(st.integers(min_value=100, max_value=5000)),
        "learning_rate": draw(st.floats(min_value=1e-6, max_value=1e-3)),
        "batch_size": draw(st.integers(min_value=1, max_value=8)),
        "gradient_accumulation_steps": draw(st.integers(min_value=1, max_value=16)),
    }


# Markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "model: marks model loading tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark tests based on their location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath) or "test_" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name or "model" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)


# Benchmarking fixtures
@pytest.fixture
def benchmark_data():
    """Provide data for benchmarking tests."""
    return {
        "small_batch": list(range(100)),
        "medium_batch": list(range(1000)),
        "large_batch": list(range(10000)),
    }
