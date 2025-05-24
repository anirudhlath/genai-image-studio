"""Example tests showcasing advanced testing features."""

import time
from datetime import datetime, timedelta

import pytest
from freezegun import freeze_time
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import image_dimensions, training_params


class TestAdvancedFeatures:
    """Demonstrate advanced testing capabilities."""

    @pytest.mark.timeout(5)  # Test must complete within 5 seconds
    def test_with_timeout(self):
        """Test that completes quickly."""
        time.sleep(0.1)
        assert True

    @pytest.mark.slow
    @pytest.mark.skip(reason="Example of slow test")
    def test_slow_operation(self):
        """Example of a slow test that would be skipped in quick runs."""
        time.sleep(10)
        assert True

    @pytest.mark.parametrize(
        "width,height,expected_pixels",
        [
            (64, 64, 4096),
            (128, 128, 16384),
            (512, 512, 262144),
            (1024, 1024, 1048576),
        ],
    )
    def test_image_sizes(self, width, height, expected_pixels):
        """Test various image size calculations."""
        assert width * height == expected_pixels

    @given(image_dimensions())
    @settings(max_examples=10)
    def test_property_based_image_dimensions(self, dimensions):
        """Property-based test for image dimensions."""
        width, height = dimensions
        # Properties that should always hold
        assert width >= 64
        assert height >= 64
        assert width <= 2048
        assert height <= 2048
        assert width % 64 == 0
        assert height % 64 == 0

    @given(training_params())
    @settings(max_examples=10)
    def test_property_based_training_params(self, params):
        """Property-based test for training parameters."""
        assert params["num_train_steps"] >= 100
        assert params["num_train_steps"] <= 5000
        assert params["learning_rate"] > 0
        assert params["batch_size"] >= 1
        assert params["gradient_accumulation_steps"] >= 1

    @freeze_time("2024-01-01 12:00:00")
    def test_with_frozen_time(self):
        """Test with mocked time."""
        start = datetime.now()
        time.sleep(1)  # This won't actually sleep
        end = datetime.now()

        # Time should not have changed
        assert start == end
        assert start.year == 2024
        assert start.month == 1
        assert start.day == 1

    def test_with_faker(self, faker):
        """Test using faker for random data."""
        # Faker is automatically available as a fixture
        name = faker.name()
        email = faker.email()
        text = faker.text()

        assert isinstance(name, str)
        assert "@" in email
        assert len(text) > 0

    @pytest.mark.benchmark
    def test_performance_benchmark(self, benchmark):
        """Benchmark a function's performance."""

        def compute_intensive_task(n):
            return sum(i**2 for i in range(n))

        # Benchmark the function
        result = benchmark(compute_intensive_task, 1000)
        assert result == sum(i**2 for i in range(1000))

    @pytest.mark.xfail(reason="Example of expected failure")
    def test_expected_failure(self):
        """Test that is expected to fail."""
        assert False, "This test is expected to fail"

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(), reason="GPU not available"
    )
    @pytest.mark.gpu
    def test_gpu_required(self):
        """Test that requires GPU."""
        import torch

        assert torch.cuda.is_available()
        device = torch.device("cuda")
        tensor = torch.zeros(1).to(device)
        assert tensor.device.type == "cuda"

    def test_with_mock_time(self, mocker):
        """Test using pytest-mock for mocking."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 1234567890.0

        import time

        assert time.time() == 1234567890.0

    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    def test_flaky_operation(self):
        """Test that might fail intermittently (would retry 3 times)."""
        import random

        # This would normally be a flaky operation
        assert random.random() > 0.1  # 90% chance of success


class TestFixtureExamples:
    """Examples using custom fixtures."""

    def test_with_temp_directory(self, tmp_dir):
        """Test using temporary directory fixture."""
        test_file = tmp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

    def test_with_sample_images(self, sample_image_paths):
        """Test using sample image paths fixture."""
        assert len(sample_image_paths) == 5
        for path in sample_image_paths:
            assert path.exists()
            assert path.suffix == ".jpg"

    def test_with_mock_model(self, mock_model):
        """Test using mock model fixture."""
        assert mock_model.device == "cpu"
        assert hasattr(mock_model, "unet")
        assert hasattr(mock_model, "vae")
        assert hasattr(mock_model, "text_encoder")


@pytest.mark.integration
class TestIntegrationExample:
    """Example integration tests."""

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test asynchronous operations."""
        import asyncio

        async def async_task():
            await asyncio.sleep(0.1)
            return "completed"

        result = await async_task()
        assert result == "completed"

    def test_api_endpoint_mock(self, mock_api_client):
        """Test API endpoint with mocked responses."""
        # Test the health endpoint
        response = mock_api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Test models endpoint
        response = mock_api_client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


# Example of custom pytest plugin
def pytest_generate_tests(metafunc):
    """Generate test cases dynamically."""
    if "dynamic_value" in metafunc.fixturenames:
        metafunc.parametrize("dynamic_value", [1, 2, 3, 4, 5])


def test_dynamic_generation(dynamic_value):
    """Test generated dynamically by pytest_generate_tests."""
    assert 1 <= dynamic_value <= 5
