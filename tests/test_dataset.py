"""Tests for dataset functionality."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from windsurf_dreambooth.training.dataset import DreamBoothDataset, ValidationDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    model_max_length = 77

    def __call__(self, text, **kwargs):
        # Return mock token IDs
        return type(
            "obj", (object,), {"input_ids": torch.randint(0, 1000, (1, self.model_max_length))}
        )


def create_test_image(path: Path, size=(512, 512)):
    """Create a test image."""
    img = Image.new("RGB", size, color="red")
    img.save(path)
    return path


class TestDreamBoothDataset:
    """Test DreamBooth dataset functionality."""

    def test_dataset_creation(self, tmp_path):
        """Test dataset can be created with valid images."""
        # Create test images
        img_paths = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.jpg"
            create_test_image(img_path)
            img_paths.append(img_path)

        # Create dataset
        dataset = DreamBoothDataset(
            instance_images=img_paths,
            instance_prompt="a photo of sks person",
            tokenizer=MockTokenizer(),
            size=512,
        )

        assert len(dataset) == 3

    def test_dataset_filters_invalid_images(self, tmp_path):
        """Test dataset filters out invalid images."""
        # Create valid image
        valid_img = tmp_path / "valid.jpg"
        create_test_image(valid_img)

        # Create invalid paths
        missing_img = tmp_path / "missing.jpg"

        # Create invalid format
        invalid_img = tmp_path / "invalid.txt"
        invalid_img.write_text("not an image")

        # Create dataset
        dataset = DreamBoothDataset(
            instance_images=[valid_img, missing_img, invalid_img],
            instance_prompt="test prompt",
            tokenizer=MockTokenizer(),
        )

        assert len(dataset) == 1

    def test_dataset_getitem(self, tmp_path):
        """Test dataset returns correct format."""
        img_path = tmp_path / "test.jpg"
        create_test_image(img_path)

        dataset = DreamBoothDataset(
            instance_images=[img_path], instance_prompt="test prompt", tokenizer=MockTokenizer()
        )

        item = dataset[0]

        assert "pixel_values" in item
        assert "input_ids" in item
        assert isinstance(item["pixel_values"], torch.Tensor)
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_dataset_with_class_images(self, tmp_path):
        """Test dataset with prior preservation."""
        # Create instance and class images
        instance_img = tmp_path / "instance.jpg"
        class_img = tmp_path / "class.jpg"
        create_test_image(instance_img)
        create_test_image(class_img)

        dataset = DreamBoothDataset(
            instance_images=[instance_img],
            instance_prompt="a photo of sks person",
            class_images=[class_img],
            class_prompt="a photo of person",
            tokenizer=MockTokenizer(),
        )

        item = dataset[0]

        assert item["pixel_values"].shape[0] == 2  # Should have both instance and class
        assert item["input_ids"].shape[0] == 2

    def test_validation_dataset(self):
        """Test validation dataset."""
        prompts = ["test prompt 1", "test prompt 2"]
        dataset = ValidationDataset(prompts, MockTokenizer())

        assert len(dataset) == 2

        item = dataset[0]
        assert "prompt" in item
        assert "input_ids" in item
        assert item["prompt"] == "test prompt 1"


if __name__ == "__main__":
    pytest.main([__file__])
