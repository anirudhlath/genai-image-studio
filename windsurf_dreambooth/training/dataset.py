"""Dataset implementations for DreamBooth training with improved error handling and validation."""

from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.logging import get_logger

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)


class DreamBoothDataset(Dataset):
    """Dataset class for DreamBooth fine-tuning with robust image handling and augmentation support."""

    def __init__(
        self,
        instance_images: list[Union[str, Path]],
        instance_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = True,
        class_images: Optional[list[Union[str, Path]]] = None,
        class_prompt: Optional[str] = None,
        random_flip: bool = True,
        color_jitter: bool = True,
    ):
        """Initialize the DreamBooth dataset with validation and augmentation options.

        Args:
            instance_images: List of paths to instance training images
            instance_prompt: Text prompt for instance images
            tokenizer: Tokenizer from the model being fine-tuned
            size: Size to resize images to (square)
            center_crop: Whether to center crop images
            class_images: Optional list of class images for prior preservation
            class_prompt: Optional prompt for class images
            random_flip: Enable random horizontal flipping
            color_jitter: Enable color jittering augmentation
        """
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # Validate and filter instance images
        self.instance_images = self._validate_images(instance_images, "instance")
        self.instance_prompt = instance_prompt

        if not self.instance_images:
            raise ValueError("No valid instance images found")

        # Handle class images for prior preservation
        self.class_images = []
        self.class_prompt = class_prompt
        if class_images and class_prompt:
            self.class_images = self._validate_images(class_images, "class")
            logger.info(f"Using {len(self.class_images)} class images for prior preservation")

        # Setup transforms
        self.instance_transform = self._create_transform(
            training=True, random_flip=random_flip, color_jitter=color_jitter
        )

        self.class_transform = self._create_transform(
            training=True,
            random_flip=random_flip,
            color_jitter=False,  # Less augmentation for class images
        )

        # Pre-compute dataset length
        self._length = len(self.instance_images)
        if self.class_images:
            self._length = max(len(self.instance_images), len(self.class_images))

        logger.info(f"Dataset initialized with {len(self.instance_images)} instance images")

    def _validate_images(self, image_paths: list[Union[str, Path]], image_type: str) -> list[Path]:
        """Validate and filter image paths."""
        valid_paths = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for path_str in image_paths:
            path = Path(path_str)

            # Check if file exists
            if not path.exists():
                logger.warning(f"{image_type} image not found: {path}")
                continue

            # Check extension
            if path.suffix.lower() not in valid_extensions:
                logger.warning(f"Unsupported {image_type} image format: {path}")
                continue

            # Try to open and validate image
            try:
                with Image.open(path) as img:
                    # Check if image is valid
                    img.verify()

                    # Re-open for actual checks (verify closes the file)
                    with Image.open(path) as img_check:
                        # Check minimum size
                        if min(img_check.size) < 64:
                            logger.warning(
                                f"{image_type} image too small: {path} ({img_check.size})"
                            )
                            continue

                        # Check if image is corrupted
                        img_check.load()

                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to validate {image_type} image {path}: {e}")
                continue

        return valid_paths

    def _create_transform(
        self, training: bool = True, random_flip: bool = True, color_jitter: bool = True
    ):
        """Create image transformation pipeline."""
        transform_list = []

        # Resize with aspect ratio preservation
        transform_list.append(
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        )

        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.size))

        # Training augmentations
        if training:
            if random_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

            if color_jitter:
                transform_list.append(
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
                )

        # Convert to tensor and normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        return transforms.Compose(transform_list)

    def _load_and_transform_image(self, image_path: Path, transform) -> torch.Tensor:
        """Load and transform an image with error handling."""
        try:
            image = Image.open(image_path).convert("RGB")
            return transform(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a placeholder image
            placeholder = Image.new("RGB", (self.size, self.size), (128, 128, 128))
            return transform(placeholder)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a text prompt."""
        return self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._length

    def __getitem__(self, index: int) -> dict:
        """Get a training sample.

        Returns:
            Dictionary with pixel_values and input_ids for the model
        """
        example = {}

        # Get instance image
        instance_index = index % len(self.instance_images)
        instance_image = self._load_and_transform_image(
            self.instance_images[instance_index], self.instance_transform
        )
        instance_input_ids = self._tokenize_prompt(self.instance_prompt)

        example["instance_images"] = instance_image
        example["instance_input_ids"] = instance_input_ids

        # Get class image if using prior preservation
        if self.class_images:
            class_index = index % len(self.class_images)
            class_image = self._load_and_transform_image(
                self.class_images[class_index], self.class_transform
            )
            class_input_ids = self._tokenize_prompt(self.class_prompt)

            example["class_images"] = class_image
            example["class_input_ids"] = class_input_ids

            # Combine for training
            example["pixel_values"] = torch.stack([instance_image, class_image])
            example["input_ids"] = torch.stack([instance_input_ids, class_input_ids])
        else:
            # No prior preservation
            example["pixel_values"] = instance_image
            example["input_ids"] = instance_input_ids

        return example


class ValidationDataset(Dataset):
    """Simple dataset for validation during training."""

    def __init__(self, prompts: list[str], tokenizer):
        """Initialize validation dataset with prompts."""
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        prompt = self.prompts[index]
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"prompt": prompt, "input_ids": input_ids}
