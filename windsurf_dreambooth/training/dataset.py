"""
Dataset implementations for DreamBooth training
"""
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DreamBoothDataset(Dataset):
    """
    Dataset class for DreamBooth fine-tuning that handles image transformations
    and token generation for training
    """
    def __init__(self, image_paths, tokenizer, prompt, size, device):
        """
        Initialize the DreamBooth dataset
        
        Args:
            image_paths: List of paths to training images
            tokenizer: Tokenizer from the model being fine-tuned
            prompt: Text prompt to associate with the images
            size: Size to resize images to (square)
            device: Device to put tensors on (cuda/cpu)
        """
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.size = size
        self.device = device
        
        # Standard image transformations for diffusion models
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary with pixel_values and input_ids for the model
        """
        # Load and transform the image
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            pixel_values = self.transform(img).to(self.device)
        except Exception as e:
            from ..utils.logging import logger
            logger.error(f"Failed to load image at {self.image_paths[idx]}: {str(e)}")
            # Return a black image as fallback
            img = Image.new("RGB", (self.size, self.size), (0, 0, 0))
            pixel_values = self.transform(img).to(self.device)
        
        # Tokenize the prompt
        input_ids = (
            self.tokenizer(
                self.prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            .input_ids.squeeze(0)
            .to(self.device)
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
