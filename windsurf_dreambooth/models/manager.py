"""
Model management functionality for loading, saving, and deleting models
"""
import os
import shutil
import torch
from ..config.constants import PIPELINE_MAPPING, FINETUNED_MODELS_DIR
from ..utils.logging import logger

class DreamBoothManager:
    """
    Manager class for DreamBooth models handling loading, caching, and precision control
    """
    def __init__(self, max_cache_size=3):
        """Initialize the model manager with device detection and empty pipeline cache"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DreamBoothManager initialized with device {self.device}")
        self.pipelines = {}
        self.max_cache_size = max_cache_size
        self.cache_order = []  # Track order of cached models

    def load(self, key, pipeline_type="Generic", precision="f16", cpu_offload=False):
        """
        Load a model with the specified pipeline type and precision settings
        
        Args:
            key: Model key or HuggingFace model ID
            pipeline_type: Type of pipeline to use (from PIPELINE_MAPPING)
            precision: Precision to use for model weights
            cpu_offload: Whether to enable CPU offloading for memory optimization
            
        Returns:
            The loaded pipeline
        """
        # Return cached pipeline if available
        cache_key = f"{key}_{pipeline_type}_{precision}"
        if cache_key in self.pipelines:
            logger.info(f"Using cached pipeline for {cache_key}")
            # Move to end of cache order (most recently used)
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            return self.pipelines[cache_key]
            
        # Check cache size and evict oldest if necessary
        if len(self.pipelines) >= self.max_cache_size:
            oldest_key = self.cache_order.pop(0)
            logger.info(f"Evicting oldest cached model: {oldest_key}")
            del self.pipelines[oldest_key]
            # Clear CUDA cache after eviction
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        logger.info(f"Loading model {key} with {pipeline_type} pipeline at {precision} precision")
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            logger.info("Clearing CUDA cache before loading new model")
            torch.cuda.empty_cache()
        
        # Handle precision settings
        torch_dtype = None
        if precision.startswith("bf"):
            if precision == "bf16":
                torch_dtype = torch.bfloat16
            elif precision == "bf32":
                torch_dtype = torch.float32
        elif precision.startswith("f"):
            if precision == "f16":
                torch_dtype = torch.float16
            elif precision == "f32":
                torch_dtype = torch.float32
        
        # Set up pipeline class based on type
        if pipeline_type not in PIPELINE_MAPPING:
            logger.warning(f"Unknown pipeline type: {pipeline_type}, falling back to Generic")
            pipeline_class = PIPELINE_MAPPING["Generic"]
        else:
            pipeline_class = PIPELINE_MAPPING[pipeline_type]
            
        # Check if this is a fine-tuned local model
        model_path = os.path.join(FINETUNED_MODELS_DIR, key)
        if os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            model_id = model_path
        else:
            logger.info(f"Loading model from HuggingFace: {key}")
            model_id = key
            
        # Load the pipeline with appropriate settings
        try:
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Apply CPU offloading if requested
            if cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                logger.info("Enabling CPU offloading")
                pipeline.enable_model_cpu_offload()
            else:
                # Move to device if not using offloading
                pipeline = pipeline.to(self.device)
                
            # Cache the loaded pipeline
            self.pipelines[cache_key] = pipeline
            self.cache_order.append(cache_key)
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading model {key}: {str(e)}")
            raise


def list_models():
    """
    List all available fine-tuned models
    
    Returns:
        List of model names
    """
    if not os.path.exists(FINETUNED_MODELS_DIR):
        logger.warning(f"Fine-tuned models directory {FINETUNED_MODELS_DIR} does not exist")
        return []
    
    try:
        return sorted(os.listdir(FINETUNED_MODELS_DIR))
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return []


def delete_model(model_name):
    """
    Delete a fine-tuned model
    
    Args:
        model_name: Name of the model to delete
        
    Returns:
        Tuple of (status message, updated model list)
    """
    from ..config.constants import AVAILABLE_MODELS
    
    if model_name in AVAILABLE_MODELS:
        return f"Cannot delete base model {model_name}", list_models()
        
    path = os.path.join(FINETUNED_MODELS_DIR, model_name)
    
    # Check if the model directory exists
    if not os.path.exists(path):
        logger.warning(f"Model {model_name} not found at {path}")
        return f"Model {model_name} not found", list_models()
    
    try:
        shutil.rmtree(path)
        logger.info(f"Deleted model {model_name}")
        return f"Deleted {model_name}", list_models()
    except Exception as e:
        logger.error(f"Failed to delete model {model_name}: {str(e)}")
        return f"Error deleting {model_name}: {str(e)}", list_models()
