"""Tests for model cache functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from windsurf_dreambooth.models.manager import ModelCache


def create_mock_pipeline():
    """Create a properly mocked pipeline for testing."""
    mock_pipeline = Mock()

    # Create mock parameters that can be iterated
    mock_param = Mock()
    mock_param.numel.return_value = 1000000  # 1M parameters
    mock_param.element_size.return_value = 4  # 4 bytes per parameter

    # Make parameters() return an iterable list
    mock_pipeline.unet.parameters.return_value = [mock_param]

    # Set other attributes to None to simulate missing components
    mock_pipeline.text_encoder = None
    mock_pipeline.text_encoder_2 = None
    mock_pipeline.vae = None

    return mock_pipeline


class TestModelCache:
    """Test model cache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = ModelCache(max_size=2, memory_threshold=0.8)

        assert cache.max_size == 2
        assert cache.memory_threshold == 0.8
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self):
        """Test putting and getting models from cache."""
        cache = ModelCache(max_size=3)

        # Create mock pipeline
        mock_pipeline = create_mock_pipeline()

        # Put model in cache
        cache.put("model1", mock_pipeline)

        # Get model from cache
        retrieved = cache.get("model1")
        assert retrieved is mock_pipeline

    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ModelCache(max_size=2)

        # Create mock pipelines
        mock1 = create_mock_pipeline()
        mock2 = create_mock_pipeline()
        mock3 = create_mock_pipeline()

        # Fill cache
        cache.put("model1", mock1)
        cache.put("model2", mock2)

        # Access model1 to make it more recent
        cache.get("model1")

        # Add model3, should evict model2 (least recently used)
        cache.put("model3", mock3)

        assert cache.get("model1") is mock1
        assert cache.get("model3") is mock3
        assert cache.get("model2") is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.get_device_properties")
    def test_cache_memory_eviction(self, mock_props, mock_allocated, mock_available):
        """Test memory-based eviction."""
        mock_available.return_value = True
        mock_allocated.return_value = 9 * 1024**3  # 9GB allocated
        mock_props.return_value = Mock(total_memory=10 * 1024**3)  # 10GB total

        cache = ModelCache(max_size=5, memory_threshold=0.8)

        # Create mock pipeline
        mock_pipeline = create_mock_pipeline()

        # This should trigger memory eviction
        cache.put("model1", mock_pipeline)

        # Check that GPU memory usage was considered
        mock_allocated.assert_called()

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = ModelCache()

        # Add some models
        for i in range(3):
            mock = create_mock_pipeline()
            cache.put(f"model{i}", mock)

        assert len(cache.cache) == 3

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0
        assert len(cache.model_sizes) == 0

    def test_cache_info(self):
        """Test getting cache information."""
        cache = ModelCache(max_size=3)

        # Add a model
        mock = create_mock_pipeline()
        cache.put("model1", mock)

        info = cache.get_info()

        assert info["size"] == 1
        assert info["max_size"] == 3
        assert "model1" in info["models"]
        assert "total_size_gb" in info
        assert "gpu_memory_usage" in info

    def test_model_size_estimation(self):
        """Test model size estimation."""
        cache = ModelCache()

        # Create a mock pipeline with known size
        mock_pipeline = Mock()

        # Mock UNet with 100M parameters, 4 bytes each
        mock_param = Mock()
        mock_param.numel.return_value = 100_000_000
        mock_param.element_size.return_value = 4
        mock_pipeline.unet.parameters.return_value = [mock_param]

        # Mock VAE with 50M parameters
        mock_vae_param = Mock()
        mock_vae_param.numel.return_value = 50_000_000
        mock_vae_param.element_size.return_value = 4
        mock_pipeline.vae = Mock()
        mock_pipeline.vae.parameters.return_value = [mock_vae_param]

        # No text encoder
        mock_pipeline.text_encoder = None
        mock_pipeline.text_encoder_2 = None

        # Calculate expected size: (100M + 50M) * 4 bytes / 1024^3 = ~0.559 GB
        size_gb = cache._estimate_model_size(mock_pipeline)
        assert 0.5 < size_gb < 0.6  # Should be around 0.559 GB


if __name__ == "__main__":
    pytest.main([__file__])
