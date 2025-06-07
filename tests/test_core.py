"""Test core functionality."""

import pytest
import torch
import torch.nn as nn
from torchslim import TorchSlim, CompressionConfig, create_test_model
from torchslim.core.registry import get_available_methods

class TestCoreComponents:
    """Test core TorchSlim components."""
    
    def test_imports(self):
        """Test that core components can be imported."""
        from torchslim import TorchSlim, CompressionConfig
        from torchslim.core.base import CompressionMethod, CompressionType
        assert TorchSlim is not None
        assert CompressionConfig is not None
        assert CompressionMethod is not None
        assert CompressionType is not None
    
    def test_available_methods(self):
        """Test that compression methods are registered."""
        methods = get_available_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert 'svd' in methods
    
    def test_compression_config(self):
        """Test compression configuration."""
        config = CompressionConfig()
        config.add_method("svd", rank_ratio=0.5)
        
        assert "svd" in config.enabled_methods
        assert config.method_configs["svd"]["rank_ratio"] == 0.5
    
    def test_model_creation(self):
        """Test model creation utilities."""
        model = create_test_model("mlp", input_size=10, hidden_sizes=[5], output_size=2)
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)

class TestCompression:
    """Test compression functionality."""
    
    def test_basic_compression(self, simple_model):
        """Test basic compression workflow."""
        config = CompressionConfig()
        config.add_method("svd", rank_ratio=0.5)
        
        compressor = TorchSlim(config)
        compressed_model = compressor.compress_model(simple_model)
        
        assert compressed_model is not None
        
        # Test that compressed model still works
        x = torch.randn(1, 10)
        original_output = simple_model(x)
        compressed_output = compressed_model(x)
        
        assert original_output.shape == compressed_output.shape
    
    def test_compression_report(self, simple_model):
        """Test compression reporting."""
        config = CompressionConfig()
        config.add_method("svd", rank_ratio=0.7)
        
        compressor = TorchSlim(config)
        compressed_model = compressor.compress_model(simple_model)
        report = compressor.get_compression_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'compression_ratio' in report['summary']
        assert report['summary']['compression_ratio'] > 0
