
# tests/test_methods.py
"""Test compression methods."""

import pytest
import torch
import torch.nn as nn
from torchslim.core.registry import create_method_instance, get_available_methods

class TestCompressionMethods:
    """Test individual compression methods."""
    
    @pytest.mark.parametrize("method_name", ["svd", "quantization", "pruning"])
    def test_method_creation(self, method_name):
        """Test that methods can be created."""
        if method_name in get_available_methods():
            method = create_method_instance(method_name)
            assert method is not None
            assert method.name == method_name
    
    def test_svd_compression(self):
        """Test SVD compression specifically."""
        if 'svd' not in get_available_methods():
            pytest.skip("SVD method not available")
            
        method = create_method_instance('svd')
        layer = nn.Linear(10, 5)
        
        can_compress = method.can_compress_layer(layer)
        assert can_compress is True
        
        compressed_layer = method.compress_layer(layer, "test_layer", rank_ratio=0.5)
        assert compressed_layer is not None
        
        # Test forward pass
        x = torch.randn(1, 10)
        original_output = layer(x)
        compressed_output = compressed_layer(x)
        assert original_output.shape == compressed_output.shape