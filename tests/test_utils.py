# tests/test_utils.py
"""Test utility functions."""

import pytest
import torch
from torchslim.utils.models import create_test_model, get_model_info
from torchslim.utils.validation import validate_compression_result

class TestUtilities:
    """Test utility functions."""
    
    def test_model_creation_variants(self):
        """Test different model creation options."""
        # Test MLP
        mlp = create_test_model("mlp", input_size=20, hidden_sizes=[10, 5], output_size=3)
        assert isinstance(mlp, torch.nn.Module)
        
        # Test CNN
        cnn = create_test_model("cnn", input_channels=3, num_classes=5)
        assert isinstance(cnn, torch.nn.Module)
    
    def test_model_info(self):
        """Test model information extraction."""
        model = create_test_model("mlp", input_size=10, hidden_sizes=[5], output_size=2)
        info = get_model_info(model)
        
        assert isinstance(info, dict)
        assert 'total_parameters' in info
        assert 'total_size_mb' in info
        assert info['total_parameters'] > 0
    
    def test_validation_utilities(self):
        """Test validation functions."""
        model1 = create_test_model("mlp", input_size=10, hidden_sizes=[8], output_size=2)
        model2 = create_test_model("mlp", input_size=10, hidden_sizes=[6], output_size=2)
        
        test_input = torch.randn(1, 10)
        result = validate_compression_result(model1, model2, test_input)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'cosine_similarity' in result
        assert 'mse' in result