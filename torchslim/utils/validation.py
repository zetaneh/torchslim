"""
Validation utilities for TorchSlim compression methods.

This module provides functions to validate compression results, check model accuracy,
and verify the integrity of compressed models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import time

logger = logging.getLogger(__name__)


def validate_compression_result(
    original_model: nn.Module,
    compressed_model: nn.Module,
    test_input: torch.Tensor,
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Validate that compressed model produces similar outputs to original model.
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model
        test_input: Test input tensor
        tolerance: Tolerance for output difference
        
    Returns:
        Dictionary with validation results
    """
    original_model.eval()
    compressed_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        compressed_output = compressed_model(test_input)
    
    # Calculate various similarity metrics
    mse = torch.mean((original_output - compressed_output) ** 2).item()
    mae = torch.mean(torch.abs(original_output - compressed_output)).item()
    
    # Cosine similarity
    original_flat = original_output.flatten()
    compressed_flat = compressed_output.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_flat.unsqueeze(0), 
        compressed_flat.unsqueeze(0)
    ).item()
    
    # Relative error
    relative_error = torch.mean(
        torch.abs(original_output - compressed_output) / 
        (torch.abs(original_output) + 1e-8)
    ).item()
    
    # Check if outputs are within tolerance
    max_diff = torch.max(torch.abs(original_output - compressed_output)).item()
    is_valid = max_diff < tolerance
    
    return {
        'is_valid': is_valid,
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cosine_sim,
        'relative_error': relative_error,
        'max_difference': max_diff,
        'tolerance': tolerance,
        'output_shape': tuple(original_output.shape),
        'validation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }


def validate_model_accuracy(
    original_model: nn.Module,
    compressed_model: nn.Module,
    test_dataset: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cpu",
    num_batches: int = 10
) -> Tuple[float, float]:
    """
    Validate model accuracy on a test dataset.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model  
        test_dataset: Test dataset loader. If None, creates dummy data
        device: Device to run validation on
        num_batches: Number of batches to test (for dummy data)
        
    Returns:
        Tuple of (original_accuracy, compressed_accuracy)
    """
    device = torch.device(device)
    original_model = original_model.to(device)
    compressed_model = compressed_model.to(device)
    
    original_model.eval()
    compressed_model.eval()
    
    if test_dataset is None:
        # Create dummy test data if none provided
        logger.warning("No test dataset provided, using dummy data for validation")
        test_data = _create_dummy_test_data(original_model, device, num_batches)
    else:
        test_data = test_dataset
    
    original_correct = 0
    compressed_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data):
            if isinstance(test_data, list) and batch_idx >= len(test_data):
                break
            elif batch_idx >= num_batches:  # Limit batches for real datasets
                break
                
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            original_output = original_model(data)
            compressed_output = compressed_model(data)
            
            # Calculate accuracy for classification
            if len(original_output.shape) > 1 and original_output.shape[1] > 1:
                original_pred = original_output.argmax(dim=1, keepdim=True)
                compressed_pred = compressed_output.argmax(dim=1, keepdim=True)
                
                original_correct += original_pred.eq(target.view_as(original_pred)).sum().item()
                compressed_correct += compressed_pred.eq(target.view_as(compressed_pred)).sum().item()
            else:
                # For regression or binary classification
                original_correct += torch.sum(torch.abs(original_output - target.float()) < 0.5).item()
                compressed_correct += torch.sum(torch.abs(compressed_output - target.float()) < 0.5).item()
            
            total += target.size(0)
    
    original_accuracy = original_correct / total if total > 0 else 0.0
    compressed_accuracy = compressed_correct / total if total > 0 else 0.0
    
    return original_accuracy, compressed_accuracy


def _create_dummy_test_data(model: nn.Module, device: torch.device, num_batches: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create dummy test data based on model input requirements."""
    batch_size = 32
    
    # Try to infer input shape from model
    try:
        first_layer = next(iter(model.modules()))
        if isinstance(first_layer, nn.Linear):
            input_shape = (batch_size, first_layer.in_features)
            num_classes = 10  # Default
        elif isinstance(first_layer, nn.Conv2d):
            input_shape = (batch_size, first_layer.in_channels, 32, 32)
            num_classes = 10  # Default
        else:
            input_shape = (batch_size, 784)  # Default
            num_classes = 10
    except:
        input_shape = (batch_size, 784)
        num_classes = 10
    
    test_data = []
    for _ in range(num_batches):
        test_input = torch.randn(input_shape).to(device)
        test_target = torch.randint(0, num_classes, (batch_size,)).to(device)
        test_data.append((test_input, test_target))
    
    return test_data


def validate_model_structure(model: nn.Module) -> Dict[str, Any]:
    """
    Validate model structure and return information.
    
    Args:
        model: PyTorch model to validate
        
    Returns:
        Dictionary with model structure information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_types = {}
    layer_count = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type = type(module).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            layer_count += 1
    
    # Check for common issues
    issues = []
    if total_params == 0:
        issues.append("Model has no parameters")
    if trainable_params == 0:
        issues.append("Model has no trainable parameters")
    if layer_count == 0:
        issues.append("Model has no layers")
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_layers': layer_count,
        'layer_types': layer_types,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
        'has_batch_norm': any('BatchNorm' in str(type(m)) for m in model.modules()),
        'has_dropout': any('Dropout' in str(type(m)) for m in model.modules()),
        'issues': issues,
        'is_valid': len(issues) == 0
    }


def check_compression_compatibility(
    model: nn.Module,
    compression_method: str
) -> Dict[str, Any]:
    """
    Check if a model is compatible with a specific compression method.
    
    Args:
        model: Model to check
        compression_method: Name of compression method
        
    Returns:
        Dictionary with compatibility information
    """
    compatible_layers = []
    incompatible_layers = []
    
    # Define layer compatibility for different methods
    compatibility_map = {
        'svd': [nn.Linear],
        'quantization': [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d],
        'pruning': [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d],
        'lora': [nn.Linear],
        'knowledge_distillation': [],  # Compatible with any model
        'tensor_decomposition': [nn.Conv2d, nn.Conv1d, nn.Conv3d],
        'weight_clustering': [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]
    }
    
    supported_types = compatibility_map.get(compression_method, [])
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            if any(isinstance(module, layer_type) for layer_type in supported_types) or not supported_types:
                compatible_layers.append((name, type(module).__name__))
            else:
                incompatible_layers.append((name, type(module).__name__))
    
    total_layers = len(compatible_layers) + len(incompatible_layers)
    compatibility_ratio = len(compatible_layers) / total_layers if total_layers > 0 else 0
    
    return {
        'compression_method': compression_method,
        'compatible_layers': compatible_layers,
        'incompatible_layers': incompatible_layers,
        'compatibility_ratio': compatibility_ratio,
        'is_fully_compatible': len(incompatible_layers) == 0,
        'recommendation': 'Compatible' if compatibility_ratio > 0.5 else 'Not recommended'
    }


def validate_compression_pipeline(
    original_model: nn.Module,
    compressed_model: nn.Module,
    compression_methods: List[str],
    test_input: Optional[torch.Tensor] = None,
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Comprehensive validation of an entire compression pipeline.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        compression_methods: List of compression methods used
        test_input: Test input tensor (optional)
        tolerance: Tolerance for validation
        
    Returns:
        Comprehensive validation results
    """
    results = {
        'compression_methods': compression_methods,
        'validation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'overall_status': 'unknown'
    }
    
    # Model structure validation
    original_structure = validate_model_structure(original_model)
    compressed_structure = validate_model_structure(compressed_model)
    
    results['original_model'] = original_structure
    results['compressed_model'] = compressed_structure
    
    # Calculate compression metrics
    original_params = original_structure['total_parameters']
    compressed_params = compressed_structure['total_parameters']
    
    results['compression_ratio'] = original_params / compressed_params if compressed_params > 0 else float('inf')
    results['parameter_reduction'] = (original_params - compressed_params) / original_params if original_params > 0 else 0
    results['size_reduction_mb'] = original_structure['model_size_mb'] - compressed_structure['model_size_mb']
    
    # Create test input if not provided
    if test_input is None:
        test_input = _create_test_input_for_model(original_model)
    
    # Output validation
    if test_input is not None:
        output_validation = validate_compression_result(
            original_model, compressed_model, test_input, tolerance
        )
        results['output_validation'] = output_validation
        results['output_similarity'] = output_validation['cosine_similarity']
    else:
        results['output_validation'] = {'error': 'Could not create test input'}
        results['output_similarity'] = 0.0
    
    # Compatibility check for each method
    compatibility_results = {}
    for method in compression_methods:
        compatibility_results[method] = check_compression_compatibility(original_model, method)
    results['method_compatibility'] = compatibility_results
    
    # Overall status assessment
    issues = []
    
    if not compressed_structure['is_valid']:
        issues.extend(compressed_structure['issues'])
    
    if results['compression_ratio'] < 1.1:
        issues.append("Low compression ratio (< 1.1x)")
    
    if 'output_validation' in results and not results['output_validation'].get('is_valid', False):
        issues.append("Output validation failed")
    
    if results['output_similarity'] < 0.9:
        issues.append("Low output similarity (< 0.9)")
    
    results['issues'] = issues
    results['overall_status'] = 'success' if len(issues) == 0 else 'warning' if len(issues) <= 2 else 'failure'
    
    return results


def _create_test_input_for_model(model: nn.Module) -> Optional[torch.Tensor]:
    """Create appropriate test input for a model."""
    try:
        # Find the first layer to infer input shape
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                first_layer = module
                break
        
        if first_layer is None:
            return None
        
        batch_size = 1
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(batch_size, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(batch_size, first_layer.in_channels, 32, 32)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(batch_size, first_layer.in_channels, 128)
        elif isinstance(first_layer, nn.Conv3d):
            return torch.randn(batch_size, first_layer.in_channels, 16, 16, 16)
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Could not create test input: {e}")
        return None


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_compression_pipeline
    """
    print("=" * 60)
    print("COMPRESSION VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {validation_results['validation_timestamp']}")
    print(f"Methods Used: {', '.join(validation_results['compression_methods'])}")
    print(f"Overall Status: {validation_results['overall_status'].upper()}")
    print()
    
    # Compression metrics
    print("Compression Metrics:")
    print("-" * 30)
    print(f"  Compression Ratio: {validation_results['compression_ratio']:.2f}x")
    print(f"  Parameter Reduction: {validation_results['parameter_reduction']:.1%}")
    print(f"  Size Reduction: {validation_results['size_reduction_mb']:.2f} MB")
    print(f"  Output Similarity: {validation_results['output_similarity']:.3f}")
    print()
    
    # Model comparison
    orig = validation_results['original_model']
    comp = validation_results['compressed_model']
    
    print("Model Comparison:")
    print("-" * 30)
    print(f"  Original Parameters: {orig['total_parameters']:,}")
    print(f"  Compressed Parameters: {comp['total_parameters']:,}")
    print(f"  Original Size: {orig['model_size_mb']:.2f} MB")
    print(f"  Compressed Size: {comp['model_size_mb']:.2f} MB")
    print()
    
    # Issues
    if validation_results['issues']:
        print("Issues Found:")
        print("-" * 30)
        for issue in validation_results['issues']:
            print(f"  ⚠️  {issue}")
        print()
    else:
        print("✅ No issues found!")
        print()
    
    print("=" * 60)


# Convenience function for quick validation
def quick_validate(original_model: nn.Module, compressed_model: nn.Module, methods: List[str] = None) -> bool:
    """
    Quick validation check - returns True if compression seems successful.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        methods: List of compression methods (optional)
        
    Returns:
        True if validation passes basic checks
    """
    if methods is None:
        methods = ['unknown']
    
    try:
        results = validate_compression_pipeline(original_model, compressed_model, methods)
        return results['overall_status'] in ['success', 'warning']
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    try:
        from .models import create_test_model
    except ImportError:
        # Handle direct script execution
        try:
            from models import create_test_model
        except ImportError:
            print("Warning: Could not import create_test_model. Creating simple test models instead.")
            # Create simple test models for demonstration
            import torch.nn as nn
            
            def create_simple_model(input_size=784, hidden_size=256, output_size=10):
                return nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                )
            
            create_test_model = lambda model_type='mlp', **kwargs: create_simple_model(**kwargs)
    
    print("Testing validation utilities...")
    
    try:
        # Create test models
        original = create_test_model('mlp', input_size=784, hidden_sizes=[512, 256], output_size=10)
        compressed = create_test_model('mlp', input_size=784, hidden_sizes=[256, 128], output_size=10)
        
        # Run validation
        results = validate_compression_pipeline(original, compressed, ['pruning', 'quantization'])
        print_validation_report(results)
        
        print("\n" + "="*50)
        print("✅ Validation utilities test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected when running directly without the full package structure.")