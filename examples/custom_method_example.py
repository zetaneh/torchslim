#!/usr/bin/env python3
"""
Custom Compression Method Example for TorchSlim

This example demonstrates how to create custom compression methods using
TorchSlim's plugin-based architecture. We'll create several different types
of custom compression methods to showcase the flexibility of the framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import logging

from torchslim.core.base import CompressionMethod, CompressionType
from torchslim.core.registry import register_method
from torchslim import TorchSlim, CompressionConfig, create_test_model
from torchslim.utils.validation import validate_compression_pipeline, print_validation_report

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Simple Weight Scaling Compression
# =============================================================================

class WeightScalingCompression(CompressionMethod):
    """
    Simple compression method that scales down weights by a constant factor.
    This is a basic example to demonstrate the core concepts.
    """
    
    def __init__(self):
        super().__init__(
            name="weight_scaling",
            description="Scale weights by a constant factor to reduce magnitude",
            compression_type=CompressionType.PARAMETRIC
        )
    
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress a layer by scaling its weights."""
        
        # Get configuration parameters
        scale_factor = kwargs.get('scale_factor', 0.8)
        
        logger.info(f"Applying weight scaling to {layer_name} with factor {scale_factor}")
        
        # Apply scaling to layer weights
        with torch.no_grad():
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data *= scale_factor
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data *= scale_factor
        
        # Store compression metadata
        self.metrics[layer_name] = {
            'scale_factor': scale_factor,
            'original_weight_norm': torch.norm(layer.weight.data / scale_factor).item() if hasattr(layer, 'weight') else 0,
            'compressed_weight_norm': torch.norm(layer.weight.data).item() if hasattr(layer, 'weight') else 0
        }
        
        return layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer can be compressed (has weights)."""
        return hasattr(layer, 'weight') and layer.weight is not None
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Weight scaling doesn't change parameter count, so ratio is 1.0."""
        return 1.0


# =============================================================================
# Example 2: Random Pruning with Custom Layer
# =============================================================================

class PrunedLinear(nn.Module):
    """Custom linear layer that supports random pruning."""
    
    def __init__(self, original_layer: nn.Linear, pruning_mask: torch.Tensor):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Store the pruning mask
        self.register_buffer('pruning_mask', pruning_mask)
        
        # Copy weights and apply mask
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Apply pruning mask during forward pass
        masked_weight = self.weight * self.pruning_mask
        return torch.nn.functional.linear(x, masked_weight, self.bias)
    
    def get_active_parameters(self):
        """Get count of non-pruned parameters."""
        active_weights = torch.sum(self.pruning_mask).item()
        active_bias = self.out_features if self.bias is not None else 0
        return int(active_weights + active_bias)


class RandomPruningCompression(CompressionMethod):
    """
    Custom pruning method that randomly removes weights.
    Demonstrates creating custom layers and structural compression.
    """
    
    def __init__(self):
        super().__init__(
            name="random_pruning",
            description="Randomly prune weights to achieve compression",
            compression_type=CompressionType.STRUCTURAL
        )
    
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress layer using random pruning."""
        
        if not isinstance(layer, nn.Linear):
            logger.warning(f"Random pruning only supports Linear layers, skipping {layer_name}")
            return layer
        
        # Get configuration
        pruning_ratio = kwargs.get('pruning_ratio', 0.3)
        seed = kwargs.get('seed', 42)
        
        logger.info(f"Applying random pruning to {layer_name} with ratio {pruning_ratio}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create random pruning mask
        weight_shape = layer.weight.shape
        total_weights = weight_shape[0] * weight_shape[1]
        num_to_prune = int(total_weights * pruning_ratio)
        
        # Create mask (1 = keep, 0 = prune)
        mask = torch.ones(weight_shape)
        flat_mask = mask.view(-1)
        
        # Randomly select weights to prune
        prune_indices = torch.randperm(total_weights)[:num_to_prune]
        flat_mask[prune_indices] = 0
        mask = flat_mask.view(weight_shape)
        
        # Create compressed layer
        compressed_layer = PrunedLinear(layer, mask)
        
        # Store metrics
        original_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        active_params = compressed_layer.get_active_parameters()
        
        self.metrics[layer_name] = {
            'pruning_ratio': pruning_ratio,
            'original_parameters': original_params,
            'active_parameters': active_params,
            'compression_ratio': original_params / active_params if active_params > 0 else float('inf')
        }
        
        return compressed_layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Only compress Linear layers."""
        return isinstance(layer, nn.Linear)
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate compression ratio based on active parameters."""
        if hasattr(compressed_layer, 'get_active_parameters'):
            original_params = sum(p.numel() for p in original_layer.parameters())
            active_params = compressed_layer.get_active_parameters()
            return original_params / active_params if active_params > 0 else float('inf')
        return 1.0


# =============================================================================
# Example 3: Adaptive Rank Compression
# =============================================================================

class AdaptiveRankLinear(nn.Module):
    """Linear layer with adaptive low-rank decomposition."""
    
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        
        # Perform SVD on original weights
        U, S, Vt = torch.svd(original_layer.weight.data)
        
        # Keep only top-k singular values
        self.U = nn.Parameter(U[:, :rank].contiguous())
        self.S = nn.Parameter(S[:rank].contiguous())
        self.Vt = nn.Parameter(Vt[:rank, :].contiguous())
        
        # Copy bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Reconstruct weight matrix: W = U @ diag(S) @ Vt
        weight = self.U @ torch.diag(self.S) @ self.Vt
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def get_compression_ratio(self, original_layer):
        """Calculate compression ratio."""
        original_params = original_layer.weight.numel() + (original_layer.bias.numel() if original_layer.bias is not None else 0)
        compressed_params = (self.U.numel() + self.S.numel() + self.Vt.numel() + 
                           (self.bias.numel() if self.bias is not None else 0))
        return original_params / compressed_params


class AdaptiveRankCompression(CompressionMethod):
    """
    Advanced compression using adaptive rank selection based on singular values.
    Demonstrates complex compression logic and parameter analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="adaptive_rank",
            description="Adaptive low-rank compression based on singular value analysis",
            compression_type=CompressionType.STRUCTURAL
        )
    
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress layer using adaptive rank selection."""
        
        if not isinstance(layer, nn.Linear):
            logger.warning(f"Adaptive rank only supports Linear layers, skipping {layer_name}")
            return layer
        
        # Configuration
        energy_threshold = kwargs.get('energy_threshold', 0.95)
        max_rank_ratio = kwargs.get('max_rank_ratio', 0.8)
        min_rank = kwargs.get('min_rank', 1)
        
        logger.info(f"Applying adaptive rank to {layer_name} with energy threshold {energy_threshold}")
        
        # Perform SVD
        U, S, Vt = torch.svd(layer.weight.data)
        
        # Calculate cumulative energy
        total_energy = torch.sum(S ** 2)
        cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
        
        # Find rank that preserves desired energy
        rank_by_energy = torch.sum(cumulative_energy < energy_threshold).item() + 1
        
        # Apply constraints
        max_rank = int(min(layer.weight.shape) * max_rank_ratio)
        rank = max(min_rank, min(rank_by_energy, max_rank))
        
        logger.info(f"Selected rank {rank} for {layer_name} (energy-based: {rank_by_energy}, max allowed: {max_rank})")
        
        # Create compressed layer
        compressed_layer = AdaptiveRankLinear(layer, rank)
        
        # Store detailed metrics
        self.metrics[layer_name] = {
            'rank': rank,
            'rank_by_energy': rank_by_energy,
            'max_rank': max_rank,
            'energy_threshold': energy_threshold,
            'energy_preserved': cumulative_energy[rank-1].item(),
            'singular_values': S[:rank].tolist(),
            'compression_ratio': compressed_layer.get_compression_ratio(layer)
        }
        
        return compressed_layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Only compress Linear layers with sufficient size."""
        return isinstance(layer, nn.Linear) and min(layer.weight.shape) > 4
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Get compression ratio from compressed layer."""
        if hasattr(compressed_layer, 'get_compression_ratio'):
            return compressed_layer.get_compression_ratio(original_layer)
        return 1.0


# =============================================================================
# Example 4: Quantization with Custom Precision
# =============================================================================

class CustomQuantizedLinear(nn.Module):
    """Linear layer with custom bit-width quantization."""
    
    def __init__(self, original_layer: nn.Linear, bits: int, scale_factor: float, zero_point: int):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.bits = bits
        
        # Store quantization parameters
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.register_buffer('zero_point', torch.tensor(zero_point))
        
        # Quantize weights
        quantized_weights = self._quantize_tensor(original_layer.weight.data, bits, scale_factor, zero_point)
        self.register_buffer('quantized_weight', quantized_weights)
        
        # Handle bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int, scale: float, zero_point: int) -> torch.Tensor:
        """Quantize tensor to specified bit width."""
        qmin = 0
        qmax = (1 << bits) - 1
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(torch.uint8 if bits <= 8 else torch.int16)
    
    def _dequantize_tensor(self, quantized: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to float."""
        return (quantized.float() - self.zero_point) * self.scale_factor
    
    def forward(self, x):
        # Dequantize weights for computation
        weight = self._dequantize_tensor(self.quantized_weight)
        return torch.nn.functional.linear(x, weight, self.bias)


class CustomQuantizationCompression(CompressionMethod):
    """
    Custom quantization method with configurable bit-width and calibration.
    """
    
    def __init__(self):
        super().__init__(
            name="custom_quantization",
            description="Custom quantization with configurable precision",
            compression_type=CompressionType.QUANTIZATION
        )
    
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress layer using custom quantization."""
        
        if not isinstance(layer, nn.Linear):
            logger.warning(f"Custom quantization only supports Linear layers, skipping {layer_name}")
            return layer
        
        # Configuration
        bits = kwargs.get('bits', 8)
        symmetric = kwargs.get('symmetric', True)
        
        logger.info(f"Applying {bits}-bit quantization to {layer_name}")
        
        # Calculate quantization parameters
        weight_data = layer.weight.data
        if symmetric:
            # Symmetric quantization
            max_val = torch.max(torch.abs(weight_data))
            scale = max_val / ((1 << (bits - 1)) - 1)
            zero_point = 0
        else:
            # Asymmetric quantization
            min_val = torch.min(weight_data)
            max_val = torch.max(weight_data)
            qmin = 0
            qmax = (1 << bits) - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, qmin, qmax).int()
        
        # Create quantized layer
        compressed_layer = CustomQuantizedLinear(layer, bits, scale.item(), zero_point.item() if not symmetric else 0)
        
        # Calculate compression metrics
        original_bits = 32  # Assuming float32
        compression_ratio = original_bits / bits
        
        self.metrics[layer_name] = {
            'bits': bits,
            'symmetric': symmetric,
            'scale_factor': scale.item(),
            'zero_point': zero_point.item() if not symmetric else 0,
            'compression_ratio': compression_ratio,
            'weight_range': (torch.min(weight_data).item(), torch.max(weight_data).item())
        }
        
        return compressed_layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Can compress Linear and Conv layers."""
        return isinstance(layer, (nn.Linear, nn.Conv2d))
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate compression ratio based on bit reduction."""
        return 32 / compressed_layer.bits  # Assuming float32 -> custom bits


# =============================================================================
# Registration and Usage Example
# =============================================================================

def register_custom_methods():
    """Register all custom compression methods."""
    print("📝 Registering custom compression methods...")
    
    # Register all our custom methods
    register_method("weight_scaling", WeightScalingCompression)
    register_method("random_pruning", RandomPruningCompression)
    register_method("adaptive_rank", AdaptiveRankCompression)
    register_method("custom_quantization", CustomQuantizationCompression)
    
    print("✅ Custom methods registered successfully!")


def demonstrate_custom_methods():
    """Demonstrate usage of custom compression methods."""
    
    print("\n" + "="*80)
    print("🚀 CUSTOM COMPRESSION METHODS DEMONSTRATION")
    print("="*80)
    
    # Create a test model
    print("\n1️⃣ Creating test model...")
    model = create_test_model('mlp', input_size=784, hidden_sizes=[512, 256, 128], output_size=10)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Original model: {original_params:,} parameters")
    
    # Test each custom method
    custom_methods = [
        ("weight_scaling", {"scale_factor": 0.7}),
        ("random_pruning", {"pruning_ratio": 0.4, "seed": 42}),
        ("adaptive_rank", {"energy_threshold": 0.9, "max_rank_ratio": 0.6}),
        ("custom_quantization", {"bits": 4, "symmetric": False})
    ]
    
    results = {}
    
    for method_name, method_config in custom_methods:
        print(f"\n2️⃣ Testing {method_name}...")
        
        try:
            # Create compression configuration
            config = CompressionConfig()
            config.add_method(method_name, **method_config)
            
            # Compress model
            compressor = TorchSlim(config)
            model_copy = torch.nn.Sequential(*[layer for layer in model])  # Simple copy
            compressed_model = compressor.compress_model(model_copy)
            
            # Get results
            report = compressor.get_compression_report()
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            
            results[method_name] = {
                'compression_ratio': report['summary']['compression_ratio'],
                'compressed_params': compressed_params,
                'original_params': original_params,
                'config': method_config,
                'report': report
            }
            
            print(f"   ✅ {method_name}: {report['summary']['compression_ratio']:.2f}x compression")
            
        except Exception as e:
            print(f"   ❌ {method_name} failed: {e}")
            results[method_name] = {'error': str(e)}
    
    return results


def comprehensive_custom_pipeline():
    """Demonstrate a comprehensive pipeline using multiple custom methods."""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE CUSTOM COMPRESSION PIPELINE")
    print("="*80)
    
    # Create model
    model = create_test_model('mlp', input_size=784, hidden_sizes=[1024, 512, 256], output_size=10)
    print(f"Original model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create multi-method configuration
    config = CompressionConfig()
    config.add_method("adaptive_rank", energy_threshold=0.95, max_rank_ratio=0.7)
    config.add_method("custom_quantization", bits=6, symmetric=True)
    config.add_method("weight_scaling", scale_factor=0.9)
    
    print(f"\nApplying methods: {config.enabled_methods}")
    
    # Compress
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    
    # Validate
    validation_results = validate_compression_pipeline(
        model, compressed_model, config.enabled_methods
    )
    
    print_validation_report(validation_results)
    
    # Detailed analysis
    report = compressor.get_compression_report()
    print(f"\n🔍 Pipeline Analysis:")
    print(f"   Overall compression: {report['summary']['compression_ratio']:.2f}x")
    print(f"   Parameter reduction: {(1 - sum(p.numel() for p in compressed_model.parameters()) / sum(p.numel() for p in model.parameters())):.1%}")
    
    # Method-specific details
    print(f"\n📊 Method Details:")
    for method_name, method_report in report.get('methods', {}).items():
        print(f"   {method_name}:")
        print(f"     - Layers processed: {method_report.get('layers_processed', 'N/A')}")
        print(f"     - Method ratio: {method_report.get('compression_ratio', 'N/A')}")
    
    return validation_results, report


def main():
    """Main execution function."""
    
    print("🎨 TorchSlim Custom Compression Methods Example")
    print("This example shows how to create and use custom compression methods")
    
    try:
        # Register custom methods
        register_custom_methods()
        
        # Demonstrate individual methods
        results = demonstrate_custom_methods()
        
        # Show summary
        print("\n📋 SUMMARY OF CUSTOM METHODS:")
        print("-" * 60)
        for method_name, result in results.items():
            if 'error' not in result:
                ratio = result['compression_ratio']
                print(f"   {method_name:20} : {ratio:6.2f}x compression")
            else:
                print(f"   {method_name:20} : ❌ {result['error']}")
        
        # Comprehensive pipeline
        print("\n" + "="*80)
        validation_results, pipeline_report = comprehensive_custom_pipeline()
        
        print(f"\n🎉 Custom method demonstration completed!")
        print(f"   Status: {validation_results['overall_status'].upper()}")
        print(f"   Overall compression: {pipeline_report['summary']['compression_ratio']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Custom method example completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Custom methods inherit from CompressionMethod")
        print("   • Implement compress_layer(), can_compress_layer(), get_compression_ratio()")
        print("   • Register methods using register_method()")
        print("   • Custom layers can implement complex compression logic")
        print("   • Methods can be combined in compression pipelines")
        print("   • Use validation tools to verify compression quality")
    else:
        print("\n❌ Custom method example failed!")