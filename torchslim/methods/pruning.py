"""Pruning-based compression method"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..core.base import CompressionMethod, CompressionType

class PruningCompression(CompressionMethod):
    """Weight pruning compression method"""
    
    def __init__(self):
        super().__init__(
            name="pruning",
            description="Remove weights based on magnitude or structured patterns",
            compression_type=CompressionType.PARAMETRIC
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Apply pruning to a layer"""
        
        pruning_ratio = kwargs.get('pruning_ratio', 0.1)
        pruning_type = kwargs.get('pruning_type', 'magnitude')
        
        if hasattr(layer, 'weight'):
            with torch.no_grad():
                if pruning_type == 'magnitude':
                    layer.weight.data = self._magnitude_pruning(layer.weight.data, pruning_ratio)
                else:
                    layer.weight.data = self._magnitude_pruning(layer.weight.data, pruning_ratio)
        
        return layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer has weights to prune"""
        return hasattr(layer, 'weight')
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate pruning compression ratio based on sparsity"""
        if not hasattr(compressed_layer, 'weight'):
            return 1.0
            
        total_weights = compressed_layer.weight.numel()
        non_zero_weights = torch.count_nonzero(compressed_layer.weight).item()
        
        return total_weights / non_zero_weights if non_zero_weights > 0 else 1.0
    
    def _magnitude_pruning(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Remove weights with smallest magnitude"""
        if ratio <= 0:
            return tensor
            
        abs_tensor = torch.abs(tensor)
        threshold = torch.quantile(abs_tensor, ratio)
        mask = abs_tensor > threshold
        return tensor * mask.float()
