"""Compressed layer implementations"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CompressedLayer(nn.Module):
    """Base class for compressed layers"""
    
    def __init__(self, original_layer: nn.Module, compression_method: str):
        super().__init__()
        self.compression_method = compression_method
        self.original_shape = None
        if hasattr(original_layer, 'weight'):
            self.original_shape = original_layer.weight.shape

class SVDLinear(CompressedLayer):
    """SVD-compressed Linear layer"""
    
    def __init__(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, 
                 bias: Optional[torch.Tensor] = None, original_layer: Optional[nn.Linear] = None):
        super().__init__(original_layer, "svd")
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)  
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(bias) if bias is not None else None

class QuantizedLinear(CompressedLayer):
    """Quantized Linear layer"""
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                 scale: float, zero_point: float, bits: int):
        super().__init__(None, "quantization")
        self.register_buffer('weight_quantized', weight)
        self.register_buffer('bias', bias)
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits

class PrunedLayer(CompressedLayer):
    """Pruned layer with sparse weights"""
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], mask: torch.Tensor):
        super().__init__(None, "pruning")
        self.register_parameter('weight', nn.Parameter(weight))
        self.register_buffer('mask', mask)
        self.bias = nn.Parameter(bias) if bias is not None else None
