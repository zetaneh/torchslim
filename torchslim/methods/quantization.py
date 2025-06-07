"""Quantization-based compression method"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ..core.base import CompressionMethod, CompressionType

class QuantizedLinear(nn.Module):
    """Quantized Linear layer"""
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                 scale: float, zero_point: float, bits: int):
        super().__init__()
        self.register_buffer('weight_quantized', weight)
        self.register_buffer('bias', bias)
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits
        
        # Store original dimensions for compatibility
        self.out_features, self.in_features = weight.shape
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization"""
        # Dequantize weights
        weight_fp = self.scale * (self.weight_quantized.float() - self.zero_point)
        return F.linear(x, weight_fp, self.bias)

class QuantizationCompression(CompressionMethod):
    """Quantization compression method"""
    
    def __init__(self):
        super().__init__(
            name="quantization",
            description="Reduce weight precision to lower bit representations",
            compression_type=CompressionType.QUANTIZATION
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Apply quantization to a layer"""
        
        if not isinstance(layer, nn.Linear):
            return layer
            
        bits = kwargs.get('bits', 8)
        
        # Quantize weights
        quantized_weight, scale, zero_point = self._quantize_tensor(layer.weight.data, bits)
        
        return QuantizedLinear(quantized_weight, layer.bias, scale, zero_point, bits)
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer can be quantized"""
        return isinstance(layer, nn.Linear)
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate quantization compression ratio"""
        if isinstance(compressed_layer, QuantizedLinear):
            return 32.0 / compressed_layer.bits  # Assuming original is float32
        return 1.0
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float, float]:
        """Quantize tensor to specified bit width"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        qmin = 0
        qmax = 2**bits - 1
        scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
        
        return quantized.to(torch.uint8), scale, zero_point
