"""LoRA (Low-Rank Adaptation) compression method"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math
from ..core.base import CompressionMethod, CompressionType

class LoRALinear(nn.Module):
    """LoRA-adapted Linear layer"""
    
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        self.register_buffer('weight_frozen', original_linear.weight.data)
        if original_linear.bias is not None:
            self.register_buffer('bias_frozen', original_linear.bias.data)
        else:
            self.bias_frozen = None
            
        # LoRA adaptation matrices
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen weights and LoRA adaptation"""
        # Original computation
        result = F.linear(x, self.weight_frozen, self.bias_frozen)
        
        # LoRA adaptation
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        
        return result + lora_output

class LoRACompression(CompressionMethod):
    """LoRA compression for efficient fine-tuning"""
    
    def __init__(self):
        super().__init__(
            name="lora",
            description="Low-Rank Adaptation for efficient fine-tuning",
            compression_type=CompressionType.STRUCTURAL
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Apply LoRA to a linear layer"""
        
        if not isinstance(layer, nn.Linear):
            return layer
            
        rank = kwargs.get('rank', 16)
        alpha = kwargs.get('alpha', 16.0)
        
        return LoRALinear(layer, rank, alpha)
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """LoRA can be applied to Linear layers"""
        return isinstance(layer, nn.Linear)
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate parameter efficiency ratio for LoRA"""
        if not isinstance(compressed_layer, LoRALinear):
            return 1.0
            
        original_trainable = original_layer.weight.numel()
        lora_trainable = compressed_layer.lora_A.numel() + compressed_layer.lora_B.numel()
        
        return original_trainable / lora_trainable
