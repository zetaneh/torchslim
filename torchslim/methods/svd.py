"""SVD-based compression method"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..core.base import CompressionMethod, CompressionType

class SVDLinear(nn.Module):
    """SVD-compressed Linear layer"""
    
    def __init__(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(bias) if bias is not None else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.V
        x = x * self.S.unsqueeze(0)
        x = x @ self.U.T
        if self.bias is not None:
            x = x + self.bias
        return x

class SVDCompression(CompressionMethod):
    """SVD compression using low-rank approximation"""
    
    def __init__(self):
        super().__init__(
            name="svd",
            description="Low-rank approximation using Singular Value Decomposition",
            compression_type=CompressionType.STRUCTURAL
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress a layer using SVD"""
        if not isinstance(layer, nn.Linear):
            return layer
            
        rank_ratio = kwargs.get('rank_ratio', 0.5)
        
        # Perform SVD
        U, S, V = torch.svd(layer.weight.data)
        rank = max(1, int(len(S) * rank_ratio))
        
        # Truncate to desired rank
        U_compressed = U[:, :rank]
        S_compressed = S[:rank]
        V_compressed = V[:, :rank]
        
        return SVDLinear(U_compressed, S_compressed, V_compressed, layer.bias)
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer can be compressed with SVD"""
        return isinstance(layer, nn.Linear)
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate SVD compression ratio"""
        if not isinstance(original_layer, nn.Linear) or not isinstance(compressed_layer, SVDLinear):
            return 1.0
            
        original_params = original_layer.weight.numel()
        if original_layer.bias is not None:
            original_params += original_layer.bias.numel()
            
        compressed_params = (compressed_layer.U.numel() + 
                           compressed_layer.S.numel() + 
                           compressed_layer.V.numel())
        if compressed_layer.bias is not None:
            compressed_params += compressed_layer.bias.numel()
            
        return original_params / compressed_params
