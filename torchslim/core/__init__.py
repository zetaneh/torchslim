"""Core components of TorchSlim framework"""

from .base import CompressionMethod, CompressionConfig, BaseCompressor, CompressionPhase
from .registry import register_method, get_available_methods, create_method_instance
from .compressor import TorchSlim
from .layers import CompressedLayer, SVDLinear, QuantizedLinear, PrunedLayer
from .scheduler import CompressionScheduler

__all__ = [
    'CompressionMethod',
    'CompressionConfig', 
    'BaseCompressor',
    'CompressionPhase',
    'TorchSlim',
    'register_method',
    'get_available_methods',
    'create_method_instance',
    'CompressedLayer',
    'SVDLinear',
    'QuantizedLinear', 
    'PrunedLayer',
    'CompressionScheduler'
]
