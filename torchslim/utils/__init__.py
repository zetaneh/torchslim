"""
Utility functions and tools for TorchSlim.
"""

# Import core utility functions
from .models import create_test_model, get_model_info
from .validation import validate_compression_result, validate_model_accuracy
# Note: benchmark_methods will be added later when benchmarks module is complete

# Make commonly used functions available at package level
__all__ = [
    'create_test_model',
    'get_model_info', 
    'validate_compression_result',
    'validate_model_accuracy',
]

# Optional import for benchmarking - only if available
try:
    from .benchmarks import benchmark_methods
    __all__.append('benchmark_methods')
except ImportError:
    pass  # benchmark_methods not available yet