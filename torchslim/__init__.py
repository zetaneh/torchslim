"""
TorchSlim - Extensible PyTorch Model Compression Library

A modular, plugin-based framework for neural network compression with easy extensibility.
"""

__version__ = "1.0.0"
__author__ = "TorchSlim Contributors"

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Core imports
try:
    from .core import (
        BaseCompressor,
        CompressionMethod,
        CompressionConfig,
        CompressionPhase,
        TorchSlim,
        register_method,
        get_available_methods,
        create_method_instance
    )
    logger.debug("✓ Core imports successful")
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    raise

# Built-in compression methods - import what's available
try:
    from .methods import (
        SVDCompression,
        QuantizationCompression,
        PruningCompression,
        LoRACompression,
        KnowledgeDistillation,
        TensorDecomposition,
        WeightClustering
    )
    logger.debug("✓ Compression methods imported")
except ImportError as e:
    logger.warning(f"Some compression methods may not be available: {e}")

# Analysis and utilities - import what exists
try:
    from .analysis import ModelAnalyzer, CompressionProfiler, VisualizationTools
    _ANALYSIS_AVAILABLE = True
    logger.debug("✓ Analysis tools imported")
except ImportError as e:
    logger.warning(f"Analysis tools not available: {e}")
    _ANALYSIS_AVAILABLE = False

# Utilities - import what actually exists
try:
    from .utils import create_test_model, get_model_info
    _CORE_UTILS_AVAILABLE = True
    logger.debug("✓ Core utilities imported")
except ImportError as e:
    logger.warning(f"Core utilities not available: {e}")
    _CORE_UTILS_AVAILABLE = False

# Optional utilities - import if available
_OPTIONAL_UTILS = {}

try:
    from .utils import benchmark_methods
    _OPTIONAL_UTILS['benchmark_methods'] = benchmark_methods
except ImportError:
    logger.debug("benchmark_methods not available")

try:
    from .utils import validate_compression_result as validate_compression
    _OPTIONAL_UTILS['validate_compression'] = validate_compression
except ImportError:
    logger.debug("validate_compression not available")

try:
    from .utils import validate_model_accuracy as model_similarity
    _OPTIONAL_UTILS['model_similarity'] = model_similarity
except ImportError:
    logger.debug("model_similarity not available")

# Compression scheduler - placeholder for now
def compression_scheduler(*args, **kwargs):
    """Placeholder for compression scheduler - to be implemented."""
    raise NotImplementedError("Compression scheduler not yet implemented")

_OPTIONAL_UTILS['compression_scheduler'] = compression_scheduler

# Export all public APIs
__all__ = [
    # Core framework
    'BaseCompressor',
    'CompressionMethod', 
    'CompressionConfig',
    'CompressionPhase',
    'TorchSlim',
    'register_method',
    'get_available_methods',
    'create_method_instance',
    
    # Built-in methods
    'SVDCompression',
    'QuantizationCompression', 
    'PruningCompression',
    'LoRACompression',
    'KnowledgeDistillation',
    'TensorDecomposition',
    'WeightClustering',
]

# Add analysis tools if available
if _ANALYSIS_AVAILABLE:
    __all__.extend([
        'ModelAnalyzer',
        'CompressionProfiler', 
        'VisualizationTools',
    ])

# Add core utilities if available
if _CORE_UTILS_AVAILABLE:
    __all__.extend([
        'create_test_model',
        'get_model_info',
    ])

# Add optional utilities that are available
for util_name in _OPTIONAL_UTILS:
    __all__.append(util_name)
    # Make them available at module level
    globals()[util_name] = _OPTIONAL_UTILS[util_name]

def print_welcome():
    """Print welcome message with available features."""
    print("🔥 TorchSlim v1.0.0 - Extensible PyTorch Model Compression")
    
    try:
        methods = get_available_methods()
        print(f"📦 Available methods: {methods}")
    except:
        print("📦 Compression methods loading...")
    
    available_features = []
    if _ANALYSIS_AVAILABLE:
        available_features.append("Analysis Tools")
    if _CORE_UTILS_AVAILABLE:
        available_features.append("Model Utilities")
    if len(_OPTIONAL_UTILS) > 0:
        available_features.append(f"{len(_OPTIONAL_UTILS)} Optional Utilities")
    
    if available_features:
        print(f"🛠️  Available features: {', '.join(available_features)}")
    
    print("🚀 Ready for compression!")

# Show welcome message
print_welcome()

# Convenience function for quick compression
def quick_compress(model, method_name="svd", **method_kwargs):
    """
    Quick compression function for simple use cases.
    
    Args:
        model: PyTorch model to compress
        method_name: Name of compression method to use
        **method_kwargs: Arguments for the compression method
        
    Returns:
        Tuple of (compressed_model, compression_report)
    """
    config = CompressionConfig()
    config.add_method(method_name, **method_kwargs)
    
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    
    return compressed_model, compressor.get_compression_report()

# Add quick_compress to exports
__all__.append('quick_compress')