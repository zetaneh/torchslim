"""Built-in compression methods for TorchSlim"""

from .svd import SVDCompression
from .quantization import QuantizationCompression
from .pruning import PruningCompression
from .lora import LoRACompression
from .knowledge_distillation import KnowledgeDistillation
from .tensor_decomposition import TensorDecomposition
from .weight_clustering import WeightClustering

# Auto-register all built-in methods
from ..core.registry import register_method

# Register methods with the global registry
register_method("svd", SVDCompression)
register_method("quantization", QuantizationCompression)
register_method("pruning", PruningCompression)
register_method("lora", LoRACompression)
register_method("knowledge_distillation", KnowledgeDistillation)
register_method("tensor_decomposition", TensorDecomposition)
register_method("weight_clustering", WeightClustering)

__all__ = [
    'SVDCompression',
    'QuantizationCompression',
    'PruningCompression', 
    'LoRACompression',
    'KnowledgeDistillation',
    'TensorDecomposition',
    'WeightClustering'
]
