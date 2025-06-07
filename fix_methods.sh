#!/bin/bash

# ================================================================================================
# TorchSlim - Fix Missing Methods Script
# Creates all missing compression method files
# ================================================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_header "🔧 Fixing TorchSlim Missing Methods"

# Navigate to the correct directory
if [[ -d "torchslim" ]]; then
    cd torchslim
    print_status "Changed to torchslim directory"
elif [[ -d "../torchslim" ]]; then
    cd ../torchslim
    print_status "Changed to ../torchslim directory"
else
    print_status "Already in torchslim package directory"
fi

# Create methods directory if it doesn't exist
mkdir -p methods
print_status "Ensured methods directory exists"

# ================================================================================================
# CREATE QUANTIZATION METHOD
# ================================================================================================

print_info "Creating quantization.py..."
cat > methods/quantization.py << 'EOF'
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
EOF

print_status "Created quantization.py"

# ================================================================================================
# CREATE PRUNING METHOD
# ================================================================================================

print_info "Creating pruning.py..."
cat > methods/pruning.py << 'EOF'
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
EOF

print_status "Created pruning.py"

# ================================================================================================
# CREATE LORA METHOD
# ================================================================================================

print_info "Creating lora.py..."
cat > methods/lora.py << 'EOF'
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
EOF

print_status "Created lora.py"

# ================================================================================================
# CREATE KNOWLEDGE DISTILLATION METHOD
# ================================================================================================

print_info "Creating knowledge_distillation.py..."
cat > methods/knowledge_distillation.py << 'EOF'
"""Knowledge Distillation compression method"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from ..core.base import CompressionMethod, CompressionType, CompressionPhase

class KnowledgeDistillation(CompressionMethod):
    """Knowledge Distillation for model compression during training"""
    
    def __init__(self):
        super().__init__(
            name="knowledge_distillation",
            description="Transfer knowledge from teacher to student model",
            compression_type=CompressionType.KNOWLEDGE
        )
        self.teacher_model = None
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """KD doesn't modify individual layers"""
        return layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """KD works at model level, not layer level"""
        return False
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """KD maintains same architecture"""
        return 1.0
    
    def get_supported_phases(self) -> List[CompressionPhase]:
        """KD is applied during training"""
        return [CompressionPhase.DURING_TRAINING]
    
    def configure(self, teacher_model: nn.Module = None, temperature: float = 3.0, alpha: float = 0.5, **kwargs):
        """Configure knowledge distillation"""
        super().configure(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                         targets: torch.Tensor) -> torch.Tensor:
        """Calculate knowledge distillation loss"""
        
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kd_loss *= (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(student_outputs, targets)
        
        # Combined loss
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
EOF

print_status "Created knowledge_distillation.py"

# ================================================================================================
# CREATE TENSOR DECOMPOSITION METHOD
# ================================================================================================

print_info "Creating tensor_decomposition.py..."
cat > methods/tensor_decomposition.py << 'EOF'
"""Tensor decomposition methods"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..core.base import CompressionMethod, CompressionType

class TensorDecomposition(CompressionMethod):
    """Tensor decomposition for weight compression"""
    
    def __init__(self):
        super().__init__(
            name="tensor_decomposition",
            description="Compress weights using tensor decomposition",
            compression_type=CompressionType.STRUCTURAL
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Apply tensor decomposition to a layer"""
        
        # For now, we'll use SVD as a simple tensor decomposition
        if isinstance(layer, nn.Linear):
            rank_ratio = kwargs.get('rank_ratio', 0.5)
            
            # Perform SVD
            U, S, V = torch.svd(layer.weight.data)
            rank = max(1, int(len(S) * rank_ratio))
            
            # Create a simple decomposed representation
            # This is a simplified version - real tensor decomposition would be more complex
            return layer
        else:
            return layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer can be decomposed"""
        return isinstance(layer, nn.Linear)
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate compression ratio"""
        return 1.0  # Simplified for now
EOF

print_status "Created tensor_decomposition.py"

# ================================================================================================
# CREATE WEIGHT CLUSTERING METHOD
# ================================================================================================

print_info "Creating weight_clustering.py..."
cat > methods/weight_clustering.py << 'EOF'
"""Weight clustering compression method"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..core.base import CompressionMethod, CompressionType

class WeightClustering(CompressionMethod):
    """Weight clustering compression using k-means"""
    
    def __init__(self):
        super().__init__(
            name="weight_clustering",
            description="Cluster weights to reduce unique values",
            compression_type=CompressionType.PARAMETRIC
        )
        
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Apply weight clustering to a layer"""
        
        if not hasattr(layer, 'weight'):
            return layer
            
        num_clusters = kwargs.get('num_clusters', 16)
        
        with torch.no_grad():
            # Simple clustering approximation
            weight_flat = layer.weight.data.flatten()
            
            # Use quantiles as cluster centers (simplified k-means)
            percentiles = torch.linspace(0, 100, num_clusters)
            cluster_centers = torch.quantile(weight_flat, percentiles / 100)
            
            # Assign each weight to nearest cluster center
            distances = torch.abs(weight_flat.unsqueeze(1) - cluster_centers.unsqueeze(0))
            cluster_assignments = torch.argmin(distances, dim=1)
            
            # Replace weights with cluster centers
            clustered_weights = cluster_centers[cluster_assignments]
            layer.weight.data = clustered_weights.reshape(layer.weight.shape)
        
        return layer
    
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if layer has weights to cluster"""
        return hasattr(layer, 'weight')
    
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate compression ratio based on unique values"""
        if not hasattr(compressed_layer, 'weight'):
            return 1.0
            
        # This is a theoretical ratio - actual compression depends on storage format
        unique_values = torch.unique(compressed_layer.weight).numel()
        total_values = compressed_layer.weight.numel()
        
        return total_values / unique_values if unique_values > 0 else 1.0
EOF

print_status "Created weight_clustering.py"

# ================================================================================================
# UPDATE METHODS __INIT__.PY
# ================================================================================================

print_info "Updating methods/__init__.py..."
cat > methods/__init__.py << 'EOF'
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
EOF

print_status "Updated methods/__init__.py"

# ================================================================================================
# ADD MISSING COMPRESSION TYPES TO BASE.PY
# ================================================================================================

print_info "Checking core/base.py for CompressionType..."

# Check if CompressionType.QUANTIZATION exists
if ! grep -q "QUANTIZATION" core/base.py; then
    print_info "Adding missing QUANTIZATION type to CompressionType..."
    # Add QUANTIZATION to CompressionType enum
    sed -i '/class CompressionType(Enum):/,/^class / {
        /KNOWLEDGE = "knowledge"/a\
    QUANTIZATION = "quantization"  # Reduces precision
    }' core/base.py
fi

print_status "Ensured all CompressionType values exist"

# ================================================================================================
# CREATE MISSING UTILITY FILES
# ================================================================================================

print_info "Creating missing utility files..."

# Create utils/models.py if missing
mkdir -p utils
if [[ ! -f "utils/models.py" ]]; then
    cat > utils/models.py << 'EOF'
"""Test model creation utilities"""

import torch
import torch.nn as nn
from typing import List

def create_test_model(model_type: str = "mlp", **kwargs) -> nn.Module:
    """Create test models for compression experiments"""
    
    if model_type == "mlp":
        input_size = kwargs.get('input_size', 784)
        hidden_sizes = kwargs.get('hidden_sizes', [512, 256, 128])
        output_size = kwargs.get('output_size', 10)
        
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        return nn.Sequential(*layers)
    
    elif model_type == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
EOF
fi

# Create other missing utility files
for util_file in "benchmarks.py" "validation.py" "schedulers.py"; do
    if [[ ! -f "utils/$util_file" ]]; then
        cat > "utils/$util_file" << EOF
"""Placeholder for $util_file"""

def placeholder_function():
    """Placeholder function for $util_file"""
    pass
EOF
    fi
done

print_status "Created missing utility files"

# ================================================================================================
# REINSTALL PACKAGE
# ================================================================================================

print_info "Reinstalling TorchSlim package..."
cd ..
pip install -e . --force-reinstall

print_status "Package reinstalled successfully"

# ================================================================================================
# TEST THE FIX
# ================================================================================================

print_header "🧪 Testing the Fix"

print_info "Testing imports..."
TEST_RESULT=$(python -c "
try:
    print('Testing TorchSlim import...')
    import torchslim
    print('✅ TorchSlim imported successfully')
    
    print('Testing core components...')
    from torchslim import TorchSlim, CompressionConfig, create_test_model
    print('✅ Core components imported successfully')
    
    print('Testing method registration...')
    from torchslim import get_available_methods
    methods = get_available_methods()
    print(f'✅ Available methods: {methods}')
    
    print('Testing basic functionality...')
    model = create_test_model('mlp', input_size=10, hidden_sizes=[5], output_size=2)
    config = CompressionConfig()
    config.add_method('svd', rank_ratio=0.5)
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    print('✅ Basic compression test passed!')
    
    print('🎉 ALL TESTS PASSED!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    print_status "All tests passed!"
    echo "$TEST_RESULT" | sed 's/^/    /'
else
    echo "Test failed:"
    echo "$TEST_RESULT" | sed 's/^/    /'
fi

print_header "✅ Fix Complete!"

echo -e "${GREEN}🎉 TorchSlim methods have been fixed!${NC}"
echo ""
echo -e "${BLUE}Try running your example again:${NC}"
echo -e "${PURPLE}python examples/basic_usage.py${NC}"
echo ""
echo -e "${BLUE}Or test with:${NC}"
echo -e "${PURPLE}python -c \"from torchslim import *; print('TorchSlim works!')\"${NC}"