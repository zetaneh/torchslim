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
