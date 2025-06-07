"""Core base classes and interfaces for TorchSlim"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)

class CompressionPhase(Enum):
    """Phases when compression can be applied"""
    PRE_TRAINING = "pre_training"
    DURING_TRAINING = "during_training" 
    POST_TRAINING = "post_training"
    INFERENCE = "inference"

class CompressionType(Enum):
    """Types of compression techniques"""
    STRUCTURAL = "structural"  # Changes model architecture
    PARAMETRIC = "parametric"  # Modifies parameters
    QUANTIZATION = "quantization"  # Reduces precision
    KNOWLEDGE = "knowledge"  # Knowledge transfer

@dataclass
class LayerCompressionInfo:
    """Information about layer compression"""
    layer_name: str
    layer_type: str
    original_parameters: int
    compressed_parameters: int
    compression_ratio: float
    method_used: str
    compression_time: float
    memory_saved_mb: float

@dataclass
class CompressionConfig:
    """Comprehensive configuration for compression methods"""
    
    # Core settings
    enabled_methods: List[str] = field(default_factory=list)
    target_compression_ratio: Optional[float] = None
    target_memory_reduction_mb: Optional[float] = None
    preserve_accuracy_threshold: float = 0.05  # 5% max accuracy drop
    
    # Method-specific configurations (extensible)
    method_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Layer selection
    preserve_first_layer: bool = True
    preserve_last_layer: bool = True
    preserve_embedding_layers: bool = True
    exclude_layers: List[str] = field(default_factory=list)
    include_layers: List[str] = field(default_factory=list)
    layer_type_filters: List[str] = field(default_factory=list)
    
    # Advanced settings
    compression_schedule: Optional[Dict[str, Any]] = None
    validation_frequency: int = 1
    early_stopping: bool = True
    parallel_compression: bool = False
    checkpoint_frequency: int = 0  # 0 = no checkpointing
    
    # Quality control
    validate_each_layer: bool = True
    rollback_on_failure: bool = True
    accuracy_validation_data: Optional[Any] = None
    
    # Logging and debugging
    verbose: bool = True
    log_layer_details: bool = False
    save_intermediate_models: bool = False
    
    def add_method(self, method_name: str, **kwargs):
        """Add a compression method with its configuration"""
        if method_name not in self.enabled_methods:
            self.enabled_methods.append(method_name)
        self.method_configs[method_name] = kwargs
        
    def remove_method(self, method_name: str):
        """Remove a compression method"""
        if method_name in self.enabled_methods:
            self.enabled_methods.remove(method_name)
        self.method_configs.pop(method_name, None)
        
    def get_method_config(self, method_name: str) -> Dict[str, Any]:
        """Get configuration for a specific method"""
        return self.method_configs.get(method_name, {})
    
    def update_method_config(self, method_name: str, **kwargs):
        """Update configuration for a specific method"""
        if method_name not in self.method_configs:
            self.method_configs[method_name] = {}
        self.method_configs[method_name].update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'enabled_methods': self.enabled_methods,
            'target_compression_ratio': self.target_compression_ratio,
            'method_configs': self.method_configs,
            'preserve_first_layer': self.preserve_first_layer,
            'preserve_last_layer': self.preserve_last_layer,
            'exclude_layers': self.exclude_layers,
            'include_layers': self.include_layers
        }

class CompressionMethod(ABC):
    """Abstract base class for all compression methods"""
    
    def __init__(self, name: str, description: str = "", compression_type: CompressionType = CompressionType.PARAMETRIC):
        self.name = name
        self.description = description
        self.compression_type = compression_type
        self.config = {}
        self.metrics = {}
        self.layer_info: List[LayerCompressionInfo] = []
        
    @abstractmethod
    def compress_layer(self, layer: nn.Module, layer_name: str, **kwargs) -> nn.Module:
        """Compress a single layer"""
        pass
    
    @abstractmethod
    def can_compress_layer(self, layer: nn.Module) -> bool:
        """Check if this method can compress the given layer type"""
        pass
    
    @abstractmethod
    def get_compression_ratio(self, original_layer: nn.Module, compressed_layer: nn.Module) -> float:
        """Calculate compression ratio for a layer"""
        pass
    
    def configure(self, **kwargs):
        """Configure the compression method with parameters"""
        self.config.update(kwargs)
        
    def get_supported_phases(self) -> List[CompressionPhase]:
        """Get phases when this method can be applied"""
        return [CompressionPhase.POST_TRAINING]
    
    def get_supported_layer_types(self) -> List[type]:
        """Get list of layer types this method supports"""
        return [nn.Linear, nn.Conv2d, nn.Conv1d]
    
    def prepare_compression(self, model: nn.Module, config: CompressionConfig):
        """Prepare for compression (called before compress_layer)"""
        self.layer_info.clear()
        self.metrics.clear()
        
    def finalize_compression(self, model: nn.Module, config: CompressionConfig):
        """Finalize compression (called after all layers processed)"""
        pass
    
    def validate_compression(self, original_layer: nn.Module, compressed_layer: nn.Module, 
                           test_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Validate the compressed layer"""
        try:
            if test_input is None:
                # Create dummy input based on layer type
                if isinstance(original_layer, nn.Linear):
                    test_input = torch.randn(1, original_layer.in_features)
                elif isinstance(original_layer, (nn.Conv2d, nn.Conv1d)):
                    in_channels = original_layer.in_channels
                    if isinstance(original_layer, nn.Conv2d):
                        test_input = torch.randn(1, in_channels, 32, 32)
                    else:
                        test_input = torch.randn(1, in_channels, 100)
                else:
                    return {"valid": True, "reason": "No test input available"}
            
            with torch.no_grad():
                original_output = original_layer(test_input)
                compressed_output = compressed_layer(test_input)
                
                mse = torch.mean((original_output - compressed_output)**2).item()
                max_diff = torch.max(torch.abs(original_output - compressed_output)).item()
                
                return {
                    "valid": mse < 1.0,  # Configurable threshold
                    "mse": mse,
                    "max_difference": max_diff,
                    "output_shape_match": original_output.shape == compressed_output.shape
                }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this compression method"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.compression_type.value,
            "supported_phases": [phase.value for phase in self.get_supported_phases()],
            "supported_layer_types": [cls.__name__ for cls in self.get_supported_layer_types()],
            "config": self.config.copy()
        }

class BaseCompressor(ABC):
    """Base class for model compressors with plugin architecture"""
    
    def __init__(self):
        self.methods: Dict[str, CompressionMethod] = {}
        self.config: Optional[CompressionConfig] = None
        self.compression_history: List[Dict] = []
        self.layer_mapping: Dict[str, str] = {}
        
    def register_method(self, method: CompressionMethod):
        """Register a compression method"""
        if not isinstance(method, CompressionMethod):
            raise TypeError("Method must inherit from CompressionMethod")
            
        self.methods[method.name] = method
        logger.info(f"✓ Registered compression method: {method.name}")
        
    def get_method(self, name: str) -> CompressionMethod:
        """Get a registered compression method"""
        if name not in self.methods:
            available = list(self.methods.keys())
            raise ValueError(f"Method '{name}' not registered. Available: {available}")
        return self.methods[name]
    
    def list_methods(self) -> List[str]:
        """List all registered compression methods"""
        return list(self.methods.keys())
    
    @abstractmethod
    def compress_model(self, model: nn.Module, config: CompressionConfig) -> nn.Module:
        """Compress a model using configured methods"""
        pass
