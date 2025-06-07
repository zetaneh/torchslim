"""Main TorchSlim compressor implementation"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import copy
import time
import logging
from tqdm import tqdm

from .base import BaseCompressor, CompressionConfig, LayerCompressionInfo
from .registry import create_method_instance, get_available_methods

logger = logging.getLogger(__name__)

class TorchSlim(BaseCompressor):
    """Main TorchSlim compressor with advanced plugin architecture"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        super().__init__()
        self.config = config or CompressionConfig()
        self.compression_stats = {}
        self.original_model = None
        self.layer_compression_info: Dict[str, LayerCompressionInfo] = {}
        
        # Auto-register all available methods
        self._auto_register_methods()
        
    def _auto_register_methods(self):
        """Automatically register all available compression methods"""
        available_methods = get_available_methods()
        for method_name in available_methods:
            try:
                method_instance = create_method_instance(method_name)
                self.register_method(method_instance)
            except Exception as e:
                logger.warning(f"Failed to register method {method_name}: {e}")
    
    def compress_model(self, model: nn.Module, config: Optional[CompressionConfig] = None) -> nn.Module:
        """Compress a model using configured methods"""
        if config:
            self.config = config
            
        if not self.config.enabled_methods:
            logger.warning("No compression methods enabled")
            return model
            
        logger.info(f"🚀 Starting compression with methods: {self.config.enabled_methods}")
        start_time = time.time()
        
        # Store original model reference
        self.original_model = model
        compressed_model = copy.deepcopy(model)
        
        # Track original statistics
        original_params = sum(p.numel() for p in model.parameters())
        
        try:
            # Apply each compression method
            for method_name in self.config.enabled_methods:
                if method_name not in self.methods:
                    logger.warning(f"Method '{method_name}' not available, skipping")
                    continue
                    
                logger.info(f"📦 Applying {method_name} compression...")
                compressed_model = self._apply_method(compressed_model, method_name)
        
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            if self.config.rollback_on_failure:
                logger.info("Rolling back to original model")
                return model
            raise
        
        # Calculate final statistics
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
        compression_time = time.time() - start_time
        
        self.compression_stats = {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compression_ratio,
            'compression_time_seconds': compression_time,
            'methods_applied': self.config.enabled_methods.copy(),
            'layer_details': dict(self.layer_compression_info)
        }
        
        logger.info(f"✅ Compression complete! Ratio: {compression_ratio:.2f}x, Time: {compression_time:.1f}s")
        return compressed_model
    
    def _apply_method(self, model: nn.Module, method_name: str) -> nn.Module:
        """Apply a single compression method to the model"""
        method = self.get_method(method_name)
        method_config = self.config.get_method_config(method_name)
        method.configure(**method_config)
        
        method.prepare_compression(model, self.config)
        
        layers_processed = 0
        for name, module in model.named_modules():
            if self._should_skip_layer(name, module):
                continue
                
            if method.can_compress_layer(module):
                try:
                    start_time = time.time()
                    original_params = sum(p.numel() for p in module.parameters())
                    
                    compressed_layer = method.compress_layer(module, name, **method_config)
                    
                    if self.config.validate_each_layer:
                        validation = method.validate_compression(module, compressed_layer)
                        if not validation.get("valid", True):
                            logger.warning(f"Validation failed for layer {name}")
                            if self.config.rollback_on_failure:
                                continue
                    
                    self._replace_module(model, name, compressed_layer)
                    
                    # Track compression info
                    compression_time = time.time() - start_time
                    compressed_params = sum(p.numel() for p in compressed_layer.parameters())
                    
                    self.layer_compression_info[name] = LayerCompressionInfo(
                        layer_name=name,
                        layer_type=module.__class__.__name__,
                        original_parameters=original_params,
                        compressed_parameters=compressed_params,
                        compression_ratio=original_params / compressed_params if compressed_params > 0 else 1.0,
                        method_used=method.name,
                        compression_time=compression_time,
                        memory_saved_mb=(original_params - compressed_params) * 4 / (1024 * 1024)
                    )
                    
                    layers_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to compress layer {name}: {e}")
        
        method.finalize_compression(model, self.config)
        logger.info(f"Applied {method.name} to {layers_processed} layers")
        return model
    
    def _should_skip_layer(self, layer_name: str, layer: nn.Module) -> bool:
        """Check if a layer should be skipped during compression"""
        if layer_name in self.config.exclude_layers:
            return True
        if self.config.include_layers and layer_name not in self.config.include_layers:
            return True
        return False
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def save_compressed_model(self, model: nn.Module, path: str):
        """Save compressed model with metadata"""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'compression_stats': self.compression_stats,
            'torchslim_version': "1.0.0"
        }
        torch.save(save_dict, path)
        logger.info(f"💾 Saved compressed model to {path}")
    
    def get_compression_report(self) -> Dict:
        """Get comprehensive compression report"""
        if not self.compression_stats:
            return {"error": "No compression performed yet"}
        
        return {
            "summary": {
                "compression_ratio": self.compression_stats['compression_ratio'],
                "parameter_reduction": self.compression_stats['original_parameters'] - self.compression_stats['compressed_parameters'],
                "methods_used": self.compression_stats['methods_applied']
            },
            "detailed_stats": self.compression_stats,
            "layer_breakdown": {name: info.__dict__ for name, info in self.layer_compression_info.items()}
        }
