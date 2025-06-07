"""Model analysis tools for TorchSlim"""

import torch
import torch.nn as nn
from typing import Dict, Any

class ModelAnalyzer:
    """Comprehensive model analysis tools"""
    
    @staticmethod
    def analyze_model_structure(model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture and structure"""
        
        layer_info = {}
        total_params = 0
        trainable_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layer_info[name] = {
                    'type': module.__class__.__name__,
                    'parameters': params,
                    'trainable_parameters': trainable,
                    'input_shape': getattr(module, 'in_features', None) or getattr(module, 'in_channels', None),
                    'output_shape': getattr(module, 'out_features', None) or getattr(module, 'out_channels', None),
                }
                
                total_params += params
                trainable_params += trainable
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'memory_mb': total_params * 4 / (1024 * 1024),  # float32
            'layer_details': layer_info
        }

class CompressionProfiler:
    """Profiling tools for compression performance"""
    
    @staticmethod
    def profile_compression_speed(compressor, model, config):
        """Profile compression speed"""
        import time
        start_time = time.time()
        compressed_model = compressor.compress_model(model, config)
        end_time = time.time()
        
        return {
            'compression_time': end_time - start_time,
            'compressed_model': compressed_model
        }

class VisualizationTools:
    """Visualization tools for compression analysis"""
    
    @staticmethod
    def plot_compression_ratios(layer_info):
        """Plot compression ratios by layer"""
        try:
            import matplotlib.pyplot as plt
            
            names = list(layer_info.keys())
            ratios = [info.compression_ratio for info in layer_info.values()]
            
            plt.figure(figsize=(12, 6))
            plt.bar(names, ratios)
            plt.xticks(rotation=45)
            plt.ylabel('Compression Ratio')
            plt.title('Compression Ratio by Layer')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for visualization")
