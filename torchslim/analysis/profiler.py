"""
TorchSlim Compression Profiler
Comprehensive profiling tools for compression performance analysis
"""

import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProfilingResult:
    """Container for profiling results"""
    method_name: str
    compression_time: float
    compression_ratio: float
    memory_reduction_mb: float
    inference_speedup: float
    accuracy_preservation: float
    model_size_mb: float
    compressed_size_mb: float
    parameters_original: int
    parameters_compressed: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryProfile:
    """Memory usage profiling data"""
    peak_memory_mb: float
    current_memory_mb: float
    memory_efficiency: float
    gc_collections: int

@dataclass
class InferenceProfile:
    """Inference performance profiling data"""
    original_time_ms: float
    compressed_time_ms: float
    speedup_factor: float
    throughput_original: float
    throughput_compressed: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

class CompressionProfiler:
    """Comprehensive profiling tools for compression performance analysis"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results: List[ProfilingResult] = []
        self.profiling_history: List[Dict] = []
        
    def profile_compression_method(
        self,
        model: nn.Module,
        compression_method: str,
        compression_config: Dict[str, Any],
        test_data: torch.Tensor,
        target_data: Optional[torch.Tensor] = None,
        num_inference_runs: int = 100,
        detailed_profiling: bool = True
    ) -> ProfilingResult:
        """
        Comprehensive profiling of a single compression method
        
        Args:
            model: Original model to compress
            compression_method: Name of compression method
            compression_config: Configuration for the compression method
            test_data: Test input data for inference profiling
            target_data: Optional target data for accuracy testing
            num_inference_runs: Number of inference runs for timing
            detailed_profiling: Whether to run detailed profiling
            
        Returns:
            ProfilingResult containing all metrics
        """
        
        logger.info(f"🔍 Profiling compression method: {compression_method}")
        
        # Move model and data to device
        model = model.to(self.device)
        test_data = test_data.to(self.device)
        if target_data is not None:
            target_data = target_data.to(self.device)
        
        # Profile original model
        original_profile = self._profile_model(model, test_data, num_inference_runs, "original")
        
        # Profile compression process
        compression_profile = self._profile_compression_process(
            model, compression_method, compression_config
        )
        
        # Profile compressed model
        compressed_model = compression_profile['compressed_model']
        compressed_profile = self._profile_model(
            compressed_model, test_data, num_inference_runs, "compressed"
        )
        
        # Calculate accuracy preservation
        accuracy_preservation = self._calculate_accuracy_preservation(
            model, compressed_model, test_data, target_data
        )
        
        # Calculate metrics
        compression_ratio = original_profile['parameters'] / compressed_profile['parameters']
        memory_reduction = original_profile['memory_mb'] - compressed_profile['memory_mb']
        inference_speedup = original_profile['inference_time_ms'] / compressed_profile['inference_time_ms']
        
        # Create profiling result
        result = ProfilingResult(
            method_name=compression_method,
            compression_time=compression_profile['compression_time'],
            compression_ratio=compression_ratio,
            memory_reduction_mb=memory_reduction,
            inference_speedup=inference_speedup,
            accuracy_preservation=accuracy_preservation,
            model_size_mb=original_profile['memory_mb'],
            compressed_size_mb=compressed_profile['memory_mb'],
            parameters_original=original_profile['parameters'],
            parameters_compressed=compressed_profile['parameters'],
            additional_metrics={
                'original_profile': original_profile,
                'compressed_profile': compressed_profile,
                'compression_profile': compression_profile
            }
        )
        
        # Add detailed profiling if requested
        if detailed_profiling:
            result.additional_metrics.update(
                self._detailed_profiling(model, compressed_model, test_data)
            )
        
        self.results.append(result)
        return result
    
    def profile_multiple_methods(
        self,
        model: nn.Module,
        methods_config: Dict[str, Dict[str, Any]],
        test_data: torch.Tensor,
        target_data: Optional[torch.Tensor] = None,
        num_inference_runs: int = 100
    ) -> Dict[str, ProfilingResult]:
        """
        Profile multiple compression methods and compare results
        
        Args:
            model: Original model to compress
            methods_config: Dictionary mapping method names to their configs
            test_data: Test input data
            target_data: Optional target data for accuracy testing
            num_inference_runs: Number of inference runs for timing
            
        Returns:
            Dictionary mapping method names to ProfilingResult
        """
        
        results = {}
        
        for method_name, config in methods_config.items():
            try:
                logger.info(f"Profiling method: {method_name}")
                result = self.profile_compression_method(
                    model.clone() if hasattr(model, 'clone') else model,
                    method_name,
                    config,
                    test_data,
                    target_data,
                    num_inference_runs
                )
                results[method_name] = result
                
            except Exception as e:
                logger.error(f"Failed to profile {method_name}: {e}")
                results[method_name] = None
        
        return results
    
    def _profile_model(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        num_runs: int,
        model_type: str
    ) -> Dict[str, Any]:
        """Profile a model's characteristics and performance"""
        
        model.eval()
        
        # Calculate model size and parameters
        total_params = sum(p.numel() for p in model.parameters())
        memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Profile inference performance
        inference_profile = self._profile_inference(model, test_data, num_runs)
        
        # Memory profiling
        memory_profile = self._profile_memory_usage(model, test_data)
        
        return {
            'parameters': total_params,
            'memory_mb': memory_mb,
            'inference_time_ms': inference_profile.original_time_ms if model_type == "original" else inference_profile.compressed_time_ms,
            'throughput': inference_profile.throughput_original if model_type == "original" else inference_profile.throughput_compressed,
            'latency_p95': inference_profile.latency_p95,
            'memory_profile': memory_profile,
            'inference_profile': inference_profile
        }
    
    def _profile_compression_process(
        self,
        model: nn.Module,
        method_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile the compression process itself"""
        
        from ..core.compressor import TorchSlim
        from ..core.base import CompressionConfig
        
        # Setup compression
        compression_config = CompressionConfig()
        compression_config.add_method(method_name, **config)
        compressor = TorchSlim(compression_config)
        
        # Profile compression time
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        compressed_model = compressor.compress_model(model)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        compression_time = end_time - start_time
        memory_overhead = end_memory - start_memory
        
        return {
            'compressed_model': compressed_model,
            'compression_time': compression_time,
            'memory_overhead_mb': memory_overhead,
            'compressor_stats': compressor.compression_stats
        }
    
    def _profile_inference(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        num_runs: int
    ) -> InferenceProfile:
        """Detailed inference performance profiling"""
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data)
        
        # Synchronize if using CUDA
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Profile inference times
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="Profiling inference", leave=False):
                start_time = time.perf_counter()
                _ = model(test_data)
                
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        throughput = 1000 / avg_time  # samples per second
        
        return InferenceProfile(
            original_time_ms=avg_time,
            compressed_time_ms=avg_time,  # Will be overridden for compressed model
            speedup_factor=1.0,
            throughput_original=throughput,
            throughput_compressed=throughput,
            latency_p50=np.percentile(times, 50),
            latency_p95=np.percentile(times, 95),
            latency_p99=np.percentile(times, 99)
        )
    
    def _profile_memory_usage(
        self,
        model: nn.Module,
        test_data: torch.Tensor
    ) -> MemoryProfile:
        """Profile memory usage during inference"""
        
        # Clear memory
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        initial_memory = self._get_memory_usage()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            _ = model(test_data)
        
        peak_memory = self._get_peak_memory_usage()
        current_memory = self._get_memory_usage()
        
        gc_before = gc.get_count()
        gc.collect()
        gc_after = gc.get_count()
        
        return MemoryProfile(
            peak_memory_mb=peak_memory,
            current_memory_mb=current_memory,
            memory_efficiency=(current_memory - initial_memory) / peak_memory if peak_memory > 0 else 1.0,
            gc_collections=sum(gc_before) - sum(gc_after)
        )
    
    def _calculate_accuracy_preservation(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.Tensor,
        target_data: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate how well the compressed model preserves accuracy"""
        
        original_model.eval()
        compressed_model.eval()
        
        with torch.no_grad():
            original_output = original_model(test_data)
            compressed_output = compressed_model(test_data)
            
            if target_data is not None:
                # Calculate actual accuracy if targets provided
                original_predictions = torch.argmax(original_output, dim=1)
                compressed_predictions = torch.argmax(compressed_output, dim=1)
                
                original_accuracy = (original_predictions == target_data).float().mean().item()
                compressed_accuracy = (compressed_predictions == target_data).float().mean().item()
                
                return compressed_accuracy / original_accuracy if original_accuracy > 0 else 0.0
            else:
                # Calculate output similarity
                mse = torch.mean((original_output - compressed_output)**2).item()
                max_val = torch.max(torch.abs(original_output)).item()
                
                # Return normalized similarity (1.0 = identical, 0.0 = completely different)
                similarity = 1.0 - min(mse / (max_val**2 + 1e-8), 1.0)
                return similarity
    
    def _detailed_profiling(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform detailed profiling analysis"""
        
        detailed_metrics = {}
        
        # Layer-wise analysis
        detailed_metrics['layer_analysis'] = self._layer_wise_analysis(
            original_model, compressed_model
        )
        
        # Gradient flow analysis (if models are in training mode)
        if original_model.training:
            detailed_metrics['gradient_analysis'] = self._gradient_flow_analysis(
                original_model, compressed_model, test_data
            )
        
        # Output distribution analysis
        detailed_metrics['output_distribution'] = self._output_distribution_analysis(
            original_model, compressed_model, test_data
        )
        
        return detailed_metrics
    
    def _layer_wise_analysis(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module
    ) -> Dict[str, Any]:
        """Analyze compression effects on individual layers"""
        
        analysis = {}
        
        original_layers = dict(original_model.named_modules())
        compressed_layers = dict(compressed_model.named_modules())
        
        for name, orig_layer in original_layers.items():
            if name in compressed_layers and hasattr(orig_layer, 'weight'):
                comp_layer = compressed_layers[name]
                
                orig_params = sum(p.numel() for p in orig_layer.parameters())
                comp_params = sum(p.numel() for p in comp_layer.parameters())
                
                analysis[name] = {
                    'original_parameters': orig_params,
                    'compressed_parameters': comp_params,
                    'compression_ratio': orig_params / comp_params if comp_params > 0 else float('inf'),
                    'layer_type': orig_layer.__class__.__name__
                }
        
        return analysis
    
    def _gradient_flow_analysis(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze gradient flow in compressed vs original model"""
        
        # This is a simplified gradient analysis
        # In practice, you'd want more sophisticated gradient flow analysis
        
        original_grads = []
        compressed_grads = []
        
        # Calculate gradients for original model
        original_model.train()
        original_output = original_model(test_data)
        dummy_loss = original_output.sum()
        dummy_loss.backward()
        
        for param in original_model.parameters():
            if param.grad is not None:
                original_grads.append(param.grad.norm().item())
        
        # Calculate gradients for compressed model
        compressed_model.train()
        compressed_output = compressed_model(test_data)
        dummy_loss = compressed_output.sum()
        dummy_loss.backward()
        
        for param in compressed_model.parameters():
            if param.grad is not None:
                compressed_grads.append(param.grad.norm().item())
        
        return {
            'original_grad_norms': original_grads,
            'compressed_grad_norms': compressed_grads,
            'gradient_preservation': np.corrcoef(original_grads[:len(compressed_grads)], compressed_grads)[0, 1] if len(compressed_grads) > 1 else 1.0
        }
    
    def _output_distribution_analysis(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze how compression affects output distributions"""
        
        original_model.eval()
        compressed_model.eval()
        
        with torch.no_grad():
            original_output = original_model(test_data)
            compressed_output = compressed_model(test_data)
            
            # Statistical analysis
            original_stats = {
                'mean': torch.mean(original_output).item(),
                'std': torch.std(original_output).item(),
                'min': torch.min(original_output).item(),
                'max': torch.max(original_output).item()
            }
            
            compressed_stats = {
                'mean': torch.mean(compressed_output).item(),
                'std': torch.std(compressed_output).item(),
                'min': torch.min(compressed_output).item(),
                'max': torch.max(compressed_output).item()
            }
            
            # Distribution similarity metrics
            mse = torch.mean((original_output - compressed_output)**2).item()
            mae = torch.mean(torch.abs(original_output - compressed_output)).item()
            
            # Correlation
            orig_flat = original_output.flatten()
            comp_flat = compressed_output.flatten()
            correlation = torch.corrcoef(torch.stack([orig_flat, comp_flat]))[0, 1].item()
        
        return {
            'original_statistics': original_stats,
            'compressed_statistics': compressed_stats,
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.startswith('cuda'):
            return torch.cuda.memory_allocated(self.device) / (1024**2)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024**2)
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB"""
        if self.device.startswith('cuda'):
            return torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            # For CPU, return current usage as approximation
            return self._get_memory_usage()
    
    def generate_report(
        self,
        results: Optional[List[ProfilingResult]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive profiling report"""
        
        if results is None:
            results = self.results
        
        if not results:
            return "No profiling results available."
        
        report = []
        report.append("🔥 TorchSlim Compression Profiling Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("📊 Summary")
        report.append("-" * 30)
        report.append(f"{'Method':<20} {'Ratio':<8} {'Speed':<8} {'Accuracy':<10} {'Time(s)':<10}")
        report.append("-" * 60)
        
        for result in results:
            report.append(
                f"{result.method_name:<20} "
                f"{result.compression_ratio:<8.2f} "
                f"{result.inference_speedup:<8.2f} "
                f"{result.accuracy_preservation:<10.3f} "
                f"{result.compression_time:<10.2f}"
            )
        
        report.append("")
        
        # Detailed analysis for each method
        for result in results:
            report.append(f"🔍 Detailed Analysis: {result.method_name}")
            report.append("-" * 40)
            report.append(f"  Compression Ratio: {result.compression_ratio:.2f}x")
            report.append(f"  Memory Reduction: {result.memory_reduction_mb:.2f} MB")
            report.append(f"  Inference Speedup: {result.inference_speedup:.2f}x")
            report.append(f"  Accuracy Preservation: {result.accuracy_preservation:.3f}")
            report.append(f"  Compression Time: {result.compression_time:.2f}s")
            report.append(f"  Original Parameters: {result.parameters_original:,}")
            report.append(f"  Compressed Parameters: {result.parameters_compressed:,}")
            report.append("")
        
        # Recommendations
        report.append("💡 Recommendations")
        report.append("-" * 20)
        
        if results:
            best_ratio = max(results, key=lambda x: x.compression_ratio)
            best_speed = max(results, key=lambda x: x.inference_speedup)
            best_accuracy = max(results, key=lambda x: x.accuracy_preservation)
            
            report.append(f"  Best Compression: {best_ratio.method_name} ({best_ratio.compression_ratio:.2f}x)")
            report.append(f"  Best Speed: {best_speed.method_name} ({best_speed.inference_speedup:.2f}x)")
            report.append(f"  Best Accuracy: {best_accuracy.method_name} ({best_accuracy.accuracy_preservation:.3f})")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def compare_methods(
        self,
        results: Optional[List[ProfilingResult]] = None
    ) -> Dict[str, Any]:
        """Compare compression methods across different metrics"""
        
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results to compare"}
        
        comparison = {
            'methods': [r.method_name for r in results],
            'compression_ratios': [r.compression_ratio for r in results],
            'inference_speedups': [r.inference_speedup for r in results],
            'accuracy_preservations': [r.accuracy_preservation for r in results],
            'compression_times': [r.compression_time for r in results],
            'memory_reductions': [r.memory_reduction_mb for r in results]
        }
        
        # Calculate rankings
        comparison['rankings'] = {}
        for metric in ['compression_ratios', 'inference_speedups', 'accuracy_preservations']:
            values = comparison[metric]
            ranked_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            comparison['rankings'][metric] = [results[i].method_name for i in ranked_indices]
        
        # Calculate Pareto efficiency
        comparison['pareto_efficient'] = self._find_pareto_efficient_methods(results)
        
        return comparison
    
    def _find_pareto_efficient_methods(
        self,
        results: List[ProfilingResult]
    ) -> List[str]:
        """Find Pareto efficient compression methods"""
        
        # Pareto efficiency based on compression ratio, speed, and accuracy
        efficient_methods = []
        
        for i, result_i in enumerate(results):
            is_dominated = False
            
            for j, result_j in enumerate(results):
                if i != j:
                    # Check if result_j dominates result_i
                    if (result_j.compression_ratio >= result_i.compression_ratio and
                        result_j.inference_speedup >= result_i.inference_speedup and
                        result_j.accuracy_preservation >= result_i.accuracy_preservation and
                        (result_j.compression_ratio > result_i.compression_ratio or
                         result_j.inference_speedup > result_i.inference_speedup or
                         result_j.accuracy_preservation > result_i.accuracy_preservation)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                efficient_methods.append(result_i.method_name)
        
        return efficient_methods
    
    def export_results(
        self,
        results: Optional[List[ProfilingResult]] = None,
        format: str = 'json',
        save_path: Optional[str] = None
    ) -> str:
        """Export profiling results in various formats"""
        
        if results is None:
            results = self.results
        
        if format == 'json':
            import json
            
            export_data = []
            for result in results:
                data = {
                    'method_name': result.method_name,
                    'compression_time': result.compression_time,
                    'compression_ratio': result.compression_ratio,
                    'memory_reduction_mb': result.memory_reduction_mb,
                    'inference_speedup': result.inference_speedup,
                    'accuracy_preservation': result.accuracy_preservation,
                    'model_size_mb': result.model_size_mb,
                    'compressed_size_mb': result.compressed_size_mb,
                    'parameters_original': result.parameters_original,
                    'parameters_compressed': result.parameters_compressed
                }
                export_data.append(data)
            
            export_text = json.dumps(export_data, indent=2)
            
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = [
                'method_name', 'compression_time', 'compression_ratio',
                'memory_reduction_mb', 'inference_speedup', 'accuracy_preservation',
                'model_size_mb', 'compressed_size_mb', 'parameters_original',
                'parameters_compressed'
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'method_name': result.method_name,
                    'compression_time': result.compression_time,
                    'compression_ratio': result.compression_ratio,
                    'memory_reduction_mb': result.memory_reduction_mb,
                    'inference_speedup': result.inference_speedup,
                    'accuracy_preservation': result.accuracy_preservation,
                    'model_size_mb': result.model_size_mb,
                    'compressed_size_mb': result.compressed_size_mb,
                    'parameters_original': result.parameters_original,
                    'parameters_compressed': result.parameters_compressed
                })
            
            export_text = output.getvalue()
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(export_text)
            logger.info(f"Results exported to {save_path}")
        
        return export_text


# Convenience functions for quick profiling
def quick_profile(
    model: nn.Module,
    compression_methods: Dict[str, Dict[str, Any]],
    test_input: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, ProfilingResult]:
    """Quick profiling of multiple compression methods"""
    
    profiler = CompressionProfiler(device=device)
    return profiler.profile_multiple_methods(
        model, compression_methods, test_input
    )

def profile_single_method(
    model: nn.Module,
    method_name: str,
    method_config: Dict[str, Any],
    test_input: torch.Tensor,
    device: str = 'cpu'
) -> ProfilingResult:
    """Profile a single compression method"""
    
    profiler = CompressionProfiler(device=device)
    return profiler.profile_compression_method(
        model, method_name, method_config, test_input
    )