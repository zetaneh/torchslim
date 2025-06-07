"""
Comprehensive benchmarking utilities for TorchSlim compression methods.

This module provides tools for benchmarking compression methods, comparing performance,
and generating detailed reports.
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import pickle
from collections import defaultdict
import psutil
import gc

from ..core.base import CompressionMethod
from ..core.registry import get_available_methods, create_method_instance
from ..core.compressor import TorchSlim
from ..core.base import CompressionConfig
from ..analysis.profiler import CompressionProfiler, ProfilingResult
from ..utils.models import create_test_model
from ..utils.validation import validate_model_accuracy

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    methods_to_test: List[str] = field(default_factory=list)
    method_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_models: List[Dict[str, Any]] = field(default_factory=list)
    num_warmup_runs: int = 3
    num_timing_runs: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    device: str = "cpu"
    save_intermediate_results: bool = True
    output_dir: Optional[str] = None
    include_accuracy_test: bool = True
    include_memory_profiling: bool = True
    include_flops_analysis: bool = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method_name: str
    model_name: str
    config: Dict[str, Any]
    
    # Compression metrics
    compression_ratio: float
    parameter_reduction: float
    model_size_reduction: float
    
    # Performance metrics
    compression_time: float
    inference_times: Dict[int, float]  # batch_size -> time
    memory_usage: Dict[str, float]
    
    # Quality metrics
    accuracy_drop: Optional[float] = None
    output_similarity: Optional[float] = None
    
    # Additional metrics
    flops_reduction: Optional[float] = None
    energy_consumption: Optional[float] = None
    
    # Metadata
    timestamp: str = ""
    device: str = "cpu"
    torch_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'model_name': self.model_name,
            'config': self.config,
            'compression_ratio': self.compression_ratio,
            'parameter_reduction': self.parameter_reduction,
            'model_size_reduction': self.model_size_reduction,
            'compression_time': self.compression_time,
            'inference_times': self.inference_times,
            'memory_usage': self.memory_usage,
            'accuracy_drop': self.accuracy_drop,
            'output_similarity': self.output_similarity,
            'flops_reduction': self.flops_reduction,
            'energy_consumption': self.energy_consumption,
            'timestamp': self.timestamp,
            'device': self.device,
            'torch_version': self.torch_version
        }


class ModelBenchmarker:
    """Benchmarks compression methods on various models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: List[BenchmarkResult] = []
        self.profiler = CompressionProfiler(device=config.device)
        
        # Set up output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path("benchmark_results")
            self.output_dir.mkdir(exist_ok=True)
            
        logger.info(f"Benchmarker initialized with device: {self.device}")
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        logger.info("Starting comprehensive benchmark suite...")
        
        total_runs = len(self.config.methods_to_test) * len(self.config.test_models)
        current_run = 0
        
        for model_config in self.config.test_models:
            model = self._create_model(model_config)
            model_name = model_config.get('name', 'unknown_model')
            
            logger.info(f"Benchmarking model: {model_name}")
            
            for method_name in self.config.methods_to_test:
                current_run += 1
                logger.info(f"Progress: {current_run}/{total_runs} - Method: {method_name}")
                
                try:
                    result = self._benchmark_method_on_model(
                        method_name, model, model_name, model_config
                    )
                    self.results.append(result)
                    
                    if self.config.save_intermediate_results:
                        self._save_intermediate_result(result)
                        
                except Exception as e:
                    logger.error(f"Failed to benchmark {method_name} on {model_name}: {e}")
                    continue
        
        logger.info("Benchmark suite completed!")
        return self.results
    
    def _create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create a model from configuration."""
        model_type = model_config.get('type', 'mlp')
        
        if model_type == 'custom':
            # For custom models, expect a 'model' key with the actual model
            return model_config['model'].to(self.device)
        else:
            # Use the create_test_model utility
            model = create_test_model(
                model_type=model_type,
                **{k: v for k, v in model_config.items() if k not in ['type', 'name']}
            )
            return model.to(self.device)
    
    def _benchmark_method_on_model(
        self, 
        method_name: str, 
        model: nn.Module, 
        model_name: str,
        model_config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Benchmark a single method on a single model."""
        
        # Get method configuration
        method_config = self.config.method_configs.get(method_name, {})
        
        # Create compression configuration
        compression_config = CompressionConfig()
        compression_config.add_method(method_name, **method_config)
        
        # Initialize result
        result = BenchmarkResult(
            method_name=method_name,
            model_name=model_name,
            config=method_config,
            compression_ratio=0.0,
            parameter_reduction=0.0,
            model_size_reduction=0.0,
            compression_time=0.0,
            inference_times={},
            memory_usage={},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            device=str(self.device),
            torch_version=torch.__version__
        )
        
        # Create a copy of the model for compression
        model_copy = self._deep_copy_model(model)
        
        # Measure compression time and perform compression
        start_time = time.time()
        compressor = TorchSlim(compression_config)
        
        try:
            compressed_model = compressor.compress_model(model_copy)
            compression_time = time.time() - start_time
            result.compression_time = compression_time
            
            # Calculate compression metrics
            original_params = sum(p.numel() for p in model.parameters())
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            
            result.compression_ratio = original_params / compressed_params if compressed_params > 0 else float('inf')
            result.parameter_reduction = (original_params - compressed_params) / original_params
            
            # Model size reduction (approximate)
            original_size = sum(p.numel() * p.element_size() for p in model.parameters())
            compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
            result.model_size_reduction = (original_size - compressed_size) / original_size
            
            # Benchmark inference times
            result.inference_times = self._benchmark_inference_times(model, compressed_model)
            
            # Memory profiling
            if self.config.include_memory_profiling:
                result.memory_usage = self._profile_memory_usage(model, compressed_model)
            
            # Accuracy testing
            if self.config.include_accuracy_test:
                result.accuracy_drop, result.output_similarity = self._test_accuracy_preservation(
                    model, compressed_model, model_config
                )
            
            # FLOPS analysis
            if self.config.include_flops_analysis:
                result.flops_reduction = self._analyze_flops_reduction(model, compressed_model)
                
        except Exception as e:
            logger.error(f"Compression failed for {method_name}: {e}")
            # Set default values for failed compression
            result.compression_ratio = 1.0
            result.parameter_reduction = 0.0
            result.model_size_reduction = 0.0
            result.compression_time = float('inf')
        
        return result
    
    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return model_copy.to(self.device)
    
    def _benchmark_inference_times(
        self, 
        original_model: nn.Module, 
        compressed_model: nn.Module
    ) -> Dict[int, float]:
        """Benchmark inference times for different batch sizes."""
        inference_times = {}
        
        original_model.eval()
        compressed_model.eval()
        
        for batch_size in self.config.batch_sizes:
            # Create dummy input
            input_shape = self._get_input_shape(original_model, batch_size)
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(self.config.num_warmup_runs):
                    _ = compressed_model(dummy_input)
            
            # Timing runs
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            times = []
            with torch.no_grad():
                for _ in range(self.config.num_timing_runs):
                    start_time = time.perf_counter()
                    _ = compressed_model(dummy_input)
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            inference_times[batch_size] = avg_time
        
        return inference_times
    
    def _get_input_shape(self, model: nn.Module, batch_size: int) -> Tuple[int, ...]:
        """Infer input shape for the model."""
        # Try to find the first layer and infer input shape
        first_layer = next(iter(model.modules()))
        
        if isinstance(first_layer, nn.Linear):
            return (batch_size, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv2d):
            # Assume square input for simplicity
            return (batch_size, first_layer.in_channels, 32, 32)
        else:
            # Default fallback
            return (batch_size, 784)
    
    def _profile_memory_usage(
        self, 
        original_model: nn.Module, 
        compressed_model: nn.Module
    ) -> Dict[str, float]:
        """Profile memory usage of models."""
        memory_usage = {}
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure model memory footprint
        original_memory = self._get_model_memory_footprint(original_model)
        compressed_memory = self._get_model_memory_footprint(compressed_model)
        
        memory_usage['original_model_mb'] = original_memory
        memory_usage['compressed_model_mb'] = compressed_memory
        memory_usage['memory_reduction'] = (original_memory - compressed_memory) / original_memory
        
        # System memory usage
        process = psutil.Process()
        memory_usage['system_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        return memory_usage
    
    def _get_model_memory_footprint(self, model: nn.Module) -> float:
        """Calculate model memory footprint in MB."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / 1024 / 1024
    
    def _test_accuracy_preservation(
        self, 
        original_model: nn.Module, 
        compressed_model: nn.Module,
        model_config: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Test accuracy preservation and output similarity."""
        try:
            # Generate test data
            batch_size = 32
            input_shape = self._get_input_shape(original_model, batch_size)
            test_input = torch.randn(input_shape).to(self.device)
            
            original_model.eval()
            compressed_model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_input)
                compressed_output = compressed_model(test_input)
            
            # Calculate output similarity (cosine similarity)
            similarity = torch.nn.functional.cosine_similarity(
                original_output.flatten(), 
                compressed_output.flatten(), 
                dim=0
            ).item()
            
            # For classification tasks, calculate accuracy drop
            accuracy_drop = None
            if hasattr(model_config, 'task') and model_config['task'] == 'classification':
                original_pred = torch.argmax(original_output, dim=1)
                compressed_pred = torch.argmax(compressed_output, dim=1)
                accuracy = (original_pred == compressed_pred).float().mean().item()
                accuracy_drop = 1.0 - accuracy
            
            return accuracy_drop, similarity
            
        except Exception as e:
            logger.warning(f"Accuracy testing failed: {e}")
            return None, None
    
    def _analyze_flops_reduction(
        self, 
        original_model: nn.Module, 
        compressed_model: nn.Module
    ) -> Optional[float]:
        """Analyze FLOPS reduction (simplified implementation)."""
        try:
            # This is a simplified FLOPS calculation
            # In practice, you might want to use a library like fvcore or thop
            
            def count_flops(model):
                total_flops = 0
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        total_flops += module.in_features * module.out_features
                    elif isinstance(module, nn.Conv2d):
                        # Simplified conv FLOPS calculation
                        total_flops += (module.in_channels * module.out_channels * 
                                      module.kernel_size[0] * module.kernel_size[1])
                return total_flops
            
            original_flops = count_flops(original_model)
            compressed_flops = count_flops(compressed_model)
            
            if original_flops > 0:
                return (original_flops - compressed_flops) / original_flops
            else:
                return None
                
        except Exception as e:
            logger.warning(f"FLOPS analysis failed: {e}")
            return None
    
    def _save_intermediate_result(self, result: BenchmarkResult):
        """Save intermediate result to disk."""
        filename = f"{result.method_name}_{result.model_name}_{int(time.time())}.json"
        filepath = self.output_dir / "intermediate" / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TorchSlim Compression Benchmark Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Device: {self.config.device}")
        report_lines.append(f"Total runs: {len(self.results)}")
        report_lines.append("")
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        # Summary statistics
        report_lines.append("Summary Statistics:")
        report_lines.append("-" * 40)
        
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            avg_compression = method_data['compression_ratio'].mean()
            avg_time = method_data['compression_time'].mean()
            
            report_lines.append(f"{method}:")
            report_lines.append(f"  Average compression ratio: {avg_compression:.2f}x")
            report_lines.append(f"  Average compression time: {avg_time:.3f}s")
            report_lines.append(f"  Parameter reduction: {method_data['parameter_reduction'].mean():.1%}")
            report_lines.append("")
        
        # Best performers
        report_lines.append("Best Performers:")
        report_lines.append("-" * 40)
        
        best_compression = df.loc[df['compression_ratio'].idxmax()]
        best_speed = df.loc[df['compression_time'].idxmin()]
        
        report_lines.append(f"Best compression ratio: {best_compression['method_name']} "
                          f"({best_compression['compression_ratio']:.2f}x)")
        report_lines.append(f"Fastest compression: {best_speed['method_name']} "
                          f"({best_speed['compression_time']:.3f}s)")
        report_lines.append("")
        
        # Detailed results table
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 40)
        
        # Create a summary table
        summary_cols = ['method_name', 'model_name', 'compression_ratio', 
                       'compression_time', 'parameter_reduction']
        summary_df = df[summary_cols].round(3)
        report_lines.append(summary_df.to_string(index=False))
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {save_path}")
        
        return report_content
    
    def save_results(self, filepath: str):
        """Save all benchmark results to file."""
        results_data = [result.to_dict() for result in self.results]
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
        elif filepath.endswith('.pickle'):
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
        else:
            # Default to CSV
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Results saved to: {filepath}")
    
    def load_results(self, filepath: str):
        """Load benchmark results from file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                results_data = json.load(f)
            # Convert back to BenchmarkResult objects
            self.results = [BenchmarkResult(**data) for data in results_data]
        elif filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pickle")
        
        logger.info(f"Loaded {len(self.results)} results from: {filepath}")


class ComparisonBenchmark:
    """Specialized benchmarking for comparing compression methods."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.benchmarker = None
    
    def quick_comparison(
        self, 
        model: nn.Module, 
        methods: List[str],
        method_configs: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Quick comparison of multiple methods on a single model."""
        
        if method_configs is None:
            method_configs = {method: {} for method in methods}
        
        # Create benchmark config
        config = BenchmarkConfig(
            methods_to_test=methods,
            method_configs=method_configs,
            test_models=[{'type': 'custom', 'model': model, 'name': 'test_model'}],
            num_timing_runs=5,
            batch_sizes=[1, 8],
            device=self.device,
            save_intermediate_results=False
        )
        
        # Run benchmark
        benchmarker = ModelBenchmarker(config)
        results = benchmarker.run_full_benchmark()
        
        # Convert to DataFrame for easy comparison
        df = pd.DataFrame([result.to_dict() for result in results])
        
        # Select key comparison metrics
        comparison_cols = [
            'method_name', 'compression_ratio', 'parameter_reduction',
            'compression_time', 'output_similarity'
        ]
        
        return df[comparison_cols].round(3)
    
    def pareto_analysis(
        self, 
        results: List[BenchmarkResult],
        x_metric: str = 'compression_ratio',
        y_metric: str = 'output_similarity'
    ) -> List[BenchmarkResult]:
        """Find Pareto optimal compression methods."""
        
        # Extract metric values
        points = []
        for result in results:
            x_val = getattr(result, x_metric, 0)
            y_val = getattr(result, y_metric, 0)
            if x_val is not None and y_val is not None:
                points.append((x_val, y_val, result))
        
        # Find Pareto frontier
        pareto_optimal = []
        for i, (x1, y1, result1) in enumerate(points):
            is_pareto = True
            for j, (x2, y2, result2) in enumerate(points):
                if i != j and x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_optimal.append(result1)
        
        return pareto_optimal


def run_standard_benchmark(
    output_dir: str = "benchmark_results",
    device: str = "cpu"
) -> ModelBenchmarker:
    """Run a standard benchmark suite with common models and methods."""
    
    # Define standard test models
    test_models = [
        {
            'type': 'mlp',
            'name': 'small_mlp',
            'input_size': 784,
            'hidden_sizes': [256, 128],
            'output_size': 10
        },
        {
            'type': 'mlp',
            'name': 'large_mlp',
            'input_size': 784,
            'hidden_sizes': [1024, 512, 256],
            'output_size': 10
        },
        {
            'type': 'cnn',
            'name': 'simple_cnn',
            'input_channels': 3,
            'num_classes': 10,
            'hidden_dims': [64, 128]
        }
    ]
    
    # Define standard method configurations
    method_configs = {
        'svd': {'rank_ratio': 0.5},
        'pruning': {'pruning_ratio': 0.2, 'pruning_type': 'magnitude'},
        'quantization': {'bits': 8, 'scheme': 'asymmetric'},
        'lora': {'rank': 16, 'alpha': 32}
    }
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        methods_to_test=list(method_configs.keys()),
        method_configs=method_configs,
        test_models=test_models,
        batch_sizes=[1, 8, 32],
        device=device,
        output_dir=output_dir,
        include_accuracy_test=True,
        include_memory_profiling=True
    )
    
    # Run benchmark
    benchmarker = ModelBenchmarker(config)
    results = benchmarker.run_full_benchmark()
    
    # Generate and save report
    report = benchmarker.generate_report()
    print(report)
    
    # Save results
    benchmarker.save_results(f"{output_dir}/standard_benchmark_results.json")
    
    return benchmarker


if __name__ == "__main__":
    # Example usage
    print("Running standard TorchSlim benchmark...")
    benchmarker = run_standard_benchmark()
    print(f"Benchmark completed with {len(benchmarker.results)} results.")