# 🔥 TorchSlim

[![PyPI version](https://badge.fury.io/py/torchslim.svg)](https://badge.fury.io/py/torchslim)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://github.com/zetaneh/torchslim/workflows/CI/badge.svg)](https://github.com/zetaneh/torchslim/actions)

**Extensible PyTorch Model Compression Library with Plugin-Based Architecture**

TorchSlim is a comprehensive framework for neural network compression that provides a modular, plugin-based architecture for easy integration of new compression methods. Built for researchers and practitioners who need flexible, extensible compression tools.

## 🌟 Key Features

- 🔌 **Plugin Architecture**: Easy to add new compression methods
- 🗜️ **Multiple Compression Methods**: SVD, Quantization, Pruning, LoRA, Knowledge Distillation, etc.
- 📊 **Comprehensive Analysis**: Profiling, visualization, and benchmarking tools
- ⚡ **Production Ready**: Complete package structure with proper Python packaging
- 🎛️ **Flexible Configuration**: Mix and match compression techniques
- 📈 **Advanced Visualizations**: Static and interactive plots for analysis

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install torchslim

# Or install from GitHub (development version)
pip install git+https://github.com/zetaneh/torchslim.git



# Install for development
git clone https://github.com/zetaneh/torchslim.git
cd torchslim
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from torchslim import TorchSlim, CompressionConfig, create_test_model

# Create a model
model = create_test_model("mlp", input_size=784, hidden_sizes=[512, 256], output_size=10)

# Configure compression
config = CompressionConfig()
config.add_method("svd", rank_ratio=0.5)
config.add_method("pruning", pruning_ratio=0.2)

# Compress the model
compressor = TorchSlim(config)
compressed_model = compressor.compress_model(model)

# Get compression report
report = compressor.get_compression_report()
print(f"Compression ratio: {report['summary']['compression_ratio']:.2f}x")
```

### Quick Compression

```python
from torchslim import quick_compress

# One-liner compression
compressed_model, report = quick_compress(model, method_name="svd", rank_ratio=0.5)
```

## 📦 Available Compression Methods

| Method | Type | Description | Best For |
|--------|------|-------------|----------|
| **SVD** | Structural | Low-rank matrix decomposition | Linear layers, 2-4x compression |
| **Quantization** | Parametric | Reduce weight precision | Memory optimization, 4x+ compression |
| **Pruning** | Structural | Remove unimportant weights | General compression, 1.5-3x |
| **LoRA** | Structural | Low-rank adaptation | Fine-tuning efficiency, 10x+ reduction |
| **Knowledge Distillation** | Training | Teacher-student training | Training-time compression |
| **Tensor Decomposition** | Structural | Tucker/CP decomposition | Convolutional layers |
| **Weight Clustering** | Parametric | Reduce unique weight values | Hardware-specific optimizations |

## 🔧 Advanced Usage

### Custom Compression Pipeline

```python
# Multi-stage compression with validation
config = CompressionConfig()
config.add_method("svd", rank_ratio=0.7, energy_threshold=0.95)
config.add_method("pruning", pruning_type="structured", pruning_ratio=0.15)
config.add_method("quantization", bits=8, scheme="asymmetric")

# Advanced options
config.preserve_first_layer = True
config.preserve_last_layer = True
config.validate_each_layer = True

compressor = TorchSlim(config)
compressed_model = compressor.compress_model(model)
```

### Analysis and Benchmarking

```python
from torchslim.analysis import CompressionProfiler, VisualizationTools
from torchslim.utils import benchmark_methods

# Comprehensive profiling
profiler = CompressionProfiler(device='cpu')
methods_config = {
    "svd": {"rank_ratio": 0.5},
    "pruning": {"pruning_ratio": 0.2},
    "quantization": {"bits": 8}
}

results = profiler.profile_multiple_methods(model, methods_config, test_data)

# Generate visualizations
viz = VisualizationTools()
viz.plot_compression_comparison(results, save_path='comparison.png')
viz.create_dashboard(results, save_path='dashboard.html')
```

### Creating Custom Methods

```python
from torchslim.core.base import CompressionMethod, CompressionType
from torchslim import register_method

class MyCustomCompression(CompressionMethod):
    def __init__(self):
        super().__init__(
            name="my_method",
            description="My custom compression technique",
            compression_type=CompressionType.PARAMETRIC
        )
    
    def compress_layer(self, layer, layer_name, **kwargs):
        # Your compression logic here
        return compressed_layer
    
    def can_compress_layer(self, layer):
        return isinstance(layer, torch.nn.Linear)
    
    def get_compression_ratio(self, original, compressed):
        return original_params / compressed_params

# Register and use
register_method("my_method", MyCustomCompression)
config.add_method("my_method", custom_param=0.5)
```

## 📊 Validation and Quality Assessment

```python
from torchslim.utils.validation import validate_compression_pipeline, print_validation_report

# Comprehensive validation
results = validate_compression_pipeline(original_model, compressed_model, methods_used)
print_validation_report(results)

# Quick validation
from torchslim.utils.validation import quick_validate
is_valid = quick_validate(original_model, compressed_model)
```

## 🏗️ Architecture Overview

```
TorchSlim Framework
├── Core Components
│   ├── CompressionMethod (Abstract Base Class)
│   ├── TorchSlim (Main Compressor)
│   ├── CompressionConfig (Configuration Management)
│   └── Registry (Method Registration & Discovery)
├── Compression Methods (Plugin System)
│   ├── SVDCompression
│   ├── QuantizationCompression
│   ├── PruningCompression
│   └── ... (Custom Methods)
├── Analysis Tools
│   ├── ModelAnalyzer
│   ├── CompressionProfiler
│   └── VisualizationTools
└── Utilities
    ├── Model Creation
    ├── Benchmarking
    └── Validation
```

## 🔬 Research and Benchmarks

TorchSlim includes comprehensive benchmarking tools for research:

```python
from torchslim.utils.benchmarks import run_standard_benchmark

# Run standard benchmark suite
benchmarker = run_standard_benchmark(device="cuda")
print(f"Completed {len(benchmarker.results)} compression experiments")
```

## 📚 Documentation

- **[API Documentation](https://torchslim.readthedocs.io/)** - Complete API reference
- **[User Guide](https://torchslim.readthedocs.io/en/latest/user_guide.html)** - Detailed usage examples
- **[Developer Guide](https://torchslim.readthedocs.io/en/latest/developer_guide.html)** - How to contribute
- **[Examples](./examples/)** - Jupyter notebooks and scripts

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
git clone https://github.com/zetaneh/torchslim.git
cd torchslim
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black torchslim tests examples
flake8 torchslim tests examples

# Build docs
cd docs && make html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research community for compression techniques and methodologies
- Contributors and users of TorchSlim

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zetaneh/torchslim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zetaneh/torchslim/discussions)
- **Email**: abraich.jons+torchslim@gmail.com

## 🎯 Roadmap

- [ ] Additional compression methods (Distillation variants, NAS-based compression)
- [ ] Hardware-specific optimizations (GPU, mobile, edge devices)
- [ ] Integration with popular ML frameworks (Hugging Face, TorchVision models)
- [ ] AutoML for compression method selection
- [ ] Distributed compression support
- [ ] Model deployment utilities

---

**Star ⭐ the project if you find it useful!**

[![Star History Chart](https://api.star-history.com/svg?repos=zetaneh/torchslim&type=Date)](https://star-history.com/#zetaneh/torchslim&Date)