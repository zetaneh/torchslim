# docs/installation.rst
Installation Guide
==================

System Requirements
-------------------

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- NumPy 1.19.0 or higher

Installation Options
--------------------

From GitHub (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install git+https://github.com/zetaneh/torchslim.git

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # All optional dependencies
   pip install "git+https://github.com/zetaneh/torchslim.git[all]"

   # Just analysis tools
   pip install "git+https://github.com/zetaneh/torchslim.git[analysis]"

   # Development dependencies
   pip install "git+https://github.com/zetaneh/torchslim.git[dev]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/zetaneh/torchslim.git
   cd torchslim
   pip install -e ".[dev]"

---

# docs/quickstart.rst
Quick Start Guide
=================

This guide will get you up and running with TorchSlim in just a few minutes.

Basic Compression
-----------------

.. code-block:: python

   from torchslim import TorchSlim, CompressionConfig, create_test_model

   # Create a test model
   model = create_test_model("mlp", input_size=784, hidden_sizes=[512, 256], output_size=10)

   # Set up compression
   config = CompressionConfig()
   config.add_method("svd", rank_ratio=0.5)

   # Compress
   compressor = TorchSlim(config)
   compressed_model = compressor.compress_model(model)

   # Check results
   report = compressor.get_compression_report()
   print(f"Achieved {report['summary']['compression_ratio']:.2f}x compression")

Multi-Method Compression
------------------------

.. code-block:: python

   # Combine multiple compression methods
   config = CompressionConfig()
   config.add_method("svd", rank_ratio=0.7)
   config.add_method("pruning", pruning_ratio=0.2)
   config.add_method("quantization", bits=8)

   compressor = TorchSlim(config)
   compressed_model = compressor.compress_model(model)

Validation and Analysis
-----------------------

.. code-block:: python

   from torchslim.utils.validation import validate_compression_pipeline

   # Validate compression quality
   results = validate_compression_pipeline(
       original_model=model,
       compressed_model=compressed_model,
       compression_methods=["svd", "pruning"]
   )

   print(f"Compression status: {results['overall_status']}")
   print(f"Output similarity: {results['output_similarity']:.3f}")

---

# docs/license.rst
License
=======

TorchSlim is released under the MIT License.

.. literalinclude:: ../LICENSE
   :language: text

---

# docs/changelog.rst
Changelog
=========

Version 1.0.0 (2025-01-XX)
---------------------------

Initial release of TorchSlim with the following features:

**Core Features:**
- Plugin-based architecture for compression methods
- Built-in compression methods: SVD, Quantization, Pruning, LoRA, Knowledge Distillation
- Comprehensive analysis and profiling tools
- Validation and benchmarking utilities

**Compression Methods:**
- SVD compression with adaptive rank selection
- Quantization with configurable bit-width
- Magnitude and structured pruning
- LoRA for efficient fine-tuning
- Knowledge distillation framework

**Analysis Tools:**
- Model structure analyzer
- Compression profiler with detailed metrics
- Visualization tools with interactive plots
- Benchmarking utilities for method comparison

**Developer Features:**
- Extensible plugin architecture
- Custom method creation support
- Comprehensive test suite
- Modern Python packaging with pyproject.toml