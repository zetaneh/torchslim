TorchSlim Documentation
=======================

Welcome to TorchSlim, an extensible PyTorch model compression library with a plugin-based architecture!

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://github.com/zetaneh/torchslim/workflows/CI/badge.svg
   :target: https://github.com/zetaneh/torchslim/actions
   :alt: Build Status

**TorchSlim** is a comprehensive framework for neural network compression that provides a modular, plugin-based architecture for easy integration of new compression methods. Built for researchers and practitioners who need flexible, extensible compression tools.

🌟 Key Features
---------------

- 🔌 **Plugin Architecture**: Easy to add new compression methods
- 🗜️ **Multiple Compression Methods**: SVD, Quantization, Pruning, LoRA, Knowledge Distillation, etc.
- 📊 **Comprehensive Analysis**: Profiling, visualization, and benchmarking tools
- ⚡ **Production Ready**: Complete package structure with proper Python packaging
- 🎛️ **Flexible Configuration**: Mix and match compression techniques
- 📈 **Advanced Visualizations**: Static and interactive plots for analysis

🚀 Quick Start
---------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install from GitHub
   pip install git+https://github.com/zetaneh/torchslim.git

   # Install with all optional dependencies
   pip install "git+https://github.com/zetaneh/torchslim.git[all]"

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

📚 Documentation Contents
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   user_guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/core
   api/methods
   api/analysis
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   developer_guide/index
   developer_guide/custom_methods
   developer_guide/contributing

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`