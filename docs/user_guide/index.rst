# docs/user_guide/index.rst
User Guide
==========

This comprehensive user guide covers all aspects of using TorchSlim for model compression.

.. toctree::
   :maxdepth: 2

   basic_usage
   compression_methods
   analysis_tools
   custom_methods
   best_practices

---

# docs/user_guide/basic_usage.rst
Basic Usage
===========

Introduction
------------

TorchSlim provides a simple yet powerful interface for compressing PyTorch models. This section covers the fundamental concepts and basic usage patterns.

Creating Models
---------------

TorchSlim includes utilities for creating test models:

.. code-block:: python

   from torchslim import create_test_model

   # Create a multi-layer perceptron
   model = create_test_model(
       "mlp", 
       input_size=784, 
       hidden_sizes=[512, 256, 128], 
       output_size=10
   )

   # Create a convolutional neural network
   cnn = create_test_model(
       "cnn",
       input_channels=3,
       num_classes=10,
       hidden_dims=[64, 128]
   )

Compression Configuration
-------------------------

Configure compression methods using the `CompressionConfig` class:

.. code-block:: python

   from torchslim import CompressionConfig

   config = CompressionConfig()
   
   # Add SVD compression
   config.add_method("svd", rank_ratio=0.5)
   
   # Add pruning
   config.add_method("pruning", pruning_ratio=0.2, pruning_type="magnitude")
   
   # Add quantization
   config.add_method("quantization", bits=8, scheme="symmetric")

Applying Compression
--------------------

Use the `TorchSlim` compressor to apply compression:

.. code-block:: python

   from torchslim import TorchSlim

   # Create compressor
   compressor = TorchSlim(config)
   
   # Compress the model
   compressed_model = compressor.compress_model(model)
   
   # Get compression report
   report = compressor.get_compression_report()
   
   print(f"Compression ratio: {report['summary']['compression_ratio']:.2f}x")
   print(f"Parameter reduction: {report['summary']['parameter_reduction']:.1%}")

---

# docs/examples/index.rst
Examples
========

This section provides practical examples of using TorchSlim for various compression tasks.

.. toctree::
   :maxdepth: 2

   basic_compression
   multi_method_compression
   custom_methods
   analysis_and_benchmarking

---

# docs/examples/basic_compression.rst
Basic Compression Example
=========================

This example demonstrates the most basic usage of TorchSlim for model compression.

Simple SVD Compression
----------------------

.. literalinclude:: ../../examples/basic_usage.py
   :language: python
   :lines: 1-30

Complete Example
----------------

.. literalinclude:: ../../examples/basic_usage.py
   :language: python

Output
------

When you run this example, you should see output similar to:

.. code-block:: text

   🚀 Testing realistic TorchSlim compression...
   Original model parameters: 567,434
   Compression methods: ['svd', 'pruning']
   Compressed model parameters: 456,705
   
   ============================================================
   COMPRESSION VALIDATION REPORT
   ============================================================
   Timestamp: 2025-01-XX XX:XX:XX
   Methods Used: svd, pruning
   Overall Status: SUCCESS
   
   Compression Metrics:
   ------------------------------
     Compression Ratio: 1.24x
     Parameter Reduction: 19.5%
     Size Reduction: 0.42 MB
     Output Similarity: 0.92

---

# docs/developer_guide/index.rst
Developer Guide
===============

Welcome to the TorchSlim developer guide! This section covers advanced topics for extending and contributing to TorchSlim.

.. toctree::
   :maxdepth: 2

   architecture
   custom_methods
   testing
   contributing

---

# docs/developer_guide/custom_methods.rst
Creating Custom Compression Methods
===================================

One of TorchSlim's key features is its extensible plugin architecture that allows you to easily create custom compression methods.

Method Interface
----------------

All compression methods must inherit from the `CompressionMethod` base class:

.. code-block:: python

   from torchslim.core.base import CompressionMethod, CompressionType

   class MyCustomMethod(CompressionMethod):
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
           # Return True if this method can compress the layer
           return isinstance(layer, torch.nn.Linear)
       
       def get_compression_ratio(self, original, compressed):
           # Calculate and return compression ratio
           return original_params / compressed_params

Registration
------------

Register your custom method to make it available:

.. code-block:: python

   from torchslim import register_method

   register_method("my_method", MyCustomMethod)

For a complete example, see the `custom_method_example.py` in the examples directory.