"""
Test model creation utilities for TorchSlim compression experiments.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_test_model(model_type: str = "mlp", **kwargs) -> nn.Module:
    """
    Create test models for compression experiments.
    
    Args:
        model_type: Type of model to create ('mlp', 'cnn', 'resnet_block', 'transformer_block')
        **kwargs: Model-specific parameters
        
    Returns:
        PyTorch model
    """
    if model_type == "mlp":
        input_size = kwargs.get('input_size', 784)
        hidden_sizes = kwargs.get('hidden_sizes', [512, 256, 128])
        output_size = kwargs.get('output_size', 10)
        dropout = kwargs.get('dropout', 0.0)
        activation = kwargs.get('activation', 'relu')
        
        return create_mlp(input_size, hidden_sizes, output_size, dropout, activation)
        
    elif model_type == "cnn":
        input_channels = kwargs.get('input_channels', 3)
        num_classes = kwargs.get('num_classes', 10)
        hidden_dims = kwargs.get('hidden_dims', [32, 64])
        
        return create_cnn(input_channels, num_classes, hidden_dims)
        
    elif model_type == "resnet_block":
        input_channels = kwargs.get('input_channels', 64)
        output_channels = kwargs.get('output_channels', 64)
        stride = kwargs.get('stride', 1)
        
        return create_resnet_block(input_channels, output_channels, stride)
        
    elif model_type == "transformer_block":
        d_model = kwargs.get('d_model', 512)
        num_heads = kwargs.get('num_heads', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 2048)
        
        return create_transformer_block(d_model, num_heads, dim_feedforward)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_mlp(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    dropout: float = 0.0,
    activation: str = 'relu'
) -> nn.Module:
    """Create a Multi-Layer Perceptron."""
    
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'gelu': nn.GELU,
        'leaky_relu': nn.LeakyReLU
    }
    
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")
    
    activation_fn = activation_map[activation]
    
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(activation_fn())
    
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(activation_fn())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    
    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    return nn.Sequential(*layers)


def create_cnn(
    input_channels: int = 3,
    num_classes: int = 10,
    hidden_dims: List[int] = None
) -> nn.Module:
    """Create a simple Convolutional Neural Network."""
    
    if hidden_dims is None:
        hidden_dims = [32, 64, 128]
    
    layers = []
    in_channels = input_channels
    
    # Convolutional layers
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        in_channels = hidden_dim
    
    # Global average pooling and classifier
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(hidden_dims[-1], num_classes)
    ])
    
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    """Basic ResNet block for testing."""
    
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


def create_resnet_block(
    input_channels: int = 64,
    output_channels: int = 64,
    stride: int = 1
) -> nn.Module:
    """Create a ResNet block for testing."""
    return ResNetBlock(input_channels, output_channels, stride)


class TransformerBlock(nn.Module):
    """Simple Transformer block for testing."""
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ff_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


def create_transformer_block(
    d_model: int = 512,
    num_heads: int = 8,
    dim_feedforward: int = 2048
) -> nn.Module:
    """Create a Transformer block for testing."""
    return TransformerBlock(d_model, num_heads, dim_feedforward)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing model information
    """
    info = {}
    
    # Basic model information
    info['model_class'] = model.__class__.__name__
    info['model_name'] = str(model)[:100] + "..." if len(str(model)) > 100 else str(model)
    
    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    info['total_parameters'] = total_params
    info['trainable_parameters'] = trainable_params
    info['non_trainable_parameters'] = non_trainable_params
    
    # Memory footprint (approximate)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    info['parameter_size_bytes'] = param_size
    info['buffer_size_bytes'] = buffer_size
    info['total_size_bytes'] = total_size
    info['total_size_mb'] = total_size / (1024 * 1024)
    
    # Layer analysis
    layer_types = {}
    layer_count = 0
    compressible_layers = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type = type(module).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            layer_count += 1
            
            # Count potentially compressible layers
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                compressible_layers += 1
    
    info['total_layers'] = layer_count
    info['layer_types'] = layer_types
    info['compressible_layers'] = compressible_layers
    info['compression_potential'] = compressible_layers / layer_count if layer_count > 0 else 0
    
    # Parameter distribution
    param_shapes = [tuple(p.shape) for p in model.parameters()]
    param_sizes = [p.numel() for p in model.parameters()]
    
    info['parameter_shapes'] = param_shapes
    info['parameter_sizes'] = param_sizes
    info['largest_layer_params'] = max(param_sizes) if param_sizes else 0
    info['smallest_layer_params'] = min(param_sizes) if param_sizes else 0
    info['avg_layer_params'] = sum(param_sizes) / len(param_sizes) if param_sizes else 0
    
    # Gradient information
    requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['requires_grad_ratio'] = requires_grad_params / total_params if total_params > 0 else 0
    
    return info


def print_model_summary(model: nn.Module) -> None:
    """
    Print a detailed summary of the model.
    
    Args:
        model: PyTorch model to summarize
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model Class: {info['model_class']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {info['non_trainable_parameters']:,}")
    print(f"Model Size: {info['total_size_mb']:.2f} MB")
    print()
    
    print("Layer Distribution:")
    print("-" * 30)
    for layer_type, count in info['layer_types'].items():
        print(f"  {layer_type}: {count}")
    print()
    
    print(f"Compressible Layers: {info['compressible_layers']}/{info['total_layers']} "
          f"({info['compression_potential']:.1%})")
    print(f"Largest Layer: {info['largest_layer_params']:,} parameters")
    print(f"Average Layer Size: {info['avg_layer_params']:.0f} parameters")
    print("=" * 60)


def create_model_suite() -> Dict[str, nn.Module]:
    """
    Create a suite of test models for comprehensive testing.
    
    Returns:
        Dictionary of model name to model instance
    """
    models = {}
    
    # Small models
    models['tiny_mlp'] = create_test_model('mlp', 
                                         input_size=128, 
                                         hidden_sizes=[64, 32], 
                                         output_size=10)
    
    models['small_mlp'] = create_test_model('mlp', 
                                          input_size=784, 
                                          hidden_sizes=[256, 128], 
                                          output_size=10)
    
    models['medium_mlp'] = create_test_model('mlp', 
                                           input_size=784, 
                                           hidden_sizes=[512, 256, 128], 
                                           output_size=10)
    
    models['large_mlp'] = create_test_model('mlp', 
                                          input_size=784, 
                                          hidden_sizes=[1024, 512, 256, 128], 
                                          output_size=10)
    
    # Convolutional models
    models['simple_cnn'] = create_test_model('cnn', 
                                           input_channels=3, 
                                           num_classes=10, 
                                           hidden_dims=[32, 64])
    
    models['medium_cnn'] = create_test_model('cnn', 
                                           input_channels=3, 
                                           num_classes=10, 
                                           hidden_dims=[64, 128, 256])
    
    # Specialized models
    models['resnet_block'] = create_test_model('resnet_block', 
                                             input_channels=64, 
                                             output_channels=64)
    
    models['transformer_block'] = create_test_model('transformer_block', 
                                                  d_model=256, 
                                                  num_heads=8)
    
    return models


def validate_model_creation():
    """Validate that all model creation functions work correctly."""
    print("Validating model creation...")
    
    try:
        # Test MLP creation
        mlp = create_test_model('mlp')
        test_input = torch.randn(1, 784)
        output = mlp(test_input)
        assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
        print("✓ MLP creation successful")
        
        # Test CNN creation
        cnn = create_test_model('cnn')
        test_input = torch.randn(1, 3, 32, 32)
        output = cnn(test_input)
        assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
        print("✓ CNN creation successful")
        
        # Test ResNet block
        resnet = create_test_model('resnet_block')
        test_input = torch.randn(1, 64, 32, 32)
        output = resnet(test_input)
        assert output.shape == (1, 64, 32, 32), f"Expected (1, 64, 32, 32), got {output.shape}"
        print("✓ ResNet block creation successful")
        
        # Test Transformer block
        transformer = create_test_model('transformer_block')
        test_input = torch.randn(1, 10, 512)  # (batch, seq_len, d_model)
        output = transformer(test_input)
        assert output.shape == (1, 10, 512), f"Expected (1, 10, 512), got {output.shape}"
        print("✓ Transformer block creation successful")
        
        # Test model info
        info = get_model_info(mlp)
        assert 'total_parameters' in info
        assert 'total_size_mb' in info
        print("✓ Model info extraction successful")
        
        print("All model creation tests passed!")
        
    except Exception as e:
        print(f"✗ Model creation validation failed: {e}")
        raise


if __name__ == "__main__":
    # Run validation
    validate_model_creation()
    
    # Create and analyze a sample model
    print("\nSample Model Analysis:")
    model = create_test_model('mlp', input_size=784, hidden_sizes=[512, 256, 128], output_size=10)
    print_model_summary(model)