"""Global method registry for TorchSlim"""

from typing import Dict, List, Type, Optional
import importlib
import logging
from .base import CompressionMethod

logger = logging.getLogger(__name__)

# Global registry of compression methods
_METHOD_REGISTRY: Dict[str, Type[CompressionMethod]] = {}
_METHOD_INSTANCES: Dict[str, CompressionMethod] = {}

def register_method(name: str, method_class: Type[CompressionMethod]):
    """Register a compression method globally"""
    if not issubclass(method_class, CompressionMethod):
        raise TypeError(f"Method class must inherit from CompressionMethod")
    
    if name in _METHOD_REGISTRY:
        logger.warning(f"Method '{name}' already registered, overwriting...")
    
    _METHOD_REGISTRY[name] = method_class
    logger.info(f"✓ Registered compression method: {name}")

def get_method_class(name: str) -> Type[CompressionMethod]:
    """Get a method class by name"""
    if name not in _METHOD_REGISTRY:
        available = list(_METHOD_REGISTRY.keys())
        raise ValueError(f"Method '{name}' not found. Available: {available}")
    return _METHOD_REGISTRY[name]

def get_available_methods() -> List[str]:
    """Get list of all available compression methods"""
    return list(_METHOD_REGISTRY.keys())

def create_method_instance(name: str, **kwargs) -> CompressionMethod:
    """Create an instance of a compression method"""
    if name in _METHOD_INSTANCES:
        instance = _METHOD_INSTANCES[name]
        instance.configure(**kwargs)
        return instance
    
    method_class = get_method_class(name)
    instance = method_class()
    instance.configure(**kwargs)
    
    _METHOD_INSTANCES[name] = instance
    return instance

def discover_methods(package_names: List[str] = None):
    """Discover and auto-register compression methods from packages"""
    if package_names is None:
        package_names = ['torchslim.methods']
    
    for package_name in package_names:
        try:
            package = importlib.import_module(package_name)
            logger.info(f"Discovered methods from {package_name}")
        except ImportError as e:
            logger.warning(f"Could not import {package_name}: {e}")

# Auto-discover built-in methods
discover_methods()
