# tests/conftest.py
"""Configuration for pytest."""

import pytest
import torch
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def device():
    """Device fixture for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def simple_model():
    """Simple model fixture for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
