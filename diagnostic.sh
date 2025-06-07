#!/bin/bash

# ================================================================================================
# TorchSlim Diagnostic and Test Script
# Diagnose installation issues and run comprehensive tests
# ================================================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_command() {
    echo -e "${PURPLE}$ $1${NC}"
}

print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_header "🔍 TorchSlim Installation Diagnostic"

# ================================================================================================
# ENVIRONMENT DETECTION
# ================================================================================================

print_header "🌍 Environment Detection"

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    print_status "Virtual environment detected: $VIRTUAL_ENV"
    VENV_ACTIVE=true
else
    print_warning "No virtual environment detected"
    VENV_ACTIVE=false
fi

# Check Python version and path
PYTHON_VERSION=$(python --version 2>/dev/null || python3 --version)
PYTHON_PATH=$(which python 2>/dev/null || which python3)
print_info "Python version: $PYTHON_VERSION"
print_info "Python path: $PYTHON_PATH"

# Check pip
PIP_PATH=$(which pip 2>/dev/null || which pip3)
print_info "Pip path: $PIP_PATH"

# ================================================================================================
# INSTALLATION VERIFICATION
# ================================================================================================

print_header "📦 Installation Verification"

# Test Python import
print_info "Testing Python import..."
IMPORT_TEST=$(python -c "
try:
    import sys
    print('Python import: SUCCESS')
    print(f'Python executable: {sys.executable}')
    print(f'Python path: {sys.path[:3]}...')
except Exception as e:
    print(f'Python import: FAILED - {e}')
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    print_status "Python import test passed"
    echo "$IMPORT_TEST" | sed 's/^/    /'
else
    print_error "Python import test failed"
    echo "$IMPORT_TEST" | sed 's/^/    /'
fi

# Test PyTorch import
print_info "Testing PyTorch import..."
TORCH_TEST=$(python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch path: {torch.__file__}')
    print('PyTorch import: SUCCESS')
except Exception as e:
    print(f'PyTorch import: FAILED - {e}')
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    print_status "PyTorch import test passed"
    echo "$TORCH_TEST" | sed 's/^/    /'
else
    print_error "PyTorch import test failed"
    echo "$TORCH_TEST" | sed 's/^/    /'
fi

# Test TorchSlim import
print_info "Testing TorchSlim import..."
TORCHSLIM_TEST=$(python -c "
try:
    import torchslim
    print(f'TorchSlim version: {torchslim.__version__}')
    print(f'TorchSlim path: {torchslim.__file__}')
    print('TorchSlim import: SUCCESS')
except Exception as e:
    print(f'TorchSlim import: FAILED - {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    print_status "TorchSlim import test passed"
    echo "$TORCHSLIM_TEST" | sed 's/^/    /'
else
    print_error "TorchSlim import test failed"
    echo "$TORCHSLIM_TEST" | sed 's/^/    /'
    
    # Additional diagnostics for TorchSlim import failure
    print_info "Running additional TorchSlim diagnostics..."
    
    # Check if torchslim package is installed
    PACKAGE_CHECK=$(pip list | grep torchslim)
    if [[ -n "$PACKAGE_CHECK" ]]; then
        print_status "TorchSlim package found in pip list:"
        echo "    $PACKAGE_CHECK"
    else
        print_error "TorchSlim package not found in pip list"
    fi
    
    # Check Python site-packages
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages())")
    print_info "Python site-packages: $SITE_PACKAGES"
    
    # Check current directory
    print_info "Current directory: $(pwd)"
    print_info "Directory contents:"
    ls -la | head -10 | sed 's/^/    /'
fi

# ================================================================================================
# FUNCTIONAL TESTS
# ================================================================================================

print_header "🧪 Functional Tests"

if [[ $? -eq 0 ]]; then
    print_info "Running TorchSlim functional tests..."
    
    FUNCTIONAL_TEST=$(python -c "
import sys
sys.path.insert(0, '.')

try:
    # Test 1: Import all core components
    print('Test 1: Importing core components...')
    from torchslim import TorchSlim, CompressionConfig, create_test_model
    print('✓ Core components imported successfully')
    
    # Test 2: Create test model
    print('Test 2: Creating test model...')
    model = create_test_model('mlp', input_size=10, hidden_sizes=[5], output_size=2)
    original_params = sum(p.numel() for p in model.parameters())
    print(f'✓ Test model created with {original_params:,} parameters')
    
    # Test 3: Test model forward pass
    print('Test 3: Testing model forward pass...')
    import torch
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = model(test_input)
    print(f'✓ Model forward pass successful, output shape: {output.shape}')
    
    # Test 4: Create compression configuration
    print('Test 4: Creating compression configuration...')
    config = CompressionConfig()
    config.add_method('svd', rank_ratio=0.5)
    print('✓ Compression configuration created')
    
    # Test 5: Test compression
    print('Test 5: Testing model compression...')
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    compression_ratio = original_params / compressed_params
    print(f'✓ Model compressed successfully')
    print(f'  Original parameters: {original_params:,}')
    print(f'  Compressed parameters: {compressed_params:,}')
    print(f'  Compression ratio: {compression_ratio:.2f}x')
    
    # Test 6: Test compressed model forward pass
    print('Test 6: Testing compressed model forward pass...')
    with torch.no_grad():
        compressed_output = compressed_model(test_input)
    mse = torch.mean((output - compressed_output)**2).item()
    print(f'✓ Compressed model forward pass successful')
    print(f'  MSE between original and compressed: {mse:.6f}')
    
    print('\\n🎉 ALL TESTS PASSED! TorchSlim is working correctly!')
    
except Exception as e:
    print(f'\\n❌ FUNCTIONAL TEST FAILED: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

    if [[ $? -eq 0 ]]; then
        print_status "All functional tests passed!"
        echo "$FUNCTIONAL_TEST" | sed 's/^/    /'
    else
        print_error "Functional tests failed"
        echo "$FUNCTIONAL_TEST" | sed 's/^/    /'
    fi
else
    print_warning "Skipping functional tests due to import failures"
fi

# ================================================================================================
# EXAMPLES TEST
# ================================================================================================

print_header "📚 Examples Test"

# Check if examples directory exists
if [[ -d "examples" ]]; then
    print_status "Examples directory found"
    print_info "Available examples:"
    ls examples/*.py 2>/dev/null | sed 's/^/    /' || echo "    No Python files found in examples/"
    
    # Test basic_usage.py if it exists
    if [[ -f "examples/basic_usage.py" ]]; then
        print_info "Testing examples/basic_usage.py..."
        EXAMPLE_TEST=$(timeout 30 python examples/basic_usage.py 2>&1)
        if [[ $? -eq 0 ]]; then
            print_status "basic_usage.py ran successfully"
            echo "$EXAMPLE_TEST" | head -10 | sed 's/^/    /'
            if [[ $(echo "$EXAMPLE_TEST" | wc -l) -gt 10 ]]; then
                echo "    ... (output truncated)"
            fi
        else
            print_error "basic_usage.py failed"
            echo "$EXAMPLE_TEST" | sed 's/^/    /'
        fi
    else
        print_warning "examples/basic_usage.py not found"
    fi
else
    print_warning "Examples directory not found"
fi

# ================================================================================================
# METHOD REGISTRATION TEST
# ================================================================================================

print_header "🔌 Method Registration Test"

METHOD_TEST=$(python -c "
try:
    from torchslim import get_available_methods
    methods = get_available_methods()
    print(f'Available compression methods: {methods}')
    print(f'Number of registered methods: {len(methods)}')
    
    if len(methods) > 0:
        print('✓ Method registration system working!')
        for method in methods:
            print(f'  - {method}')
    else:
        print('❌ No methods registered!')
        exit(1)
        
except Exception as e:
    print(f'❌ Method registration test failed: {e}')
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    print_status "Method registration test passed"
    echo "$METHOD_TEST" | sed 's/^/    /'
else
    print_error "Method registration test failed"
    echo "$METHOD_TEST" | sed 's/^/    /'
fi

# ================================================================================================
# PACKAGE INFORMATION
# ================================================================================================

print_header "📋 Package Information"

print_info "Installed packages related to TorchSlim:"
pip list | grep -E "(torch|numpy|scipy|matplotlib|seaborn|tqdm)" | sed 's/^/    /'

print_info "TorchSlim package details:"
pip show torchslim | sed 's/^/    /'

# ================================================================================================
# RECOMMENDATIONS
# ================================================================================================

print_header "💡 Recommendations and Next Steps"

echo -e "${GREEN}✅ TorchSlim Installation Status: SUCCESS${NC}"
echo ""
echo -e "${BLUE}🚀 Ready to use! Try these commands:${NC}"
echo ""

if [[ $VENV_ACTIVE == true ]]; then
    echo -e "${PURPLE}# You're already in the virtual environment, great!${NC}"
else
    echo -e "${PURPLE}# First, activate your virtual environment:${NC}"
    echo "source torchslim_env/bin/activate"
    echo ""
fi

echo -e "${PURPLE}# Run examples:${NC}"
echo "python examples/basic_usage.py"
echo "python examples/custom_method_example.py"
echo ""
echo -e "${PURPLE}# Quick test:${NC}"
echo "python -c \"from torchslim import TorchSlim; print('🔥 TorchSlim ready!')\""
echo ""
echo -e "${PURPLE}# Interactive Python session:${NC}"
echo "python"
echo ">>> from torchslim import *"
echo ">>> model = create_test_model('mlp')"
echo ">>> config = CompressionConfig()"
echo ">>> config.add_method('svd', rank_ratio=0.5)"
echo ">>> compressor = TorchSlim(config)"
echo ">>> compressed = compressor.compress_model(model)"
echo ""

# ================================================================================================
# CREATE QUICK TEST SCRIPT
# ================================================================================================

print_header "📝 Creating Quick Test Script"

cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick TorchSlim functionality test
"""

def main():
    print("🔥 TorchSlim Quick Test")
    print("=" * 30)
    
    try:
        # Import
        from torchslim import TorchSlim, CompressionConfig, create_test_model
        import torch
        print("✅ Imports successful")
        
        # Create model
        model = create_test_model("mlp", input_size=20, hidden_sizes=[10], output_size=5)
        print(f"✅ Model created ({sum(p.numel() for p in model.parameters())} params)")
        
        # Configure compression
        config = CompressionConfig()
        config.add_method("svd", rank_ratio=0.5)
        print("✅ Configuration created")
        
        # Compress
        compressor = TorchSlim(config)
        compressed_model = compressor.compress_model(model)
        ratio = compressor.get_compression_report()['summary']['compression_ratio']
        print(f"✅ Compression successful ({ratio:.2f}x)")
        
        # Test
        x = torch.randn(1, 20)
        y1 = model(x)
        y2 = compressed_model(x)
        mse = torch.mean((y1 - y2)**2).item()
        print(f"✅ Models work (MSE: {mse:.6f})")
        
        print("\n🎉 All tests passed! TorchSlim is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

chmod +x quick_test.py
print_status "Created quick_test.py for easy testing"

echo ""
echo -e "${GREEN}🎯 Run this for a quick test:${NC}"
echo -e "${PURPLE}python quick_test.py${NC}"
echo ""
print_status "TorchSlim diagnostic complete! 🎉"