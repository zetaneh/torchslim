#!/bin/bash

# ================================================================================================
# TorchSlim - Virtual Environment Setup Script
# Solves "externally-managed-environment" error on Ubuntu/Debian
# ================================================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_command() {
    echo -e "${PURPLE}$ $1${NC}"
}

print_header "🐍 TorchSlim Virtual Environment Setup"

# Detect the current directory
CURRENT_DIR=$(pwd)
PROJECT_NAME="torchslim"

# Find the project root directory
if [[ "$CURRENT_DIR" == *"torchslim"* ]]; then
    # Navigate to the actual project root
    while [[ $(basename "$PWD") != "$PROJECT_NAME" ]] && [[ "$PWD" != "/" ]]; do
        cd ..
    done
    
    if [[ $(basename "$PWD") == "$PROJECT_NAME" ]]; then
        PROJECT_ROOT="$PWD"
        print_status "Found project root: $PROJECT_ROOT"
    else
        print_error "Could not find project root directory"
        exit 1
    fi
else
    PROJECT_ROOT="$CURRENT_DIR"
    print_warning "Assuming current directory is project root: $PROJECT_ROOT"
fi

print_status "Working in: $PROJECT_ROOT"

# ================================================================================================
# CHECK SYSTEM REQUIREMENTS
# ================================================================================================

print_header "🔍 Checking System Requirements"

# Check if python3-venv is installed
if ! python3 -m venv --help &> /dev/null; then
    print_warning "python3-venv not installed, installing..."
    print_command "sudo apt update && sudo apt install python3-venv python3-full -y"
    sudo apt update && sudo apt install python3-venv python3-full -y
else
    print_status "✅ python3-venv is available"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version)
print_status "✅ Python version: $PYTHON_VERSION"

# ================================================================================================
# CREATE VIRTUAL ENVIRONMENT
# ================================================================================================

print_header "🏗️ Creating Virtual Environment"

VENV_NAME="torchslim_env"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"

# Remove existing virtual environment if it exists
if [[ -d "$VENV_PATH" ]]; then
    print_warning "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    else
        print_status "Using existing virtual environment..."
    fi
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_PATH" ]]; then
    print_status "Creating virtual environment: $VENV_NAME"
    print_command "python3 -m venv $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    
    if [[ $? -eq 0 ]]; then
        print_status "✅ Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_status "✅ Virtual environment already exists"
fi

# ================================================================================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ================================================================================================

print_header "🔌 Activating Virtual Environment"

print_status "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "✅ Virtual environment activated: $VIRTUAL_ENV"
    print_status "Python path: $(which python)"
    print_status "Pip path: $(which pip)"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# ================================================================================================
# INSTALL DEPENDENCIES
# ================================================================================================

print_header "📦 Installing Dependencies"

# Install PyTorch first (CPU version for compatibility)
print_status "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
if [[ -f "requirements.txt" ]]; then
    print_status "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    print_status "requirements.txt not found, installing essential packages..."
    pip install numpy>=1.19.0 tqdm>=4.60.0 matplotlib>=3.3.0 seaborn>=0.11.0 scipy>=1.7.0
fi

# Install development dependencies if available
if [[ -f "requirements-dev.txt" ]]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
else
    print_status "Installing essential development tools..."
    pip install pytest black flake8 jupyter notebook
fi

# ================================================================================================
# INSTALL TORCHSLIM
# ================================================================================================

print_header "🔥 Installing TorchSlim"

# Install in development mode
print_status "Installing TorchSlim in development mode..."
pip install -e .

# Verify installation
if python -c "import torchslim" 2>/dev/null; then
    print_status "✅ TorchSlim installed successfully"
    TORCHSLIM_VERSION=$(python -c "import torchslim; print(torchslim.__version__)")
    print_status "TorchSlim version: $TORCHSLIM_VERSION"
else
    print_error "Failed to install TorchSlim"
    exit 1
fi

# ================================================================================================
# CREATE ACTIVATION SCRIPT
# ================================================================================================

print_header "📝 Creating Activation Script"

# Create a convenient activation script
ACTIVATE_SCRIPT="$PROJECT_ROOT/activate_torchslim.sh"

cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# TorchSlim Virtual Environment Activation Script

echo "🔥 Activating TorchSlim Environment..."
source "$VENV_PATH/bin/activate"

if [[ "\$VIRTUAL_ENV" != "" ]]; then
    echo "✅ TorchSlim environment activated!"
    echo "📍 Virtual environment: \$VIRTUAL_ENV"
    echo "🐍 Python: \$(which python)"
    echo "📦 Pip: \$(which pip)"
    echo ""
    echo "🚀 Quick commands:"
    echo "  python examples/basic_usage.py           # Run basic example"
    echo "  python examples/custom_method_example.py # Custom methods"
    echo "  pytest tests/                           # Run tests"
    echo "  deactivate                              # Exit environment"
    echo ""
else
    echo "❌ Failed to activate environment"
    exit 1
fi
EOF

chmod +x "$ACTIVATE_SCRIPT"
print_status "✅ Created activation script: $ACTIVATE_SCRIPT"

# ================================================================================================
# CREATE CONVENIENT WRAPPER SCRIPTS
# ================================================================================================

print_header "🛠️ Creating Wrapper Scripts"

# Create run_example.sh
cat > "$PROJECT_ROOT/run_example.sh" << EOF
#!/bin/bash
# TorchSlim Example Runner

# Activate environment
source "$VENV_PATH/bin/activate"

# Run the requested script
if [[ \$# -eq 0 ]]; then
    echo "Usage: ./run_example.sh <script_name>"
    echo "Available examples:"
    ls examples/*.py | sed 's/examples\///g' | sed 's/\.py//g'
    exit 1
fi

SCRIPT_NAME="\$1"
if [[ ! "\$SCRIPT_NAME" == *.py ]]; then
    SCRIPT_NAME="\$SCRIPT_NAME.py"
fi

SCRIPT_PATH="examples/\$SCRIPT_NAME"

if [[ -f "\$SCRIPT_PATH" ]]; then
    echo "🚀 Running \$SCRIPT_PATH in TorchSlim environment..."
    python "\$SCRIPT_PATH"
else
    echo "❌ Script not found: \$SCRIPT_PATH"
    echo "Available examples:"
    ls examples/*.py | sed 's/examples\///g'
fi
EOF

chmod +x "$PROJECT_ROOT/run_example.sh"
print_status "✅ Created example runner: run_example.sh"

# Create test runner
cat > "$PROJECT_ROOT/run_tests.sh" << EOF
#!/bin/bash
# TorchSlim Test Runner

echo "🧪 Running TorchSlim tests..."
source "$VENV_PATH/bin/activate"

if [[ -d "tests" ]] && [[ \$(ls tests/*.py 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "Running pytest..."
    pytest tests/ -v
else
    echo "No tests directory found, creating basic test..."
    mkdir -p tests
    
    cat > tests/test_basic.py << 'EOFTEST'
import pytest
import torch
from torchslim import TorchSlim, CompressionConfig, create_test_model

def test_basic_functionality():
    """Test basic TorchSlim functionality"""
    model = create_test_model("mlp", input_size=10, hidden_sizes=[5], output_size=2)
    
    config = CompressionConfig()
    config.add_method("svd", rank_ratio=0.5)
    
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    
    assert compressed_model is not None
    print("✅ Basic functionality test passed!")

if __name__ == "__main__":
    test_basic_functionality()
EOFTEST
    
    echo "Running basic test..."
    python tests/test_basic.py
fi
EOF

chmod +x "$PROJECT_ROOT/run_tests.sh"
print_status "✅ Created test runner: run_tests.sh"

# ================================================================================================
# RUN TESTS
# ================================================================================================

print_header "🧪 Testing Installation"

# Test basic import
print_status "Testing basic import..."
if python -c "import torchslim; print('TorchSlim import: ✅')" 2>/dev/null; then
    print_status "✅ Import test passed"
else
    print_error "❌ Import test failed"
    exit 1
fi

# Test basic functionality
print_status "Testing basic functionality..."
python << 'EOF'
try:
    from torchslim import TorchSlim, CompressionConfig, create_test_model
    import torch
    
    print("📦 Creating test model...")
    model = create_test_model("mlp", input_size=10, hidden_sizes=[5], output_size=2)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {original_params:,} parameters")
    
    print("⚙️ Configuring compression...")
    config = CompressionConfig()
    config.add_method("svd", rank_ratio=0.5)
    
    print("🗜️ Compressing model...")
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(model)
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    
    compression_ratio = original_params / compressed_params
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print("✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")
    import traceback
    traceback.print_exc()
EOF

# ================================================================================================
# PROVIDE USAGE INSTRUCTIONS
# ================================================================================================

print_header "🎉 Installation Complete!"

echo -e "${GREEN}✅ TorchSlim is now installed in a virtual environment!${NC}"
echo ""
echo -e "${BLUE}🚀 Quick Start:${NC}"
echo ""
echo -e "${PURPLE}# Activate the environment:${NC}"
echo "source ./activate_torchslim.sh"
echo ""
echo -e "${PURPLE}# Or manually:${NC}"
echo "source $VENV_NAME/bin/activate"
echo ""
echo -e "${PURPLE}# Run examples:${NC}"
echo "./run_example.sh basic_usage"
echo "./run_example.sh custom_method_example"
echo ""
echo -e "${PURPLE}# Run tests:${NC}"
echo "./run_tests.sh"
echo ""
echo -e "${PURPLE}# Manual commands (after activation):${NC}"
echo "python examples/basic_usage.py"
echo "python -c \"from torchslim import TorchSlim; print('TorchSlim ready!')\""
echo ""
echo -e "${BLUE}📁 Project Structure:${NC}"
echo "  $PROJECT_ROOT/"
echo "  ├── $VENV_NAME/              # Virtual environment"
echo "  ├── activate_torchslim.sh    # Environment activation"
echo "  ├── run_example.sh          # Example runner"
echo "  ├── run_tests.sh            # Test runner"
echo "  ├── torchslim/              # Main library"
echo "  ├── examples/               # Usage examples"
echo "  └── tests/                  # Test suite"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "• Always activate the environment before working: source ./activate_torchslim.sh"
echo "• Use './run_example.sh <name>' for quick testing"
echo "• Deactivate with 'deactivate' when done"
echo "• The environment is isolated and won't affect your system Python"
echo ""
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo "1. Run: source ./activate_torchslim.sh"
echo "2. Test: ./run_example.sh basic_usage"
echo "3. Explore: ls examples/"
echo "4. Develop: Create your own compression methods!"
echo ""
print_status "Happy compressing with TorchSlim! 🔥⚡"

# ================================================================================================
# SAVE ENVIRONMENT INFO
# ================================================================================================

cat > "$PROJECT_ROOT/environment_info.txt" << EOF
TorchSlim Virtual Environment Information
=========================================

Created: $(date)
Python Version: $(python --version)
Virtual Environment: $VENV_PATH
Project Root: $PROJECT_ROOT

Installed Packages:
$(pip list)

Activation Command:
source $VENV_PATH/bin/activate

Or use the convenience script:
source ./activate_torchslim.sh
EOF

print_status "Environment info saved to: environment_info.txt"