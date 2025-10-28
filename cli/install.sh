#!/bin/bash
# Installation script for ailang CLI

set -e

echo "=== ailang CLI Installation ==="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Check for pip
if ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not installed"
    echo "Please install pip: python3 -m ensurepip"
    exit 1
fi

echo "Found pip $(python3 -m pip --version | cut -d' ' -f2)"
echo ""

# Install in development mode
echo "Installing ailang CLI in development mode..."
python3 -m pip install -e .

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verify installation:"
echo "  ailang --version"
echo "  ailang --help"
echo ""
echo "Quick start:"
echo "  ailang models recommend coding"
echo "  ailang examples list"
echo ""
echo "See QUICKSTART.md for more information"
