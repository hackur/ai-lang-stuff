#!/bin/bash
# Demo script for ailang CLI

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║          ailang CLI - Feature Demonstration          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if ailang is installed
if ! command -v ailang &> /dev/null; then
    echo "Error: ailang is not installed"
    echo "Run: ./install.sh"
    exit 1
fi

# 1. Version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Version Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ailang --version
echo ""

# 2. Model recommendations
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Model Recommendations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ailang models recommend coding
echo ""

# 3. List examples
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Available Examples"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ailang examples list | head -25
echo ""

# 4. MCP servers
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. MCP Servers"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ailang mcp list
echo ""

# 5. RAG collections
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. RAG Collections"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ailang rag list
echo ""

# 6. Help for each command
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Command Groups"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Models:"
ailang models --help | grep "  " | head -10
echo ""
echo "Examples:"
ailang examples --help | grep "  " | head -10
echo ""
echo "MCP:"
ailang mcp --help | grep "  " | head -10
echo ""
echo "RAG:"
ailang rag --help | grep "  " | head -10
echo ""

echo "╔═══════════════════════════════════════════════════════╗"
echo "║                  Demo Complete!                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "Try these commands:"
echo "  ailang models list"
echo "  ailang examples list --category 01-foundation"
echo "  ailang models recommend vision"
echo "  ailang rag index ./docs --name my-docs"
echo ""
echo "For more information: cat README.md"
