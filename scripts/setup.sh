#!/bin/bash
# Setup script for local AI development environment
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}Local AI Development - Setup Script${NC}\n"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo -e "${BOLD}Checking prerequisites...${NC}"

if ! command_exists brew; then
    print_error "Homebrew not found. Please install from https://brew.sh"
    exit 1
fi
print_status "Homebrew installed"

# Install required tools
echo -e "\n${BOLD}Installing required tools...${NC}"

if ! command_exists uv; then
    echo "Installing uv..."
    brew install uv
fi
print_status "uv installed"

if ! command_exists node; then
    echo "Installing Node.js..."
    brew install node
fi
print_status "Node.js installed"

if ! command_exists ollama; then
    echo "Installing Ollama..."
    brew install ollama
fi
print_status "Ollama installed"

if ! command_exists python3; then
    echo "Installing Python..."
    brew install python@3.13
fi
print_status "Python installed"

# Create directory structure
echo -e "\n${BOLD}Creating directory structure...${NC}"
mkdir -p data logs data/chroma_db models
print_status "Directories created"

# Setup Python environment
echo -e "\n${BOLD}Setting up Python environment...${NC}"
uv sync
print_status "Python dependencies installed"

# Setup Node.js dependencies
echo -e "\n${BOLD}Setting up Node.js environment...${NC}"
npm install
print_status "Node.js dependencies installed"

# Create .env file
if [ ! -f ".env" ]; then
    echo -e "\n${BOLD}Creating .env file...${NC}"
    cp config/.env.example .env
    print_status ".env file created from template"
    print_warning "Please edit .env file with your settings"
else
    print_warning ".env file already exists"
fi

# Start Ollama server
echo -e "\n${BOLD}Starting Ollama server...${NC}"
if pgrep -x "ollama" > /dev/null; then
    print_status "Ollama server already running"
else
    ollama serve > /dev/null 2>&1 &
    sleep 2
    print_status "Ollama server started"
fi

# Pull recommended models
echo -e "\n${BOLD}Pulling recommended models...${NC}"
echo "This may take several minutes depending on your internet connection..."

MODELS=("qwen3:8b" "qwen3:30b-a3b" "gemma3:4b" "qwen3-embedding")

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        print_status "$model already installed"
    else
        echo "Pulling $model..."
        if ollama pull "$model"; then
            print_status "$model installed"
        else
            print_warning "Failed to pull $model (you can install it later)"
        fi
    fi
done

# Run basic test
echo -e "\n${BOLD}Running basic test...${NC}"
if uv run python -c "from langchain_ollama import ChatOllama; print('Import successful')"; then
    print_status "Python imports working"
else
    print_error "Python import test failed"
fi

# Test Ollama connection
if curl -s http://localhost:11434/api/tags > /dev/null; then
    print_status "Ollama server responding"
else
    print_warning "Ollama server not responding at http://localhost:11434"
fi

# Print summary
echo -e "\n${BOLD}${GREEN}Setup Complete!${NC}\n"
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Review config/models.yaml for model settings"
echo "3. Run example: uv run python examples/01-foundation/simple_chat.py"
echo "4. Launch LangGraph Studio: npx langgraph@latest dev"
echo ""
echo "Documentation:"
echo "- Quick start: README.md"
echo "- Examples: plans/3-kitchen-sink-plan.md"
echo "- Architecture: plans/1-research-plan.md"
echo ""
echo "Useful commands:"
echo "- List models: ollama list"
echo "- Pull model: ollama pull <model-name>"
echo "- Run model: ollama run <model-name>"
echo "- Check Ollama: curl http://localhost:11434/api/tags"
echo ""
print_status "Setup script completed successfully!"
