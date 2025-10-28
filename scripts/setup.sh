#!/bin/bash
# Complete environment setup script for local AI development
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help text
show_help() {
    cat << EOF
${BOLD}Environment Setup Script${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -f, --full          Full setup (dependencies + models + tests)
    -q, --quick         Quick setup (dependencies only)
    -m, --models        Pull recommended models
    -t, --test          Run verification tests
    -s, --skip-models   Skip model downloads
    --ci                CI/CD mode (non-interactive)

Examples:
    $0                  # Interactive setup
    $0 --full           # Complete setup
    $0 --quick          # Dependencies only
    $0 --models         # Pull models only

EOF
}

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

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
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

# Run verification tests
run_verification() {
    echo -e "\n${BOLD}Running verification tests...${NC}"

    # Test Python imports
    if uv run python -c "from langchain_ollama import ChatOllama; from langgraph.graph import StateGraph; print('Imports OK')"; then
        print_status "Python dependencies verified"
    else
        print_error "Python import verification failed"
        return 1
    fi

    # Test Ollama connection
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_status "Ollama connection verified"
    else
        print_warning "Ollama not responding (may need restart)"
    fi

    # Check if any model is available
    if ollama list | tail -n +2 | grep -q .; then
        print_status "Models available"
    else
        print_warning "No models installed (run with --models)"
    fi

    # Test basic functionality
    if [ -f "examples/01-foundation/simple_chat.py" ]; then
        print_info "Run example to test: uv run python examples/01-foundation/simple_chat.py"
    fi
}

# Print summary
print_summary() {
    echo -e "\n${BOLD}${GREEN}Setup Complete!${NC}\n"

    echo -e "${BOLD}Next Steps:${NC}"
    echo "1. Edit .env file: ${BLUE}nano .env${NC}"
    echo "2. Run example: ${BLUE}uv run python examples/01-foundation/simple_chat.py${NC}"
    echo "3. Start dev mode: ${BLUE}./scripts/dev.sh --all${NC}"
    echo "4. Pull more models: ${BLUE}./scripts/pull_models.sh --all${NC}"
    echo ""

    echo -e "${BOLD}Documentation:${NC}"
    echo "- Quick start: README.md"
    echo "- Development plan: docs/DEVELOPMENT-PLAN-20-POINTS.md"
    echo "- Examples: plans/3-kitchen-sink-plan.md"
    echo ""

    echo -e "${BOLD}Useful Scripts:${NC}"
    echo "- ${BLUE}./scripts/dev.sh --all${NC}      - Start development environment"
    echo "- ${BLUE}./scripts/pull_models.sh${NC}    - Manage Ollama models"
    echo "- ${BLUE}./scripts/run_tests.sh${NC}      - Run test suite"
    echo "- ${BLUE}./scripts/benchmark.sh${NC}      - Run benchmarks"
    echo "- ${BLUE}./scripts/clean.sh${NC}          - Clean temporary files"
    echo ""

    echo -e "${BOLD}Quick Commands:${NC}"
    echo "- ${BLUE}ollama list${NC}                 - List installed models"
    echo "- ${BLUE}ollama pull <model>${NC}         - Download a model"
    echo "- ${BLUE}npx langgraph@latest dev${NC}    - Start LangGraph Studio"
    echo "- ${BLUE}uv run pytest tests/${NC}        - Run tests"
    echo ""

    print_status "Setup completed successfully!"
}

# Main function
main() {
    # Parse arguments
    FULL_SETUP=false
    QUICK_SETUP=false
    PULL_MODELS=true
    RUN_TESTS=false
    CI_MODE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -f|--full)
                FULL_SETUP=true
                PULL_MODELS=true
                RUN_TESTS=true
                shift
                ;;
            -q|--quick)
                QUICK_SETUP=true
                PULL_MODELS=false
                shift
                ;;
            -m|--models)
                PULL_MODELS=true
                shift
                ;;
            -t|--test)
                RUN_TESTS=true
                shift
                ;;
            -s|--skip-models)
                PULL_MODELS=false
                shift
                ;;
            --ci)
                CI_MODE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Execute setup based on options
    if [ "$RUN_TESTS" = true ]; then
        run_verification
    fi

    print_summary
}

# Run main if arguments provided, otherwise run original flow
if [ $# -gt 0 ]; then
    main "$@"
else
    # Original setup flow continues...
    # Print summary
    print_summary
fi
