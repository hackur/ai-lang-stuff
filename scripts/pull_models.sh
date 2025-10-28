#!/bin/bash
# Model management script - Pull and verify all recommended models
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
${BOLD}Model Management Script${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -a, --all           Pull all recommended models
    -f, --fast          Pull only fast models (qwen3:30b-a3b, gemma3:4b)
    -v, --vision        Pull vision models
    -e, --embedding     Pull embedding models only
    -l, --list          List currently installed models
    -s, --size          Show disk usage for models
    -r, --remove MODEL  Remove a specific model

Examples:
    $0 --all            # Pull all recommended models
    $0 --fast           # Pull only fast models
    $0 --list           # List installed models
    $0 --size           # Show disk usage
    $0 --remove qwen3:8b  # Remove specific model

Model Categories:
    Fast:       qwen3:30b-a3b, gemma3:4b
    Standard:   qwen3:8b, gemma3:12b
    Vision:     qwen3-vl:8b
    Embedding:  qwen3-embedding, nomic-embed-text
    Large:      deepseek-coder:33b, qwen3:72b

EOF
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print functions
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
check_prereqs() {
    if ! command_exists ollama; then
        print_error "Ollama not found. Please install from https://ollama.ai"
        exit 1
    fi

    # Check if Ollama server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_warning "Ollama server not running. Starting..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_status "Ollama server started"
        else
            print_error "Failed to start Ollama server"
            exit 1
        fi
    fi
}

# Progress bar for downloads
show_progress() {
    local model=$1
    echo -e "${BLUE}Pulling ${model}...${NC}"
}

# Pull a single model with error handling
pull_model() {
    local model=$1

    # Check if already installed
    if ollama list | grep -q "^${model}"; then
        print_status "${model} already installed"
        return 0
    fi

    show_progress "$model"

    if ollama pull "$model" 2>&1; then
        print_status "${model} installed successfully"
        return 0
    else
        print_error "Failed to pull ${model}"
        return 1
    fi
}

# Get model size
get_model_size() {
    local model=$1
    local size=$(ollama list | grep "^${model}" | awk '{print $2}')
    echo "$size"
}

# List installed models
list_models() {
    echo -e "\n${BOLD}Installed Models:${NC}\n"
    ollama list
    echo ""
}

# Show disk usage
show_disk_usage() {
    echo -e "\n${BOLD}Disk Usage by Models:${NC}\n"

    local total_size=0
    while IFS= read -r line; do
        if [[ $line =~ ^[^[:space:]] ]]; then
            echo "$line"
        fi
    done < <(ollama list)

    echo ""
    local model_dir="$HOME/.ollama/models"
    if [ -d "$model_dir" ]; then
        local total=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
        print_info "Total disk usage: ${total}"
    fi
    echo ""
}

# Remove a model
remove_model() {
    local model=$1

    if ! ollama list | grep -q "^${model}"; then
        print_warning "${model} not installed"
        return 1
    fi

    echo -e "${YELLOW}Removing ${model}...${NC}"
    if ollama rm "$model"; then
        print_status "${model} removed successfully"
        return 0
    else
        print_error "Failed to remove ${model}"
        return 1
    fi
}

# Pull all recommended models
pull_all_models() {
    echo -e "${BOLD}Pulling all recommended models...${NC}\n"
    print_warning "This will download ~50GB of data. Ensure you have sufficient disk space and bandwidth."
    echo ""

    local models=(
        "qwen3:8b"
        "qwen3:30b-a3b"
        "qwen3:72b"
        "gemma3:4b"
        "gemma3:12b"
        "qwen3-vl:8b"
        "qwen3-embedding"
        "nomic-embed-text"
        "deepseek-coder:33b"
    )

    local success=0
    local failed=0

    for model in "${models[@]}"; do
        if pull_model "$model"; then
            ((success++))
        else
            ((failed++))
        fi
        echo ""
    done

    echo -e "\n${BOLD}Summary:${NC}"
    print_status "$success models installed successfully"
    if [ $failed -gt 0 ]; then
        print_warning "$failed models failed to install"
    fi
}

# Pull fast models only
pull_fast_models() {
    echo -e "${BOLD}Pulling fast models...${NC}\n"

    local models=(
        "qwen3:30b-a3b"
        "gemma3:4b"
    )

    for model in "${models[@]}"; do
        pull_model "$model"
        echo ""
    done
}

# Pull vision models
pull_vision_models() {
    echo -e "${BOLD}Pulling vision models...${NC}\n"

    local models=(
        "qwen3-vl:8b"
    )

    for model in "${models[@]}"; do
        pull_model "$model"
        echo ""
    done
}

# Pull embedding models
pull_embedding_models() {
    echo -e "${BOLD}Pulling embedding models...${NC}\n"

    local models=(
        "qwen3-embedding"
        "nomic-embed-text"
    )

    for model in "${models[@]}"; do
        pull_model "$model"
        echo ""
    done
}

# Verify installations
verify_models() {
    echo -e "\n${BOLD}Verifying model installations...${NC}\n"

    local models=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")

    if [ -z "$models" ]; then
        print_warning "No models installed"
        return 1
    fi

    while IFS= read -r model; do
        if [ -n "$model" ]; then
            # Test model with simple query
            if echo "Hi" | ollama run "$model" > /dev/null 2>&1; then
                print_status "${model} verified"
            else
                print_warning "${model} installed but not responding"
            fi
        fi
    done <<< "$models"

    echo ""
}

# Main execution
main() {
    check_prereqs

    # Parse arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        -h|--help)
            show_help
            ;;
        -a|--all)
            pull_all_models
            verify_models
            show_disk_usage
            ;;
        -f|--fast)
            pull_fast_models
            verify_models
            ;;
        -v|--vision)
            pull_vision_models
            verify_models
            ;;
        -e|--embedding)
            pull_embedding_models
            verify_models
            ;;
        -l|--list)
            list_models
            ;;
        -s|--size)
            show_disk_usage
            ;;
        -r|--remove)
            if [ -z "$2" ]; then
                print_error "Please specify a model to remove"
                exit 1
            fi
            remove_model "$2"
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
