#!/bin/bash
# Cleanup script - Remove temporary files and reset to clean state
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
${BOLD}Cleanup Script${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -a, --all           Clean everything (excludes models)
    -v, --vectors       Remove vector stores
    -c, --checkpoints   Remove LangGraph checkpoints
    -p, --pycache       Clean Python cache files
    -l, --logs          Clear log files
    -t, --temp          Remove temporary files
    -d, --data          Remove data directory (preserves configs)
    -m, --models        Remove downloaded models (CAUTION!)
    --deep              Deep clean (all + node_modules, .venv)
    --dry-run           Show what would be deleted without deleting

Examples:
    $0                  # Interactive mode
    $0 --all            # Clean everything except models
    $0 --vectors        # Remove vector stores only
    $0 --deep           # Deep clean including dependencies
    $0 --dry-run        # Preview what will be deleted

EOF
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

# Get directory size
get_size() {
    local path=$1
    if [ -d "$path" ] || [ -f "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Clean vector stores
clean_vectors() {
    echo -e "\n${BOLD}Cleaning Vector Stores...${NC}\n"

    local paths=(
        "data/chroma_db"
        "data/faiss_index"
        "data/vectorstores"
    )

    local total_size=0

    for path in "${paths[@]}"; do
        if [ -d "$path" ]; then
            local size=$(get_size "$path")
            print_info "Removing $path ($size)"

            if [ "$DRY_RUN" = false ]; then
                rm -rf "$path"
                print_status "Removed $path"
            fi
        fi
    done

    if [ "$DRY_RUN" = false ]; then
        print_status "Vector stores cleaned"
    fi
}

# Clean checkpoints
clean_checkpoints() {
    echo -e "\n${BOLD}Cleaning LangGraph Checkpoints...${NC}\n"

    local paths=(
        "checkpoints"
        "data/checkpoints"
        ".langgraph"
    )

    for path in "${paths[@]}"; do
        if [ -d "$path" ]; then
            local size=$(get_size "$path")
            print_info "Removing $path ($size)"

            if [ "$DRY_RUN" = false ]; then
                rm -rf "$path"
                print_status "Removed $path"
            fi
        fi
    done

    if [ "$DRY_RUN" = false ]; then
        print_status "Checkpoints cleaned"
    fi
}

# Clean Python cache
clean_pycache() {
    echo -e "\n${BOLD}Cleaning Python Cache...${NC}\n"

    local count=0

    # Find and remove __pycache__ directories
    while IFS= read -r -d '' dir; do
        if [ "$DRY_RUN" = false ]; then
            rm -rf "$dir"
        fi
        ((count++))
    done < <(find . -type d -name "__pycache__" -print0 2>/dev/null)

    print_info "Found $count __pycache__ directories"

    # Remove .pyc files
    local pyc_count=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l | tr -d ' ')
    print_info "Found $pyc_count .pyc files"

    if [ "$DRY_RUN" = false ]; then
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
    fi

    # Remove pytest cache
    if [ -d ".pytest_cache" ]; then
        if [ "$DRY_RUN" = false ]; then
            rm -rf .pytest_cache
        fi
        print_info "Removed .pytest_cache"
    fi

    # Remove coverage files
    if [ -d "htmlcov" ]; then
        if [ "$DRY_RUN" = false ]; then
            rm -rf htmlcov
        fi
        print_info "Removed htmlcov"
    fi

    if [ -f ".coverage" ]; then
        if [ "$DRY_RUN" = false ]; then
            rm -f .coverage
        fi
        print_info "Removed .coverage"
    fi

    if [ "$DRY_RUN" = false ]; then
        print_status "Python cache cleaned"
    fi
}

# Clean logs
clean_logs() {
    echo -e "\n${BOLD}Cleaning Logs...${NC}\n"

    if [ -d "logs" ]; then
        local size=$(get_size "logs")
        print_info "Logs directory size: $size"

        if [ "$DRY_RUN" = false ]; then
            # Keep directory structure but remove log files
            find logs -type f -name "*.log" -delete 2>/dev/null || true
            print_status "Log files removed"
        fi
    else
        print_info "No logs directory found"
    fi
}

# Clean temporary files
clean_temp() {
    echo -e "\n${BOLD}Cleaning Temporary Files...${NC}\n"

    local temp_patterns=(
        "*.tmp"
        "*.temp"
        ".DS_Store"
        "Thumbs.db"
    )

    for pattern in "${temp_patterns[@]}"; do
        local count=$(find . -name "$pattern" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$count" -gt 0 ]; then
            print_info "Found $count $pattern files"
            if [ "$DRY_RUN" = false ]; then
                find . -name "$pattern" -delete 2>/dev/null || true
            fi
        fi
    done

    if [ "$DRY_RUN" = false ]; then
        print_status "Temporary files cleaned"
    fi
}

# Clean data directory
clean_data() {
    echo -e "\n${BOLD}Cleaning Data Directory...${NC}\n"

    print_warning "This will remove all data except configuration files"

    if [ "$DRY_RUN" = true ]; then
        print_info "Would remove: data/* (excluding configs)"
        return
    fi

    # Prompt for confirmation if not in auto mode
    if [ "$AUTO_CONFIRM" = false ]; then
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping data cleanup"
            return
        fi
    fi

    if [ -d "data" ]; then
        # Remove subdirectories but keep structure
        find data -mindepth 1 -maxdepth 1 -type d ! -name "config" -exec rm -rf {} + 2>/dev/null || true

        # Remove files except .gitkeep and configs
        find data -type f ! -name ".gitkeep" ! -name "*.yaml" ! -name "*.json" -delete 2>/dev/null || true

        print_status "Data directory cleaned"
    fi
}

# Clean models
clean_models() {
    echo -e "\n${BOLD}Cleaning Downloaded Models...${NC}\n"

    print_warning "This will remove ALL downloaded Ollama models"
    print_warning "Models will need to be re-downloaded"

    if [ "$DRY_RUN" = true ]; then
        print_info "Would remove all Ollama models"
        ollama list
        return
    fi

    # Prompt for confirmation
    read -p "Are you ABSOLUTELY sure? (type 'yes' to confirm): " -r
    echo
    if [[ ! $REPLY == "yes" ]]; then
        print_info "Skipping model cleanup"
        return
    fi

    # Get list of models
    local models=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")

    if [ -z "$models" ]; then
        print_info "No models to remove"
        return
    fi

    echo "$models" | while read -r model; do
        print_info "Removing $model..."
        ollama rm "$model" 2>/dev/null || true
    done

    print_status "All models removed"
}

# Deep clean
deep_clean() {
    echo -e "\n${BOLD}Deep Clean Mode${NC}\n"

    print_warning "This will remove dependencies and require full reinstall"

    if [ "$DRY_RUN" = true ]; then
        print_info "Would remove: node_modules, .venv, uv.lock"
        return
    fi

    if [ "$AUTO_CONFIRM" = false ]; then
        read -p "Continue with deep clean? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping deep clean"
            return
        fi
    fi

    # Remove Node modules
    if [ -d "node_modules" ]; then
        local size=$(get_size "node_modules")
        print_info "Removing node_modules ($size)..."
        rm -rf node_modules
        print_status "node_modules removed"
    fi

    # Remove Python virtual environment
    if [ -d ".venv" ]; then
        local size=$(get_size ".venv")
        print_info "Removing .venv ($size)..."
        rm -rf .venv
        print_status ".venv removed"
    fi

    # Remove lock files
    if [ -f "uv.lock" ]; then
        rm -f uv.lock
        print_info "uv.lock removed"
    fi

    if [ -f "package-lock.json" ]; then
        rm -f package-lock.json
        print_info "package-lock.json removed"
    fi

    print_status "Deep clean complete"
}

# Show disk space summary
show_summary() {
    echo -e "\n${BOLD}Disk Space Summary${NC}\n"

    local total=0

    echo "Current directory sizes:"
    echo ""

    local dirs=("data" "logs" "node_modules" ".venv" "checkpoints")

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            local size=$(get_size "$dir")
            printf "  %-20s %s\n" "$dir:" "$size"
        fi
    done

    echo ""

    # Ollama models
    if command -v ollama >/dev/null 2>&1; then
        local model_dir="$HOME/.ollama/models"
        if [ -d "$model_dir" ]; then
            local model_size=$(get_size "$model_dir")
            printf "  %-20s %s\n" "Ollama models:" "$model_size"
        fi
    fi

    echo ""
}

# Interactive mode
interactive_clean() {
    echo -e "${BOLD}Interactive Cleanup${NC}\n"

    PS3=$'\nSelect cleanup option: '
    options=(
        "Vector stores"
        "Checkpoints"
        "Python cache"
        "Logs"
        "Temporary files"
        "Data directory"
        "All (excludes models)"
        "Deep clean"
        "Show summary"
        "Exit"
    )

    select opt in "${options[@]}"; do
        case $opt in
            "Vector stores")
                clean_vectors
                ;;
            "Checkpoints")
                clean_checkpoints
                ;;
            "Python cache")
                clean_pycache
                ;;
            "Logs")
                clean_logs
                ;;
            "Temporary files")
                clean_temp
                ;;
            "Data directory")
                clean_data
                ;;
            "All (excludes models)")
                clean_vectors
                clean_checkpoints
                clean_pycache
                clean_logs
                clean_temp
                ;;
            "Deep clean")
                deep_clean
                ;;
            "Show summary")
                show_summary
                ;;
            "Exit")
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Main execution
main() {
    # Default options
    DRY_RUN=false
    AUTO_CONFIRM=false
    ACTION=""

    # Parse arguments
    if [ $# -eq 0 ]; then
        interactive_clean
        exit 0
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                ACTION="all"
                shift
                ;;
            -v|--vectors)
                ACTION="vectors"
                shift
                ;;
            -c|--checkpoints)
                ACTION="checkpoints"
                shift
                ;;
            -p|--pycache)
                ACTION="pycache"
                shift
                ;;
            -l|--logs)
                ACTION="logs"
                shift
                ;;
            -t|--temp)
                ACTION="temp"
                shift
                ;;
            -d|--data)
                ACTION="data"
                AUTO_CONFIRM=false
                shift
                ;;
            -m|--models)
                ACTION="models"
                shift
                ;;
            --deep)
                ACTION="deep"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Show dry run notice
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No files will be deleted"
        echo ""
    fi

    # Execute action
    case "$ACTION" in
        vectors)
            clean_vectors
            ;;
        checkpoints)
            clean_checkpoints
            ;;
        pycache)
            clean_pycache
            ;;
        logs)
            clean_logs
            ;;
        temp)
            clean_temp
            ;;
        data)
            clean_data
            ;;
        models)
            clean_models
            ;;
        all)
            AUTO_CONFIRM=true
            clean_vectors
            clean_checkpoints
            clean_pycache
            clean_logs
            clean_temp
            ;;
        deep)
            clean_vectors
            clean_checkpoints
            clean_pycache
            clean_logs
            clean_temp
            deep_clean
            ;;
        *)
            print_error "No action specified"
            show_help
            exit 1
            ;;
    esac

    if [ "$DRY_RUN" = false ]; then
        echo -e "\n${GREEN}Cleanup complete!${NC}\n"
        show_summary
    fi
}

# Run main
main "$@"
