#!/bin/bash
# Development mode script - Start services and watch for changes
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
${BOLD}Development Mode Script${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -s, --start         Start all services
    -w, --watch         Watch mode (auto-reload examples)
    -l, --langgraph     Start LangGraph Studio
    -j, --jupyter       Start Jupyter Lab
    -o, --ollama        Start Ollama server
    -a, --all           Start all services + watch
    --port PORT         Specify Jupyter port (default: 8888)
    --host HOST         Specify host (default: localhost)
    --no-browser        Don't open browser automatically

Examples:
    $0 --all            # Start all services
    $0 --watch          # Watch mode only
    $0 --jupyter        # Start Jupyter Lab
    $0 -l -j            # Start LangGraph + Jupyter

Services:
    - Ollama (local LLM server)
    - LangGraph Studio (agent workflows)
    - Jupyter Lab (notebooks)
    - File watcher (auto-reload)

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

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"

    # Kill background processes
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi

    if [ -n "$LANGGRAPH_PID" ]; then
        kill $LANGGRAPH_PID 2>/dev/null || true
    fi

    if [ -n "$JUPYTER_PID" ]; then
        kill $JUPYTER_PID 2>/dev/null || true
    fi

    if [ -n "$WATCHER_PID" ]; then
        kill $WATCHER_PID 2>/dev/null || true
    fi

    print_status "Services stopped"
    exit 0
}

# Trap cleanup on exit
trap cleanup EXIT INT TERM

# Check prerequisites
check_prereqs() {
    print_info "Checking prerequisites..."

    # Check if in project directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "Must be run from project root directory"
        exit 1
    fi

    # Check required tools
    local missing=()

    if ! command -v uv >/dev/null 2>&1; then
        missing+=("uv")
    fi

    if ! command -v node >/dev/null 2>&1; then
        missing+=("node")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        print_info "Run: ./scripts/setup.sh"
        exit 1
    fi

    print_status "Prerequisites OK"
}

# Start Ollama server
start_ollama() {
    if pgrep -x "ollama" > /dev/null; then
        print_status "Ollama already running"
        return 0
    fi

    print_info "Starting Ollama server..."

    ollama serve > logs/ollama.log 2>&1 &
    OLLAMA_PID=$!

    # Wait for server to start
    local max_attempts=10
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_status "Ollama server started (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 1
        ((attempt++))
    done

    print_error "Failed to start Ollama server"
    return 1
}

# Start LangGraph Studio
start_langgraph() {
    print_info "Starting LangGraph Studio..."

    # Create logs directory if needed
    mkdir -p logs

    # Start LangGraph dev server
    npx langgraph@latest dev > logs/langgraph.log 2>&1 &
    LANGGRAPH_PID=$!

    sleep 3

    if ps -p $LANGGRAPH_PID > /dev/null; then
        print_status "LangGraph Studio started (PID: $LANGGRAPH_PID)"
        print_info "Access at: http://localhost:8123"

        if [ "$OPEN_BROWSER" = true ]; then
            sleep 2
            open http://localhost:8123 2>/dev/null || true
        fi
    else
        print_error "Failed to start LangGraph Studio"
        return 1
    fi
}

# Start Jupyter Lab
start_jupyter() {
    local port=${JUPYTER_PORT:-8888}
    local host=${JUPYTER_HOST:-localhost}

    print_info "Starting Jupyter Lab on ${host}:${port}..."

    mkdir -p logs

    local cmd="uv run jupyter lab --ip=$host --port=$port --no-browser"

    if [ "$OPEN_BROWSER" = true ]; then
        cmd="uv run jupyter lab --ip=$host --port=$port"
    fi

    $cmd > logs/jupyter.log 2>&1 &
    JUPYTER_PID=$!

    sleep 3

    if ps -p $JUPYTER_PID > /dev/null; then
        print_status "Jupyter Lab started (PID: $JUPYTER_PID)"
        print_info "Access at: http://${host}:${port}"

        # Extract token from logs
        sleep 2
        local token=$(grep -oE "token=[a-f0-9]+" logs/jupyter.log | head -1 | cut -d= -f2)
        if [ -n "$token" ]; then
            print_info "Token: $token"
        fi
    else
        print_error "Failed to start Jupyter Lab"
        return 1
    fi
}

# Watch mode - auto-reload examples
start_watcher() {
    print_info "Starting file watcher..."

    if ! command -v fswatch >/dev/null 2>&1; then
        print_warning "fswatch not found. Install with: brew install fswatch"
        print_info "Falling back to manual mode"
        return 1
    fi

    {
        fswatch -o examples/ src/ | while read; do
            clear
            echo -e "${BOLD}Files changed - $(date)${NC}\n"
            print_info "Changes detected in examples/ or src/"

            # Optionally run linter
            if command -v ruff >/dev/null 2>&1; then
                echo -e "\n${BOLD}Running linter...${NC}"
                uv run ruff check examples/ src/ --fix 2>/dev/null || true
            fi

            echo -e "\n${BOLD}Ready for testing${NC}"
            echo -e "Run example: ${BLUE}uv run python examples/...${NC}\n"
        done
    } &

    WATCHER_PID=$!
    print_status "File watcher started (PID: $WATCHER_PID)"
}

# Show service status
show_status() {
    echo -e "\n${BOLD}Service Status:${NC}\n"

    # Ollama
    if pgrep -x "ollama" > /dev/null; then
        print_status "Ollama: Running"
        echo "           http://localhost:11434"
    else
        print_warning "Ollama: Not running"
    fi

    # LangGraph
    if [ -n "$LANGGRAPH_PID" ] && ps -p $LANGGRAPH_PID > /dev/null; then
        print_status "LangGraph: Running"
        echo "           http://localhost:8123"
    else
        print_info "LangGraph: Not running"
    fi

    # Jupyter
    if [ -n "$JUPYTER_PID" ] && ps -p $JUPYTER_PID > /dev/null; then
        print_status "Jupyter: Running"
        echo "           http://localhost:${JUPYTER_PORT:-8888}"
    else
        print_info "Jupyter: Not running"
    fi

    # Watcher
    if [ -n "$WATCHER_PID" ] && ps -p $WATCHER_PID > /dev/null; then
        print_status "Watcher: Active"
    else
        print_info "Watcher: Not running"
    fi

    echo ""
}

# Show useful commands
show_commands() {
    echo -e "${BOLD}Useful Commands:${NC}\n"

    cat << EOF
Development:
    uv run python examples/...       Run example
    uv run pytest tests/             Run tests
    uv run ruff check .              Lint code
    uv run ruff format .             Format code

Ollama:
    ollama list                      List models
    ollama pull <model>              Download model
    ollama run <model>               Chat with model

LangGraph:
    npx langgraph@latest dev         Start studio

Logs:
    tail -f logs/ollama.log          View Ollama logs
    tail -f logs/langgraph.log       View LangGraph logs
    tail -f logs/jupyter.log         View Jupyter logs

EOF
}

# Main execution
main() {
    # Default options
    START_OLLAMA=false
    START_LANGGRAPH=false
    START_JUPYTER=false
    START_WATCHER=false
    OPEN_BROWSER=true
    JUPYTER_PORT=8888
    JUPYTER_HOST="localhost"

    # Parse arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--start)
                START_OLLAMA=true
                shift
                ;;
            -w|--watch)
                START_WATCHER=true
                shift
                ;;
            -l|--langgraph)
                START_LANGGRAPH=true
                shift
                ;;
            -j|--jupyter)
                START_JUPYTER=true
                shift
                ;;
            -o|--ollama)
                START_OLLAMA=true
                shift
                ;;
            -a|--all)
                START_OLLAMA=true
                START_LANGGRAPH=true
                START_JUPYTER=true
                START_WATCHER=true
                shift
                ;;
            --port)
                JUPYTER_PORT="$2"
                shift 2
                ;;
            --host)
                JUPYTER_HOST="$2"
                shift 2
                ;;
            --no-browser)
                OPEN_BROWSER=false
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    echo -e "${BOLD}Development Mode${NC}\n"

    check_prereqs

    # Create logs directory
    mkdir -p logs

    # Start services
    if [ "$START_OLLAMA" = true ]; then
        start_ollama
    fi

    if [ "$START_LANGGRAPH" = true ]; then
        start_langgraph
    fi

    if [ "$START_JUPYTER" = true ]; then
        start_jupyter
    fi

    if [ "$START_WATCHER" = true ]; then
        start_watcher
    fi

    # Show status
    show_status
    show_commands

    echo -e "${BOLD}${GREEN}Development environment ready!${NC}\n"
    print_info "Press Ctrl+C to stop all services"
    echo ""

    # Keep script running
    while true; do
        sleep 60

        # Check if services are still running
        if [ "$START_OLLAMA" = true ] && ! pgrep -x "ollama" > /dev/null; then
            print_warning "Ollama stopped unexpectedly"
            start_ollama
        fi

        if [ "$START_LANGGRAPH" = true ] && [ -n "$LANGGRAPH_PID" ] && ! ps -p $LANGGRAPH_PID > /dev/null; then
            print_warning "LangGraph stopped unexpectedly"
            start_langgraph
        fi

        if [ "$START_JUPYTER" = true ] && [ -n "$JUPYTER_PID" ] && ! ps -p $JUPYTER_PID > /dev/null; then
            print_warning "Jupyter stopped unexpectedly"
            start_jupyter
        fi
    done
}

# Run main
main "$@"
