#!/bin/bash
# Test runner script with coverage reporting
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
${BOLD}Test Runner Script${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -a, --all           Run all tests (default)
    -u, --unit          Run unit tests only
    -i, --integration   Run integration tests only
    -c, --coverage      Generate coverage report
    -f, --fast          Run tests without coverage
    -v, --verbose       Verbose output
    -m, --mark MARK     Run tests with specific pytest mark
    -k, --keyword EXPR  Run tests matching keyword expression
    --ci                CI/CD mode (strict, no warnings)
    --html              Generate HTML coverage report
    --watch             Watch mode (re-run on file changes)

Examples:
    $0                  # Run all tests with default settings
    $0 --unit           # Run unit tests only
    $0 --coverage       # Run with detailed coverage report
    $0 -m slow          # Run tests marked as 'slow'
    $0 -k "test_agent"  # Run tests matching 'test_agent'
    $0 --ci             # Run in CI mode

Test Markers:
    unit        - Unit tests (fast, isolated)
    integration - Integration tests (require external services)
    slow        - Slow tests
    requires_ollama - Tests requiring Ollama server
    requires_mcp - Tests requiring MCP servers

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

# Check prerequisites
check_prereqs() {
    # Check if in project directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "Must be run from project root directory"
        exit 1
    fi

    # Check if uv is installed
    if ! command -v uv >/dev/null 2>&1; then
        print_error "uv not found. Please install: brew install uv"
        exit 1
    fi
}

# Setup test environment
setup_test_env() {
    print_info "Setting up test environment..."

    # Create necessary directories
    mkdir -p data/test logs/test

    # Set test environment variables
    export TESTING=1
    export LOG_LEVEL=WARNING
    export OLLAMA_BASE_URL=http://localhost:11434

    print_status "Test environment ready"
}

# Check Ollama availability (for integration tests)
check_ollama() {
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama server available"
        return 0
    else
        print_warning "Ollama server not available (integration tests will be skipped)"
        return 1
    fi
}

# Run unit tests
run_unit_tests() {
    echo -e "\n${BOLD}Running Unit Tests...${NC}\n"

    local cmd="uv run pytest tests/ -m unit"

    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=term-missing"
    fi

    eval "$cmd"
}

# Run integration tests
run_integration_tests() {
    echo -e "\n${BOLD}Running Integration Tests...${NC}\n"

    check_ollama

    local cmd="uv run pytest tests/ -m integration"

    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-append --cov-report=term-missing"
    fi

    eval "$cmd"
}

# Run all tests
run_all_tests() {
    echo -e "\n${BOLD}Running All Tests...${NC}\n"

    local cmd="uv run pytest tests/"

    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=term-missing"

        if [ "$HTML_REPORT" = true ]; then
            cmd="$cmd --cov-report=html"
        fi
    fi

    if [ "$CI_MODE" = true ]; then
        cmd="$cmd --strict-markers --tb=short -W error"
    fi

    if [ -n "$MARK" ]; then
        cmd="$cmd -m $MARK"
    fi

    if [ -n "$KEYWORD" ]; then
        cmd="$cmd -k $KEYWORD"
    fi

    eval "$cmd"
}

# Generate coverage report
generate_coverage_report() {
    echo -e "\n${BOLD}Generating Coverage Report...${NC}\n"

    uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

    print_status "Coverage report generated"
    print_info "HTML report: htmlcov/index.html"

    # Show coverage summary
    local coverage=$(uv run coverage report --precision=2 | tail -1 | awk '{print $NF}')
    echo ""
    print_info "Total coverage: ${coverage}"

    # Check coverage threshold
    local threshold=80
    local coverage_num=${coverage%\%}
    if (( $(echo "$coverage_num >= $threshold" | bc -l) )); then
        print_status "Coverage above ${threshold}% threshold"
    else
        print_warning "Coverage below ${threshold}% threshold"
    fi
}

# Watch mode - re-run tests on file changes
watch_tests() {
    echo -e "\n${BOLD}Watch Mode - Press Ctrl+C to stop${NC}\n"
    print_info "Watching for changes in tests/ and src/..."

    # Use fswatch if available, otherwise polling
    if command -v fswatch >/dev/null 2>&1; then
        fswatch -o tests/ src/ | while read; do
            clear
            echo -e "${BOLD}Files changed, re-running tests...${NC}\n"
            run_all_tests || true
            echo -e "\n${BOLD}Waiting for changes...${NC}"
        done
    else
        print_warning "fswatch not found. Install with: brew install fswatch"
        print_info "Falling back to manual mode. Press Enter to re-run tests."
        while true; do
            read -r
            clear
            run_all_tests || true
            echo -e "\n${BOLD}Press Enter to re-run tests...${NC}"
        done
    fi
}

# Cleanup test artifacts
cleanup() {
    print_info "Cleaning up test artifacts..."
    rm -rf data/test logs/test .pytest_cache __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_status "Cleanup complete"
}

# Print test summary
print_summary() {
    echo -e "\n${BOLD}${GREEN}Test Run Complete!${NC}\n"

    if [ "$COVERAGE" = true ] || [ "$HTML_REPORT" = true ]; then
        print_info "View coverage report: open htmlcov/index.html"
    fi

    echo ""
}

# Main execution
main() {
    # Default options
    TEST_TYPE="all"
    COVERAGE=false
    VERBOSE=false
    CI_MODE=false
    HTML_REPORT=false
    WATCH_MODE=false
    MARK=""
    KEYWORD=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                TEST_TYPE="all"
                shift
                ;;
            -u|--unit)
                TEST_TYPE="unit"
                shift
                ;;
            -i|--integration)
                TEST_TYPE="integration"
                shift
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -f|--fast)
                COVERAGE=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -m|--mark)
                MARK="$2"
                shift 2
                ;;
            -k|--keyword)
                KEYWORD="$2"
                shift 2
                ;;
            --ci)
                CI_MODE=true
                COVERAGE=true
                shift
                ;;
            --html)
                HTML_REPORT=true
                COVERAGE=true
                shift
                ;;
            --watch)
                WATCH_MODE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    check_prereqs
    setup_test_env

    # Handle watch mode
    if [ "$WATCH_MODE" = true ]; then
        watch_tests
        exit 0
    fi

    # Run tests based on type
    case "$TEST_TYPE" in
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        all)
            run_all_tests
            ;;
    esac

    # Generate coverage report if requested
    if [ "$HTML_REPORT" = true ] && [ "$COVERAGE" = false ]; then
        generate_coverage_report
    fi

    print_summary
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main
main "$@"
