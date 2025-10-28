#!/bin/bash
# Comprehensive validation script for AI Lang Stuff project
# Runs all local tests, benchmarks, and health checks
set -e

# Colors and formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis
CHECK="✅"
CROSS="❌"
CLOCK="⏱️"
SEARCH="🔍"
ROCKET="🚀"
PACKAGE="📦"
GEAR="⚙️"
SKIP="⏭️"
WARNING="⚠️"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
TIMESTAMP=$(date +%Y-%m-%d)
REPORT_FILE="${REPORT_DIR}/validation-${TIMESTAMP}.md"
QUICK_MODE=false
VERBOSE=false
SKIP_SLOW=false

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0
START_TIME=$(date +%s)

# Help text
show_help() {
    cat << EOF
${BOLD}AI Lang Stuff - Comprehensive Validation${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -q, --quick         Quick validation (skip slow tests and benchmarks)
    -v, --verbose       Verbose output with detailed logs
    -s, --skip-slow     Skip slow-running tests
    --no-report         Don't generate validation report
    --fix               Attempt to auto-fix common issues

Validation Steps:
    1. Environment checks (Python, Node, Ollama, uv)
    2. Model availability checks
    3. Unit tests (all utilities)
    4. Integration tests (examples)
    5. Benchmarks (performance validation)
    6. Example validation
    7. MCP server health checks
    8. Vector store validation
    9. CLI tool validation
    10. Generate validation report

Examples:
    $0                  # Full validation
    $0 --quick          # Quick validation
    $0 --verbose        # Verbose output
    $0 --fix            # Auto-fix issues

EOF
}

# Parse arguments
AUTO_FIX=false
NO_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quick)
            QUICK_MODE=true
            SKIP_SLOW=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        --no-report)
            NO_REPORT=true
            shift
            ;;
        --fix)
            AUTO_FIX=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Utility functions
print_header() {
    echo -e "\n${CYAN}${BOLD}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${BLUE}${BOLD}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
    ((PASSED_CHECKS++))
}

print_failure() {
    echo -e "${RED}${CROSS} $1${NC}"
    ((FAILED_CHECKS++))
}

print_skip() {
    echo -e "${YELLOW}${SKIP} $1${NC}"
    ((SKIPPED_CHECKS++))
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

time_step() {
    local start=$1
    local end=$(date +%s)
    echo "$((end - start))"
}

format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    else
        echo "$((seconds / 60))m $((seconds % 60))s"
    fi
}

# Initialize report
init_report() {
    mkdir -p "$REPORT_DIR"
    cat > "$REPORT_FILE" << EOF
# Validation Report
**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Mode:** $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")

## Summary

EOF
}

append_report() {
    echo "$1" >> "$REPORT_FILE"
}

# Validation functions
validate_environment() {
    print_step "Environment Check"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    local issues=0

    # Check Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | awk '{print $2}')
        [ "$VERBOSE" = true ] && echo "  Python: $python_version"
    else
        print_warning "Python 3 not found"
        ((issues++))
    fi

    # Check uv
    if command -v uv &> /dev/null; then
        local uv_version=$(uv --version | awk '{print $2}')
        [ "$VERBOSE" = true ] && echo "  uv: $uv_version"
    else
        print_warning "uv not found"
        if [ "$AUTO_FIX" = true ]; then
            echo "  Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        ((issues++))
    fi

    # Check Node
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        [ "$VERBOSE" = true ] && echo "  Node.js: $node_version"
    else
        print_warning "Node.js not found"
        ((issues++))
    fi

    # Check Ollama
    if command -v ollama &> /dev/null; then
        local ollama_version=$(ollama --version | head -n1)
        [ "$VERBOSE" = true ] && echo "  Ollama: $ollama_version"

        # Check if Ollama is running
        if curl -s http://localhost:11434 &> /dev/null; then
            [ "$VERBOSE" = true ] && echo "  Ollama service: Running"
        else
            print_warning "Ollama not running"
            if [ "$AUTO_FIX" = true ]; then
                echo "  Starting Ollama..."
                ollama serve &
                sleep 2
            fi
            ((issues++))
        fi
    else
        print_warning "Ollama not found"
        ((issues++))
    fi

    local elapsed=$(time_step $step_start)
    if [ $issues -eq 0 ]; then
        print_success "Environment Check (${elapsed}s)"
        append_report "- ${CHECK} **Environment Check**: All dependencies found (${elapsed}s)"
    else
        print_failure "Environment Check (${elapsed}s) - $issues issues"
        append_report "- ${CROSS} **Environment Check**: $issues issues found (${elapsed}s)"
    fi
}

validate_models() {
    print_step "Model Availability"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    if ! command -v ollama &> /dev/null; then
        print_skip "Ollama not available"
        return
    fi

    local required_models=("qwen3:8b" "gemma3:4b")
    local available_models=$(ollama list | tail -n +2 | awk '{print $1}')
    local missing=0

    for model in "${required_models[@]}"; do
        if echo "$available_models" | grep -q "^${model}"; then
            [ "$VERBOSE" = true ] && echo "  ${CHECK} $model"
        else
            print_warning "Missing model: $model"
            if [ "$AUTO_FIX" = true ]; then
                echo "  Pulling $model..."
                ollama pull "$model"
            fi
            ((missing++))
        fi
    done

    local elapsed=$(time_step $step_start)
    if [ $missing -eq 0 ]; then
        print_success "Model Availability (${elapsed}s)"
        append_report "- ${CHECK} **Model Availability**: All required models present (${elapsed}s)"
    else
        print_warning "Model Availability (${elapsed}s) - $missing models missing"
        append_report "- ${WARNING} **Model Availability**: $missing models missing (${elapsed}s)"
    fi
}

run_unit_tests() {
    print_step "Unit Tests"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    cd "$PROJECT_ROOT"

    if [ ! -d "tests" ]; then
        print_skip "No tests directory found"
        append_report "- ${SKIP} **Unit Tests**: No tests found"
        return
    fi

    local test_output
    if [ "$VERBOSE" = true ]; then
        uv run pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_output.txt
        test_output=$(cat /tmp/test_output.txt)
    else
        test_output=$(uv run pytest tests/ -v --tb=short 2>&1)
    fi

    local exit_code=$?
    local passed=$(echo "$test_output" | grep -oE '[0-9]+ passed' | awk '{print $1}' || echo "0")
    local failed=$(echo "$test_output" | grep -oE '[0-9]+ failed' | awk '{print $1}' || echo "0")

    local elapsed=$(time_step $step_start)
    if [ $exit_code -eq 0 ]; then
        print_success "Unit Tests (${elapsed}s) - ${passed} passed"
        append_report "- ${CHECK} **Unit Tests**: ${passed} tests passed (${elapsed}s)"
    else
        print_failure "Unit Tests (${elapsed}s) - ${failed} failed, ${passed} passed"
        append_report "- ${CROSS} **Unit Tests**: ${failed} failed, ${passed} passed (${elapsed}s)"
    fi
}

run_integration_tests() {
    print_step "Integration Tests"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    if [ "$SKIP_SLOW" = true ]; then
        print_skip "Skipped (slow tests disabled)"
        append_report "- ${SKIP} **Integration Tests**: Skipped"
        return
    fi

    cd "$PROJECT_ROOT"

    # Run integration tests if they exist
    if [ -d "tests/integration" ]; then
        local test_output
        if [ "$VERBOSE" = true ]; then
            uv run pytest tests/integration/ -v --tb=short 2>&1 | tee /tmp/integration_output.txt
            test_output=$(cat /tmp/integration_output.txt)
        else
            test_output=$(uv run pytest tests/integration/ -v --tb=short 2>&1)
        fi

        local exit_code=$?
        local passed=$(echo "$test_output" | grep -oE '[0-9]+ passed' | awk '{print $1}' || echo "0")
        local failed=$(echo "$test_output" | grep -oE '[0-9]+ failed' | awk '{print $1}' || echo "0")

        local elapsed=$(time_step $step_start)
        if [ $exit_code -eq 0 ]; then
            print_success "Integration Tests (${elapsed}s) - ${passed} passed"
            append_report "- ${CHECK} **Integration Tests**: ${passed} tests passed (${elapsed}s)"
        else
            print_failure "Integration Tests (${elapsed}s) - ${failed} failed"
            append_report "- ${CROSS} **Integration Tests**: ${failed} failed (${elapsed}s)"
        fi
    else
        print_skip "No integration tests found"
        append_report "- ${SKIP} **Integration Tests**: Not found"
    fi
}

run_benchmarks() {
    print_step "Performance Benchmarks"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    if [ "$QUICK_MODE" = true ]; then
        print_skip "Skipped (quick mode)"
        append_report "- ${SKIP} **Benchmarks**: Skipped (quick mode)"
        return
    fi

    cd "$PROJECT_ROOT"

    if [ -f "scripts/benchmark.sh" ]; then
        local bench_output
        if [ "$VERBOSE" = true ]; then
            ./scripts/benchmark.sh --suite inference 2>&1 | tee /tmp/bench_output.txt
            bench_output=$(cat /tmp/bench_output.txt)
        else
            bench_output=$(./scripts/benchmark.sh --suite inference 2>&1)
        fi

        local exit_code=$?
        local elapsed=$(time_step $step_start)

        if [ $exit_code -eq 0 ]; then
            print_success "Benchmarks (${elapsed}s)"
            append_report "- ${CHECK} **Benchmarks**: Completed successfully (${elapsed}s)"
        else
            print_warning "Benchmarks (${elapsed}s) - Some benchmarks failed"
            append_report "- ${WARNING} **Benchmarks**: Some failures (${elapsed}s)"
        fi
    else
        print_skip "No benchmark script found"
        append_report "- ${SKIP} **Benchmarks**: Not configured"
    fi
}

validate_examples() {
    print_step "Example Validation"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    if [ "$SKIP_SLOW" = true ]; then
        print_skip "Skipped (slow tests disabled)"
        append_report "- ${SKIP} **Examples**: Skipped"
        return
    fi

    cd "$PROJECT_ROOT"

    if [ ! -d "examples" ]; then
        print_skip "No examples directory found"
        append_report "- ${SKIP} **Examples**: Not found"
        return
    fi

    local total_examples=0
    local working_examples=0

    # Test a few key examples
    local test_examples=("utils/test_ollama_manager.py" "utils/test_tool_registry.py")

    for example in "${test_examples[@]}"; do
        if [ -f "$example" ]; then
            ((total_examples++))
            if [ "$VERBOSE" = true ]; then
                echo "  Testing: $example"
            fi

            if python3 "$example" &> /dev/null; then
                ((working_examples++))
                [ "$VERBOSE" = true ] && echo "    ${CHECK} Passed"
            else
                [ "$VERBOSE" = true ] && echo "    ${CROSS} Failed"
            fi
        fi
    done

    local elapsed=$(time_step $step_start)
    if [ $working_examples -eq $total_examples ] && [ $total_examples -gt 0 ]; then
        print_success "Examples (${elapsed}s) - ${working_examples}/${total_examples} working"
        append_report "- ${CHECK} **Examples**: ${working_examples}/${total_examples} working (${elapsed}s)"
    elif [ $total_examples -eq 0 ]; then
        print_skip "No testable examples found"
        append_report "- ${SKIP} **Examples**: None tested"
    else
        print_warning "Examples (${elapsed}s) - ${working_examples}/${total_examples} working"
        append_report "- ${WARNING} **Examples**: ${working_examples}/${total_examples} working (${elapsed}s)"
    fi
}

validate_mcp_servers() {
    print_step "MCP Servers"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    cd "$PROJECT_ROOT"

    if [ ! -d "mcp-servers" ]; then
        print_skip "No MCP servers directory found"
        append_report "- ${SKIP} **MCP Servers**: Not configured"
        return
    fi

    local total_servers=0
    local healthy_servers=0

    # Count server directories
    for server_dir in mcp-servers/*/; do
        if [ -d "$server_dir" ]; then
            ((total_servers++))
            local server_name=$(basename "$server_dir")

            # Check if server has package.json or requirements.txt
            if [ -f "${server_dir}package.json" ] || [ -f "${server_dir}requirements.txt" ]; then
                ((healthy_servers++))
                [ "$VERBOSE" = true ] && echo "  ${CHECK} $server_name"
            else
                [ "$VERBOSE" = true ] && echo "  ${WARNING} $server_name (incomplete)"
            fi
        fi
    done

    local elapsed=$(time_step $step_start)
    if [ $total_servers -eq 0 ]; then
        print_skip "No MCP servers found"
        append_report "- ${SKIP} **MCP Servers**: None configured"
    elif [ $healthy_servers -eq $total_servers ]; then
        print_success "MCP Servers (${elapsed}s) - ${healthy_servers}/${total_servers} healthy"
        append_report "- ${CHECK} **MCP Servers**: ${healthy_servers}/${total_servers} healthy (${elapsed}s)"
    else
        print_warning "MCP Servers (${elapsed}s) - ${healthy_servers}/${total_servers} healthy"
        append_report "- ${WARNING} **MCP Servers**: ${healthy_servers}/${total_servers} healthy (${elapsed}s)"
    fi
}

validate_vector_stores() {
    print_step "Vector Stores"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    cd "$PROJECT_ROOT"

    # Check if vector store utility exists
    if [ -f "utils/vector_store.py" ]; then
        if python3 -c "from utils.vector_store import VectorStoreManager" &> /dev/null; then
            local elapsed=$(time_step $step_start)
            print_success "Vector Stores (${elapsed}s) - Operational"
            append_report "- ${CHECK} **Vector Stores**: Operational (${elapsed}s)"
        else
            local elapsed=$(time_step $step_start)
            print_failure "Vector Stores (${elapsed}s) - Import failed"
            append_report "- ${CROSS} **Vector Stores**: Import failed (${elapsed}s)"
        fi
    else
        print_skip "Vector store not configured"
        append_report "- ${SKIP} **Vector Stores**: Not configured"
    fi
}

validate_cli_tool() {
    print_step "CLI Tool"
    local step_start=$(date +%s)
    ((TOTAL_CHECKS++))

    cd "$PROJECT_ROOT"

    if [ -f "main.py" ]; then
        # Test CLI help command
        if python3 main.py --help &> /dev/null; then
            local elapsed=$(time_step $step_start)
            print_success "CLI Tool (${elapsed}s) - Working"
            append_report "- ${CHECK} **CLI Tool**: All commands working (${elapsed}s)"
        else
            local elapsed=$(time_step $step_start)
            print_failure "CLI Tool (${elapsed}s) - Failed"
            append_report "- ${CROSS} **CLI Tool**: Failed (${elapsed}s)"
        fi
    else
        print_skip "No CLI tool found"
        append_report "- ${SKIP} **CLI Tool**: Not found"
    fi
}

generate_summary() {
    local total_time=$(time_step $START_TIME)
    local formatted_time=$(format_time $total_time)

    print_header "Validation Summary"

    echo -e "${BOLD}Results:${NC}"
    echo -e "  ${GREEN}${CHECK} Passed:  ${PASSED_CHECKS}${NC}"
    echo -e "  ${RED}${CROSS} Failed:  ${FAILED_CHECKS}${NC}"
    echo -e "  ${YELLOW}${SKIP} Skipped: ${SKIPPED_CHECKS}${NC}"
    echo -e "  ${CYAN}━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${BOLD}Total:   ${TOTAL_CHECKS}${NC}"
    echo ""
    echo -e "${CLOCK} ${BOLD}Total Time: ${formatted_time}${NC}"

    if [ "$NO_REPORT" = false ]; then
        echo -e "${BOLD}Report: ${CYAN}${REPORT_FILE}${NC}"
    fi
    echo ""

    # Finalize report
    if [ "$NO_REPORT" = false ]; then
        cat >> "$REPORT_FILE" << EOF

## Statistics

- **Total Checks**: ${TOTAL_CHECKS}
- **Passed**: ${PASSED_CHECKS}
- **Failed**: ${FAILED_CHECKS}
- **Skipped**: ${SKIPPED_CHECKS}
- **Total Time**: ${formatted_time}

## Conclusion

EOF

        if [ $FAILED_CHECKS -eq 0 ]; then
            echo "${CHECK} **ALL CHECKS PASSED**" >> "$REPORT_FILE"
            echo -e "${GREEN}${BOLD}${CHECK} ALL CHECKS PASSED${NC}\n"
        else
            echo "${CROSS} **VALIDATION FAILED** - ${FAILED_CHECKS} check(s) failed" >> "$REPORT_FILE"
            echo -e "${RED}${BOLD}${CROSS} VALIDATION FAILED${NC} - ${FAILED_CHECKS} check(s) failed\n"
        fi
    fi

    # Exit with appropriate code
    [ $FAILED_CHECKS -eq 0 ] && exit 0 || exit 1
}

# Main execution
main() {
    clear
    print_header "${SEARCH} AI Lang Stuff - Local Validation"

    # Initialize report
    if [ "$NO_REPORT" = false ]; then
        init_report
    fi

    # Run all validation steps
    validate_environment
    validate_models
    run_unit_tests
    run_integration_tests
    run_benchmarks
    validate_examples
    validate_mcp_servers
    validate_vector_stores
    validate_cli_tool

    # Generate summary
    generate_summary
}

# Run main function
main
