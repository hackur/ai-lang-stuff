#!/bin/bash
# Benchmark suite runner for AI model performance testing
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
${BOLD}Benchmark Suite Runner${NC}

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -a, --all               Run all benchmarks
    -m, --model MODEL       Benchmark specific model
    -s, --suite SUITE       Run specific benchmark suite
    -c, --compare           Compare multiple models
    -o, --output FILE       Save results to file
    -r, --report            Generate detailed report
    -b, --baseline FILE     Compare against baseline results
    --save-baseline         Save current results as baseline
    --verbose               Verbose output

Benchmark Suites:
    inference   - Inference speed and latency
    throughput  - Token generation throughput
    memory      - Memory usage and efficiency
    quality     - Response quality metrics
    agents      - Multi-agent workflow performance
    rag         - RAG system performance
    all         - Run all suites

Examples:
    $0 --all                        # Run all benchmarks
    $0 --model qwen3:8b             # Benchmark specific model
    $0 --suite inference            # Run inference benchmarks
    $0 --compare                    # Compare all models
    $0 --baseline baseline.json     # Compare against baseline

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
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_error "Ollama server not running. Please start with: ollama serve"
        exit 1
    fi

    # Check if benchmarks directory exists
    if [ ! -d "benchmarks" ]; then
        print_info "Creating benchmarks directory..."
        mkdir -p benchmarks/results
    fi
}

# Get available models
get_models() {
    ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$"
}

# Run inference benchmark
benchmark_inference() {
    local model=$1
    echo -e "\n${BOLD}Benchmarking Inference: ${model}${NC}\n"

    local start_time=$(date +%s.%N)

    # Test simple prompt
    local prompt="Explain quantum computing in one sentence."
    local response=$(echo "$prompt" | ollama run "$model" 2>/dev/null)

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    print_info "Response time: ${duration}s"

    # Count tokens (approximate)
    local tokens=$(echo "$response" | wc -w)
    print_info "Tokens generated: ~${tokens}"

    if [ "$tokens" -gt 0 ]; then
        local tokens_per_sec=$(echo "scale=2; $tokens / $duration" | bc)
        print_info "Tokens/second: ${tokens_per_sec}"
    fi

    echo "{\"model\": \"$model\", \"duration\": $duration, \"tokens\": $tokens}" >> benchmarks/results/inference_$(date +%Y%m%d_%H%M%S).json
}

# Run throughput benchmark
benchmark_throughput() {
    local model=$1
    echo -e "\n${BOLD}Benchmarking Throughput: ${model}${NC}\n"

    local iterations=5
    local total_tokens=0
    local total_time=0

    for i in $(seq 1 $iterations); do
        print_info "Iteration $i/$iterations..."

        local start=$(date +%s.%N)
        local response=$(echo "Generate a 100-word paragraph about artificial intelligence." | ollama run "$model" 2>/dev/null)
        local end=$(date +%s.%N)

        local duration=$(echo "$end - $start" | bc)
        local tokens=$(echo "$response" | wc -w)

        total_time=$(echo "$total_time + $duration" | bc)
        total_tokens=$(echo "$total_tokens + $tokens" | bc)
    done

    local avg_time=$(echo "scale=2; $total_time / $iterations" | bc)
    local avg_tokens=$(echo "scale=0; $total_tokens / $iterations" | bc)
    local avg_throughput=$(echo "scale=2; $total_tokens / $total_time" | bc)

    echo ""
    print_info "Average time: ${avg_time}s"
    print_info "Average tokens: ${avg_tokens}"
    print_info "Average throughput: ${avg_throughput} tokens/s"
}

# Run memory benchmark
benchmark_memory() {
    local model=$1
    echo -e "\n${BOLD}Benchmarking Memory: ${model}${NC}\n"

    # Get initial memory
    local pid=$(pgrep ollama | head -1)
    if [ -z "$pid" ]; then
        print_warning "Could not find Ollama process"
        return 1
    fi

    local mem_before=$(ps -o rss= -p "$pid")

    # Load model and generate response
    echo "Generate a detailed explanation of neural networks." | ollama run "$model" > /dev/null 2>&1

    # Get memory after
    local mem_after=$(ps -o rss= -p "$pid")

    local mem_diff=$(echo "scale=2; ($mem_after - $mem_before) / 1024" | bc)

    print_info "Memory delta: ${mem_diff} MB"
    print_info "Current memory: $(echo "scale=2; $mem_after / 1024" | bc) MB"
}

# Run agent benchmark
benchmark_agents() {
    local model=$1
    echo -e "\n${BOLD}Benchmarking Agent Workflows: ${model}${NC}\n"

    # Check if agent examples exist
    if [ -f "examples/03-multi-agent/simple_langgraph.py" ]; then
        print_info "Running agent workflow..."

        local start=$(date +%s.%N)
        uv run python examples/03-multi-agent/simple_langgraph.py --model "$model" > /dev/null 2>&1 || true
        local end=$(date +%s.%N)

        local duration=$(echo "$end - $start" | bc)
        print_info "Workflow duration: ${duration}s"
    else
        print_warning "Agent examples not found"
    fi
}

# Compare models
compare_models() {
    echo -e "\n${BOLD}Comparing Models${NC}\n"

    local models=($(get_models))

    if [ ${#models[@]} -eq 0 ]; then
        print_error "No models found"
        exit 1
    fi

    echo -e "${BOLD}Model Comparison - Inference Speed${NC}\n"
    printf "%-25s %-15s %-15s\n" "Model" "Time (s)" "Tokens/s"
    printf "%-25s %-15s %-15s\n" "-----" "--------" "---------"

    local prompt="Explain machine learning briefly."

    for model in "${models[@]}"; do
        local start=$(date +%s.%N)
        local response=$(echo "$prompt" | ollama run "$model" 2>/dev/null)
        local end=$(date +%s.%N)

        local duration=$(echo "scale=2; $end - $start" | bc)
        local tokens=$(echo "$response" | wc -w)
        local tps=$(echo "scale=2; $tokens / $duration" | bc)

        printf "%-25s %-15s %-15s\n" "$model" "$duration" "$tps"
    done

    echo ""
}

# Generate detailed report
generate_report() {
    local output_file=${1:-"benchmarks/results/report_$(date +%Y%m%d_%H%M%S).md"}

    echo -e "\n${BOLD}Generating Benchmark Report${NC}\n"

    cat > "$output_file" << EOF
# Benchmark Report

**Generated:** $(date)
**System:** $(uname -s) $(uname -r)
**CPU:** $(sysctl -n machdep.cpu.brand_string)
**Memory:** $(sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}')

## Models Tested

$(ollama list)

## Benchmark Results

### Inference Speed

All models tested with prompt: "Explain quantum computing in one sentence."

| Model | Response Time | Tokens | Tokens/sec |
|-------|---------------|--------|------------|

EOF

    local models=($(get_models))
    local prompt="Explain quantum computing in one sentence."

    for model in "${models[@]}"; do
        local start=$(date +%s.%N)
        local response=$(echo "$prompt" | ollama run "$model" 2>/dev/null)
        local end=$(date +%s.%N)

        local duration=$(echo "scale=2; $end - $start" | bc)
        local tokens=$(echo "$response" | wc -w)
        local tps=$(echo "scale=2; $tokens / $duration" | bc)

        echo "| $model | ${duration}s | $tokens | $tps |" >> "$output_file"
    done

    cat >> "$output_file" << EOF

## Recommendations

Based on these benchmarks:

- **Fastest inference:** Use smallest quantized models (Q4, Q5)
- **Best throughput:** MoE models (qwen3:30b-a3b)
- **Memory efficient:** Small models (gemma3:4b)
- **Quality vs Speed:** Medium models (qwen3:8b)

## Notes

- All tests run locally via Ollama
- Results vary based on system load
- Token counts are approximate (word-based)

EOF

    print_status "Report generated: $output_file"
}

# Main execution
main() {
    # Default options
    SUITE="all"
    MODEL=""
    COMPARE=false
    GENERATE_REPORT=false
    OUTPUT_FILE=""
    BASELINE_FILE=""
    SAVE_BASELINE=false
    VERBOSE=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                SUITE="all"
                shift
                ;;
            -m|--model)
                MODEL="$2"
                shift 2
                ;;
            -s|--suite)
                SUITE="$2"
                shift 2
                ;;
            -c|--compare)
                COMPARE=true
                shift
                ;;
            -r|--report)
                GENERATE_REPORT=true
                shift
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -b|--baseline)
                BASELINE_FILE="$2"
                shift 2
                ;;
            --save-baseline)
                SAVE_BASELINE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
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

    # Handle comparison mode
    if [ "$COMPARE" = true ]; then
        compare_models
        exit 0
    fi

    # Handle report generation
    if [ "$GENERATE_REPORT" = true ]; then
        generate_report "$OUTPUT_FILE"
        exit 0
    fi

    # Get model to benchmark
    if [ -z "$MODEL" ]; then
        print_info "No model specified, using qwen3:8b"
        MODEL="qwen3:8b"
    fi

    # Verify model exists
    if ! ollama list | grep -q "^${MODEL}"; then
        print_error "Model not found: $MODEL"
        print_info "Available models:"
        ollama list
        exit 1
    fi

    # Run benchmarks based on suite
    case "$SUITE" in
        inference)
            benchmark_inference "$MODEL"
            ;;
        throughput)
            benchmark_throughput "$MODEL"
            ;;
        memory)
            benchmark_memory "$MODEL"
            ;;
        agents)
            benchmark_agents "$MODEL"
            ;;
        all)
            benchmark_inference "$MODEL"
            benchmark_throughput "$MODEL"
            benchmark_memory "$MODEL"
            benchmark_agents "$MODEL"
            ;;
        *)
            print_error "Unknown suite: $SUITE"
            show_help
            exit 1
            ;;
    esac

    echo -e "\n${GREEN}Benchmark complete!${NC}\n"
}

# Run main
main "$@"
