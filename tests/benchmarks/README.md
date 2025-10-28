# Performance Benchmark Suite

Comprehensive performance benchmarking for the ai-lang-stuff project. This suite measures and compares the performance characteristics of local LLMs, vector stores, and multi-agent workflows.

## Overview

The benchmark suite consists of four main components:

1. **Model Performance** (`model_performance.py`) - Benchmarks different local LLM models
2. **Vector Store Performance** (`vector_store_performance.py`) - Compares Chroma vs FAISS
3. **Agent Workflow Performance** (`agent_workflow_performance.py`) - Measures multi-agent system performance
4. **Benchmark Runner** (`benchmark_runner.py`) - Unified interface and reporting

## Quick Start

### Run All Benchmarks

```bash
# Full benchmark suite (takes 15-30 minutes)
python tests/benchmarks/benchmark_runner.py --all

# Quick benchmark suite (5-10 minutes, fewer tests)
python tests/benchmarks/benchmark_runner.py --all --quick
```

### Run Individual Benchmarks

```bash
# Model performance only
python tests/benchmarks/benchmark_runner.py --models

# Vector stores only
python tests/benchmarks/benchmark_runner.py --vector-stores

# Agent workflows only
python tests/benchmarks/benchmark_runner.py --workflows
```

### Run Standalone

Each benchmark can also be run independently:

```bash
python tests/benchmarks/model_performance.py
python tests/benchmarks/vector_store_performance.py
python tests/benchmarks/agent_workflow_performance.py
```

## Benchmark Details

### 1. Model Performance Benchmarks

**File**: `model_performance.py`

**Models Tested**:
- `qwen3:8b` - Baseline dense model
- `qwen3:30b-a3b` - MoE model optimized for speed
- `gemma3:4b` - Edge/mobile optimized
- `gemma3:12b` - Multilingual model
- `deepseek-r1:8b` - Reasoning-focused model

**Metrics**:
- **Latency**: Time to complete generation (ms)
- **Throughput**: Tokens generated per second
- **Memory Usage**: RAM consumption during inference (MB)
- **Quality Score**: Optional subjective quality rating

**Test Prompts**:
- Simple: Basic arithmetic
- Short: One-sentence explanations
- Medium: Function generation with docstrings
- Long: Multi-part API design
- Reasoning: Step-by-step problem solving
- Code Generation: Complex algorithm implementation

**Example Output**:
```json
{
  "by_model": {
    "qwen3:8b": {
      "avg_latency_ms": 2500,
      "avg_tokens_per_second": 45.2,
      "avg_memory_mb": 8192
    }
  }
}
```

### 2. Vector Store Performance Benchmarks

**File**: `vector_store_performance.py`

**Stores Tested**:
- **Chroma**: Persistent, production-ready vector database
- **FAISS**: In-memory, high-performance similarity search

**Metrics**:
- **Indexing Speed**: Documents indexed per second
- **Query Latency**: Time to retrieve similar documents (ms)
- **Memory Usage**: RAM consumption for index
- **Scaling**: Performance across 100, 500, 1K documents

**Operations Tested**:
- Document indexing
- Similarity search (k=5)
- Batch operations

**Example Output**:
```json
{
  "by_store": {
    "chroma": {
      "avg_latency_ms": 150,
      "avg_throughput": 85.3,
      "avg_memory_mb": 256
    },
    "faiss": {
      "avg_latency_ms": 45,
      "avg_throughput": 320.1,
      "avg_memory_mb": 128
    }
  }
}
```

### 3. Agent Workflow Performance Benchmarks

**File**: `agent_workflow_performance.py`

**Workflow Types**:
- **Sequential**: Linear agent pipeline
- **Parallel**: Concurrent agent execution
- **Tool Calling**: Agents with tool integration
- **Supervisor**: Supervisor-worker pattern
- **Conditional**: Dynamic routing based on conditions

**Metrics**:
- **Total Latency**: End-to-end workflow time (ms)
- **Step Latency**: Average time per agent step (ms)
- **State Overhead**: State management overhead (ms)
- **Tool Call Latency**: Time for tool invocations (ms)

**Example Output**:
```json
{
  "by_workflow": {
    "sequential": {
      "avg_total_latency_ms": 5000,
      "avg_step_latency_ms": 1250
    },
    "tool_calling": {
      "avg_total_latency_ms": 800,
      "avg_step_latency_ms": 160
    }
  }
}
```

## Benchmark Runner

**File**: `benchmark_runner.py`

The unified runner provides:
- Single command to run all benchmarks
- Consolidated reporting (JSON, CSV)
- Visualization generation (charts and graphs)
- Result comparison between runs
- CI/CD integration support

### Command-Line Options

```bash
python tests/benchmarks/benchmark_runner.py [OPTIONS]

Options:
  --all              Run all benchmarks
  --models           Run model performance benchmarks only
  --vector-stores    Run vector store benchmarks only
  --workflows        Run agent workflow benchmarks only
  --quick            Run quick benchmarks (fewer tests)
  --output-dir DIR   Output directory for results (default: benchmark_results)
  --compare F1 F2    Compare two result files
  --help             Show help message
```

### Output Files

All results are saved to `benchmark_results/` directory:

```
benchmark_results/
├── model_benchmark_YYYYMMDD_HHMMSS.json
├── model_benchmark_YYYYMMDD_HHMMSS.csv
├── vector_store_benchmark_YYYYMMDD_HHMMSS.json
├── vector_store_benchmark_YYYYMMDD_HHMMSS.csv
├── agent_workflow_benchmark_YYYYMMDD_HHMMSS.json
├── agent_workflow_benchmark_YYYYMMDD_HHMMSS.csv
├── benchmark_summary_YYYYMMDD_HHMMSS.json
├── model_performance_YYYYMMDD_HHMMSS.png
├── vector_store_performance_YYYYMMDD_HHMMSS.png
└── workflow_performance_YYYYMMDD_HHMMSS.png
```

## Comparison Mode

Compare results from two different runs:

```bash
python tests/benchmarks/benchmark_runner.py --compare \
  benchmark_results/benchmark_summary_20250101_120000.json \
  benchmark_results/benchmark_summary_20250102_120000.json
```

Output shows percentage changes:
```
Model Performance Changes:
  qwen3:8b           :   12.5% faster
  gemma3:4b          :    3.2% slower

Vector Store Performance Changes:
  chroma             :   18.0% faster
  faiss              :    2.1% faster
```

## Pytest Integration

All benchmarks include pytest integration for automated testing:

```bash
# Run all benchmark tests
pytest tests/benchmarks/ -v --slow

# Run specific benchmark tests
pytest tests/benchmarks/test_model_performance.py -v
pytest tests/benchmarks/test_vector_store_performance.py -v
pytest tests/benchmarks/test_agent_workflow_performance.py -v

# Skip slow tests
pytest tests/benchmarks/ -m "not slow"
```

**Test Markers**:
- `@pytest.mark.integration` - Requires Ollama server running
- `@pytest.mark.slow` - Long-running tests (> 30 seconds)

## Methodology

### Measurement Approach

1. **Warm-up**: Each benchmark runs a warm-up iteration (not counted)
2. **Multiple Runs**: Each test runs 3 times by default, results are averaged
3. **Memory Measurement**: Process RSS memory before and after each test
4. **Statistical Analysis**: Mean, median, and standard deviation calculated
5. **Error Handling**: Failures are logged but don't halt benchmark suite

### Best Practices

- **Close Other Applications**: Minimize background processes during benchmarking
- **Consistent Hardware**: Run benchmarks on the same hardware for comparisons
- **Ollama Status**: Ensure Ollama is running and models are pulled
- **Disk Space**: Ensure adequate disk space for vector store persistence
- **CPU Temperature**: Monitor CPU temperature to avoid thermal throttling

### Reproducibility

To ensure reproducible results:

```bash
# Pull all required models first
ollama pull qwen3:8b
ollama pull qwen3:30b-a3b
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text

# Restart Ollama to clear cache
pkill ollama
ollama serve &

# Run benchmarks
python tests/benchmarks/benchmark_runner.py --all
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Install Ollama
        run: |
          brew install ollama
          ollama serve &
          sleep 5
          ollama pull qwen3:8b

      - name: Run quick benchmarks
        run: |
          python tests/benchmarks/benchmark_runner.py --all --quick

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

## Performance Targets

### Expected Performance (MacBook Pro M3, 32GB RAM)

**Model Performance**:
- qwen3:8b: ~40-50 tokens/sec
- qwen3:30b-a3b: ~30-40 tokens/sec (MoE advantage)
- gemma3:4b: ~60-80 tokens/sec
- gemma3:12b: ~35-45 tokens/sec

**Vector Store Performance**:
- Chroma indexing: 50-100 docs/sec
- Chroma query: 100-200 queries/sec
- FAISS indexing: 200-500 docs/sec
- FAISS query: 500-1000 queries/sec

**Workflow Performance**:
- Sequential (3 agents): < 10 seconds
- Tool calling (5 calls): < 2 seconds
- State update: < 1ms per update

## Visualization

Charts generated (requires matplotlib):

1. **Model Latency Comparison** - Bar chart of average latency per model
2. **Model Throughput Comparison** - Bar chart of tokens/second per model
3. **Vector Store Throughput** - Comparison of Chroma vs FAISS
4. **Workflow Performance** - Latency across different workflow types

Example visualization command:
```bash
python tests/benchmarks/benchmark_runner.py --all
# Charts saved to benchmark_results/*.png
```

## Troubleshooting

### Common Issues

**"Connection refused to localhost:11434"**
```bash
# Start Ollama
ollama serve
```

**"Model not found"**
```bash
# Pull missing model
ollama pull qwen3:8b
```

**"Out of memory"**
- Reduce document counts in vector store benchmarks
- Use smaller models (gemma3:4b instead of qwen3:30b-a3b)
- Close other applications

**"Matplotlib import error"**
```bash
# Install matplotlib for visualizations
pip install matplotlib
```

## Contributing

To add new benchmarks:

1. Create new benchmark file following existing patterns
2. Implement benchmark class with proper error handling
3. Add pytest integration tests
4. Update `benchmark_runner.py` to include new benchmark
5. Update this README with new benchmark details

## References

- [LangChain Benchmarking](https://python.langchain.com/docs/guides/benchmarking)
- [Ollama Performance](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-fast-is-ollama)
- [FAISS Performance](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)
- [Chroma Performance](https://docs.trychroma.com/guides/performance)

## License

Same as parent project (see root LICENSE file).
