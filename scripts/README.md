# Deployment Automation Scripts

Comprehensive automation scripts for managing the AI Lang Stuff local development environment.

## Available Scripts

### 1. setup.sh - Complete Environment Setup

**Purpose**: Initialize the entire development environment from scratch.

**Usage**:
```bash
./scripts/setup.sh                 # Interactive setup
./scripts/setup.sh --full          # Complete setup with models and tests
./scripts/setup.sh --quick         # Dependencies only (no models)
./scripts/setup.sh --models        # Pull models only
./scripts/setup.sh --test          # Run verification tests
./scripts/setup.sh --skip-models   # Skip model downloads
./scripts/setup.sh --ci            # CI/CD mode (non-interactive)
```

**Features**:
- Installs Homebrew dependencies (uv, Node.js, Ollama, Python)
- Creates required directory structure
- Sets up Python environment with uv
- Installs Node.js dependencies
- Creates .env from template
- Starts Ollama server
- Optionally pulls recommended models
- Runs verification tests
- Idempotent (safe to run multiple times)

**What it installs**:
- uv (Python package manager)
- Node.js and npm
- Ollama (local LLM server)
- Python 3.13+
- All Python dependencies from pyproject.toml
- All Node.js dependencies from package.json

---

### 2. pull_models.sh - Model Management

**Purpose**: Download, verify, and manage Ollama models.

**Usage**:
```bash
./scripts/pull_models.sh --help           # Show help
./scripts/pull_models.sh --all            # Pull all recommended models
./scripts/pull_models.sh --fast           # Pull fast models only
./scripts/pull_models.sh --vision         # Pull vision models
./scripts/pull_models.sh --embedding      # Pull embedding models
./scripts/pull_models.sh --list           # List installed models
./scripts/pull_models.sh --size           # Show disk usage
./scripts/pull_models.sh --remove qwen3:8b  # Remove specific model
```

**Model Categories**:
- **Fast**: qwen3:30b-a3b (MoE), gemma3:4b (small)
- **Standard**: qwen3:8b, gemma3:12b
- **Vision**: qwen3-vl:8b
- **Embedding**: qwen3-embedding, nomic-embed-text
- **Large**: deepseek-coder:33b, qwen3:72b

**Features**:
- Progress display during downloads
- Automatic verification after installation
- Disk usage reporting
- Resume interrupted downloads
- Safe model removal
- Checks Ollama server availability

**Disk Space**:
- Fast models: ~15GB
- Standard models: ~25GB
- All models: ~50GB

---

### 3. run_tests.sh - Test Suite Runner

**Purpose**: Execute test suites with coverage reporting.

**Usage**:
```bash
./scripts/run_tests.sh                    # Run all tests
./scripts/run_tests.sh --unit             # Unit tests only
./scripts/run_tests.sh --integration      # Integration tests only
./scripts/run_tests.sh --coverage         # Generate coverage report
./scripts/run_tests.sh --fast             # Run without coverage
./scripts/run_tests.sh --verbose          # Verbose output
./scripts/run_tests.sh -m slow            # Run tests marked 'slow'
./scripts/run_tests.sh -k "test_agent"    # Run tests matching keyword
./scripts/run_tests.sh --ci               # CI/CD mode (strict)
./scripts/run_tests.sh --html             # Generate HTML coverage report
./scripts/run_tests.sh --watch            # Watch mode (auto-rerun)
```

**Test Markers**:
- `unit` - Fast, isolated unit tests
- `integration` - Tests requiring external services
- `slow` - Long-running tests
- `requires_ollama` - Tests needing Ollama server
- `requires_mcp` - Tests needing MCP servers

**Features**:
- Pytest-based test runner
- Coverage reporting (term + HTML)
- CI/CD friendly output
- Watch mode for TDD workflow
- Automatic test environment setup
- Parallel test execution
- Coverage threshold checking (80%)

**Outputs**:
- Terminal coverage report
- HTML report: `htmlcov/index.html`
- JUnit XML (CI mode)

---

### 4. benchmark.sh - Performance Benchmarking

**Purpose**: Run comprehensive performance benchmarks on AI models.

**Usage**:
```bash
./scripts/benchmark.sh --help                      # Show help
./scripts/benchmark.sh --all                       # All benchmarks
./scripts/benchmark.sh --model qwen3:8b            # Specific model
./scripts/benchmark.sh --suite inference           # Specific suite
./scripts/benchmark.sh --compare                   # Compare models
./scripts/benchmark.sh --report                    # Generate report
./scripts/benchmark.sh --baseline baseline.json    # Compare vs baseline
./scripts/benchmark.sh --save-baseline             # Save as baseline
```

**Benchmark Suites**:
- **inference** - Response time and latency
- **throughput** - Token generation speed
- **memory** - Memory usage and efficiency
- **quality** - Response quality metrics
- **agents** - Multi-agent workflow performance
- **rag** - RAG system performance

**Metrics Tracked**:
- Response time (seconds)
- Tokens per second
- Memory usage (MB)
- First token latency
- Total tokens generated

**Report Outputs**:
- JSON results: `benchmarks/results/`
- Markdown reports with charts
- Comparison tables
- Performance recommendations

**Example Report**:
```
Model Comparison - Inference Speed
Model                     Time (s)    Tokens/s
qwen3:8b                 2.34        45.2
qwen3:30b-a3b           1.87        58.3
gemma3:4b               0.98        72.1
```

---

### 5. clean.sh - Cleanup Utility

**Purpose**: Remove temporary files and reset to clean state.

**Usage**:
```bash
./scripts/clean.sh                    # Interactive mode
./scripts/clean.sh --all              # Clean everything (excludes models)
./scripts/clean.sh --vectors          # Remove vector stores
./scripts/clean.sh --checkpoints      # Remove LangGraph checkpoints
./scripts/clean.sh --pycache          # Clean Python cache
./scripts/clean.sh --logs             # Clear log files
./scripts/clean.sh --temp             # Remove temp files
./scripts/clean.sh --data             # Remove data directory
./scripts/clean.sh --models           # Remove models (CAUTION!)
./scripts/clean.sh --deep             # Deep clean (all + deps)
./scripts/clean.sh --dry-run          # Preview without deleting
```

**What Gets Cleaned**:
- **Vector stores**: ChromaDB, FAISS indexes
- **Checkpoints**: LangGraph state snapshots
- **Python cache**: `__pycache__`, `.pyc`, `.pytest_cache`
- **Logs**: All `.log` files
- **Temp files**: `.tmp`, `.DS_Store`, etc.
- **Data**: Test data, temporary databases
- **Deep clean**: node_modules, .venv, lock files

**Safety Features**:
- Dry-run mode to preview changes
- Confirmation prompts for destructive operations
- Preserves configuration files
- Shows disk space recovered
- Idempotent (safe to run multiple times)

**Example Output**:
```
Disk Space Summary
  data:                 1.2 GB
  logs:                 45 MB
  checkpoints:          320 MB
  __pycache__:         12 MB
  Ollama models:        48 GB
```

---

### 6. dev.sh - Development Mode

**Purpose**: Start all development services with auto-reload.

**Usage**:
```bash
./scripts/dev.sh --help               # Show help
./scripts/dev.sh --all                # Start all services
./scripts/dev.sh --watch              # Watch mode only
./scripts/dev.sh --langgraph          # Start LangGraph Studio
./scripts/dev.sh --jupyter            # Start Jupyter Lab
./scripts/dev.sh --ollama             # Start Ollama server
./scripts/dev.sh -l -j                # LangGraph + Jupyter
./scripts/dev.sh --port 9999          # Custom Jupyter port
./scripts/dev.sh --no-browser         # Don't open browser
```

**Services Managed**:
1. **Ollama** - Local LLM server (port 11434)
2. **LangGraph Studio** - Agent workflow UI (port 8123)
3. **Jupyter Lab** - Interactive notebooks (port 8888)
4. **File Watcher** - Auto-reload on changes

**Features**:
- Automatic service health checks
- Auto-restart on crash
- Colored log output
- Graceful shutdown (Ctrl+C)
- Browser auto-open
- Log aggregation in `logs/`

**Service URLs**:
- Ollama: http://localhost:11434
- LangGraph Studio: http://localhost:8123
- Jupyter Lab: http://localhost:8888

**Logs**:
```bash
tail -f logs/ollama.log        # View Ollama logs
tail -f logs/langgraph.log     # View LangGraph logs
tail -f logs/jupyter.log       # View Jupyter logs
```

---

### 7. validate.sh - Comprehensive Validation

**Purpose**: Single command to verify all components and tests are working.

**Usage**:
```bash
./scripts/validate.sh                 # Full validation
./scripts/validate.sh --quick         # Quick validation (skip slow tests)
./scripts/validate.sh --verbose       # Verbose output with details
./scripts/validate.sh --skip-slow     # Skip slow-running tests
./scripts/validate.sh --fix           # Auto-fix common issues
./scripts/validate.sh --no-report     # Don't generate report
```

**Validation Steps**:
1. **Environment Checks** - Python, Node, Ollama, uv
2. **Model Availability** - Required models installed
3. **Unit Tests** - All utility tests
4. **Integration Tests** - Examples and workflows
5. **Benchmarks** - Performance validation
6. **Example Validation** - Test example scripts
7. **MCP Servers** - Health checks
8. **Vector Stores** - Operational status
9. **CLI Tool** - Command validation
10. **Report Generation** - Validation summary

**Features**:
- Progress indicators with emojis
- Time tracking for each step
- Colored output (pass/fail/skip)
- Auto-fix common issues (with `--fix`)
- Summary report at end
- Saves results to `reports/validation-YYYY-MM-DD.md`
- Quick mode for fast feedback
- Full mode for comprehensive testing

**Example Output**:
```
🔍 AI Lang Stuff - Local Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Environment Check (2.3s)
✅ Model Availability (1.1s)
✅ Unit Tests (12.4s) - 142/142 passed
✅ Integration Tests (45.2s) - 74/74 passed
⏱️  Benchmarks (120.5s) - 8 benchmarks complete
✅ Example Validation (23.1s) - 13/13 working
✅ MCP Servers (3.4s) - 2/2 healthy
✅ Vector Stores (5.2s) - operational
✅ CLI Tool (1.8s) - all commands work

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ALL CHECKS PASSED
Total time: 215.0s
Report: reports/validation-2025-10-26.md
```

**Report Contents**:
- Summary of all checks
- Pass/fail/skip counts
- Time for each validation step
- Detailed findings
- Recommendations for failures

**Use Cases**:
- Before committing code
- After pulling updates
- CI/CD integration
- Pre-deployment checks
- Troubleshooting issues

---

## Quick Start Workflows

### New Developer Setup
```bash
# 1. Clone and setup
git clone <repo>
cd ai-lang-stuff

# 2. Run full setup
./scripts/setup.sh --full

# 3. Validate everything works
./scripts/validate.sh

# 4. Start development environment
./scripts/dev.sh --all
```

### Daily Development
```bash
# Start development services
./scripts/dev.sh --all

# In another terminal:
# - Edit code
# - Run examples: uv run python examples/...
# - Run tests: ./scripts/run_tests.sh --watch

# Before committing
./scripts/validate.sh --quick

# Clean up when done
./scripts/clean.sh --temp
```

### Pull Additional Models
```bash
# Pull all recommended models
./scripts/pull_models.sh --all

# Or pull specific categories
./scripts/pull_models.sh --vision
./scripts/pull_models.sh --embedding

# Check disk usage
./scripts/pull_models.sh --size
```

### Run Benchmarks
```bash
# Compare all installed models
./scripts/benchmark.sh --compare

# Benchmark specific model
./scripts/benchmark.sh --model qwen3:8b --suite all

# Generate detailed report
./scripts/benchmark.sh --report
```

### Clean Up
```bash
# Interactive cleanup (safest)
./scripts/clean.sh

# Clean everything except models
./scripts/clean.sh --all

# Preview without deleting
./scripts/clean.sh --deep --dry-run

# Deep clean (removes dependencies)
./scripts/clean.sh --deep
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup environment
        run: ./scripts/setup.sh --ci --skip-models
      - name: Run tests
        run: ./scripts/run_tests.sh --ci
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Script Design Principles

All scripts follow these principles:

1. **Idempotent**: Safe to run multiple times
2. **Error Handling**: Proper error checking and reporting
3. **Help Text**: Clear usage documentation
4. **Color Output**: Visual feedback with color-coded messages
5. **macOS Native**: Optimized for macOS tools and paths
6. **Logging**: Detailed logs for debugging
7. **Cleanup**: Proper resource cleanup on exit
8. **Non-Interactive**: Support for CI/CD environments
9. **Dry Run**: Preview mode for destructive operations
10. **Progress**: Clear progress indication for long operations

---

## Troubleshooting

### Script Won't Execute
```bash
# Make executable
chmod +x scripts/*.sh

# Check shell
echo $SHELL  # Should be /bin/bash or /bin/zsh
```

### Ollama Issues
```bash
# Check if running
pgrep ollama

# Restart
killall ollama
ollama serve
```

### Permission Errors
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix directory permissions
chmod -R u+w data/ logs/
```

### Disk Space
```bash
# Check available space
df -h

# See model sizes
./scripts/pull_models.sh --size

# Clean up
./scripts/clean.sh --all
```

---

## Environment Variables

Scripts respect these environment variables:

- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `TESTING` - Enable test mode
- `LOG_LEVEL` - Logging verbosity
- `CI` - CI/CD mode flag

---

## Dependencies

### Required
- macOS (Darwin)
- Homebrew
- bash 4.0+

### Installed by Scripts
- uv (Python package manager)
- Node.js and npm
- Ollama
- Python 3.13+

### Optional
- fswatch (for watch mode) - `brew install fswatch`
- bc (for benchmarks) - Usually pre-installed

---

## Contributing

When adding new scripts:

1. Follow existing naming convention
2. Include comprehensive help text
3. Add error handling
4. Make idempotent
5. Support dry-run mode
6. Add to this README
7. Test on clean macOS install

---

## License

Part of the ai-lang-stuff project. See main LICENSE file.
