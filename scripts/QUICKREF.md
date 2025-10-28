# Scripts Quick Reference

## One-Liners for Common Tasks

### Setup & Installation
```bash
# First time setup
./scripts/setup.sh --full

# Quick setup (no models)
./scripts/setup.sh --quick

# Just pull models
./scripts/pull_models.sh --fast
```

### Development
```bash
# Start everything
./scripts/dev.sh --all

# Just Ollama
./scripts/dev.sh --ollama

# Watch files
./scripts/dev.sh --watch
```

### Testing & Validation
```bash
# Validate everything
./scripts/validate.sh

# Quick validation
./scripts/validate.sh --quick

# Run all tests
./scripts/run_tests.sh

# Quick test
./scripts/run_tests.sh --fast

# Watch mode
./scripts/run_tests.sh --watch

# Coverage report
./scripts/run_tests.sh --html
```

### Benchmarking
```bash
# Compare models
./scripts/benchmark.sh --compare

# Full report
./scripts/benchmark.sh --report

# Single model
./scripts/benchmark.sh --model qwen3:8b
```

### Cleanup
```bash
# Interactive
./scripts/clean.sh

# Quick clean
./scripts/clean.sh --all

# Preview
./scripts/clean.sh --all --dry-run

# Deep clean
./scripts/clean.sh --deep
```

### Model Management
```bash
# List models
./scripts/pull_models.sh --list

# Check disk usage
./scripts/pull_models.sh --size

# Pull all
./scripts/pull_models.sh --all

# Remove model
./scripts/pull_models.sh --remove qwen3:8b
```

## Daily Workflow

### Morning Startup
```bash
# Start development environment
./scripts/dev.sh --all

# Check models are ready
./scripts/pull_models.sh --list
```

### During Development
```bash
# Run tests continuously
./scripts/run_tests.sh --watch

# In another terminal:
# Edit code, run examples...
```

### Before Committing
```bash
# Validate everything works
./scripts/validate.sh --quick

# Or run full validation
./scripts/validate.sh

# Clean temporary files
./scripts/clean.sh --temp
```

### End of Day
```bash
# Clean up
./scripts/clean.sh --logs

# Stop services (Ctrl+C on dev.sh terminal)
```

## Troubleshooting One-Liners

```bash
# Restart Ollama
killall ollama && ollama serve

# Clear Python cache
./scripts/clean.sh --pycache

# Fresh start
./scripts/clean.sh --deep && ./scripts/setup.sh --full

# Check service status
ps aux | grep -E "ollama|jupyter|langgraph"

# View logs
tail -f logs/*.log

# Test Ollama connection
curl http://localhost:11434/api/tags
```

## Cheat Sheet

| Task | Command |
|------|---------|
| Full setup | `./scripts/setup.sh --full` |
| Validate all | `./scripts/validate.sh` |
| Quick validate | `./scripts/validate.sh --quick` |
| Start dev | `./scripts/dev.sh --all` |
| Run tests | `./scripts/run_tests.sh` |
| Pull models | `./scripts/pull_models.sh --all` |
| Benchmark | `./scripts/benchmark.sh --compare` |
| Clean all | `./scripts/clean.sh --all` |
| Help | Add `--help` to any script |
