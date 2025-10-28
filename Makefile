.PHONY: help install test lint format benchmark clean setup docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	@echo "Installing dependencies with uv..."
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"
	@echo "✓ Dependencies installed"

setup: install  ## Setup development environment
	@echo "Setting up pre-commit hooks..."
	. .venv/bin/activate && pip install pre-commit
	. .venv/bin/activate && pre-commit install
	@echo "✓ Development environment ready"

test:  ## Run test suite
	@echo "Running tests..."
	. .venv/bin/activate && pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Tests complete. Coverage report: htmlcov/index.html"

test-fast:  ## Run tests without coverage
	@echo "Running fast tests..."
	. .venv/bin/activate && pytest tests/ -v -x
	@echo "✓ Fast tests complete"

lint:  ## Run all linters
	@echo "Running Ruff linter..."
	. .venv/bin/activate && ruff check src/ examples/ tests/
	@echo "Running Black formatter check..."
	. .venv/bin/activate && black --check src/ examples/ tests/
	@echo "Running mypy type checker..."
	. .venv/bin/activate && mypy src/ || true
	@echo "✓ Linting complete"

format:  ## Format code with Ruff and Black
	@echo "Formatting code with Ruff..."
	. .venv/bin/activate && ruff format src/ examples/ tests/
	@echo "Formatting code with Black..."
	. .venv/bin/activate && black src/ examples/ tests/
	@echo "Sorting imports..."
	. .venv/bin/activate && ruff check --select I --fix src/ examples/ tests/
	@echo "✓ Code formatted"

format-check:  ## Check code formatting without modifying
	@echo "Checking Ruff formatting..."
	. .venv/bin/activate && ruff format --check src/ examples/ tests/
	@echo "Checking Black formatting..."
	. .venv/bin/activate && black --check src/ examples/ tests/
	@echo "✓ Format check complete"

benchmark:  ## Run performance benchmarks
	@echo "Running benchmarks..."
	. .venv/bin/activate && pytest tests/benchmarks/ --benchmark-only
	@echo "✓ Benchmarks complete"

benchmark-compare:  ## Compare benchmark with baseline
	@echo "Running benchmark comparison..."
	. .venv/bin/activate && pytest tests/benchmarks/ --benchmark-only --benchmark-compare
	@echo "✓ Comparison complete"

docs:  ## Build documentation
	@echo "Building documentation..."
	. .venv/bin/activate && mkdocs build
	@echo "✓ Documentation built to site/"

docs-serve:  ## Serve documentation locally
	@echo "Serving documentation at http://127.0.0.1:8000"
	. .venv/bin/activate && mkdocs serve

clean:  ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf .venv/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf site/
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"

check-ollama:  ## Check if Ollama is running
	@echo "Checking Ollama status..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is running" || echo "✗ Ollama is not running. Start it with: ollama serve"

pull-models:  ## Pull required Ollama models
	@echo "Pulling Ollama models..."
	ollama pull qwen2.5:0.5b
	ollama pull qwen3:8b
	@echo "✓ Models pulled"

ci:  ## Run CI checks locally (lint + test)
	@echo "Running CI checks..."
	@$(MAKE) lint
	@$(MAKE) test
	@echo "✓ All CI checks passed"

all: clean install lint test  ## Clean, install, lint, and test

.DEFAULT_GOAL := help
