.PHONY: install dev test lint format clean help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the CLI tool
	pip install .

dev:  ## Install in development mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests with coverage
	pytest tests/ --cov=ailang --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting checks
	ruff check ailang/
	mypy ailang/

format:  ## Format code with ruff
	ruff format ailang/
	ruff check --fix ailang/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

release: clean build  ## Build and upload to PyPI
	python -m twine upload dist/*

check:  ## Run all checks (lint + test)
	$(MAKE) lint
	$(MAKE) test

demo:  ## Run a demo of all CLI features
	@echo "=== ailang CLI Demo ==="
	@echo ""
	@echo "1. Checking version..."
	ailang --version
	@echo ""
	@echo "2. Listing model recommendations..."
	ailang models recommend coding
	@echo ""
	@echo "3. Listing examples..."
	ailang examples list | head -n 20
	@echo ""
	@echo "4. Listing MCP servers..."
	ailang mcp list
	@echo ""
	@echo "5. Listing RAG collections..."
	ailang rag list
	@echo ""
	@echo "Demo complete!"

.DEFAULT_GOAL := help
