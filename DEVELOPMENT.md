# Development Guide - Local-First AI Toolkit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000)
![Type Checker](https://img.shields.io/badge/type%20checker-mypy-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)

**Status**: 🚀 Active Development | **Version**: 0.1.0-alpha | **Last Updated**: 2025-10-28

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- **macOS** (primary), Linux, or Windows (WSL2)
- **Python 3.10+** (3.14 recommended)
- **uv** package manager
- **Ollama** (for local LLMs)
- **Git** with LFS support

### One-Command Setup

```bash
# Clone and setup
git clone https://github.com/hackur/ai-lang-stuff.git
cd ai-lang-stuff
./scripts/setup.sh
```

This installs all dependencies, downloads models, and validates the environment.

### Manual Setup

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# 3. Install Ollama (macOS)
brew install ollama
ollama serve  # Start Ollama server

# 4. Pull recommended models
ollama pull qwen3:8b
ollama pull gemma3:4b

# 5. Install pre-commit hooks
pre-commit install

# 6. Verify installation
make test
```

---

## 🛠️ Development Environment

### Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **uv** | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Ollama** | Local LLM runtime | `brew install ollama` |
| **pre-commit** | Git hooks for quality | Included in dev dependencies |
| **ruff** | Fast Python linter/formatter | Included in dev dependencies |
| **mypy** | Type checker | Included in dev dependencies |
| **pytest** | Testing framework | Included in dev dependencies |

### Optional Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **LangGraph Studio** | Visual workflow debugging | See [LANGGRAPH-STUDIO-QUICKSTART.md](LANGGRAPH-STUDIO-QUICKSTART.md) |
| **LangSmith** | Tracing and observability | Create account at smith.langchain.com |
| **GitHub CLI** | Git operations | `brew install gh` |
| **Docker** | Containerized deployment | `brew install docker` |

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
OLLAMA_HOST=http://localhost:11434

# Optional (for advanced features)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=ai-lang-stuff
LANGCHAIN_API_KEY=your_key_here

# Model selection
DEFAULT_MODEL=qwen3:8b
EMBEDDING_MODEL=qwen3-embedding
VISION_MODEL=qwen3-vl:8b
```

---

## 📁 Project Structure

```
ai-lang-stuff/
├── .claude/                    # Claude Code configuration
│   ├── agents/                # Agent definitions
│   ├── commands/              # Custom slash commands
│   └── skills/                # Reusable skills
├── .github/                   # CI/CD workflows
│   ├── workflows/            # GitHub Actions
│   └── ISSUE_TEMPLATE/       # Issue templates
├── cli/                       # Command-line interface
│   ├── ailang/               # CLI implementation
│   └── tests/                # CLI tests
├── config/                    # Configuration files
│   └── models.yaml           # Model configurations
├── docs/                      # Documentation
│   ├── adr/                  # Architecture Decision Records
│   ├── api-reference/        # API documentation
│   └── *.md                  # Guides and tutorials
├── examples/                  # Working examples
│   ├── 01-foundation/        # Basic examples
│   ├── 02-mcp/               # MCP integration
│   ├── 03-multi-agent/       # Multi-agent workflows
│   ├── 04-rag/               # RAG systems
│   ├── 05-interpretability/  # Model analysis
│   ├── 06-production/        # Production patterns
│   └── 07-advanced/          # Advanced features
├── mcp-servers/               # MCP server implementations
│   ├── custom/               # Custom servers
│   └── template/             # Server template
├── plans/                     # Research and planning
├── scripts/                   # Automation scripts
│   ├── setup.sh              # Environment setup
│   ├── dev.sh                # Development server
│   ├── test.sh               # Test runner
│   └── benchmark.sh          # Performance tests
├── src/                       # Source code (when refactored)
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── benchmarks/           # Performance tests
├── utils/                     # Core utilities
│   ├── ollama_manager.py     # Ollama integration
│   ├── mcp_client.py         # MCP client
│   ├── vector_store.py       # Vector databases
│   ├── state_manager.py      # State persistence
│   └── tool_registry.py      # Tool management
├── workflows/                 # LangGraph workflows
├── .gitignore                 # Git exclusions
├── .pre-commit-config.yaml    # Pre-commit hooks
├── CLAUDE.md                  # Claude instructions
├── DEVELOPMENT.md             # This file
├── Makefile                   # Common tasks
├── pyproject.toml             # Python project config
└── README.md                  # User documentation
```

---

## 🔄 Development Workflow

### Daily Development

```bash
# 1. Start development environment
./scripts/dev.sh start

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes
# ... edit files ...

# 4. Run quality checks (automatic via pre-commit)
pre-commit run --all-files

# 5. Run tests
make test

# 6. Commit changes
git add .
git commit -m "feat: add your feature"

# 7. Push and create PR
git push origin feature/your-feature-name
gh pr create
```

### Common Tasks (via Makefile)

```bash
# Install dependencies
make install

# Run tests
make test                    # All tests
make test-unit              # Unit tests only
make test-integration       # Integration tests
make test-benchmarks        # Performance benchmarks

# Code quality
make lint                    # Run ruff linter
make format                  # Format code
make type-check              # Run mypy
make pre-commit              # Run all pre-commit hooks

# Development
make dev                     # Start dev environment
make clean                   # Clean artifacts
make docs                    # Build documentation

# Benchmarks
make benchmark               # Run all benchmarks
make benchmark-models        # Benchmark models
make benchmark-vector        # Benchmark vector stores

# Examples
make examples-list           # List all examples
make examples-run NAME=mcp  # Run specific example
```

### Using Development Scripts

```bash
# Setup and validation
./scripts/setup.sh           # Complete environment setup
./scripts/validate.sh        # Validate project health
./scripts/verify-ci-setup.sh # Check CI/CD config

# Development
./scripts/dev.sh start       # Start development server
./scripts/dev.sh stop        # Stop all services
./scripts/dev.sh status      # Check service status
./scripts/dev.sh menu        # Interactive menu

# Testing
./scripts/run_tests.sh all   # Run all tests
./scripts/run_tests.sh unit  # Unit tests only
./scripts/benchmark.sh model # Benchmark models

# Cleanup
./scripts/clean.sh cache     # Clean caches
./scripts/clean.sh all       # Full cleanup

# Models
./scripts/pull_models.sh qwen3:8b gemma3:4b  # Download models
```

---

## ✅ Code Quality

### Pre-Commit Hooks

Automatically run on every commit:

```yaml
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Large file detection
- Python syntax checking
- Ruff linting and formatting
- mypy type checking
- pytest execution (on changed files)
```

### Linting with Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .

# Configuration in pyproject.toml
```

**Ruff Rules Enabled:**
- E (pycodestyle errors)
- F (pyflakes)
- I (isort)
- N (pep8-naming)
- UP (pyupgrade)
- S (flake8-bandit)
- B (flake8-bugbear)
- C4 (flake8-comprehensions)

### Type Checking with MyPy

```bash
# Check specific directory
mypy utils/ --strict

# Check entire project
mypy . --strict

# Configuration in pyproject.toml
```

**MyPy Configuration:**
- `strict = true`
- `warn_unused_ignores = true`
- `disallow_untyped_defs = true`
- `check_untyped_defs = true`

### Code Style Guide

**Python:**
- Follow PEP 8
- Use type hints for all functions
- Docstrings in Google style
- Maximum line length: 100 characters
- Use f-strings for formatting

**Example:**
```python
from typing import Dict, List, Optional

def process_data(
    input_data: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """Process input data according to configuration.

    Args:
        input_data: List of strings to process
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping categories to processed strings

    Raises:
        ValueError: If input_data is empty
    """
    if not input_data:
        raise ValueError("input_data cannot be empty")

    # Implementation
    return {"processed": input_data}
```

---

## 🧪 Testing

### Test Organization

```
tests/
├── conftest.py                # Shared fixtures
├── test_*.py                  # Unit tests
├── integration/               # Integration tests
│   ├── conftest.py
│   └── test_*.py
└── benchmarks/                # Performance tests
    ├── benchmark_runner.py
    └── test_*_performance.py
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=utils --cov=workflows --cov-report=html

# Specific test file
pytest tests/test_ollama_manager.py

# Specific test
pytest tests/test_ollama_manager.py::test_check_running

# Integration tests only
pytest tests/integration/

# Benchmarks only
pytest tests/benchmarks/

# Parallel execution
pytest -n auto

# Verbose output
pytest -v -s
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from utils.ollama_manager import OllamaManager

@pytest.fixture
def ollama_manager():
    """Provide OllamaManager instance."""
    return OllamaManager()

def test_check_ollama_running(ollama_manager):
    """Test Ollama server status check."""
    # Arrange - done by fixture

    # Act
    result = ollama_manager.check_ollama_running()

    # Assert
    assert isinstance(result, bool)
```

**Integration Test Example:**
```python
@pytest.mark.integration
@pytest.mark.requires_ollama
def test_mcp_filesystem_integration():
    """Test MCP filesystem server integration."""
    from utils.mcp_client import FilesystemMCP

    client = FilesystemMCP()
    result = client.list_files("./examples")

    assert "files" in result
    assert len(result["files"]) > 0
```

### Test Markers

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.slow           # Slow test (>5s)
@pytest.mark.requires_ollama  # Requires Ollama running
@pytest.mark.requires_model   # Requires specific model
```

### Coverage Goals

- **Overall**: 80%+
- **Core Utilities**: 90%+
- **Workflows**: 85%+
- **Examples**: Not required (documentation purpose)

---

## 📚 Documentation

### Documentation Standards

All documentation should be:
- Clear and concise
- Include examples
- Keep up-to-date with code
- Use consistent formatting

### Where to Document

| Type | Location | Format |
|------|----------|--------|
| API Reference | `docs/api-reference/` | Markdown |
| Guides | `docs/*.md` | Markdown |
| Architecture Decisions | `docs/adr/` | ADR format |
| Code Documentation | Inline docstrings | Google style |
| Examples | `examples/*/README.md` | Markdown |
| CLI Help | `cli/ailang/*.py` | Docstrings |

### Building Documentation

```bash
# Install MkDocs
uv add mkdocs mkdocs-material

# Build docs
mkdocs build

# Serve locally
mkdocs serve  # http://localhost:8000

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Writing Docstrings

```python
def example_function(param1: str, param2: int = 0) -> Dict[str, Any]:
    """One-line summary of function.

    More detailed description if needed. Explain what the function
    does, why it exists, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2, defaults to 0

    Returns:
        Dictionary containing results with keys:
        - key1: Description
        - key2: Description

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is negative

    Examples:
        >>> result = example_function("test", 5)
        >>> result["key1"]
        "value1"
    """
    pass
```

---

## 🤝 Contributing

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/ai-lang-stuff.git`
3. **Create a branch**: `git checkout -b feature/your-feature`
4. **Make your changes** following the guidelines
5. **Run tests**: `make test`
6. **Commit**: `git commit -m "feat: your feature"`
7. **Push**: `git push origin feature/your-feature`
8. **Create Pull Request** on GitHub

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance
- `perf`: Performance improvement

**Examples:**
```
feat(mcp): add filesystem MCP server integration

Implements a new MCP server for filesystem operations including
read, write, list, and search capabilities.

Closes #123
```

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure CI passes** (all checks green)
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested

### Code Review Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

---

## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection Failed

```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Verify endpoint
curl http://localhost:11434/api/tags
```

#### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull qwen3:8b

# Verify
ollama run qwen3:8b "test"
```

#### Import Errors

```bash
# Ensure in project root
cd /path/to/ai-lang-stuff

# Activate virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

#### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

#### Tests Failing

```bash
# Clear caches
./scripts/clean.sh cache

# Reinstall dependencies
uv sync

# Run specific failing test
pytest tests/test_specific.py -v -s
```

### Performance Issues

#### Slow Model Inference

- Use quantized models (Q4_K_M recommended)
- Check available RAM/VRAM
- Reduce context window size
- Use smaller models for simple tasks

#### High Memory Usage

```bash
# Check model memory
ollama ps

# Clear model cache
ollama stop <model-name>

# Use smaller quantizations
ollama pull qwen3:8b-q4_k_m
```

### Getting Help

1. **Check documentation**: Browse `docs/` directory
2. **Search issues**: Look for similar problems
3. **Ask in discussions**: GitHub Discussions
4. **File an issue**: Use issue templates
5. **Read troubleshooting guide**: `docs/TROUBLESHOOTING-RUNBOOK.md`

---

## 📊 Project Status

### Current Phase: **Phase 2 - Validation** ✅

- [x] Core utilities implemented
- [x] Documentation infrastructure
- [x] CI/CD setup
- [x] Examples created
- [ ] Full test coverage (in progress)
- [ ] Performance benchmarks
- [ ] Community ready

### Roadmap

See [MASTER-PLAN-SEQUENTIAL.md](MASTER-PLAN-SEQUENTIAL.md) for detailed 35-point roadmap.

**Next Milestones:**
1. Achieve 80%+ test coverage
2. Complete skill/agent documentation
3. Launch community channels
4. First stable release (0.1.0)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for the orchestration framework
- **Ollama** for local LLM runtime
- **FastAPI** for inspiration on project structure
- **Contributors** for their valuable input

---

**Happy Developing!** 🚀

For questions or discussions, open an issue or start a discussion on GitHub.
