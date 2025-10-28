# GitHub Actions Workflows

This directory contains CI/CD workflows for the ai-lang-stuff project.

## Workflows

### Test Suite (`test.yml`)
**Trigger**: Push/PR to main or develop branches

Runs the full test suite across multiple Python versions (3.10, 3.11, 3.12):
- Sets up Python environment with uv
- Installs and starts Ollama
- Pulls required test models
- Runs pytest with coverage
- Uploads coverage to Codecov
- Archives test results as artifacts

**Status**: ![Test Suite](https://github.com/yourusername/ai-lang-stuff/workflows/Test%20Suite/badge.svg)

### Code Quality (`lint.yml`)
**Trigger**: Push/PR to main or develop branches

Performs comprehensive code quality checks:
- Ruff linting and formatting
- Black formatting verification
- mypy type checking
- Import sorting validation

**Status**: ![Code Quality](https://github.com/yourusername/ai-lang-stuff/workflows/Code%20Quality/badge.svg)

### Documentation (`docs.yml`)
**Trigger**: Push/PR to main branch

Builds and deploys documentation:
- Validates documentation links
- Builds MkDocs site
- Deploys to GitHub Pages (main branch only)

**Status**: ![Documentation](https://github.com/yourusername/ai-lang-stuff/workflows/Documentation/badge.svg)

### Performance Benchmarks (`benchmarks.yml`)
**Trigger**: Push/PR to main or develop branches

Tracks performance over time:
- Runs benchmark test suite
- Compares with baseline
- Posts results as PR comments
- Tracks performance regressions
- Archives benchmark data

**Status**: ![Benchmarks](https://github.com/yourusername/ai-lang-stuff/workflows/Performance%20Benchmarks/badge.svg)

## Local Development

Before pushing, run CI checks locally:

```bash
# Install pre-commit hooks
make setup

# Run all CI checks
make ci

# Or run individually
make lint
make test
make benchmark
```

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:
- Ruff linting and formatting
- Black formatting
- Type checking with mypy
- Security checks with bandit
- File validation (no large files, trailing whitespace, etc.)
- Markdown and shell script linting

Install hooks:
```bash
pip install pre-commit
pre-commit install
```

Run manually:
```bash
pre-commit run --all-files
```

## Workflow Configuration

### Test Matrix
Tests run on Python 3.10, 3.11, and 3.12 to ensure compatibility.

### Ollama Setup
Workflows automatically install and configure Ollama with test models:
- `qwen2.5:0.5b` - Fast, lightweight model for CI

### Caching
- Python dependencies cached per Python version
- Ollama models cached between runs
- Pre-commit environments cached

### Artifacts
Test runs upload artifacts for debugging:
- Coverage reports (HTML and XML)
- Test results
- Benchmark data

## Secrets Required

None required for basic functionality. Optional:
- `CODECOV_TOKEN` - For private repo coverage uploads
- `LANGCHAIN_API_KEY` - For LangSmith tracing (optional)

## Badge URLs

Add these to your README.md:

```markdown
[![Test Suite](https://github.com/yourusername/ai-lang-stuff/workflows/Test%20Suite/badge.svg)](https://github.com/yourusername/ai-lang-stuff/actions/workflows/test.yml)
[![Code Quality](https://github.com/yourusername/ai-lang-stuff/workflows/Code%20Quality/badge.svg)](https://github.com/yourusername/ai-lang-stuff/actions/workflows/lint.yml)
[![Documentation](https://github.com/yourusername/ai-lang-stuff/workflows/Documentation/badge.svg)](https://github.com/yourusername/ai-lang-stuff/actions/workflows/docs.yml)
[![Benchmarks](https://github.com/yourusername/ai-lang-stuff/workflows/Performance%20Benchmarks/badge.svg)](https://github.com/yourusername/ai-lang-stuff/actions/workflows/benchmarks.yml)
```

## Troubleshooting

### Tests Failing Locally But Pass in CI
- Check Python version matches CI matrix
- Ensure Ollama is running: `ollama serve`
- Verify models are pulled: `ollama list`
- Check environment variables

### Pre-commit Hooks Failing
- Update hooks: `pre-commit autoupdate`
- Clear cache: `pre-commit clean`
- Run specific hook: `pre-commit run ruff --all-files`

### Benchmark Regressions
- Review changes that impact performance
- Update baseline if intentional: `cp benchmark-results.json .benchmark-baseline.json`
- Investigate unexpected slowdowns in PR comments

## Contributing

When adding new workflows:
1. Test locally first with `act` (GitHub Actions runner)
2. Use caching for expensive operations
3. Add workflow status badge to this README
4. Document any new secrets or configuration
5. Keep workflows focused and fast (<10 minutes)
