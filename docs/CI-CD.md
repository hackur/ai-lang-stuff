# CI/CD Pipeline Documentation

This document describes the continuous integration and continuous deployment (CI/CD) setup for the ai-lang-stuff project.

## Overview

The project uses GitHub Actions for automated testing, linting, documentation building, and performance benchmarking. All workflows are designed to work with local-first AI tools (Ollama) and maintain the project's zero-dependency philosophy.

## Workflows

### 1. Test Suite (`test.yml`)

**Purpose**: Ensure code quality and functionality across Python versions

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Matrix Testing**:
- Python 3.10
- Python 3.11
- Python 3.12

**Steps**:
1. Checkout code
2. Setup Python environment
3. Install Ollama
4. Pull test models (qwen2.5:0.5b)
5. Install dependencies with uv
6. Run pytest with coverage
7. Upload coverage to Codecov
8. Archive test results

**Artifacts**:
- Coverage reports (HTML and XML)
- Test results per Python version

### 2. Code Quality (`lint.yml`)

**Purpose**: Enforce code standards and catch potential issues

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Checks**:
- **Ruff**: Fast Python linter (replaces Flake8, isort, pyupgrade)
- **Ruff Format**: Code formatting check
- **Black**: Additional formatting verification
- **mypy**: Type checking (informational, doesn't fail build)
- **Import Sorting**: Verify imports are properly sorted

**Exit Codes**:
- Non-zero on linting or formatting issues
- mypy continues on error (informational only)

### 3. Documentation (`docs.yml`)

**Purpose**: Build and deploy project documentation

**Triggers**:
- Push to `main` branch (deploy)
- Pull requests to `main` branch (build only)

**Steps**:
1. Install MkDocs and dependencies
2. Check documentation links
3. Build MkDocs site
4. Deploy to GitHub Pages (main branch only)

**Access**: Documentation available at `https://yourusername.github.io/ai-lang-stuff/`

### 4. Performance Benchmarks (`benchmarks.yml`)

**Purpose**: Track performance regressions over time

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Steps**:
1. Setup environment and Ollama
2. Run benchmark tests
3. Compare with baseline
4. Post results as PR comment
5. Store benchmark data

**Metrics Tracked**:
- LLM completion time
- Batch processing performance
- Streaming performance
- Async operation timing
- Memory usage (future)

## Local Development

### Setup Pre-commit Hooks

Install and configure pre-commit hooks:

```bash
make setup
```

This installs:
- Ruff (linting and formatting)
- Black (formatting)
- mypy (type checking)
- Bandit (security scanning)
- General file checks (trailing whitespace, large files, etc.)

### Running CI Checks Locally

Run all CI checks before pushing:

```bash
make ci
```

Or run individually:

```bash
# Linting
make lint

# Testing
make test

# Formatting
make format

# Check formatting without changes
make format-check

# Benchmarks
make benchmark
```

### Pre-commit Hook Behavior

Hooks run automatically on `git commit`:

```bash
git commit -m "Your message"
# Hooks run automatically
# - Ruff fixes issues
# - Black formats code
# - mypy checks types
# - Bandit scans for security issues
# - File validation runs
```

Bypass hooks (not recommended):

```bash
git commit --no-verify -m "Skip hooks"
```

Run hooks manually:

```bash
# All files
pre-commit run --all-files

# Specific hook
pre-commit run ruff --all-files

# Specific files
pre-commit run --files src/some_file.py
```

## Makefile Commands

### Installation
```bash
make install          # Install dependencies
make setup            # Setup dev environment + pre-commit
```

### Testing
```bash
make test             # Run full test suite with coverage
make test-fast        # Run tests without coverage
make benchmark        # Run performance benchmarks
make benchmark-compare # Compare with baseline
```

### Code Quality
```bash
make lint             # Run all linters
make format           # Auto-format code
make format-check     # Check formatting without changes
```

### Documentation
```bash
make docs             # Build documentation
make docs-serve       # Serve docs locally at http://127.0.0.1:8000
```

### Utilities
```bash
make check-ollama     # Check if Ollama is running
make pull-models      # Pull required models
make clean            # Clean generated files
make ci               # Run all CI checks
```

## Pre-commit Configuration

Configuration in `.pre-commit-config.yaml`:

### Hooks Enabled

1. **Ruff** - Fast linting and formatting
2. **Black** - Code formatting (backup)
3. **mypy** - Type checking (skipped in CI)
4. **File checks**:
   - Prevent large files (>10MB)
   - Check for case conflicts
   - Fix end-of-file
   - Prevent commits to main
   - Trim trailing whitespace
5. **Format validation**:
   - YAML, TOML, JSON syntax
6. **Security**:
   - Bandit security scanner
   - Private key detection
   - Merge conflict detection
7. **Code quality**:
   - AST validation
   - Docstring checks
8. **Documentation**:
   - Markdown linting
   - Shell script linting

### Customization

Edit `.pre-commit-config.yaml` to:
- Add/remove hooks
- Change hook arguments
- Adjust file exclusions
- Configure auto-fixing

## Benchmark Tracking

### Running Benchmarks

```bash
# Run benchmarks
make benchmark

# Compare with baseline
make benchmark-compare
```

### Benchmark Results

Results stored in:
- `benchmark-results.json` - Current run
- `.benchmark-baseline.json` - Baseline for comparison

### Setting New Baseline

After intentional performance changes:

```bash
cp benchmark-results.json .benchmark-baseline.json
git add .benchmark-baseline.json
git commit -m "chore: update performance baseline"
```

### Interpreting Results

Benchmark output shows:
- **Mean time**: Average execution time
- **Std dev**: Standard deviation
- **Min/Max**: Fastest/slowest execution
- **Iterations**: Number of test runs

Example output:
```
test_simple_completion: 0.1234s ± 0.0056s
test_batch_completion: 0.5678s ± 0.0123s
```

## Coverage Reports

### Local Coverage

After running `make test`:
- **Terminal**: Summary printed
- **HTML**: Open `htmlcov/index.html` in browser

### CI Coverage

Coverage uploaded to Codecov:
- View at: `https://codecov.io/gh/yourusername/ai-lang-stuff`
- Badges available for README
- PR comments show coverage changes

### Coverage Goals

- **Target**: 80%+ overall coverage
- **Core modules**: 90%+ coverage
- **Acceptable**: 70%+ for experimental features

## Troubleshooting

### Tests Fail in CI But Pass Locally

**Causes**:
- Python version mismatch
- Ollama not running locally
- Environment variable differences
- Cached dependencies

**Solutions**:
```bash
# Check Python version
python --version  # Should match CI matrix

# Ensure Ollama running
ollama serve
ollama list

# Clear caches
make clean
make install
```

### Pre-commit Hooks Slow

**Causes**:
- First run installs hook environments
- mypy type checking is slow
- Large number of files

**Solutions**:
```bash
# Skip slow hooks for quick commits
SKIP=mypy git commit -m "message"

# Update hook environments
pre-commit autoupdate

# Run specific hooks
pre-commit run ruff --all-files
```

### Benchmark Regressions

**Investigating**:
1. Check recent changes in PR
2. Review benchmark comparison comment
3. Run locally: `make benchmark`
4. Profile slow operations

**Acceptable Regressions**:
- <10%: Generally acceptable
- 10-20%: Needs justification
- >20%: Requires investigation

**Updating Baseline**:
```bash
# After confirming intentional change
cp benchmark-results.json .benchmark-baseline.json
```

### Documentation Build Fails

**Common Issues**:
- Broken markdown links
- Missing files referenced in nav
- Invalid YAML in mkdocs.yml

**Solutions**:
```bash
# Build locally
make docs

# Serve and check
make docs-serve

# Validate links
find docs/ -name "*.md" -exec grep -H "http" {} \;
```

## Security

### Secrets Management

**Never Commit**:
- API keys
- Credentials
- `.env` files with secrets
- Private keys

**Pre-commit Protection**:
- `detect-private-key` hook enabled
- `.gitignore` configured for common secret files

### Security Scanning

**Bandit**:
- Scans for common security issues
- Runs in pre-commit hooks
- Configured in `pyproject.toml`

**Skip False Positives**:
```python
# nosec B101
assert something  # Bandit normally flags asserts
```

## Performance Optimization

### CI Speed

**Current Workflow Times** (approximate):
- Test Suite: 5-8 minutes
- Code Quality: 1-2 minutes
- Documentation: 2-3 minutes
- Benchmarks: 3-5 minutes

**Optimization Strategies**:
1. Cache Python dependencies
2. Cache Ollama models
3. Use smaller test models
4. Parallel test execution
5. Skip slow hooks in CI

### Local Development Speed

**Faster Feedback**:
```bash
# Quick test
make test-fast

# Lint only
make lint

# Format without full check
make format
```

## Future Enhancements

### Planned Additions

1. **Docker Integration**:
   - Containerized test environment
   - Consistent CI/local parity

2. **Extended Benchmarks**:
   - Memory profiling
   - Token usage tracking
   - Response quality metrics

3. **Advanced Coverage**:
   - Branch coverage
   - Integration test coverage
   - Example script coverage

4. **Release Automation**:
   - Semantic versioning
   - Automated changelog
   - PyPI publishing

5. **Performance Tracking**:
   - Historical performance graphs
   - Regression trend analysis
   - Performance budgets

## Best Practices

### Before Committing

1. Run `make ci` locally
2. Ensure all tests pass
3. Check coverage didn't decrease
4. Review benchmark changes
5. Update documentation if needed

### Before Creating PR

1. Rebase on latest main/develop
2. Run full test suite
3. Update CHANGELOG if applicable
4. Fill out PR template completely
5. Link related issues

### PR Review Checklist

- [ ] All CI checks passing
- [ ] Code coverage maintained/improved
- [ ] No performance regressions
- [ ] Documentation updated
- [ ] Tests added for new features
- [ ] Breaking changes documented

## Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Pre-commit Framework](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [MkDocs Documentation](https://www.mkdocs.org/)

## Support

For CI/CD issues:
1. Check this documentation
2. Review workflow logs in GitHub Actions
3. Run locally with `make ci`
4. Open issue with [BUG] tag
