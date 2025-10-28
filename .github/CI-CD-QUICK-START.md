# CI/CD Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install Pre-commit Hooks
```bash
make setup
```

### 2. Verify Installation
```bash
make ci
```

That's it! The pre-commit hooks will now run automatically on every commit.

---

## 📋 Daily Workflow

### Before Starting Work
```bash
git pull origin main
make install  # Update dependencies if needed
```

### During Development
```bash
# Auto-format your code
make format

# Run tests quickly
make test-fast
```

### Before Committing
```bash
# Run full CI suite
make ci
```

Git commit will automatically:
- Format code with Ruff/Black
- Check types with mypy
- Scan for security issues
- Validate file formats

### Creating a Pull Request
```bash
# Ensure everything passes
make ci

# Create your PR - CI will run automatically
```

---

## 🔧 Common Commands

### Testing
```bash
make test              # Full test suite with coverage
make test-fast         # Quick tests, no coverage
make benchmark         # Performance tests
```

### Code Quality
```bash
make lint              # Check code quality
make format            # Auto-fix formatting
make format-check      # Check without modifying
```

### Documentation
```bash
make docs              # Build docs
make docs-serve        # View at http://127.0.0.1:8000
```

### Utilities
```bash
make clean             # Clean generated files
make check-ollama      # Verify Ollama is running
make pull-models       # Download required models
```

---

## 🚨 Troubleshooting

### "Pre-commit hook failed"
```bash
# Fix automatically
make format

# Or commit without hooks (not recommended)
git commit --no-verify
```

### "Tests failing"
```bash
# Check Ollama is running
make check-ollama

# Clean and reinstall
make clean
make install
make test
```

### "Import errors"
```bash
# Reinstall dependencies
make install

# Activate venv manually if needed
source .venv/bin/activate
```

---

## 📊 Understanding CI Results

### Green Checkmarks = All Good ✅
Your PR is ready for review!

### Red X = Something Failed ❌

Click the details link to see:
- **Test failures**: Fix the failing test
- **Lint errors**: Run `make lint` locally
- **Format issues**: Run `make format`
- **Type errors**: Fix mypy warnings (informational only)

---

## 🎯 Best Practices

### DO ✅
- Run `make ci` before pushing
- Keep tests passing
- Maintain >80% coverage
- Update docs with code changes
- Use meaningful commit messages

### DON'T ❌
- Skip pre-commit hooks
- Commit directly to main
- Ignore failing tests
- Commit secrets/API keys
- Push without running CI locally

---

## 📚 Need More Help?

- **Detailed Docs**: See `docs/CI-CD.md`
- **Workflow Details**: See `.github/workflows/README.md`
- **Makefile Reference**: Run `make help`
- **Pre-commit Config**: See `.pre-commit-config.yaml`

---

## 🏃 Speed Tips

### Faster Commits
```bash
# Skip slow type checking
SKIP=mypy git commit -m "message"

# Format only changed files
git diff --name-only | xargs black
```

### Faster Tests
```bash
# Run specific test
pytest tests/test_specific.py -v

# Stop on first failure
pytest -x

# Use test-fast target
make test-fast
```

### Faster Development
```bash
# Keep Ollama running
ollama serve &

# Cache model pulls
ollama pull qwen2.5:0.5b
```

---

## 📈 Workflow Status

Check status at: `https://github.com/yourusername/ai-lang-stuff/actions`

### Current Workflows
- ✅ Test Suite (Python 3.10, 3.11, 3.12)
- ✅ Code Quality (Ruff, Black, mypy)
- ✅ Documentation Build
- ✅ Performance Benchmarks

---

## 🔄 Update Pre-commit Hooks

```bash
# Update to latest versions
pre-commit autoupdate

# Reinstall hooks
pre-commit install --install-hooks
```

---

## 💡 Pro Tips

1. **Alias for CI**: Add to `.bashrc` or `.zshrc`:
   ```bash
   alias mci='make ci'
   ```

2. **Watch Tests**: Auto-run tests on file changes:
   ```bash
   pytest-watch tests/
   ```

3. **Coverage Report**: After tests, open:
   ```bash
   open htmlcov/index.html
   ```

4. **Benchmark Baseline**: After performance improvements:
   ```bash
   cp benchmark-results.json .benchmark-baseline.json
   git add .benchmark-baseline.json
   git commit -m "chore: update performance baseline"
   ```

---

## 📞 Getting Help

1. Check this guide first
2. Run `make help` for command list
3. Read detailed docs in `docs/CI-CD.md`
4. Check workflow logs in GitHub Actions
5. Open issue with [BUG] tag if stuck

---

**Remember**: Running `make ci` before pushing saves everyone time! 🎉
