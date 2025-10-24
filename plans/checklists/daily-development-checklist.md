# Daily Development Checklist

## Morning Setup (2-3 minutes)

### Start Development Session
- [ ] Open terminal in project directory
- [ ] Start Ollama server in separate terminal: `ollama serve`
- [ ] Verify server running: `curl http://localhost:11434/api/tags`
- [ ] Check available models: `ollama list`
- [ ] Pull updates if on shared project: `git pull`
- [ ] Update dependencies if needed: `uv sync && npm install`

### Environment Check
- [ ] Verify Python environment: `uv run python --version`
- [ ] Check imports work: `uv run python -c "from langchain_ollama import ChatOllama"`
- [ ] Review .env settings for today's work
- [ ] Check disk space if working with large models: `df -h`

---

## Before Starting New Feature (5 minutes)

### Planning
- [ ] Review relevant example in `plans/3-kitchen-sink-plan.md`
- [ ] Check if similar pattern exists in `examples/`
- [ ] Read relevant section in `CLAUDE.md` for best practices
- [ ] Identify which models you'll need
- [ ] Estimate token usage and response times

### Code Setup
- [ ] Create new branch if needed: `git checkout -b feature/your-feature`
- [ ] Create new file in appropriate examples directory
- [ ] Copy template from similar example
- [ ] Add comprehensive docstring with purpose, prerequisites, expected output

### Testing
- [ ] Write test first in `tests/` if appropriate
- [ ] Plan verification steps
- [ ] Identify edge cases

---

## During Development (Continuous)

### Code Quality
- [ ] Add type hints to functions
- [ ] Write descriptive variable names
- [ ] Include error handling with try/except
- [ ] Add helpful error messages with troubleshooting steps
- [ ] Keep functions under 50 lines
- [ ] Add comments for complex logic

### Testing
- [ ] Test with smallest model first (gemma3:4b) for speed
- [ ] Verify with production model (qwen3:8b or qwen3:30b-a3b)
- [ ] Test error cases (server down, model missing, bad input)
- [ ] Run existing tests: `uv run pytest`
- [ ] Add new test if adding functionality

### Performance
- [ ] Monitor response times (should be <10s for simple queries)
- [ ] Check memory usage in Activity Monitor if slow
- [ ] Consider using streaming for long responses
- [ ] Cache repeated calls if appropriate

---

## Before Committing (5-10 minutes)

### Code Review Self-Check
- [ ] Code runs without errors
- [ ] All tests pass: `uv run pytest`
- [ ] No hardcoded credentials or API keys
- [ ] Docstrings updated
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Code formatted: `uv run ruff format .`
- [ ] Linting passes: `uv run ruff check .`

### Documentation
- [ ] Update relevant README sections if needed
- [ ] Add example to appropriate plan document
- [ ] Update IMPLEMENTATION_SUMMARY.md if major change
- [ ] Add comments for any non-obvious code

### Testing
- [ ] All tests pass including integration tests
- [ ] Manual testing completed
- [ ] Edge cases tested
- [ ] Performance acceptable

### Git Workflow
- [ ] Review changes: `git diff`
- [ ] Stage related changes together
- [ ] Write descriptive commit message
- [ ] Commit message explains "why" not just "what"
- [ ] Push to appropriate branch

---

## End of Day (5 minutes)

### Save Work
- [ ] Commit all work in progress
- [ ] Push to remote: `git push`
- [ ] Document any blockers in notes
- [ ] Save any useful prompts or configurations

### Cleanup
- [ ] Stop Ollama server: `killall ollama` (or leave running if convenient)
- [ ] Close unnecessary terminals
- [ ] Note disk space used by models
- [ ] Back up any important data

### Planning
- [ ] Review what was accomplished today
- [ ] Note any issues encountered
- [ ] Plan tomorrow's work
- [ ] Update PROJECT_STATUS.md if milestones completed

---

## Weekly Maintenance (Friday, 15 minutes)

### Updates
- [ ] Check for LangChain updates: Review changelog
- [ ] Check for new Ollama models
- [ ] Update dependencies: `uv sync --upgrade`
- [ ] Test after updates
- [ ] Review deprecation warnings

### Documentation
- [ ] Update README if project scope changed
- [ ] Document any new patterns discovered
- [ ] Update troubleshooting section with new issues found
- [ ] Review and update CLAUDE.md with new learnings

### Code Health
- [ ] Run full test suite: `uv run pytest --cov`
- [ ] Review test coverage report
- [ ] Check for unused code
- [ ] Review TODO comments
- [ ] Refactor duplicated code

### Performance
- [ ] Check model performance on key tasks
- [ ] Document any degradation
- [ ] Consider model upgrades if available
- [ ] Clean up old model files: Check `~/.ollama/models/`

---

## Monthly Review (30 minutes)

### Project Health
- [ ] Review all milestones progress
- [ ] Update PROJECT_STATUS.md
- [ ] Archive old branches
- [ ] Clean up unused experiments
- [ ] Review disk usage

### Code Quality
- [ ] Run comprehensive linting
- [ ] Review and update dependencies
- [ ] Check security advisories
- [ ] Update documentation
- [ ] Refactor technical debt

### Learning
- [ ] Review LangChain/LangGraph releases
- [ ] Check for new local models
- [ ] Read recent research papers
- [ ] Update skills based on learnings
- [ ] Document new best practices

---

## Quick Commands

### Fast Development
```bash
# Start everything
ollama serve &
npm run studio &

# Quick test
uv run python your_file.py

# Fast model for iteration
export DEFAULT_MODEL=gemma3:4b
```

### Debugging
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
export LANGCHAIN_VERBOSE=true

# Check what's running
ps aux | grep -E "ollama|python|node"

# Monitor logs
tail -f logs/app.log
```

### Quick Tests
```bash
# Run fast tests only
uv run pytest -m "not integration" -x

# Test specific file
uv run pytest tests/test_basic.py -v

# With coverage
uv run pytest --cov=examples --cov-report=html
```

---

## Productivity Tips

1. **Keep Ollama running** - Start once, use all day
2. **Use smaller models for iteration** - Switch to larger for final testing
3. **Cache expensive calls** - Use @lru_cache for repeated prompts
4. **Test incrementally** - Don't wait until done to test
5. **Commit frequently** - Small, focused commits are easier to review
6. **Document as you go** - Don't leave it for later
7. **Use streaming for feedback** - Better UX during development
8. **Monitor performance** - Watch response times

---

## Common Daily Issues

### Slow responses
- Switch to gemma3:4b for development
- Check Activity Monitor for CPU hogs
- Restart Ollama: `killall ollama && ollama serve`

### Import errors
- Re-sync dependencies: `uv sync`
- Check virtual environment: `uv run which python`

### Port conflicts
- Find process: `lsof -i :11434`
- Kill if needed: `kill <PID>`

### Out of memory
- Close unused applications
- Use smaller models
- Reduce context window in config

---

## Success Metrics

Daily development is on track when:
- [ ] Can start working in <5 minutes
- [ ] Tests run in <2 minutes
- [ ] Simple queries respond in <5 seconds
- [ ] No unexplained errors
- [ ] Code committed at least once
- [ ] Documentation up to date
