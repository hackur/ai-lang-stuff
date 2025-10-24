# Milestone 1: Foundation

## Goal
Get basic local LLM + LangChain working with at least 2-3 models and verify core functionality.

## Timeline
**Estimated Duration**: 2-4 hours
**Priority**: P0 (Critical - blocks all other work)

---

## Tasks

### Task 1.1: Environment Setup (30 min)
- [ ] Install Homebrew (if not already installed)
- [ ] Install uv package manager: `brew install uv`
- [ ] Install Node.js: `brew install node`
- [ ] Install Python 3.13+: `brew install python@3.13`
- [ ] Verify installations:
  - [ ] `uv --version`
  - [ ] `node --version`
  - [ ] `python3 --version`

**Expected Output**: All tools installed and version commands work

### Task 1.2: Ollama Installation (20 min)
- [ ] Install Ollama: `brew install ollama`
- [ ] Start Ollama server: `ollama serve`
- [ ] Verify server running: `curl http://localhost:11434/api/tags`
- [ ] Open separate terminal for Ollama server (keep running)

**Expected Output**: Ollama server responds to API requests

### Task 1.3: Model Download (30-60 min, depends on internet speed)
- [ ] Pull Qwen3 8B: `ollama pull qwen3:8b`
- [ ] Pull Qwen3 30B MoE: `ollama pull qwen3:30b-a3b`
- [ ] Pull Gemma3 4B: `ollama pull gemma3:4b`
- [ ] Pull Qwen3 Embedding: `ollama pull qwen3-embedding`
- [ ] Verify models: `ollama list`
- [ ] Test model: `ollama run qwen3:8b "What is 2+2?"`

**Expected Output**: All models downloaded and can respond to simple prompts

**Disk Space Required**: ~15-20GB total

### Task 1.4: Python Dependencies (15 min)
- [ ] Navigate to project directory
- [ ] Create .env file: `cp config/.env.example .env`
- [ ] Install dependencies: `uv sync`
- [ ] Verify installation: `uv pip list | grep langchain`
- [ ] Check imports: `uv run python -c "from langchain_ollama import ChatOllama"`

**Expected Output**: All imports successful, no errors

### Task 1.5: Simple Chat Example (20 min)
- [ ] Create examples/01-foundation/simple_chat.py
- [ ] Implement basic ChatOllama usage
- [ ] Run: `uv run python examples/01-foundation/simple_chat.py`
- [ ] Verify response is coherent
- [ ] Test with different models (qwen3:8b, gemma3:4b)

**Code**:
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain Python list comprehensions.")
]

response = llm.invoke(messages)
print(response.content)
```

**Expected Output**: Clear explanation of list comprehensions

### Task 1.6: Streaming Response Example (15 min)
- [ ] Create examples/01-foundation/streaming_chat.py
- [ ] Implement streaming response
- [ ] Verify tokens stream in real-time
- [ ] Test latency vs batch response

**Code**:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b")

for chunk in llm.stream("Write a haiku about coding"):
    print(chunk.content, end="", flush=True)
print()
```

**Expected Output**: Response appears token-by-token

### Task 1.7: Multi-Model Comparison (20 min)
- [ ] Create examples/01-foundation/compare_models.py
- [ ] Test same prompt on all downloaded models
- [ ] Measure response time for each
- [ ] Compare response quality subjectively
- [ ] Document findings

**Expected Output**: Speed/quality comparison table

### Task 1.8: Error Handling (15 min)
- [ ] Test behavior when Ollama server is down
- [ ] Test behavior with non-existent model
- [ ] Test behavior with malformed prompts
- [ ] Implement proper try/except blocks
- [ ] Add informative error messages

**Expected Output**: Graceful error handling with helpful messages

### Task 1.9: Configuration Management (20 min)
- [ ] Create config loader in config/loader.py
- [ ] Load settings from config/models.yaml
- [ ] Load environment variables from .env
- [ ] Validate configuration
- [ ] Add configuration examples

**Expected Output**: Centralized configuration system

### Task 1.10: Documentation (15 min)
- [ ] Document setup process in README.md
- [ ] Add troubleshooting section
- [ ] Include example commands
- [ ] Document expected outputs
- [ ] Add common error solutions

**Expected Output**: Clear setup documentation

---

## Dependencies
- **Prerequisites**: macOS with Homebrew installed
- **Blocks**: All subsequent milestones depend on this foundation

---

## Success Criteria
- [ ] Ollama server running and responding
- [ ] At least 3 models downloaded and functional
- [ ] Python environment configured correctly
- [ ] Simple chat example works end-to-end
- [ ] Streaming responses work
- [ ] Error handling in place
- [ ] Response time < 5 seconds for simple queries
- [ ] Documentation clear enough for beginner to follow

---

## Verification Steps

### 1. Environment Check
```bash
# Run all these commands successfully
uv --version
node --version
python3 --version
ollama --version
```

### 2. Server Check
```bash
# Ollama server responding
curl http://localhost:11434/api/tags
# Should return JSON with model list
```

### 3. Model Check
```bash
# Models available
ollama list | grep qwen3
ollama list | grep gemma3
# Should show at least 3 models
```

### 4. Python Check
```bash
# Imports work
uv run python -c "from langchain_ollama import ChatOllama; print('Success')"
```

### 5. End-to-End Check
```bash
# Full example runs
uv run python examples/01-foundation/simple_chat.py
# Should output coherent response
```

---

## Common Issues & Solutions

### Issue: Ollama server won't start
**Symptoms**: "Connection refused" errors
**Solutions**:
1. Check if already running: `ps aux | grep ollama`
2. Kill existing process: `killall ollama`
3. Start fresh: `ollama serve`
4. Check port availability: `lsof -i :11434`

### Issue: Model download fails
**Symptoms**: Download stalls or errors out
**Solutions**:
1. Check internet connection
2. Try different model first (smaller)
3. Check disk space: `df -h`
4. Restart Ollama: `killall ollama && ollama serve`

### Issue: Python imports fail
**Symptoms**: "ModuleNotFoundError"
**Solutions**:
1. Verify uv environment: `uv sync`
2. Check Python version: `python3 --version` (must be 3.10+)
3. Reinstall langchain: `uv add langchain-ollama`
4. Verify installation: `uv pip list | grep langchain`

### Issue: Model responds but output is gibberish
**Symptoms**: Incoherent or truncated responses
**Solutions**:
1. Check if model fully downloaded: `ollama list`
2. Re-pull model: `ollama pull <model-name>`
3. Try different model
4. Check context window not exceeded

### Issue: Very slow responses
**Symptoms**: Queries take 30+ seconds
**Solutions**:
1. Use smaller model (gemma3:4b instead of qwen3:30b-a3b)
2. Reduce context window in config
3. Check CPU usage: `top`
4. Close other applications
5. Consider using MoE model (qwen3:30b-a3b) which is faster

---

## Performance Benchmarks

Expected performance on Apple Silicon (M1/M2/M3):

| Model | Tokens/sec | 100-token response time |
|-------|-----------|------------------------|
| gemma3:4b | 50-80 | ~2 seconds |
| qwen3:8b | 30-50 | ~3 seconds |
| qwen3:30b-a3b | 40-60 | ~2.5 seconds (MoE) |

If your performance is significantly worse, check:
- Other processes using CPU/GPU
- Thermal throttling
- Available RAM
- Model quantization level

---

## Next Steps

After completing this milestone:
1. Proceed to Milestone 2: MCP Integration
2. Explore more examples in plans/3-kitchen-sink-plan.md
3. Experiment with prompt engineering
4. Try fine-tuning temperature and other parameters

---

## Completion Checklist

Before marking this milestone complete:
- [ ] All 10 tasks completed
- [ ] All success criteria met
- [ ] All verification steps pass
- [ ] Documentation updated
- [ ] Examples running correctly
- [ ] Performance meets benchmarks (±20%)
- [ ] No unresolved errors or warnings
- [ ] Ready to proceed to Milestone 2

**Estimated Completion Time**: 2-4 hours for experienced developer, 4-6 hours for beginner

---

## Notes
- Keep Ollama server running in separate terminal throughout development
- If switching between different projects, remember to activate correct Python environment
- Model files stored in ~/.ollama/models - can be deleted to free space
- For faster iteration, use gemma3:4b during development, switch to larger models for production
