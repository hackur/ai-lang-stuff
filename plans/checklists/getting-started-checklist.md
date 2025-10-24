# Getting Started Checklist

## Complete Setup (20-30 minutes)

### Prerequisites Check
- [ ] Running macOS (Apple Silicon or Intel)
- [ ] Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- [ ] At least 10GB free disk space
- [ ] At least 16GB RAM (8GB minimum)
- [ ] Stable internet connection for model downloads

### Step 1: Install Dependencies (5 minutes)
```bash
cd /Volumes/JS-DEV/ai-lang-stuff
./scripts/setup.sh
```

**What this does:**
- Installs uv (Python package manager)
- Installs Node.js
- Installs Ollama
- Installs Python 3.13+
- Creates directory structure
- Installs all Python and Node dependencies
- Creates .env file
- Starts Ollama server
- Downloads recommended models (qwen3:8b, qwen3:30b-a3b, gemma3:4b, qwen3-embedding)

**Expected output:**
- All tools installed successfully
- Ollama server running
- Models downloaded
- Tests passing

### Step 2: Verify Installation (2 minutes)
```bash
./scripts/test-setup.sh
```

**What this does:**
- Runs 10 verification tests
- Checks all tools are installed
- Verifies Ollama server is running
- Confirms models are available
- Tests Python environment
- Validates project structure

**Expected output:**
- All 10 tests pass
- Summary shows "All tests passed!"

### Step 3: Configure Environment (3 minutes)
```bash
# Review and customize .env file
cat .env
# Edit if needed
nano .env
```

**Key settings to review:**
- `OLLAMA_BASE_URL` - Usually correct at http://localhost:11434
- `DEFAULT_MODEL` - Set your preferred model (qwen3:8b recommended)
- `LOG_LEVEL` - INFO for normal use, DEBUG for troubleshooting

**Optional settings:**
- `LANGCHAIN_TRACING_V2=true` - Enable LangSmith (requires account)
- `LMSTUDIO_ENABLED=true` - If using LM Studio instead of Ollama

### Step 4: Run Your First Example (2 minutes)
```bash
# Simple chat example
uv run python examples/01-foundation/simple_chat.py
```

**Expected output:**
- Model loads
- Request sent
- Coherent response about list comprehensions
- Takes 2-5 seconds

**If this works, congratulations! Your environment is ready.**

### Step 5: Try More Examples (10 minutes)

**Streaming responses:**
```bash
uv run python examples/01-foundation/streaming_chat.py
```

**Model comparison:**
```bash
uv run python examples/01-foundation/compare_models.py
```

**Launch LangGraph Studio:**
```bash
npm run studio
# Open http://localhost:3000
```

---

## Quick Command Reference

### Ollama Management
```bash
# Start server (if not running)
ollama serve

# Check what's running
ps aux | grep ollama

# List installed models
ollama list

# Pull a new model
ollama pull gemma3:12b

# Test a model
ollama run qwen3:8b "Write a haiku"

# Check server
curl http://localhost:11434/api/tags
```

### Development
```bash
# Run any example
uv run python examples/path/to/example.py

# Run tests
uv run pytest

# Run only unit tests
uv run pytest -m "not integration"

# Start Jupyter
uv run jupyter lab

# Update dependencies
uv sync
npm install
```

### Troubleshooting
```bash
# Restart Ollama
killall ollama
ollama serve

# Check logs
tail -f logs/app.log

# Verify Python environment
uv run python -c "from langchain_ollama import ChatOllama; print('OK')"

# Check disk space
df -h

# Check memory
top
```

---

## What to Do Next

### If you're new to local LLMs (2-4 hours):
1. Read through `plans/0-readme.md` for project vision
2. Complete Milestone 1: `plans/milestones/milestone-1-foundation.md`
3. Try all three foundation examples
4. Experiment with different models
5. Adjust temperature and other parameters
6. Read `plans/1-research-plan.md` for the roadmap

### If you want to build something (1-2 days):
1. Review `plans/3-kitchen-sink-plan.md` for 10 examples
2. Pick an example that matches your use case:
   - Example 2: Agent with tools (MCP integration)
   - Example 3: Multi-agent workflow
   - Example 4: RAG for document QA
   - Example 5: Vision model for images
3. Copy the example code
4. Customize for your data/task
5. Refer to `CLAUDE.md` for development patterns

### If you're experienced with AI (today):
1. Jump to advanced examples in `plans/3-kitchen-sink-plan.md`
2. Example 6: Mechanistic interpretability
3. Example 8: Model comparison framework
4. Example 9: Fine-tuning
5. Example 10: Production deployment
6. Build custom MCP servers in `mcp-servers/custom/`

---

## Common First-Time Issues

### Issue: "Command not found: uv"
**Solution:**
```bash
brew install uv
# Then re-run setup.sh
```

### Issue: "Connection refused to localhost:11434"
**Solution:**
```bash
# Start Ollama in a separate terminal
ollama serve
# Leave this running
```

### Issue: "Model not found"
**Solution:**
```bash
# Pull the model first
ollama pull qwen3:8b
# Verify
ollama list
```

### Issue: Setup script fails
**Solution:**
```bash
# Run commands manually
brew install uv node ollama python@3.13
cd /Volumes/JS-DEV/ai-lang-stuff
uv sync
npm install
cp config/.env.example .env
ollama serve &
ollama pull qwen3:8b
```

### Issue: Very slow performance
**Solution:**
- Use smaller models: `ollama pull gemma3:4b`
- Close other applications
- Check Activity Monitor for CPU/memory usage
- Consider qwen3:30b-a3b (MoE is faster than dense)

---

## Success Indicators

You've successfully set up when:
- [ ] `./scripts/test-setup.sh` shows all tests pass
- [ ] `uv run python examples/01-foundation/simple_chat.py` works
- [ ] Model responds in under 10 seconds
- [ ] Response is coherent and relevant
- [ ] No error messages in output
- [ ] `ollama list` shows 3+ models

---

## Getting Help

1. **Check troubleshooting in README.md**
   - Common issues and solutions
   - Performance optimization tips

2. **Review milestone documentation**
   - `plans/milestones/milestone-1-foundation.md`
   - Detailed troubleshooting for setup

3. **Check official documentation**
   - LangChain: https://python.langchain.com/
   - Ollama: https://ollama.com/
   - LangGraph: https://langchain-ai.github.io/langgraph/

4. **Verify each component separately**
   ```bash
   # Test Ollama
   curl http://localhost:11434/api/tags

   # Test Python
   uv run python --version

   # Test imports
   uv run python -c "import langchain; print(langchain.__version__)"

   # Test model
   ollama run qwen3:8b "test"
   ```

---

## Next Steps

Once setup is complete:
1. **Explore examples** in order (01-foundation → 06-production)
2. **Read documentation** to understand concepts
3. **Experiment freely** - everything runs locally, safe to break
4. **Build your project** using examples as templates
5. **Share your work** - this is open source!

**Remember:** Everything runs locally. No data leaves your machine. You can't break anything that can't be fixed by re-running setup.
