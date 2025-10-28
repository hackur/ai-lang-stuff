# Local-First AI Toolkit 🤖

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-alpha-orange)

**A complete toolkit for building production AI applications entirely on your local machine** - no API keys, no cloud dependencies, complete privacy.

Build sophisticated agent workflows, RAG systems, and multi-modal AI applications using local LLMs (Qwen, Gemma, Llama) with LangChain, LangGraph, and full mechanistic interpretability.

---

## 🚀 Quick Start (macOS - 10 Minutes)

### Prerequisites Check

```bash
# Check if you have Python 3.10-3.12 (NOT 3.13)
python3 --version

# If you have Python 3.13, install 3.12:
brew install python@3.12
```

### One-Command Install

```bash
# Clone and setup everything
git clone https://github.com/hackur/ai-lang-stuff.git
cd ai-lang-stuff
./scripts/setup.sh
```

### Manual Setup

#### Step 1: Install Prerequisites

```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12 (recommended)
brew install python@3.12

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Ollama (local LLM runtime)
brew install ollama
```

#### Step 2: Start Ollama

```bash
# Start Ollama server
ollama serve

# Or run as background service
brew services start ollama

# Verify it's running
curl http://localhost:11434/api/tags
```

#### Step 3: Setup Project

```bash
# Clone repository
git clone https://github.com/hackur/ai-lang-stuff.git
cd ai-lang-stuff

# Sync dependencies (uv way - recommended)
uv sync --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python --version  # Should show 3.12.x
```

#### Step 4: Download Models

```bash
# Download recommended models
ollama pull qwen3:8b        # General purpose (4.4 GB)
ollama pull gemma3:4b       # Fast, lightweight (2.7 GB)
ollama pull qwen3-embedding # For RAG (274 MB)

# Verify models
ollama list
```

#### Step 5: Run First Example

```bash
# Test basic example
uv run python examples/error_handling_demo.py

# Test tool registry
uv run python examples/tool_registry_demo.py

# Success! You're ready to build.
```

---

## 📦 Using uv for Python Package Management

This project uses **uv** - a fast, modern Python package manager written in Rust. It's 10-100x faster than pip and handles dependency resolution better.

### Installing uv

```bash
# macOS/Linux (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

### Common uv Commands

```bash
# Sync dependencies from pyproject.toml (like pip install -r requirements.txt)
uv sync

# Sync with specific Python version
uv sync --python 3.12

# Add a new dependency
uv add langchain-openai

# Add a development dependency
uv add --dev pytest

# Remove a dependency
uv remove package-name

# Run a script with uv (uses project's virtual environment)
uv run python examples/error_handling_demo.py

# Run a command in the virtual environment
uv run pytest

# Create/recreate virtual environment
uv venv --python 3.12

# Pin Python version for the project
uv python pin 3.12

# List installed packages
uv pip list

# Show outdated packages
uv pip list --outdated
```

### uv vs pip Comparison

| Task | pip | uv |
|------|-----|-----|
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install package && pip freeze` | `uv add package` |
| Run script | `python script.py` | `uv run python script.py` |
| Speed | ~30s for 300 packages | ~3s for 300 packages |
| Lock file | pip-tools required | Built-in (uv.lock) |

### Installing Optional Dependencies

```bash
# Install with development tools
uv sync --extra dev

# Install with GPU support (CUDA required)
uv sync --extra gpu

# Install multiple extras
uv sync --extra dev --extra gpu

# Install all extras
uv sync --all-extras
```

### Why uv?

- **10-100x faster** than pip for dependency resolution
- **Built-in lock file** (uv.lock) for reproducible installs
- **Better dependency resolution** - catches conflicts early
- **Project-aware** - automatically uses correct Python version
- **Drop-in replacement** for pip in most cases
- **Written in Rust** - blazing fast, memory efficient

---

## 🎮 How to Use This Toolkit

### Basic Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Run any example
uv run python examples/01-foundation/basic_llm_interaction.py

# 4. Or use the utilities in your own code
python
>>> from utils.ollama_manager import OllamaManager
>>> manager = OllamaManager()
>>> manager.list_models()
```

### Running Examples

All examples are organized by category and can be run with `uv run`:

```bash
# Foundation examples (basic LLM usage)
uv run python examples/01-foundation/basic_llm_interaction.py
uv run python examples/01-foundation/prompt_engineering.py
uv run python examples/01-foundation/model_comparison.py

# MCP integration examples
uv run python examples/02-mcp/filesystem_mcp.py
uv run python examples/02-mcp/web_search_mcp.py

# Multi-agent workflows
uv run python examples/03-multi-agent/research_pipeline.py
uv run python examples/03-multi-agent/code_review_pipeline.py

# RAG systems
uv run python examples/04-rag/document_qa.py
uv run python examples/04-rag/code_search.py

# Interpretability
uv run python examples/05-interpretability/activation_patching.py
uv run python examples/05-interpretability/attention_visualization.py

# Production patterns
uv run python examples/06-production/monitoring_setup.py
uv run python examples/06-production/config_management.py

# Advanced features
uv run python examples/07-advanced/vision_agent.py
uv run python examples/07-advanced/audio_transcription.py
```

### Using Core Utilities

The `utils/` directory contains production-ready utilities:

```python
# Ollama Manager - model operations and health checks
from utils.ollama_manager import OllamaManager

manager = OllamaManager()
if manager.check_ollama_running():
    manager.ensure_model_available("qwen3:8b")
    models = manager.list_models()
    stats = manager.benchmark_model("qwen3:8b")

# MCP Client - filesystem and web search tools
from utils.mcp_client import FilesystemMCP, WebSearchMCP

fs = FilesystemMCP()
files = fs.list_files("/path/to/dir")
content = fs.read_file("file.txt")

search = WebSearchMCP()
results = search.search("local LLMs 2024", num_results=5)

# Vector Store Manager - RAG and document search
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager(store_type="chroma")
store = manager.create_from_documents(docs, "collection_name")
results = manager.similarity_search("query", k=5)

# State Manager - persist agent state
from utils.state_manager import StateManager

manager = StateManager()
manager.save_state("agent_id", {"messages": [...], "context": {...}})
state = manager.load_state("agent_id")

# Tool Registry - centralized tool management
from utils.tool_registry import get_registry

registry = get_registry()
registry.register_tool("my_tool", function, description="...")
tools = registry.get_langchain_tools(categories=["web", "filesystem"])

# Error Recovery - production error handling
from utils.error_recovery import RetryStrategy, CircuitBreaker, GracefulDegradation

retry = RetryStrategy(max_retries=3, base_delay=1.0)
result = retry.execute(risky_function)

circuit = CircuitBreaker(threshold=5, recovery_timeout=60)
result = circuit.call(external_service)

degradation = GracefulDegradation(fallbacks=["qwen3:8b", "gemma3:4b"])
model = degradation.try_with_fallback(primary_model)
```

### Building Your Own Agents

```python
# Example: Simple research agent
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from utils.mcp_client import WebSearchMCP
from utils.ollama_manager import OllamaManager

# Setup
manager = OllamaManager()
manager.ensure_model_available("qwen3:8b")

llm = ChatOllama(model="qwen3:8b")
search = WebSearchMCP()
tools = search.get_langchain_tools()

# Create agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({"input": "Research local LLMs in 2024"})
print(result)
```

---

## 🔒 Sandbox Mode for Safe Execution

Run examples in isolated environments for testing and experimentation.

### Using Docker Sandbox (Recommended)

```bash
# Build sandbox image
docker build -t ai-lang-stuff-sandbox -f Dockerfile.sandbox .

# Run example in sandbox
docker run --rm \
  -v $(pwd)/examples:/workspace/examples:ro \
  -v $(pwd)/utils:/workspace/utils:ro \
  ai-lang-stuff-sandbox \
  python examples/error_handling_demo.py

# Run with Ollama access (expose host network)
docker run --rm \
  --network host \
  -v $(pwd)/examples:/workspace/examples:ro \
  -v $(pwd)/utils:/workspace/utils:ro \
  ai-lang-stuff-sandbox \
  python examples/01-foundation/basic_llm_interaction.py
```

### Using Python venv Sandbox

```bash
# Create isolated environment
python3.12 -m venv sandbox_env
source sandbox_env/bin/activate

# Install minimal dependencies
pip install langchain langchain-ollama

# Run example with limited access
python examples/error_handling_demo.py

# Deactivate when done
deactivate
```

### Using firejail (Linux)

```bash
# Install firejail
sudo apt install firejail  # Ubuntu/Debian
sudo dnf install firejail  # Fedora

# Run in sandbox with limited network
firejail --net=none python examples/error_handling_demo.py

# Run with filesystem restrictions
firejail --private --read-only=. python examples/error_handling_demo.py
```

### Safe Execution Best Practices

1. **Read-only mounts** when using Docker
2. **No write access** to sensitive directories
3. **Network isolation** for examples that don't need Ollama
4. **Resource limits** (CPU, memory) for long-running tasks
5. **Separate environment** for untested code

### Example: Sandboxed Development Workflow

```bash
# 1. Create development sandbox
uv venv sandbox --python 3.12
source sandbox/bin/activate

# 2. Install minimal deps
uv pip install langchain langchain-ollama

# 3. Test new example in isolation
python my_experimental_agent.py

# 4. If successful, add to main project
deactivate
uv add langchain-experimental  # Add to main project

# 5. Clean up sandbox
rm -rf sandbox/
```

---

## ⚠️ Important: Python Version

**Use Python 3.10, 3.11, or 3.12 - NOT 3.13**

Some dependencies (transformer-lens) don't support Python 3.13 yet.

### If You Have Python 3.13:

```bash
# Install Python 3.12
brew install python@3.12

# Create venv with correct Python
uv venv --python 3.12

# Activate and sync
source .venv/bin/activate
uv sync
```

---

## 🌟 What's Included

### Core Utilities (`utils/`)

Battle-tested utilities for production AI:

- **ollama_manager.py** - Ollama operations and model management
- **mcp_client.py** - Model Context Protocol integration
- **vector_store.py** - ChromaDB & FAISS for RAG
- **state_manager.py** - Persistent agent state
- **tool_registry.py** - Tool discovery and management
- **error_recovery.py** - Production error handling

### 30+ Working Examples

**Foundation (01/):**
- Basic LLM interactions
- Prompt engineering
- Model comparison

**MCP Integration (02/):**
- Filesystem operations
- Web search integration
- Combined tool usage

**Multi-Agent (03/):**
- Research pipelines
- Code review workflows
- Parallel agent execution

**RAG Systems (04/):**
- Document Q&A
- Code search
- Reranking & streaming
- Vision RAG

**Interpretability (05/):**
- Activation patching
- Circuit discovery
- Attention visualization

**Production (06/):**
- Monitoring & logging
- Config management
- Deployment patterns

**Advanced (07/):**
- Vision agents
- Audio transcription
- Multimodal RAG
- Document understanding

### CLI Tool

```bash
# Install CLI
cd cli && pip install -e .

# Model operations
ailang models list
ailang models pull qwen3:8b
ailang models benchmark

# Run examples
ailang examples list
ailang examples run mcp-filesystem

# RAG operations
ailang rag index ./docs
ailang rag query "How do I..."
```

---

## 📋 System Requirements

### Minimum

- macOS 11+ (Big Sur)
- Python 3.10-3.12 (⚠️ NOT 3.13)
- 8 GB RAM (16 GB recommended)
- 10 GB free disk space
- Intel with AVX2 or Apple Silicon

### Recommended

- macOS 14+ (Sonoma)
- Python 3.12
- 16-32 GB RAM
- 50 GB disk space
- Apple Silicon (M1/M2/M3)

### Performance

| Hardware | qwen3:8b | Notes |
|----------|----------|-------|
| M1 Max | ~40 tok/s | Excellent |
| M2 Pro | ~35 tok/s | Very good |
| M3 Pro | ~45 tok/s | Best |
| Intel i7 | ~15 tok/s | Workable |

---

## 🛠️ Troubleshooting

### Python 3.13 Dependency Error

```bash
# Install Python 3.12
brew install python@3.12

# Recreate venv
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Ollama Connection Error

```bash
# Check if running
ps aux | grep ollama

# Start Ollama
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### Model Not Found

```bash
# Pull model
ollama pull qwen3:8b

# List available
ollama list

# Test
ollama run qwen3:8b "Hello"
```

### Import Errors

```bash
# Activate venv
source .venv/bin/activate

# Reinstall
uv sync

# Verify
python -c "import langchain; print('OK')"
```

---

## 📚 Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete dev guide
- **[MASTER-PLAN-SEQUENTIAL.md](MASTER-PLAN-SEQUENTIAL.md)** - 35-point roadmap
- **[docs/api-reference/](docs/api-reference/)** - API docs
- **[docs/adr/](docs/adr/)** - Architecture decisions

---

## 🎯 Example Use Cases

### Document Q&A

```bash
uv run python examples/04-rag/document_qa.py
# Index documents, ask questions
```

### Code Review

```bash
uv run python examples/03-multi-agent/code_review_pipeline.py
# Multi-agent code analysis
```

### Vision Understanding

```bash
# Requires: ollama pull qwen3-vl:8b
uv run python examples/07-advanced/vision_agent.py
```

---

## 🔧 Development

### Setup

```bash
# Install dev dependencies
uv sync

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .
```

### Project Structure

```
ai-lang-stuff/
├── utils/          # Core utilities
├── examples/       # 30+ examples
├── tests/          # Test suite
├── docs/           # Documentation
├── cli/            # CLI tool
├── workflows/      # LangGraph workflows
└── scripts/        # Automation
```

---

## 🚀 Performance Tips

### Model Selection

| Task | Model | Speed | Quality |
|------|-------|-------|---------|
| Fast | gemma3:4b | 🚀🚀🚀 | ⭐⭐⭐ |
| Balanced | qwen3:8b | 🚀🚀 | ⭐⭐⭐⭐ |
| Best | qwen3:70b | 🚀 | ⭐⭐⭐⭐⭐ |
| Vision | qwen3-vl:8b | 🚀🚀 | ⭐⭐⭐⭐ |

### Quantization

```bash
# Recommended (good balance)
ollama pull qwen3:8b-q4_k_m

# Better quality
ollama pull qwen3:8b-q5_k_m

# Best quality (slower)
ollama pull qwen3:8b-q8_0
```

---

## 🤝 Contributing

Contributions welcome! See [DEVELOPMENT.md](DEVELOPMENT.md).

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/ai-lang-stuff.git

# Create branch
git checkout -b feature/your-feature

# Make changes, test
make test

# Commit and push
git commit -m "feat: your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **LangChain** - Orchestration framework
- **Ollama** - Local LLM runtime
- **TransformerLens** - Interpretability
- **Community** - Contributors and users

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hackur/ai-lang-stuff/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hackur/ai-lang-stuff/discussions)
- **Docs**: [DEVELOPMENT.md](DEVELOPMENT.md)

---

## 🗺️ Roadmap

**Current**: Alpha (0.1.0)
- [x] Core utilities
- [x] 30+ examples
- [x] CLI tool
- [x] Documentation
- [ ] 80% test coverage
- [ ] Community launch

**Next**: Beta (0.2.0)
- [ ] PyPI package
- [ ] Homebrew formula
- [ ] Video tutorials
- [ ] Example projects

See [MASTER-PLAN-SEQUENTIAL.md](MASTER-PLAN-SEQUENTIAL.md) for full roadmap.

---

**Status**: 🚀 Active Development | **Version**: 0.1.0-alpha

Built with ❤️ for the local-first AI community.

**Zero API keys. Zero cloud. 100% local. Complete privacy.**
