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
