# Quick Start Guide
## Local-First AI Experimentation Toolkit

**Last Updated**: 2025-10-26

---

## What Is This?

A complete toolkit for building AI agents that run **entirely on your machine** using:
- **Local LLMs** via Ollama (Qwen3, Gemma3)
- **LangChain** for agent framework
- **LangGraph** for multi-agent orchestration
- **MCP Servers** for tool integration
- **Local Vector Stores** for RAG systems

**Zero cloud dependencies. Privacy-first. Fully local.**

---

## Quick Start (5 Minutes)

### 1. Install Prerequisites

```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install tools
brew install ollama uv node
```

### 2. Start Ollama & Pull Models

```bash
# Start Ollama server (keep this running)
ollama serve

# In a new terminal, pull models
ollama pull qwen3:8b # Primary model (6GB)
ollama pull qwen3-embedding # For RAG (2GB)
ollama pull gemma3:4b # Alternative (3GB)
```

### 3. Install Python Dependencies

```bash
cd ai-lang-stuff
uv sync
```

### 4. Run Your First Example

```bash
# Simple chat with local model
uv run python examples/01-foundation/simple_chat.py

# Streaming response
uv run python examples/01-foundation/streaming_chat.py

# Compare models
uv run python examples/01-foundation/compare_models.py
```

---

## What's Available Now

### Examples (Milestone 1 )
- `examples/01-foundation/simple_chat.py` - Basic LLM interaction
- `examples/01-foundation/streaming_chat.py` - Token streaming
- `examples/01-foundation/compare_models.py` - Model comparison

### MCP Servers (Built )
- `mcp-servers/custom/filesystem/` - File operations
- `mcp-servers/custom/web-search/` - Web search

### Agents (Updated )
- `orchestration-specialist` - Master coordinator
- `local-model-manager` - Ollama expertise
- `mcp-integration-specialist` - Tool integration
- `langgraph-orchestrator` - Multi-agent workflows
- `rag-system-builder` - RAG systems

---

## Development Roadmap

### Currently Building
See **`docs/DEVELOPMENT-PLAN-PHASE-2.md`** for the complete 36-task plan.

### Next 4 Weeks
- **Week 1**: Core utilities (ollama_manager, mcp_client, etc.)
- **Week 2**: Milestone 2 & 3 examples (MCP integration, multi-agent)
- **Week 3**: Milestone 4 & documentation (RAG systems)
- **Week 4**: Testing & advanced features

### Future Milestones
- **Milestone 2**: MCP Integration Examples
- **Milestone 3**: Multi-Agent Orchestration
- **Milestone 4**: RAG Systems
- **Milestone 5**: Interpretability Analysis
- **Milestone 6**: Production Patterns

---

## Project Structure

```
ai-lang-stuff/
 .claude/ # Claude Code configuration
 agents/ # Specialized agent docs
 commands/ # Slash commands
 skills/ # Reusable skills
 docs/ # Documentation
 DEVELOPMENT-PLAN-PHASE-2.md # 36-task plan
 DEVELOPMENT-PLAN-20-POINTS.md # Original plan
 examples/ # Runnable examples
 01-foundation/ # Basic examples
 02-mcp/ # MCP integration
 03-multi-agent/ # Orchestration
 04-rag/ # RAG systems
 05-interpretability/ # Model analysis
 06-production/ # Production patterns
 mcp-servers/ # Custom MCP servers
 custom/
 filesystem/ # File operations
 web-search/ # Web search
 utils/ # Utilities (building)
 tests/ # Test suites
 CLAUDE.md # Claude's instructions
 QUICKSTART.md # This file
 README.md # Main documentation (coming)
```

**Legend**: Complete | In Progress | Planned

---

## Recommended Models

### For General Use
```bash
ollama pull qwen3:8b # Best balance (6GB)
ollama pull qwen3:30b-a3b # Fast MoE (8GB)
```

### For RAG Systems
```bash
ollama pull qwen3-embedding # Embeddings (2GB)
```

### For Vision Tasks
```bash
ollama pull qwen3-vl:8b # Vision model (7GB)
```

### For Constrained Resources
```bash
ollama pull gemma3:4b # Smallest (3GB)
```

---

## Common Commands

### Check Ollama Status
```bash
ollama list # List installed models
ps aux | grep ollama # Check if running
curl http://localhost:11434/api/tags # Test API
```

### Run Examples
```bash
uv run python examples/01-foundation/simple_chat.py
```

### Start Development Server (Future)
```bash
npx langgraph@latest dev # LangGraph Studio
```

---

## Documentation

### Main Guides
- **CLAUDE.md** - Complete instructions for Claude Code
- **docs/DEVELOPMENT-PLAN-PHASE-2.md** - 36-task development plan
- **plans/1-research-plan.md** - Research findings & model info
- **plans/3-kitchen-sink-plan.md** - Use cases & examples

### Agent Documentation
All agents have comprehensive guides in `.claude/agents/`:
- How to use their expertise
- Code examples
- Common patterns
- Troubleshooting

---

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama
ollama serve

# Or check if already running
ps aux | grep ollama
```

### Model Not Found
```bash
# Pull the model
ollama pull qwen3:8b

# Verify
ollama list
```

### Python Import Errors
```bash
# Reinstall dependencies
uv sync

# Verify environment
uv run python -c "from langchain_ollama import ChatOllama; print('OK')"
```

### Slow Performance
1. Use smaller model: `gemma3:4b` instead of `qwen3:8b`
2. Use MoE model: `qwen3:30b-a3b` (activates only 3B)
3. Close other applications
4. Check thermal throttling

---

## Learning Path

### Beginner
1. Run `simple_chat.py` - Understand basic LLM interaction
2. Run `streaming_chat.py` - See token-by-token generation
3. Run `compare_models.py` - Compare different models

### Intermediate (Coming Soon)
1. MCP integration examples - Use filesystem and web search tools
2. Multi-agent examples - Build orchestrated workflows
3. RAG examples - Question answering over documents

### Advanced (Coming Soon)
1. Interpretability - Analyze model internals
2. Production patterns - Deploy-ready code
3. Fine-tuning - Customize models

---

## Next Steps

### Immediate
1. Ensure Ollama running with models installed
2. Run the 3 foundation examples
3. Read agent documentation in `.claude/agents/`

### This Week
1. Wait for core utilities to be built (`utils/`)
2. Try MCP integration examples (coming soon)
3. Explore agent coordination patterns

### This Month
1. Complete all milestone examples
2. Build your own agent workflows
3. Contribute improvements

---

## Key Principles

### 1. Local-First
Everything runs on your machine. No API keys, no cloud services, no internet required (after initial setup).

### 2. Privacy-Preserving
Your data never leaves your machine. Perfect for sensitive work.

### 3. Educational
Every example is documented. Every utility has clear purpose. Learn by building.

### 4. Production-Quality
Type hints, tests, error handling. This isn't just toy code.

---

## Contributing

This is an active development project. See `docs/DEVELOPMENT-PLAN-PHASE-2.md` for:
- 36 specific tasks
- Timeline and priorities
- What needs building
- Where to contribute

---

## Getting Help

### Documentation
- Check `CLAUDE.md` for comprehensive instructions
- Read agent docs in `.claude/agents/`
- Review example READMEs (when created)

### Common Issues
- See troubleshooting section above
- Check `plans/1-research-plan.md` for model info
- Review development plan for context

---

## Current Status

**Phase**: Foundation Complete, Building Phase 2
**Next Up**: Core utilities and MCP examples
**Timeline**: 4-week sprint to complete Milestones 2-6

See `docs/DEVELOPMENT-PLAN-PHASE-2.md` for complete details.

---

**Built with Claude Sonnet 4.5 using Claude Code**

Last Updated: 2025-10-26
