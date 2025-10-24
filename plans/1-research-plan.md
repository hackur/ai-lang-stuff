# Research Plan & Online Data Collection

## Executive Summary
This document consolidates research findings from January 2025 on the latest versions, capabilities, and integration patterns for local AI development tools. All components are designed to run locally without cloud dependencies.

---

## 1. Core Framework Versions (January 2025)

### LangChain Python
- **Latest Version**: 1.0.2 (October 21, 2025)
- **Python Requirements**: Python >=3.10.0, <4.0.0, with Python 3.14 support
- **Related Packages**:
  - `langchain-core`: 0.3.79 (interfaces for chat models, LLMs, vector stores, retrievers)
  - `langchain-community`: 0.4 (community integrations)
- **Major Changes**: Official 1.0 release stabilized API with production-ready guarantees
- **Installation**: `pip install langchain langchain-core langchain-community`

### LangGraph Python
- **Latest Version**: 1.0.1 (October 20, 2025)
- **Python Compatibility**: Requires Python >=3.10, supports 3.10-3.13
- **Production Proof**: Battle-tested by Uber, LinkedIn, Klarna
- **Recent Features** (June 2025 additions):
  - **Node Caching**: Cache results of individual nodes, skip redundant computation
  - **Deferred Nodes**: Delay execution until all upstream paths complete
  - **Pre/Post Model Hooks**: Custom logic before/after model calls
- **Installation**: `pip install langgraph langgraph-cli`

### LangGraph Studio (Node.js)
- **Type**: Visual workflow editor for LangGraph
- **Runtime**: Node.js via npx
- **Launch**: `npx langgraph@latest dev`
- **Port**: Default localhost:3000
- **Purpose**: Visual debugging and orchestration of multi-agent workflows

---

## 2. Local Model Ecosystem

### Qwen3 Models (Alibaba Cloud)
- **Release**: April 2025
- **Model Family**:
  - **MoE Models**:
    - `qwen3:235b-a22b` (235B total params, 22B activated)
    - `qwen3:30b-a3b` (30B total params, 3B activated)
  - **Dense Models**: 32B, 14B, 8B, 4B, 1.7B, 0.6B variants
- **Context Window**: 256K tokens native, expandable to 1M tokens
- **Vision Model**: Qwen3-VL (most powerful vision LLM in Qwen series)
  - Supports hundreds of pages of documents or 2-hour videos
- **Ollama Usage**: `ollama pull qwen3:30b-a3b` then `ollama run qwen3:30b-a3b`
- **Performance**: Competitive with DeepSeek-R1, GPT-4, Gemini-2.5-Pro on coding/math/general tasks
- **Special Models**:
  - `qwen3-embedding` for semantic search
  - `qwen3-reranker` for ranking

### Gemma 3 Models (Google)
- **Release**: March 12, 2025
- **Foundation**: Built from Gemini 2.0 research and technology
- **Model Sizes**: 1B, 4B, 12B, 27B
- **Capabilities**:
  - Multimodal (text + images) for 4B+ sizes
  - Text-only for 1B variant
  - 128K context window
  - 140+ language support
- **Ollama Usage**: `ollama pull gemma3:27b` (defaults to Q4_0 quantization)
- **QAT Models**: Quantization-aware training for consumer GPU optimization
- **Best For**: Single GPU/TPU deployment scenarios

### Model Comparison Framework Needs
| Model | Best Use Case | Context | Speed | Multimodal |
|-------|--------------|---------|-------|------------|
| Qwen3-235B-A22B | Complex reasoning, long documents | 1M tokens | Medium | Yes (VL) |
| Qwen3-30B-A3B | Fast general tasks, coding | 256K-1M | Fast | Yes (VL) |
| Gemma 3 27B | Multilingual, resource-constrained | 128K | Fast | Yes |
| Gemma 3 4B | Edge devices, rapid inference | 128K | Very Fast | Yes |

---

## 3. Model Context Protocol (MCP) Ecosystem

### Overview
- **Created By**: Anthropic (November 2024)
- **Purpose**: Universal standard for connecting AI systems with data sources
- **Major Adoption**:
  - OpenAI (March 2025)
  - Google DeepMind/Gemini (April 2025 confirmation)
- **Security**: June 2025 spec update added authorization and Resource Indicators

### Official MCP Servers (Anthropic)
- **Google Drive**: Document access and search
- **Slack**: Message history, channel operations
- **GitHub**: Repository operations, PR management
- **Git**: Local repository operations
- **Postgres**: Database queries and operations
- **Puppeteer**: Web scraping and browser automation

### AWS MCP Servers (May 2025)
- **Lambda**: Serverless function invocation
- **ECS**: Container orchestration
- **EKS**: Kubernetes cluster management
- **Finch**: Local container runtime

### Google MCP Server
- **Data Commons**: Access to public datasets, reduce LLM hallucinations

### Community MCP Servers (GitHub modelcontextprotocol/servers)
- File system operations
- SQLite databases
- Web search APIs
- Calculator and math tools
- Custom tool extensions

### MCP Integration Strategy for This Project
1. Use official MCP servers for standard operations (GitHub, filesystem, Postgres)
2. Create custom MCP servers for project-specific tools:
   - Local model management (Ollama/LM Studio)
   - Vector store operations (Chroma/FAISS)
   - Document processing pipelines
   - Mechanistic interpretability tools
3. Implement MCP clients in both Python (LangChain agents) and Node.js (LangGraph Studio)

---

## 4. Mechanistic Interpretability Tools

### TransformerLens
- **Latest Version**: v3 (Alpha as of September 2025)
- **Major Improvement**: Works well with large models (previous versions struggled >9B params)
- **Installation**: `pip install transformer-lens`
- **Capabilities**:
  - Hook-based activation analysis
  - Attention pattern visualization
  - Circuit discovery and validation
  - Causal interventions (ablations, patching)
- **Recommended For**:
  - Small models (≤9B) with complex experiments
  - Multi-model comparative analysis
  - First experiments in mechanistic interpretability
- **Companion Tools**:
  - **Neuronpedia**: Neuron visualization and documentation
  - **NNSight**: Alternative interpretability library
- **Community**: Active development through September 2025

### Interpretability Research Applications (2025)
- Attention pattern analysis
- Activation manipulation for prose generation
- Literary style transfer via atomic interventions
- Circuit discovery in reasoning tasks

### Integration Plan
1. Jupyter notebooks for interactive exploration
2. Scripts for automated interpretability sweeps
3. Visualization dashboards for activation patterns
4. Integration with LangSmith for production model monitoring

---

## 5. Observability: LangSmith

### Purpose
- **Tracing**: Full execution traces for multi-agent workflows
- **Debugging**: Step-by-step agent decision analysis
- **Performance**: Latency and token usage tracking
- **Evaluation**: Automated test suite for agent behavior

### Local-First Strategy
- LangSmith can run with local models while still providing observability
- Use environment variables to connect LangChain to LangSmith
- All data stays local (model inference) but traces can optionally sync to cloud
- For fully local: Use LangSmith's local mode (if available) or alternative tracing

### Configuration
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"  # Optional for cloud sync
export LANGCHAIN_PROJECT="local-ai-experiments"
```

---

## 6. Local Model Hosting Options

### Ollama
- **Installation**: `brew install ollama`
- **Server**: `ollama serve` (port 11434)
- **API**: OpenAI-compatible endpoint
- **Model Management**: Simple pull/run commands
- **Best For**: Command-line workflows, quick experimentation
- **Latest Models**: Qwen3, Gemma 3, Llama 3.3, DeepSeek-R1

### LM Studio
- **Type**: GUI application for macOS
- **Features**:
  - Visual model browser and downloader
  - Built-in chat interface
  - Model quantization options
  - Server mode with OpenAI-compatible API
- **Best For**: Users who prefer GUI, testing multiple model variants
- **Port**: Configurable (default 1234)

### Integration Strategy
- Use Ollama as primary backend (automation-friendly)
- Use LM Studio for visual model comparison
- Both expose OpenAI-compatible APIs → same code works with both
- LangChain supports both via `Ollama` and `ChatOpenAI` (pointed to localhost)

---

## 7. Development Milestones

### Milestone 1: Foundation (Week 1)
**Goal**: Get basic local LLM + LangChain working
- [ ] Install Ollama and pull 2-3 models (Qwen3, Gemma 3)
- [ ] Configure LangChain with local model endpoints
- [ ] Create simple single-agent script (query → response)
- [ ] Verify LangSmith tracing (optional)
- [ ] Document setup process

**Success Metric**: Run "Hello World" agent script in <30 seconds

### Milestone 2: MCP Integration (Week 2)
**Goal**: Connect agents to external tools via MCP
- [ ] Set up official MCP servers (filesystem, GitHub)
- [ ] Create custom MCP server for Ollama model management
- [ ] Integrate MCP with LangChain agent (tool calling)
- [ ] Build example: Agent that searches codebase and answers questions
- [ ] Document MCP server creation process

**Success Metric**: Agent successfully uses 3+ MCP tools in single workflow

### Milestone 3: Multi-Agent Orchestration (Week 3)
**Goal**: Build complex workflows with LangGraph
- [ ] Install LangGraph Studio
- [ ] Create visual workflow with multiple agent types (researcher, coder, reviewer)
- [ ] Implement state persistence with SQLite
- [ ] Add conditional routing based on agent outputs
- [ ] Test human-in-the-loop workflows
- [ ] Document workflow patterns

**Success Metric**: 3-agent workflow completes research → code → review cycle

### Milestone 4: Advanced Capabilities (Week 4)
**Goal**: RAG, memory, and vision models
- [ ] Set up local vector store (Chroma)
- [ ] Implement document ingestion pipeline
- [ ] Add RAG to agent for document QA
- [ ] Integrate Qwen3-VL for image understanding
- [ ] Build multi-modal agent (text + images)
- [ ] Document RAG and vision patterns

**Success Metric**: Agent answers questions from 100+ page PDF using RAG

### Milestone 5: Mechanistic Interpretability (Week 5)
**Goal**: Analyze and understand model internals
- [ ] Install TransformerLens v3
- [ ] Load local model for analysis
- [ ] Create attention visualization notebook
- [ ] Run activation patching experiments
- [ ] Build circuit discovery script
- [ ] Integrate interpretability with LangSmith
- [ ] Document interpretability workflows

**Success Metric**: Identify and visualize 1 complete circuit in local model

### Milestone 6: Production Patterns (Week 6)
**Goal**: Move from experiments to deployable code
- [ ] Create configuration management system
- [ ] Add comprehensive error handling
- [ ] Implement logging and monitoring
- [ ] Build automated test suite
- [ ] Create deployment scripts
- [ ] Write production best practices guide

**Success Metric**: All examples run with <1% failure rate over 100 iterations

---

## 8. Technical Architecture Decisions

### Python Environment Management
- **Tool**: `uv` (Astral's fast package manager)
- **Rationale**: 10-100x faster than pip, better resolver, built-in virtual envs
- **Command Pattern**: `uv run python script.py` (auto-creates venv)

### Node.js Execution
- **Tool**: `npx` for ephemeral execution
- **Rationale**: No global installs, always latest versions, clean environment
- **Command Pattern**: `npx langgraph@latest dev`

### State Management
- **Simple State**: JSON files for configuration
- **Structured State**: SQLite for agent memory and checkpoints
- **Vector Storage**: Chroma (local, no server needed) or FAISS

### API Standards
- **Model APIs**: OpenAI-compatible (Ollama and LM Studio both support)
- **Tool APIs**: MCP for all external tool integrations
- **Agent APIs**: LangGraph for multi-agent orchestration

---

## 9. Dependencies Matrix

### Python Core
```
langchain==1.0.2
langchain-core==0.3.79
langchain-community==0.4
langgraph==1.0.1
langgraph-cli
```

### Model Integrations
```
langchain-ollama
openai  # For OpenAI-compatible local endpoints
```

### Vector Stores
```
chromadb
faiss-cpu  # or faiss-gpu
langchain-chroma
```

### Interpretability
```
transformer-lens>=3.0.0
torch
numpy
matplotlib
plotly
```

### Utilities
```
python-dotenv
pydantic
httpx
sqlalchemy
```

### Development Tools
```
jupyter
jupyterlab
ipython
pytest
ruff  # Linting and formatting
```

### Node.js Core
```json
{
  "@langchain/langgraph": "latest",
  "@modelcontextprotocol/sdk": "latest"
}
```

---

## 10. Research Questions & Experiments

### Week 1-2: Foundation Questions
- How do Qwen3 and Gemma 3 compare on coding tasks?
- What's the optimal quantization for speed vs quality?
- How reliable are MCP servers for production use?

### Week 3-4: Architecture Questions
- What's the best state persistence pattern for long-running agents?
- How do you handle partial failures in multi-agent workflows?
- Can vision models (Qwen3-VL) effectively replace OCR pipelines?

### Week 5-6: Advanced Questions
- Can we fine-tune small models to specialize in specific tasks?
- How do attention patterns differ between Qwen3 and Gemma 3?
- What circuits are responsible for coding abilities in local models?

---

## 11. Success Criteria

### Technical Success
- [ ] All examples run locally without internet
- [ ] <10GB total disk space for complete setup
- [ ] <10 second cold start for basic workflows
- [ ] <60 second execution for complex multi-agent workflows
- [ ] 100% MCP server reliability for standard operations

### Documentation Success
- [ ] Complete setup guide (beginner → advanced)
- [ ] Every capability has runnable example
- [ ] Troubleshooting guide covers 90% of issues
- [ ] Architecture decision records (ADRs) for major choices

### Community Success
- [ ] Project serves as reference for local-first AI
- [ ] Examples cited in other projects
- [ ] Contributions from external developers
- [ ] Positive feedback on ease of setup

---

## 12. Key Resources & References

### Official Documentation
- LangChain Python: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- MCP Specification: https://github.com/modelcontextprotocol
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- Ollama: https://ollama.com/
- Qwen3: https://qwenlm.github.io/blog/qwen3/
- Gemma 3: https://blog.google/technology/developers/gemma-3/

### Community Resources
- MCP Servers: https://github.com/modelcontextprotocol/servers
- LangChain Templates: https://github.com/langchain-ai/langchain/tree/master/templates
- Interpretability Research: https://transformer-circuits.pub/

### Tools & Utilities
- UV Package Manager: https://github.com/astral-sh/uv
- LM Studio: https://lmstudio.ai/
- Neuronpedia: https://neuronpedia.org/

---

## Next Steps

1. **Immediate**: Update pyproject.toml and package.json with researched versions
2. **Short-term**: Create 3-kitchen-sink-plan.md with concrete examples
3. **Medium-term**: Build Milestone 1 examples
4. **Long-term**: Complete all 6 milestones and publish comprehensive guide

This research plan provides the foundation for building a complete local-first AI development environment with cutting-edge tools and models, all running on-device without cloud dependencies.
