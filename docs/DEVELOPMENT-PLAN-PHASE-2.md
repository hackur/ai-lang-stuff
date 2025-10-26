# Development Plan Phase 2: 20+ Point Task Plan
## Local-First AI Experimentation Toolkit

**Generated**: 2025-10-26
**Updated Model Context**: Claude Sonnet 4.5, Haiku 4.5, Opus 4.1
**Status**: Ready to Begin Development
**Estimated Total Time**: 35-40 hours

---

## Executive Summary

This plan outlines the immediate next steps to build out the local-first AI experimentation toolkit. The focus is on creating **working examples**, **core utilities**, and **comprehensive documentation** that demonstrates the power of running AI entirely on-device.

### Context
- **Foundation Complete**: Milestone 1 done (Ollama integration, basic examples)
- **Agent Architecture**: Updated with 4 specialized agents
- **MCP Servers Built**: Filesystem and web-search servers ready
- **Ready to Build**: Examples, utilities, and integration patterns

---

## 📊 Current State Assessment

### ✅ Completed
- [x] Orchestration specialist agent (updated for this project)
- [x] Local model manager agent
- [x] MCP integration specialist agent
- [x] LangGraph orchestrator agent
- [x] RAG system builder agent
- [x] Basic examples: simple_chat.py, streaming_chat.py, compare_models.py
- [x] Custom MCP servers: filesystem, web-search

### 🔄 In Progress
- [ ] Interpretability researcher agent (90% complete)
- [ ] Example creator agent
- [ ] Core utilities implementation

### 📝 Pending
- [ ] Milestone 2-6 examples
- [ ] Comprehensive documentation
- [ ] Test suites
- [ ] Advanced features

---

## 🎯 20+ POINT TASK PLAN

## Phase 1: Complete Agent Architecture (Priority: P0)
**Timeline**: 2-3 hours

### ✅ 1. Complete Interpretability Researcher Agent
**Status**: 90% complete, needs final save
**File**: `.claude/agents/interpretability-researcher.md`
**Action**: Finish and save the agent documentation
**Timeline**: 10 min

### 📝 2. Create Example Creator Agent
**File**: `.claude/agents/example-creator.md`
**Purpose**: Specialist for building runnable examples following project patterns
**Includes**: Example structure, documentation standards, testing approaches
**Timeline**: 30 min

### 📝 3. Clean Up Legacy Slash Commands
**Action**: Remove Laravel/Nova-specific commands
**Remove**:
- `.claude/commands/nova-resource.md`
- `.claude/commands/qdb.md`
- `.claude/commands/migration-builder.md`
- `.claude/commands/fresh-seed.md`
- `.claude/commands/deploy-check.md`
- `.claude/commands/qnova.md`
- `.claude/commands/review-staged.md`

**Keep**: ultrathink.md, code-quality.md, refactor-plan.md, qfix.md, qplan.md, search-pattern.md
**Timeline**: 20 min

### 📝 4. Clean Up Legacy Skills
**Action**: Remove Laravel-specific skills
**Remove**:
- `.claude/skills/laravel-package-specialist/`
- `.claude/skills/nova-resource/`
- `.claude/skills/nova-damage-assessment-field-development/`

**Keep & Update**:
- `debug-agent.md` → Update with local model context
- `model-comparison.md` → Update with Qwen3/Gemma3 models
- `documentation-writer.md` → Update for this project

**Timeline**: 30 min

### 📝 5. Update Model References Throughout
**Action**: Replace all model references with latest Claude models
**Context**: Current Claude models are Sonnet 4.5, Haiku 4.5, Opus 4.1
**Files to Update**:
- CLAUDE.md
- All agent files
- README.md (when created)

**Note**: This is about **documentation** references to Claude (the AI assistant), not local Ollama models
**Timeline**: 20 min

---

## Phase 2: Core Utilities Implementation (Priority: P1)
**Timeline**: 6-8 hours

### 📝 6. Create Ollama Manager Utility
**File**: `utils/ollama_manager.py`
**Functions**:
```python
def check_ollama_running() -> bool
def ensure_model_available(model: str) -> bool
def list_models() -> List[str]
def pull_model(model: str) -> bool
def get_model_info(model: str) -> dict
def benchmark_model(model: str, prompt: str) -> dict
def recommend_model(task_type: str) -> str
```
**Timeline**: 1.5 hours

### 📝 7. Create MCP Client Wrapper Library
**File**: `utils/mcp_client.py`
**Classes**:
```python
class MCPClient:
    """Base MCP client with connection management"""

class FilesystemMCP(MCPClient):
    """Filesystem MCP server wrapper"""

class WebSearchMCP(MCPClient):
    """Web search MCP server wrapper"""
```
**Features**: Connection pooling, retry logic, error handling, LangChain tool conversion
**Timeline**: 2 hours

### 📝 8. Create Vector Store Utilities
**File**: `utils/vector_store.py`
**Classes**:
```python
class VectorStoreManager:
    """Manage local vector stores (Chroma/FAISS)"""

    def create_from_documents(...)
    def load_existing(...)
    def add_documents(...)
    def delete_collection(...)
    def list_collections(...)
```
**Timeline**: 1.5 hours

### 📝 9. Create State Manager Utility
**File**: `utils/state_manager.py`
**Purpose**: LangGraph state persistence helpers
**Features**: SQLite checkpointing, state schemas, common state patterns
**Timeline**: 1.5 hours

### 📝 10. Create Tool Registry
**File**: `utils/tool_registry.py`
**Purpose**: Central registry of all available tools/utilities
**Features**: Auto-discovery, tool metadata, agent-friendly API
**Timeline**: 1 hour

### 📝 11. Create Utils __init__.py
**File**: `utils/__init__.py`
**Purpose**: Clean imports and package initialization
**Timeline**: 15 min

---

## Phase 3: Milestone 2 - MCP Integration Examples (Priority: P1)
**Timeline**: 3-4 hours

### 📝 12. Filesystem Agent Example
**File**: `examples/02-mcp/filesystem_agent.py`
**Purpose**: Demonstrate agent using filesystem MCP server
**Features**:
- Read/write files via MCP
- Search for files
- Create directory structures
- Integration with LangChain agent

**Example Usage**:
```bash
uv run python examples/02-mcp/filesystem_agent.py
# Agent: "Read README.md and summarize it"
# Agent: "Create a new file called notes.txt with these points..."
```
**Timeline**: 1 hour

### 📝 13. Web Search Agent Example
**File**: `examples/02-mcp/web_search_agent.py`
**Purpose**: Demonstrate agent using web search MCP server
**Features**:
- Search web for information
- Process search results
- Answer questions with web context

**Example Usage**:
```bash
uv run python examples/02-mcp/web_search_agent.py
# Agent: "What are the latest developments in LangGraph?"
```
**Timeline**: 1 hour

### 📝 14. Combined Tools Agent Example
**File**: `examples/02-mcp/combined_tools_agent.py`
**Purpose**: Agent using multiple MCP tools together
**Features**:
- Filesystem + web search
- Research web → save to file
- Complex multi-tool workflows

**Example Usage**:
```bash
uv run python examples/02-mcp/combined_tools_agent.py
# Agent: "Research LangGraph features and save a summary to research.md"
```
**Timeline**: 1.5 hours

### 📝 15. Create Milestone 2 README
**File**: `examples/02-mcp/README.md`
**Content**: Overview, prerequisites, usage instructions, expected output
**Timeline**: 30 min

---

## Phase 4: Milestone 3 - Multi-Agent Orchestration (Priority: P1)
**Timeline**: 4-5 hours

### 📝 16. Sequential Research Pipeline Example
**File**: `examples/03-multi-agent/research_pipeline.py`
**Purpose**: Demonstrate sequential agent workflow
**Pattern**: Researcher → Analyzer → Summarizer
**Features**:
- LangGraph state machine
- Sequential node execution
- State persistence

**Example Output**: Research report generated through 3-stage pipeline
**Timeline**: 1.5 hours

### 📝 17. Parallel Model Comparison Example
**File**: `examples/03-multi-agent/parallel_comparison.py`
**Purpose**: Demonstrate parallel execution with LangGraph Send API
**Pattern**: Launch 3 models in parallel → Merge results
**Features**:
- Parallel fan-out
- Result aggregation
- Performance comparison

**Example Output**: Side-by-side comparison of qwen3:8b, gemma3:4b, qwen3:30b-a3b
**Timeline**: 1.5 hours

### 📝 18. Code Review Pipeline Example
**File**: `examples/03-multi-agent/code_review_pipeline.py`
**Purpose**: Multi-agent code review workflow
**Pattern**: Code Analyzer → Reviewer → Documenter
**Features**:
- Conditional routing
- Quality checks
- Automated documentation generation

**Timeline**: 1.5 hours

### 📝 19. Create Milestone 3 README
**File**: `examples/03-multi-agent/README.md`
**Content**: Orchestration patterns explained, diagrams, usage
**Timeline**: 30 min

---

## Phase 5: Milestone 4 - RAG Systems (Priority: P1)
**Timeline**: 4-5 hours

### 📝 20. Document QA Example
**File**: `examples/04-rag/document_qa.py`
**Purpose**: Question answering over PDF documents
**Features**:
- PDF ingestion
- Chroma vector store
- qwen3-embedding
- Retrieval-augmented generation

**Example Usage**:
```bash
uv run python examples/04-rag/document_qa.py path/to/document.pdf
# Ask: "What are the main findings?"
```
**Timeline**: 1.5 hours

### 📝 21. Codebase Search Example
**File**: `examples/04-rag/codebase_search.py`
**Purpose**: Semantic search over code repository
**Features**:
- Code-aware chunking
- Repository indexing
- Function/class finding
- Code explanation

**Example Usage**:
```bash
uv run python examples/04-rag/codebase_search.py ./project
# Ask: "Where is the authentication logic?"
```
**Timeline**: 2 hours

### 📝 22. Multimodal RAG Example (Optional)
**File**: `examples/04-rag/multimodal_rag.py`
**Purpose**: RAG with text + images using qwen3-vl
**Features**:
- Image description generation
- Text + image indexing
- Multimodal retrieval

**Timeline**: 2 hours (if time permits)

### 📝 23. Create Milestone 4 README
**File**: `examples/04-rag/README.md`
**Content**: RAG concepts, chunking strategies, usage examples
**Timeline**: 30 min

---

## Phase 6: Documentation & Polish (Priority: P2)
**Timeline**: 4-5 hours

### 📝 24. Create Comprehensive Main README
**File**: `README.md`
**Sections**:
- Project overview
- Quick start (< 5 min setup)
- Installation guide
- Example directory tour
- Model recommendations
- Troubleshooting
- Architecture overview

**Timeline**: 2 hours

### 📝 25. Create Agent Tool Usage Guide
**File**: `docs/agent-tool-usage.md`
**Content**:
- How agents use utilities
- Code patterns
- Integration examples
- Best practices

**Timeline**: 1 hour

### 📝 26. Create Agent Coordination Guide
**File**: `docs/agent-coordination-patterns.md`
**Content**:
- Sequential pipelines
- Parallel execution
- Hierarchical delegation
- Human-in-the-loop
- Real examples from milestone 3

**Timeline**: 1 hour

### 📝 27. Update CLAUDE.md with Current State
**File**: `CLAUDE.md`
**Updates**:
- Remove Laravel references
- Update with built utilities
- Add agent usage instructions
- Update model references to Claude Sonnet 4.5

**Timeline**: 30 min

### 📝 28. Create Architecture Decision Records
**Directory**: `docs/adr/`
**Files**:
- `001-local-first-architecture.md`
- `002-langgraph-choice.md`
- `003-mcp-protocol.md`
- `004-vector-store-selection.md`

**Timeline**: 1 hour

---

## Phase 7: Testing & Validation (Priority: P2)
**Timeline**: 5-6 hours

### 📝 29. Create MCP Server Tests
**File**: `tests/test_mcp_servers.py`
**Coverage**:
- Filesystem server operations
- Web search functionality
- Error handling
- Connection management

**Timeline**: 1.5 hours

### 📝 30. Create Utility Tests
**Files**:
- `tests/test_ollama_manager.py`
- `tests/test_mcp_client.py`
- `tests/test_vector_store.py`
- `tests/test_state_manager.py`

**Timeline**: 2 hours

### 📝 31. Create Example Integration Tests
**File**: `tests/test_examples.py`
**Purpose**: Ensure all examples run successfully
**Approach**: Smoke tests with mock responses
**Timeline**: 1.5 hours

### 📝 32. Create Performance Benchmarks
**File**: `tests/benchmarks/model_performance.py`
**Metrics**: Speed, memory, quality for different models
**Timeline**: 1.5 hours

---

## Phase 8: Advanced Features (Priority: P3)
**Timeline**: 6-8 hours (optional)

### 📝 33. Milestone 5 - Interpretability Examples
**File**: `examples/05-interpretability/attention_viz.ipynb`
**Purpose**: Jupyter notebook for attention visualization
**Features**: TransformerLens integration, interactive plots
**Timeline**: 2 hours

### 📝 34. Milestone 6 - Production Patterns
**File**: `examples/06-production/production_agent.py`
**Purpose**: Production-ready agent template
**Features**: Error handling, logging, monitoring, config management
**Timeline**: 2 hours

### 📝 35. LangGraph Studio Integration
**Action**: Setup visual workflow editor
**Files**: Configuration, example workflows
**Timeline**: 1.5 hours

### 📝 36. Observability Setup
**Action**: LangSmith or local tracing alternative
**Timeline**: 2 hours

---

## 📊 Implementation Summary

| Phase | Tasks | Est. Time | Priority | Focus |
|-------|-------|-----------|----------|-------|
| **Phase 1: Agents** | 5 | 2-3h | P0 | Complete architecture |
| **Phase 2: Utilities** | 6 | 6-8h | P1 | Core infrastructure |
| **Phase 3: Milestone 2** | 4 | 3-4h | P1 | MCP examples |
| **Phase 4: Milestone 3** | 4 | 4-5h | P1 | Multi-agent |
| **Phase 5: Milestone 4** | 4 | 4-5h | P1 | RAG systems |
| **Phase 6: Documentation** | 5 | 4-5h | P2 | Comprehensive docs |
| **Phase 7: Testing** | 4 | 5-6h | P2 | Quality assurance |
| **Phase 8: Advanced** | 4 | 6-8h | P3 | Optional extras |
| **TOTAL** | **36 tasks** | **35-44h** | - | - |

---

## 🎯 Recommended Execution Order

### Week 1 (8-10 hours)
**Goal**: Complete foundation and first working examples

1. ✅ Complete agent architecture (Phase 1: 2-3h)
2. ✅ Build core utilities (Phase 2: 6-8h)
3. ✅ Start with ollama_manager and mcp_client first

**Deliverable**: Working utilities that agents can use

### Week 2 (10-12 hours)
**Goal**: Milestone 2 & 3 complete

1. ✅ Complete Milestone 2 - MCP examples (Phase 3: 3-4h)
2. ✅ Complete Milestone 3 - Multi-agent (Phase 4: 4-5h)
3. ✅ Start documentation (Phase 6: 2-3h)

**Deliverable**: 7 working examples demonstrating core capabilities

### Week 3 (8-10 hours)
**Goal**: RAG and documentation complete

1. ✅ Complete Milestone 4 - RAG (Phase 5: 4-5h)
2. ✅ Complete documentation (Phase 6: 4-5h)
3. ✅ Create main README

**Deliverable**: Complete documentation, 10+ examples

### Week 4 (8-10 hours)
**Goal**: Testing and polish

1. ✅ Complete testing suite (Phase 7: 5-6h)
2. ✅ Advanced features as time permits (Phase 8: 3-4h)
3. ✅ Final polish and validation

**Deliverable**: Production-ready toolkit with tests

---

## 🔧 Technical Decisions

### Local Models (Ollama)
- **Primary**: qwen3:8b (balanced)
- **Fast**: qwen3:30b-a3b (MoE)
- **Embeddings**: qwen3-embedding
- **Vision**: qwen3-vl:8b
- **Alternative**: gemma3:4b (resource-constrained)

### Vector Stores
- **Primary**: Chroma (local-first, easy persistence)
- **Alternative**: FAISS (faster for large datasets)

### State Management
- SQLite via LangGraph checkpointers
- JSON for simple config

### Testing
- **Framework**: pytest
- **Coverage Target**: 80%+ for core utilities
- **Integration**: Smoke tests with mocks

---

## ✅ Success Criteria

### Technical
- [ ] All utilities working and tested
- [ ] 10+ runnable examples
- [ ] All examples documented
- [ ] < 5 min setup for new users
- [ ] Tests passing with 80%+ coverage

### Documentation
- [ ] README comprehensive and clear
- [ ] Every example has usage instructions
- [ ] Troubleshooting guide complete
- [ ] Architecture documented

### Code Quality
- [ ] Type hints throughout
- [ ] Docstrings for all functions
- [ ] Follows PEP 8
- [ ] No hardcoded credentials
- [ ] Proper error handling

---

## 🚀 Getting Started Right Now

### Immediate Actions (Next 30 minutes)

1. **Complete interpretability agent** (10 min)
2. **Create example-creator agent** (20 min)

### First Utility to Build (Next 1.5 hours)

**`utils/ollama_manager.py`** - This unblocks everything else

```python
# Skeleton to start with
class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def check_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        pass

    def ensure_model_available(self, model: str) -> bool:
        """Ensure model is available, pull if needed"""
        pass

    # ... implement other methods
```

### First Example to Build (Next hour after utilities)

**`examples/02-mcp/filesystem_agent.py`** - Demonstrates MCP integration

---

## 📝 Notes

### Project Principles
1. **Local-First**: Everything runs on-device
2. **No Cloud Dependencies**: Ollama + local tools only
3. **Educational**: Clear examples and documentation
4. **Production-Quality**: Type hints, tests, error handling

### Common Pitfalls to Avoid
- ❌ Don't assume Ollama is running - check first
- ❌ Don't hardcode paths - use config
- ❌ Don't skip docstrings - document everything
- ❌ Don't create examples without README
- ❌ Don't build utilities without tests

### Model Context Update
**Important**: When this plan references "Claude models" in documentation, we're referring to:
- **Claude Sonnet 4.5** (primary, smartest)
- **Claude Haiku 4.5** (fastest)
- **Claude Opus 4.1** (specialized reasoning)

These are the AI assistants (not local Ollama models). Update all documentation to reflect current Claude model lineup.

---

## 📚 Resources

### Official Documentation
- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Ollama: https://ollama.com/
- Chroma: https://docs.trychroma.com/
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens

### Project Files
- Previous plan: `docs/DEVELOPMENT-PLAN-20-POINTS.md`
- Research: `plans/1-research-plan.md`
- Use cases: `plans/3-kitchen-sink-plan.md`
- Agents: `.claude/agents/`

---

**End of Development Plan Phase 2**

This plan provides a **clear, actionable roadmap** for the next 4 weeks of development. Start with Phase 1 to complete the foundation, then build utilities and examples in sequence. Each phase builds on the previous, creating a complete local-first AI experimentation toolkit.

**Ready to begin? Start with Task #1: Complete the interpretability researcher agent.**
