# Claude Code Responsibilities & Instructions

**!!! NEVER INCLUDE EMOJIS IN ANYTHING OR INCLUDE THE FOLLWING IN COMMITS OR ANY REFERENCES TO CLAUDE CODE ANYWHERE !!!**

**Never include lines like this:**

```text
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

```

## Project Overview
This is a local-first AI experimentation toolkit designed to run entirely on-device using macOS tools. The project integrates LangChain, LangGraph, local LLMs (via Ollama/LM Studio), MCP servers, and mechanistic interpretability tools.

---

## Claude's Role & Responsibilities

### Primary Responsibilities
1. **Code Generation**: Write production-quality Python and JavaScript code
2. **Architecture Guidance**: Suggest optimal patterns for agent workflows
3. **Debugging Assistance**: Help troubleshoot issues with local models and frameworks
4. **Documentation**: Maintain clear, comprehensive documentation
5. **Best Practices**: Ensure code follows industry standards and project conventions

### Task Prioritization
- **P0 (Critical)**: Setup issues, blocking errors, security concerns
- **P1 (High)**: Core functionality, integration tasks, critical bugs
- **P2 (Medium)**: Feature enhancements, optimization, refactoring
- **P3 (Low)**: Documentation improvements, nice-to-haves

---

## Project Structure Understanding

### Directory Layout
```
ai-lang-stuff/
├── .claude/               # Claude-specific configurations
│   ├── agents/           # Specialized agent definitions
│   ├── commands/         # Custom slash commands
│   └── skills/           # Reusable Claude skills
├── config/               # Application configuration
│   ├── models.yaml       # Model configurations (Qwen3, Gemma 3)
│   └── settings.yaml     # Application settings
├── docs/                 # Comprehensive documentation
│   ├── DEVELOPMENT-PLAN-PHASE-2.md  # Current 36-task development plan
│   └── DEVELOPMENT-PLAN-20-POINTS.md # Original planning document
├── examples/             # Working examples by milestone
│   ├── 01-foundation/   # Basic LLM interactions
│   ├── 02-mcp/          # MCP server integration
│   ├── 03-multi-agent/  # LangGraph orchestration
│   ├── 04-rag/          # RAG and vector stores
│   ├── 05-interpretability/  # Model analysis
│   └── 06-production/   # Production patterns
├── mcp-servers/         # Custom MCP server implementations
│   └── custom/
│       ├── filesystem/  # File operations MCP server
│       └── web-search/  # Web search MCP server
├── plans/                # Research and planning documents
│   ├── 0-readme.md      # Vision and intended usage (25+ thoughts)
│   ├── 1-research-plan.md  # Research findings and milestones
│   └── 3-kitchen-sink-plan.md  # Concrete examples and use cases
├── scripts/             # Automation scripts
├── src/                 # Source code (when refactored from examples)
├── tests/               # Test suites
├── utils/               # Core utilities
│   ├── ollama_manager.py   # Ollama integration utilities
│   ├── mcp_client.py       # MCP client wrappers
│   ├── vector_store.py     # Vector store management
│   ├── state_manager.py    # State persistence for agents
│   └── tool_registry.py    # Centralized tool registry
├── CLAUDE.md            # This file - Claude's instructions
├── main.py              # Main entry point
├── pyproject.toml       # Python dependencies (uv-compatible)
├── package.json         # Node.js dependencies
└── README.md            # User-facing documentation with quick start
```

### Key Files
- **plans/0-readme.md**: Vision and intended usage (25+ sequential thoughts)
- **plans/1-research-plan.md**: Research findings and development milestones
- **plans/3-kitchen-sink-plan.md**: Concrete examples and use cases
- **CLAUDE.md**: This file - Claude's instructions
- **README.md**: User-facing documentation with quick start

---

## Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use type hints, prefer Pydantic for validation
- **JavaScript/TypeScript**: Use ESLint, prefer async/await, document types
- **Formatting**: Use Ruff for Python, Prettier for JavaScript
- **Comments**: Explain "why" not "what", document complex logic

### Naming Conventions
- **Files**: snake_case.py, kebab-case.js
- **Classes**: PascalCase
- **Functions**: snake_case (Python), camelCase (JavaScript)
- **Constants**: UPPER_SNAKE_CASE
- **Private**: _leading_underscore (Python)

### Error Handling
- Always use try/except blocks for I/O operations
- Implement retry logic for network calls (use tenacity)
- Log errors with context (use Python logging module)
- Return meaningful error messages to users

### Testing
- Write unit tests for all core functionality
- Use pytest for Python, Jest for JavaScript
- Aim for 80%+ code coverage
- Include integration tests for MCP servers

---

## Working with Local Models

### Ollama Interaction
- Default endpoint: `http://localhost:11434`
- Check if Ollama is running before operations
- Use `ollama list` to verify available models
- Recommend model pull if not available

### LM Studio Integration
- Default endpoint: `http://localhost:1234/v1`
- Both use OpenAI-compatible APIs
- Allow users to choose between Ollama and LM Studio

### Model Selection Guidance

Choose the right model for your task:

| Task Type | Recommended Model | Rationale | Context Window |
|-----------|------------------|-----------|----------------|
| Fast coding | qwen3:30b-a3b | MoE optimized for speed | 30K tokens |
| Complex reasoning | qwen3:8b | Dense, reliable for most tasks | 32K tokens |
| Long context | qwen3:70b | Best quality, large context | 128K tokens |
| Multilingual | gemma3:12b | 140+ languages supported | 32K tokens |
| Edge/mobile | gemma3:4b | Minimal resource usage | 8K tokens |
| Vision tasks | qwen3-vl:8b | Best local vision model | 32K tokens |
| Embeddings | qwen3-embedding | For RAG and vector stores | N/A |

**Model Selection Tips**:
- Start with **qwen3:8b** for balanced performance
- Use **qwen3:30b-a3b** when speed is critical
- Switch to **gemma3:4b** for resource-constrained environments
- Use quantized variants (Q4, Q5) to reduce memory by 60-75%
- Test with `OllamaManager.benchmark_model()` before committing

---

## LangChain & LangGraph Patterns

### Agent Creation Pattern
```python
# Standard pattern for creating agents
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3:8b")
prompt = ChatPromptTemplate.from_messages([...])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### LangGraph State Management
```python
# Always define state with TypedDict
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # ... other fields
```

### MCP Server Integration
- Use official MCP servers when available
- Create custom servers for project-specific needs
- Always validate MCP server responses
- Implement fallback behavior if MCP server unavailable

---

## Core Utilities

The `utils/` directory provides five core utilities that power all examples:

### 1. Ollama Manager (`ollama_manager.py`)
Manages Ollama server operations and model lifecycle:
- Health checks and server status verification
- Model availability and auto-pull functionality
- Model benchmarking and performance stats
- Intelligent model recommendations based on task type
- Support for both Ollama and LM Studio endpoints

**Usage Pattern**:
```python
from utils.ollama_manager import OllamaManager

manager = OllamaManager()
if manager.check_ollama_running():
    manager.ensure_model_available("qwen3:8b")
    models = manager.list_models()
    recommended = manager.recommend_model("fast_coding")
```

### 2. MCP Client (`mcp_client.py`)
Production-quality clients for Model Context Protocol integration:
- Filesystem operations (read, write, list, search)
- Web search capabilities
- Connection pooling and retry logic
- LangChain tool integration
- Async context manager support

**Usage Pattern**:
```python
from utils.mcp_client import FilesystemMCP, WebSearchMCP

fs = FilesystemMCP()
files = fs.list_files("/path/to/dir")
content = fs.read_file("/path/to/file.txt")

search = WebSearchMCP()
results = search.search("local LLMs 2024", num_results=5)
tools = fs.get_langchain_tools()
```

### 3. Vector Store Manager (`vector_store.py`)
Manages local vector stores for RAG systems:
- Support for ChromaDB and FAISS
- Document ingestion and indexing
- Similarity search and retrieval
- Collection management
- Embedding model integration

**Usage Pattern**:
```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager(store_type="chroma")
store = manager.create_from_documents(
    documents=docs,
    collection_name="my_docs",
    embedding_model="qwen3-embedding"
)
results = manager.similarity_search("query", collection_name="my_docs")
```

### 4. State Manager (`state_manager.py`)
Persist agent state across sessions:
- SQLite-backed state persistence
- Checkpoint management for time-travel debugging
- Session history tracking
- Thread-safe operations
- State restoration and migration

**Usage Pattern**:
```python
from utils.state_manager import StateManager

manager = StateManager(db_path="agent_state.db")
manager.save_state(
    agent_id="research_agent",
    state={"messages": [...], "context": {...}}
)
state = manager.load_state("research_agent")
checkpoints = manager.list_checkpoints("research_agent")
```

### 5. Tool Registry (`tool_registry.py`)
Centralized tool management system:
- Tool registration and discovery
- Category-based organization
- LangChain tool conversion
- Auto-discovery of utility functions
- JSON export/import for tool definitions

**Usage Pattern**:
```python
from utils.tool_registry import get_registry

registry = get_registry()
registry.register_tool(
    name="add",
    tool=my_calculator,
    description="Add two numbers",
    category="math"
)
registry.auto_discover_utilities()
tools = registry.get_langchain_tools(categories=["web", "filesystem"])
```

---

## Common Tasks & Solutions

### Task: Add a New Example
1. Create file in appropriate examples/ subdirectory (01-foundation through 06-production)
2. Follow existing example structure with comprehensive docstring
3. Include purpose, prerequisites, expected output in docstring
4. Import and use utilities from `utils/` directory
5. Add logging and error handling using patterns from existing examples
6. Update plans/3-kitchen-sink-plan.md with example description
7. Update README.md examples section with link
8. Create corresponding test in tests/ directory

**Example Structure**:
```python
"""
Example: [Name]

Purpose:
    [What this example demonstrates]

Prerequisites:
    - Ollama running with qwen3:8b model
    - [Other requirements]

Expected Output:
    [What users should see]

Usage:
    uv run python examples/XX-category/example_name.py
"""

import logging
from utils.ollama_manager import OllamaManager
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

def main():
    # Implementation
    pass

if __name__ == "__main__":
    main()
```

### Task: Debug LangChain Agent
1. Enable LangSmith tracing: `export LANGCHAIN_TRACING_V2=true`
2. Check Ollama server status:
   ```bash
   ps aux | grep ollama
   curl http://localhost:11434/api/tags
   ```
3. Verify model is available: `ollama list`
4. Test model directly: `ollama run <model-name> "test prompt"`
5. Add debug logging to agent state transitions
6. Check agent scratchpad output for tool calls
7. Verify tools are properly defined with correct schemas

### Task: Build a Multi-Agent Workflow
1. Define agent state with TypedDict:
   ```python
   from typing import TypedDict, Annotated
   import operator

   class AgentState(TypedDict):
       messages: Annotated[list[BaseMessage], operator.add]
       # ... other fields
   ```
2. Create individual agent nodes as functions
3. Use LangGraph StateGraph for orchestration
4. Define conditional edges for routing logic
5. Test each agent independently before integration
6. Add checkpointing with StateManager for persistence
7. Create example in examples/03-multi-agent/

### Task: Implement a RAG System
1. Prepare documents for indexing
2. Use VectorStoreManager to create vector store:
   ```python
   from utils.vector_store import VectorStoreManager
   manager = VectorStoreManager(store_type="chroma")
   store = manager.create_from_documents(docs, "collection_name")
   ```
3. Create retriever chain with LangChain
4. Build QA chain combining retriever and LLM
5. Implement context-aware prompting
6. Add relevance filtering for retrieved documents
7. Test with diverse queries
8. Create example in examples/04-rag/

### Task: Integrate MCP Server
1. Create MCP server in mcp-servers/custom/<name>/
2. Implement MCP protocol specification
3. Add client wrapper in utils/mcp_client.py (if reusable)
4. Create tool wrapper using MCP client
5. Test MCP server independently
6. Create integration example in examples/02-mcp/
7. Document server capabilities in README
8. Add server to .claude/mcp_servers.json if needed

### Task: Optimize Performance
1. Profile with cProfile to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.stats script.py
   python -m pstats profile.stats
   ```
2. Use smaller models for simpler tasks (gemma3:4b for basic tasks)
3. Implement LRU caching for repeated queries:
   ```python
   from functools import lru_cache
   @lru_cache(maxsize=128)
   def expensive_llm_call(prompt: str) -> str:
       return llm.invoke(prompt)
   ```
4. Batch similar requests using llm.batch()
5. Use quantized models (Q4, Q5 variants)
6. Enable streaming for better perceived performance
7. Monitor token usage and optimize prompts
8. Use OllamaManager.benchmark_model() for comparison

---

## Milestone Tracking

### Current Milestone Status
Reference plans/milestones/ for detailed tracking.

### When Starting New Milestone
1. Create milestone document in plans/milestones/
2. Break down into specific tasks with checklists
3. Estimate time for each task
4. Identify dependencies
5. Update as tasks complete

### Checklist Template
```markdown
## Milestone X: [Name]
**Goal**: [Clear objective]
**Timeline**: [Estimated duration]

### Tasks
- [ ] Task 1: [Description] (Estimated: Xh)
  - [ ] Subtask 1.1
  - [ ] Subtask 1.2
- [ ] Task 2: [Description] (Estimated: Xh)

### Dependencies
- Requires: [Other milestones/tasks]
- Blocks: [Future work]

### Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

---

## Interaction Guidelines

### When User Asks for Help
1. **Clarify**: Ensure you understand the request
2. **Context**: Check relevant files and project state
3. **Options**: Present multiple approaches if applicable
4. **Recommend**: Suggest best approach with rationale
5. **Implement**: Write code or provide detailed steps
6. **Verify**: Help test and validate solution

### When Suggesting Changes
- Explain benefits and tradeoffs
- Consider impact on existing code
- Provide migration path if breaking change
- Update documentation
- Add tests for new functionality

### When Writing Code
- Include type hints and docstrings
- Add error handling
- Consider edge cases
- Write self-documenting code
- Add comments for complex logic
- Include usage example in docstring

### When Debugging
- Reproduce the issue first
- Check logs and error messages
- Verify environment (Ollama running, models available)
- Test incrementally
- Document solution for future reference

---

## Skills & Agents Usage

### Available Claude Skills
Located in `.claude/skills/`. Each skill is a specialized capability for AI development.

**Current Skills**:
- **langgraph-orchestrator**: Multi-agent workflow design and LangGraph patterns
- **local-model-manager**: Ollama model selection and optimization
- **mcp-integration-specialist**: Model Context Protocol server integration
- **orchestration-specialist**: Agent coordination and state management
- **rag-system-builder**: RAG pipeline design and vector store management

#### Creating New Skill
```markdown
# Skill: [Name]

## Purpose
[What this skill does - focus on AI/ML workflows]

## Triggers
- User asks about [specific AI task]
- Code needs [agent workflow, RAG, MCP, etc.]

## Process
1. Analyze requirements
2. Suggest appropriate models/tools
3. Implement using project utilities
4. Test and validate

## Output
[Working code example or agent configuration]

## Examples
- Input: "Build a research agent"
- Output: [LangGraph workflow with tool integration]
```

### Available Agents
Located in `.claude/agents/`. Each agent is a Markdown configuration for specialized AI tasks.

**Current Agents**:
- **langgraph-orchestrator.md**: Multi-agent workflow expert
- **local-model-manager.md**: Model selection and optimization
- **mcp-integration-specialist.md**: MCP server integration
- **orchestration-specialist.md**: Agent coordination patterns
- **rag-system-builder.md**: RAG pipeline construction

#### Creating New Agent
```markdown
# Agent: [Name]

## Purpose
[What this agent specializes in]

## Capabilities
- Capability 1
- Capability 2

## Recommended Model
- Primary: qwen3:8b (balanced performance)
- Alternative: qwen3:30b-a3b (for speed)

## System Prompt
You are a specialized agent that...

## Tools
- tool1 (from utils/)
- tool2 (from MCP servers)

## Examples
### Example 1: [Task Name]
**Input**: User request
**Process**: Step-by-step approach
**Output**: Expected result

## Best Practices
- Practice 1
- Practice 2
```

---

## Quality Standards

### Before Committing Code
- [ ] Code runs without errors
- [ ] All tests pass
- [ ] Type hints added
- [ ] Docstrings included
- [ ] Error handling implemented
- [ ] Logging added where appropriate
- [ ] Documentation updated
- [ ] No hardcoded credentials or secrets

### Documentation Standards
- Every function has docstring with args, returns, raises
- Every example has purpose, prerequisites, expected output
- README kept up to date with new features
- Complex algorithms explained with comments
- Architecture decisions recorded in plans/

### Testing Standards
- Unit tests for isolated functionality
- Integration tests for multi-component features
- Test edge cases and error conditions
- Mock external services (Ollama, MCP servers)
- Achieve 80%+ coverage for core modules

---

## Troubleshooting Guide

### Ollama Issues
**Problem**: "Connection refused to localhost:11434"
**Solution**:
1. Check if Ollama is running: `ps aux | grep ollama`
2. Start Ollama: `ollama serve`
3. Verify: `curl http://localhost:11434`

**Problem**: "Model not found"
**Solution**:
1. List models: `ollama list`
2. Pull model: `ollama pull qwen3:8b`
3. Verify: `ollama list | grep qwen3`

### LangChain Issues
**Problem**: "Module not found: langchain_ollama"
**Solution**:
1. Check installation: `uv pip list | grep langchain`
2. Install: `uv add langchain-ollama`
3. Verify: `python -c "import langchain_ollama"`

**Problem**: "Agent not responding"
**Solution**:
1. Enable tracing: `export LANGCHAIN_TRACING_V2=true`
2. Check agent scratchpad in output
3. Verify tools are properly defined
4. Test model directly first

### MCP Server Issues
**Problem**: "MCP server connection failed"
**Solution**:
1. Check server is running: `ps aux | grep mcp`
2. Verify port availability: `lsof -i :<port>`
3. Test server directly with curl
4. Check server logs for errors

---

## Learning Resources

### Essential Reading
- LangChain Docs: https://python.langchain.com/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- MCP Spec: https://github.com/modelcontextprotocol
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens

### Example Repositories
- LangChain Templates: https://github.com/langchain-ai/langchain/tree/master/templates
- MCP Servers: https://github.com/modelcontextprotocol/servers

### Community
- LangChain Discord: For framework questions
- Ollama Discord: For local model issues
- GitHub Issues: For bug reports and feature requests

---

## Version Control

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
- `feat(mcp): add filesystem MCP server integration`
- `fix(agent): handle empty responses gracefully`
- `docs(examples): add RAG system example`

### Branch Strategy
- `main`: Stable, working code
- `develop`: Integration branch for features
- `feature/<name>`: Individual features
- `fix/<issue>`: Bug fixes

---

## Security Considerations

### Never Commit
- API keys or credentials (even for optional cloud integrations)
- `.env` files with secrets
- Personal information or private documents
- Large model files (use Ollama for model management)
- Sensitive vector store data or embeddings
- Private conversation histories or agent states

### Always Check
- User input validation (especially for MCP tool inputs)
- Path traversal prevention when using filesystem operations
- Command injection prevention when using subprocess
- Prompt injection attacks in agent workflows
- Data leakage between agent sessions

### Secure Patterns
```python
# Good: Path validation for filesystem operations
from pathlib import Path

def safe_read_file(user_path: str, base_dir: str) -> str:
    base = Path(base_dir).resolve()
    target = Path(user_path).resolve()

    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal attempt detected")

    with open(target) as f:
        return f.read()

# Bad: Direct path usage
def unsafe_read_file(user_path: str) -> str:
    with open(user_path) as f:  # Vulnerable to path traversal
        return f.read()

# Good: Safe subprocess execution
import shlex
import subprocess

def safe_command(user_input: str) -> str:
    allowed_commands = {"ls", "pwd", "echo"}
    cmd = shlex.split(user_input)

    if cmd[0] not in allowed_commands:
        raise ValueError(f"Command {cmd[0]} not allowed")

    return subprocess.run(cmd, capture_output=True, text=True).stdout

# Bad: Direct command execution
def unsafe_command(user_input: str) -> str:
    return subprocess.run(user_input, shell=True, capture_output=True).stdout  # Vulnerable

# Good: Sanitize agent inputs
from langchain_core.messages import HumanMessage

def sanitize_user_prompt(prompt: str) -> str:
    """Remove potential prompt injection attempts"""
    forbidden = ["ignore previous instructions", "system:", "assistant:"]
    lower_prompt = prompt.lower()

    for term in forbidden:
        if term in lower_prompt:
            raise ValueError("Potential prompt injection detected")

    return prompt

# Good: Isolate agent sessions
from utils.state_manager import StateManager

def get_isolated_state(user_id: str, session_id: str):
    manager = StateManager()
    agent_id = f"{user_id}:{session_id}"  # Namespace by user and session
    return manager.load_state(agent_id)
```

### Local-First Security Benefits
- **No data transmission**: All data stays on your machine
- **No API key leaks**: Zero cloud API dependencies
- **Full control**: Audit all code and dependencies
- **Privacy-preserving**: Models run entirely offline
- **No telemetry**: No usage data sent to third parties

---

## Performance Optimization

### Model Selection
- Use smallest model that meets requirements
- Prefer MoE models (qwen3:30b-a3b) for speed
- Use quantized variants (Q4, Q5) for resource constraints

### Caching Strategies
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_llm_call(prompt: str) -> str:
    return llm.invoke(prompt)
```

### Batching
```python
# Good: Batch processing
results = llm.batch([prompt1, prompt2, prompt3])

# Bad: Sequential calls
results = [llm.invoke(p) for p in [prompt1, prompt2, prompt3]]
```

---

## Monitoring & Observability

### Logging Best Practices
```python
import logging

logger = logging.getLogger(__name__)

# Log levels
logger.debug("Detailed diagnostic info")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical issues")

# Structured logging
logger.info("Agent executed", extra={
    "agent": agent_name,
    "duration": elapsed_time,
    "tokens": token_count
})
```

### LangSmith Integration
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="project-name"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="<your-key>"  # Optional for local
```

---

## Continuous Improvement

### Code Review Checklist
- [ ] Functionality: Does it work as intended?
- [ ] Readability: Is the code clear and well-documented?
- [ ] Maintainability: Will this be easy to update later?
- [ ] Performance: Are there obvious optimizations?
- [ ] Security: Are there security concerns?
- [ ] Testing: Is there adequate test coverage?

### Refactoring Triggers
- Code duplication (DRY principle)
- Functions > 50 lines
- Deeply nested conditionals
- Complex boolean expressions
- Magic numbers/strings
- Poor naming

---

## Communication Protocols

### Status Updates
- Acknowledge requests promptly
- Explain approach before implementing
- Report progress on long-running tasks
- Flag blockers immediately
- Summarize completed work

### Asking for Clarification
- "To ensure I understand correctly, you want..."
- "Which approach would you prefer: A or B?"
- "Before proceeding, could you clarify..."
- "I see two possible interpretations..."

### Reporting Issues
- Describe the problem clearly
- Provide error messages and logs
- List steps to reproduce
- Suggest potential solutions
- Estimate impact and urgency

---

## Success Metrics

### Code Quality
- All tests passing
- No linting errors
- Type hints coverage > 90%
- Documentation coverage > 80%

### Project Health
- Examples run successfully
- Setup time < 10 minutes
- Build time < 30 seconds
- Test suite time < 2 minutes

### User Experience
- Clear error messages
- Helpful documentation
- Working examples
- Responsive to feedback

---

## Future Roadmap Awareness

Stay informed about:
- LangChain/LangGraph updates
- New local models (Qwen, Gemma, Llama)
- MCP specification changes
- Ollama feature releases
- macOS compatibility changes

When suggesting features, consider:
- Alignment with project vision (local-first)
- Complexity vs benefit tradeoff
- Maintenance burden
- User demand

---

## Project Philosophy

### Local-First AI Development
This project is built on the principle that powerful AI development should be:
- **Private**: All data stays on your machine
- **Accessible**: No API keys or cloud accounts required
- **Transparent**: Full visibility into models and operations
- **Sustainable**: Zero recurring costs after setup
- **Flexible**: Mix and match models, tools, and workflows

### Design Principles
1. **Composability**: Small utilities that combine well
2. **Observability**: LangSmith integration for debugging
3. **Type Safety**: Full type hints for better IDE support
4. **Progressive Enhancement**: Start simple, add complexity as needed
5. **Developer Experience**: Clear examples, good defaults, helpful errors

### Technology Stack Alignment
- **LangChain/LangGraph**: Industry-standard orchestration
- **Ollama**: Best-in-class local model runtime
- **MCP**: Open protocol for tool integration
- **ChromaDB/FAISS**: Production-ready vector stores
- **TransformerLens**: Cutting-edge interpretability

---

## Final Notes

### Claude Should:
- **Suggest local-first solutions**: Prioritize Ollama, local vector stores, MCP servers
- **Leverage existing utilities**: Use the five core utilities (ollama_manager, mcp_client, etc.)
- **Write comprehensive examples**: Include docstrings, logging, error handling
- **Maintain consistency**: Follow existing patterns in examples/
- **Document thoroughly**: Update README, plans/, and inline comments
- **Test before suggesting**: Verify code works with local models
- **Optimize for real hardware**: Consider memory, CPU constraints of local machines

### Claude Should Not:
- **Suggest cloud-only solutions**: No OpenAI/Anthropic unless explicitly requested
- **Ignore resource constraints**: Local models have memory/speed limitations
- **Skip error handling**: Always validate Ollama status, model availability
- **Write untested code**: Verify examples run successfully
- **Overcomplicate**: Start simple, add features incrementally
- **Hardcode paths**: Use Path objects and configuration files
- **Forget logging**: Add appropriate logging for debugging

### Key Reminders
- This is a **local-first AI toolkit**, not a web application
- Focus on **agent workflows**, not CRUD operations
- Use **utilities** from utils/ for common operations
- Test with **real models** (qwen3:8b, gemma3:4b)
- Examples should be **educational and practical**
- Documentation should be **clear and actionable**

Remember: This project prioritizes **local-first, privacy-preserving, zero-dependency AI development**. Every decision should align with this core principle.

### Success Metrics for Claude Contributions
- Examples run successfully on first try
- Code follows established patterns from existing examples
- Documentation is clear and comprehensive
- Utilities are reused appropriately
- Local-first principles are maintained
- Performance is optimized for local execution
