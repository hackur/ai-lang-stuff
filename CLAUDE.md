# Claude Code Responsibilities & Instructions

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
│   ├── skills/           # Reusable Claude skills
│   └── agents/           # Agent configurations
├── plans/                # Planning and design documents
│   ├── milestones/      # Milestone tracking
│   └── checklists/      # Detailed task checklists
├── config/               # Application configuration
├── examples/             # Example implementations
│   ├── 01-foundation/   # Basic LLM examples
│   ├── 02-mcp/          # MCP integration examples
│   ├── 03-multi-agent/  # LangGraph orchestration
│   ├── 04-rag/          # RAG and vision examples
│   ├── 05-interpretability/  # TransformerLens examples
│   └── 06-production/   # Production patterns
├── mcp-servers/         # Custom MCP server implementations
├── scripts/             # Automation scripts
├── tests/               # Test suites
├── main.py              # Main entry point
├── pyproject.toml       # Python dependencies
└── package.json         # Node.js dependencies
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
| Task Type | Recommended Model | Rationale |
|-----------|------------------|-----------|
| Fast coding | qwen3:30b-a3b | MoE optimized for speed |
| Complex reasoning | qwen3:8b | Dense, reliable |
| Multilingual | gemma3:12b | 140+ languages |
| Edge/mobile | gemma3:4b | Minimal resource usage |
| Vision tasks | qwen3-vl:8b | Best local vision model |

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

## Common Tasks & Solutions

### Task: Add a New Example
1. Create file in appropriate examples/ subdirectory
2. Follow existing example structure
3. Include docstring with purpose, prerequisites, expected output
4. Add to plans/3-kitchen-sink-plan.md
5. Update README.md with link
6. Create test in tests/ directory

### Task: Debug LangChain Agent
1. Enable LangSmith tracing: `export LANGCHAIN_TRACING_V2=true`
2. Check Ollama logs: `ollama logs`
3. Verify model is pulled: `ollama list`
4. Test model directly: `ollama run <model-name> "test prompt"`
5. Add debug prints for state transitions

### Task: Optimize Performance
1. Use smaller models for simpler tasks
2. Implement caching for repeated queries
3. Batch similar requests
4. Use quantized models (Q4, Q5)
5. Profile with cProfile, optimize bottlenecks

### Task: Add MCP Server
1. Create in mcp-servers/custom/<name>/
2. Implement MCP protocol specification
3. Add tool wrapper in examples/02-mcp/
4. Document in README
5. Test integration with agent

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
Located in `.claude/skills/`. Each skill is a specialized capability.

#### Creating New Skill
```markdown
# Skill: [Name]

## Purpose
[What this skill does]

## Triggers
- User asks about [topic]
- Code needs [specific functionality]

## Process
1. Step 1
2. Step 2
3. ...

## Output
[What the skill produces]
```

### Available Agents
Located in `.claude/agents/`. Each agent is a configuration for specialized tasks.

#### Creating New Agent
```yaml
# .claude/agents/[name].yaml
name: "Agent Name"
purpose: "What this agent does"
model: "qwen3:8b"
tools:
  - tool1
  - tool2
system_prompt: |
  You are a specialized agent that...
examples:
  - input: "Example input"
    output: "Example output"
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
- API keys or credentials
- `.env` files with secrets
- Personal information
- Large model files

### Always Check
- User input validation
- SQL injection prevention (if using databases)
- Path traversal prevention (if accessing files)
- Command injection prevention (if using subprocess)

### Secure Patterns
```python
# Good: Parameterized queries
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# Bad: String concatenation
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Good: Path validation
safe_path = Path(user_path).resolve()
if not str(safe_path).startswith(str(base_path)):
    raise ValueError("Invalid path")

# Bad: Direct path usage
with open(user_path) as f:
    content = f.read()
```

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

## Final Notes

Claude should:
- Be proactive in suggesting improvements
- Maintain consistency with existing code
- Prioritize user experience
- Document decisions and rationale
- Stay within project scope (local-first AI)

Claude should not:
- Suggest cloud-only solutions
- Ignore security considerations
- Skip documentation
- Write untested code
- Overcomplicate simple tasks

Remember: This project prioritizes **local-first, privacy-preserving, zero-dependency AI development**. Every decision should align with this core principle.
