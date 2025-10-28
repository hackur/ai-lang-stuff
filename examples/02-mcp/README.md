# MCP Integration Examples

This directory contains examples demonstrating Model Context Protocol (MCP) integration with LangChain agents and local LLMs. MCP is a universal standard created by Anthropic for connecting AI systems with external tools and data sources.

## Overview

MCP provides a standardized way to connect language models with external capabilities like:
- File system operations
- GitHub repository access
- Database queries
- Web scraping
- Custom tools and APIs

All examples run entirely locally using Ollama models, with no cloud dependencies required.

---

## Prerequisites

### 1. Ollama Installation and Setup

```bash
# Install Ollama
brew install ollama

# Start Ollama server (in separate terminal)
ollama serve

# Pull recommended models
ollama pull qwen3:8b          # Fast, reliable for tool use
ollama pull qwen3:30b-a3b     # Best for complex reasoning with tools
ollama pull gemma3:4b         # Lightweight for quick iterations
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 2. Python Dependencies

This project uses `uv` for fast, modern Python package management:

```bash
# Install core dependencies
uv add langchain langchain-ollama langchain-community
uv add langchain-core

# Install MCP client (if needed)
uv add httpx pydantic
```

### 3. MCP Servers

MCP servers provide the actual tool implementations. Install official servers:

```bash
# Filesystem operations
npm install -g @modelcontextprotocol/server-filesystem

# GitHub integration
npm install -g @modelcontextprotocol/server-github

# SQLite database access
npm install -g @modelcontextprotocol/server-sqlite

# Puppeteer (web scraping)
npm install -g @modelcontextprotocol/server-puppeteer
```

Verify installation:
```bash
which mcp-server-filesystem
```

---

## Available Examples

### Current Status
The MCP examples directory is currently being set up as part of Milestone 2. Examples will include:

1. **filesystem_agent.py** - Agent that searches and reads local files
2. **github_agent.py** - Agent that queries GitHub repositories
3. **database_agent.py** - SQL query generation and execution
4. **web_scraper_agent.py** - Web content extraction and analysis
5. **custom_mcp_server/** - Build your own MCP server

### Example Structure
Each example follows this pattern:

```python
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool

# 1. Initialize local LLM
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")

# 2. Define tools that wrap MCP server calls
tools = [...]

# 3. Create agent with prompt template
prompt = ChatPromptTemplate.from_messages([...])
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. Execute with AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "your query here"})
```

---

## Usage Instructions

### Basic Workflow

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run an example**:
   ```bash
   uv run python examples/02-mcp/filesystem_agent.py
   ```

3. **Watch the agent work**:
   - The agent will analyze your query
   - Determine which tools to use
   - Execute tool calls via MCP servers
   - Synthesize results into a response

### Customizing Examples

Edit the examples to:
- Change the model (qwen3:8b, qwen3:30b-a3b, gemma3:4b)
- Modify temperature and generation parameters
- Add custom tools
- Adjust system prompts

Example configuration:
```python
llm = ChatOllama(
    model="qwen3:30b-a3b",      # Use faster MoE model
    temperature=0.0,             # Deterministic for tool use
    num_predict=2048,            # Max tokens
    base_url="http://localhost:11434"
)
```

---

## Expected Output

### Successful Execution
```
> Entering new AgentExecutor chain...

Thought: I need to list the files in the current directory to find Python files.
Action: list_directory
Action Input: "."

Observation: ['main.py', 'config.yaml', 'examples/', 'tests/']

Thought: I found main.py. Now I'll read its contents.
Action: read_file
Action Input: "main.py"

Observation: [file contents...]

Thought: I now have all the information to answer.
Final Answer: The current directory contains these Python files:
- main.py (entry point, contains...)

> Finished chain.
```

### Performance Metrics
On Apple Silicon (M1/M2/M3):
- Simple file operation: 2-4 seconds
- Complex multi-tool query: 5-10 seconds
- GitHub repository analysis: 8-15 seconds

---

## MCP Server Configuration

### Custom MCP Server Setup

MCP servers can be configured in `.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["--root", "/path/to/project"],
      "env": {}
    },
    "github": {
      "command": "mcp-server-github",
      "env": {
        "GITHUB_TOKEN": "your_token_here"
      }
    },
    "custom": {
      "command": "node",
      "args": ["./mcp-servers/custom/my-server.js"]
    }
  }
}
```

### Creating Custom MCP Servers

See `mcp-servers/custom/README.md` for:
- MCP protocol specification
- Server implementation templates
- Tool definition patterns
- Testing and debugging

Example custom tool:
```javascript
// mcp-servers/custom/ollama-manager.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'ollama-manager',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {},
  },
});

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'list_models',
        description: 'List all available Ollama models',
        inputSchema: { type: 'object', properties: {} }
      }
    ]
  };
});

// ... implement tool handlers
```

---

## Troubleshooting

### Common Issues

#### 1. "Connection refused to localhost:11434"

**Cause**: Ollama server not running

**Solution**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start server
ollama serve

# Verify
curl http://localhost:11434
```

#### 2. "Model not found: qwen3:8b"

**Cause**: Model not pulled

**Solution**:
```bash
# List installed models
ollama list

# Pull required model
ollama pull qwen3:8b

# Verify
ollama list | grep qwen3
```

#### 3. "MCP server not found"

**Cause**: MCP server not installed or not in PATH

**Solution**:
```bash
# Install missing MCP server
npm install -g @modelcontextprotocol/server-filesystem

# Verify installation
which mcp-server-filesystem

# Check npm global path
npm config get prefix
```

#### 4. "Tool execution failed"

**Cause**: MCP server error or invalid parameters

**Solution**:
```bash
# Test MCP server directly
echo '{"method": "tools/list"}' | mcp-server-filesystem

# Check server logs
# Enable verbose mode in agent:
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

#### 5. Agent not using tools

**Cause**: Model not following tool calling format

**Solutions**:
- Use qwen3:30b-a3b or qwen3:8b (best tool calling support)
- Set temperature=0.0 for deterministic behavior
- Improve system prompt with examples
- Add tool descriptions with clear input/output formats

Example improved prompt:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.

    When you need information, use the available tools by following this format:
    Thought: [your reasoning]
    Action: [tool name]
    Action Input: [tool parameters]

    Always use tools when they can help answer the question."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
```

#### 6. Slow performance

**Causes & Solutions**:
- Use smaller/faster model: `qwen3:30b-a3b` (MoE, very fast)
- Reduce max tokens: `num_predict=512`
- Use quantized models: Models ending in Q4, Q5
- Close other applications
- Check system resources: `top` or Activity Monitor

#### 7. Import errors

**Cause**: Missing dependencies

**Solution**:
```bash
# Reinstall all dependencies
uv sync

# Or install specific packages
uv add langchain-ollama

# Verify installation
uv run python -c "from langchain_ollama import ChatOllama; print('OK')"
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain.agents")
logger.setLevel(logging.DEBUG)

# Run agent with verbose output
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True  # Graceful error handling
)
```

### Testing MCP Servers Independently

Before integrating with agents, test MCP servers directly:

```bash
# Test filesystem server
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | \
  mcp-server-filesystem

# Test with specific operation
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", \
       "params": {"name": "read_file", "arguments": {"path": "README.md"}}}' | \
  mcp-server-filesystem
```

---

## Model Selection Guide

### Recommended Models for MCP Integration

| Model | Use Case | Tool Calling | Speed | Resource Usage |
|-------|----------|--------------|-------|----------------|
| qwen3:30b-a3b | Best overall for tool use | Excellent | Very Fast | 16GB RAM |
| qwen3:8b | Balanced performance | Good | Fast | 8GB RAM |
| gemma3:4b | Quick iterations, testing | Fair | Very Fast | 4GB RAM |
| qwen3:14b | Complex reasoning + tools | Excellent | Medium | 10GB RAM |

### Temperature Settings

- **Tool calling tasks**: 0.0-0.1 (deterministic)
- **Creative synthesis**: 0.5-0.7
- **Exploratory queries**: 0.7-1.0

### Context Window Considerations

Most local models support 128K-256K tokens, sufficient for:
- Long conversation histories
- Large file contents
- Multiple tool call results
- Comprehensive system prompts

---

## Integration Patterns

### Pattern 1: Single-Tool Agent
Simple agent with one tool (e.g., file reader)

```python
# Minimal example
tool = Tool(name="read", description="Read file", func=read_file)
agent = create_tool_calling_agent(llm, [tool], prompt)
executor = AgentExecutor(agent=agent, tools=[tool])
```

### Pattern 2: Multi-Tool Agent
Agent with multiple related tools

```python
# Filesystem suite
tools = [
    Tool(name="read", description="...", func=read_file),
    Tool(name="write", description="...", func=write_file),
    Tool(name="list", description="...", func=list_dir),
]
agent = create_tool_calling_agent(llm, tools, prompt)
```

### Pattern 3: Hierarchical Agents
Multiple specialized agents coordinated by supervisor

```python
# See examples/03-multi-agent/ for full implementation
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("file_agent", file_agent)
workflow.add_node("github_agent", github_agent)
workflow.add_node("supervisor", supervisor)
# ... routing logic
```

### Pattern 4: Streaming Responses
Stream agent thoughts and tool calls in real-time

```python
# Streaming example
for chunk in executor.stream({"input": "your query"}):
    if "agent" in chunk:
        print(chunk["agent"]["messages"][0].content)
    elif "tools" in chunk:
        print(f"Tool: {chunk['tools']}")
```

---

## Best Practices

### 1. Tool Design
- **Clear names**: Use descriptive, action-oriented names (read_file, not file_op)
- **Detailed descriptions**: Explain when to use the tool and what it returns
- **Input validation**: Validate parameters before calling MCP server
- **Error handling**: Catch and return helpful error messages

### 2. Prompt Engineering
- **System context**: Explain what tools are available and when to use them
- **Examples**: Include few-shot examples of tool usage
- **Constraints**: Specify limits (e.g., "only read files under 10MB")
- **Output format**: Guide the final answer structure

### 3. Agent Configuration
- **Verbose mode**: Always enable for development/debugging
- **Max iterations**: Set reasonable limit (default 15) to prevent loops
- **Error handling**: Use `handle_parsing_errors=True`
- **Timeouts**: Set timeouts for long-running tools

### 4. Performance Optimization
- **Model selection**: Use qwen3:30b-a3b for speed with quality
- **Prompt optimization**: Keep prompts concise and focused
- **Tool caching**: Cache expensive tool results when possible
- **Batch operations**: Group similar tool calls

### 5. Security Considerations
- **Path validation**: Sanitize file paths to prevent traversal attacks
- **Rate limiting**: Limit tool call frequency if needed
- **Access control**: Restrict tool access to authorized directories
- **Input sanitization**: Validate all user inputs before tool execution

---

## Testing

### Unit Tests for Tools

```python
# tests/test_mcp_tools.py
import pytest
from examples.02-mcp.filesystem_agent import read_file_tool

def test_read_file_success():
    result = read_file_tool.func("README.md")
    assert "MCP Integration" in result

def test_read_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_file_tool.func("nonexistent.txt")
```

### Integration Tests for Agents

```python
# tests/test_mcp_agents.py
def test_filesystem_agent():
    executor = create_filesystem_agent()
    result = executor.invoke({
        "input": "List Python files in examples/"
    })
    assert "filesystem_agent.py" in result["output"]
```

### Running Tests

```bash
# Run all MCP tests
uv run pytest tests/test_mcp_*.py -v

# Run specific test
uv run pytest tests/test_mcp_agents.py::test_filesystem_agent -v

# Run with coverage
uv run pytest --cov=examples/02-mcp tests/
```

---

## Advanced Topics

### Custom MCP Protocol Implementation

For advanced users building custom MCP servers:

1. **Server Initialization**: Implement MCP protocol handshake
2. **Tool Registration**: Define available tools and schemas
3. **Request Handling**: Process tool/list and tool/call requests
4. **Error Handling**: Return structured error responses
5. **Streaming**: Support streaming for long-running operations

See official spec: https://github.com/modelcontextprotocol/specification

### LangSmith Integration

Enable observability for MCP-based agents:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="mcp-experiments"
export LANGCHAIN_API_KEY="your-key"  # Optional
```

This traces all agent actions, tool calls, and responses for debugging.

### Multi-Agent Coordination

Combine MCP tools with LangGraph for complex workflows:

```python
# Supervisor delegates to specialized agents
from langgraph.graph import StateGraph

def should_use_filesystem(state):
    return "file" in state["query"].lower()

workflow = StateGraph(State)
workflow.add_node("filesystem", filesystem_agent)
workflow.add_node("github", github_agent)
workflow.add_conditional_edges("supervisor", should_use_filesystem)
```

See examples/03-multi-agent/ for complete implementations.

---

## Next Steps

### After completing MCP examples:

1. **Milestone 3: Multi-Agent Orchestration**
   - Build complex workflows with LangGraph
   - Coordinate multiple specialized agents
   - Implement state persistence
   - Add human-in-the-loop patterns
   - See: `examples/03-multi-agent/README.md`

2. **Build Custom MCP Servers**
   - Create project-specific tools
   - Integrate with internal APIs
   - Add authentication and authorization
   - See: `mcp-servers/custom/README.md`

3. **Combine MCP with RAG**
   - Use MCP tools to fetch documents
   - Build RAG systems with tool-augmented retrieval
   - See: `examples/04-rag/README.md`

### Learning Resources

- **MCP Specification**: https://github.com/modelcontextprotocol/specification
- **Official MCP Servers**: https://github.com/modelcontextprotocol/servers
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **Tool Calling Guide**: https://python.langchain.com/docs/modules/agents/tools/
- **Project Plans**: `plans/1-research-plan.md` (Milestone 2 details)

---

## Example Checklist

To verify your MCP setup is complete:

- [ ] Ollama installed and running (`ollama serve`)
- [ ] At least one model pulled (`ollama list`)
- [ ] MCP servers installed (`which mcp-server-filesystem`)
- [ ] Python dependencies installed (`uv sync`)
- [ ] Can run basic LLM query (`uv run python examples/01-foundation/simple_chat.py`)
- [ ] MCP server responds to test query
- [ ] Agent successfully uses tools in verbose mode
- [ ] All tests pass (`uv run pytest tests/test_mcp_*.py`)

---

## Support and Resources

### Documentation
- This README for MCP integration overview
- `plans/1-research-plan.md` for research and milestones
- `plans/3-kitchen-sink-plan.md` for comprehensive examples
- `CLAUDE.md` for development guidelines

### Troubleshooting
- Check Ollama logs: `ollama logs`
- Enable verbose agent output
- Test MCP servers independently
- Verify model supports tool calling

### Community
- LangChain Discord: Framework questions
- Ollama Discord: Local model issues
- MCP GitHub: Protocol and server issues
- Project GitHub: Example-specific questions

---

## Quick Reference

### Start Development Session

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run examples
cd /Volumes/JS-DEV/ai-lang-stuff
uv run python examples/02-mcp/filesystem_agent.py
```

### Check System Status

```bash
# Verify Ollama
curl http://localhost:11434/api/tags

# List models
ollama list

# Test model
ollama run qwen3:8b "test query"

# Check MCP servers
which mcp-server-filesystem
npm list -g --depth=0 | grep mcp
```

### Common Commands

```bash
# Pull new model
ollama pull qwen3:14b

# Update dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format examples/02-mcp/

# Type check
uv run mypy examples/02-mcp/
```

---

**Ready to get started?** Run your first MCP agent:

```bash
uv run python examples/02-mcp/filesystem_agent.py
```

Happy building with local AI and MCP!
