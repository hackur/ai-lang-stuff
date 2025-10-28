# LangGraph Studio Guide

Complete guide to using LangGraph Studio for visual workflow editing and debugging.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Project Configuration](#project-configuration)
4. [Running Workflows](#running-workflows)
5. [Visual Debugging](#visual-debugging)
6. [Studio Features](#studio-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Start

**Run LangGraph Studio in one command:**

```bash
# From project root
npm run studio

# Or directly
npx langgraph@latest dev
```

This will:
- Start the LangGraph API server (default port: 8123)
- Open Studio UI in your browser
- Load all workflows from `langgraph.json`

---

## Installation

### Prerequisites

1. **Node.js 18+**
   ```bash
   node --version  # Should be >= 18.0.0
   ```

2. **Python 3.10+**
   ```bash
   python --version  # Should be >= 3.10
   ```

3. **Ollama Running**
   ```bash
   ollama serve
   ollama list  # Verify models installed
   ```

### Install LangGraph CLI

```bash
# Global installation (recommended)
npm install -g @langchain/langgraph-cli

# Or use npx (no installation)
npx langgraph@latest dev
```

### Verify Installation

```bash
langgraph --version
```

---

## Project Configuration

### langgraph.json

The `langgraph.json` file defines your Studio project:

```json
{
  "name": "ai-lang-stuff",
  "version": "0.1.0",
  "python_version": "3.10",
  "graphs": {
    "research_agent": {
      "path": "./workflows/research_agent.py",
      "graph": "research_agent",
      "description": "Sequential research pipeline"
    },
    "code_reviewer": {
      "path": "./workflows/code_reviewer.py",
      "graph": "code_reviewer",
      "description": "Code review with conditional routing"
    },
    "rag_pipeline": {
      "path": "./workflows/rag_pipeline.py",
      "graph": "rag_pipeline",
      "description": "RAG system with retrieval"
    }
  },
  "env": {
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "DEFAULT_MODEL": "qwen3:8b"
  },
  "checkpointer": {
    "type": "sqlite",
    "path": "./checkpoints"
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8123
  }
}
```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `graphs` | Maps workflow names to Python files | Required |
| `env` | Environment variables for workflows | `{}` |
| `checkpointer.type` | Persistence type (sqlite/postgres) | `sqlite` |
| `checkpointer.path` | Database location | `./checkpoints` |
| `api.port` | API server port | `8123` |

---

## Running Workflows

### Start Studio

```bash
# From project root
cd /Volumes/JS-DEV/ai-lang-stuff
npx langgraph@latest dev
```

**Expected output:**
```
Starting LangGraph API server...
- API server running at http://localhost:8123
- Studio UI available at http://localhost:8123/studio

Available workflows:
  - research_agent
  - code_reviewer
  - rag_pipeline
```

### Studio UI Overview

1. **Workflow Selector** (left sidebar)
   - Choose which workflow to run
   - View workflow descriptions
   - See available inputs

2. **Graph Visualization** (center)
   - Visual representation of nodes and edges
   - Real-time execution highlighting
   - Click nodes to see details

3. **Input Panel** (right sidebar)
   - Configure initial state
   - Set configuration options
   - Start execution

4. **Execution Panel** (bottom)
   - Step-by-step output
   - State at each checkpoint
   - Message history

### Running a Workflow

1. **Select Workflow**: Click "research_agent" in left sidebar

2. **Configure Input**: In right panel, enter initial state:
   ```json
   {
     "question": "What are the benefits of local LLMs?",
     "research_findings": "",
     "analysis": "",
     "summary": "",
     "messages": [],
     "iteration": 0
   }
   ```

3. **Set Config** (optional):
   ```json
   {
     "configurable": {
       "thread_id": "demo-001"
     }
   }
   ```

4. **Click "Run"**: Watch execution in real-time

5. **View Results**: Inspect final state in execution panel

---

## Visual Debugging

### Step-by-Step Execution

Studio allows you to:

1. **Pause execution** at any node
2. **Inspect state** at each checkpoint
3. **Modify state** and resume
4. **Replay executions** from checkpoints

### Debugging Features

#### 1. Breakpoints

Click any node to set a breakpoint:
- Execution pauses before node executes
- Inspect current state
- Continue or step forward

#### 2. State Inspection

Click on any executed node to see:
- Input state received
- Output state produced
- Messages generated
- Errors (if any)

#### 3. Checkpoint Navigation

Use timeline slider to:
- Jump to any checkpoint
- Compare states across steps
- Identify where issues occurred

#### 4. Live Logs

Bottom panel shows:
- LLM requests/responses
- Node execution times
- State transitions
- Error traces

### Example Debug Session

**Scenario**: Research agent summary is empty

1. **Run workflow** and notice summary is blank
2. **Check timeline**: See all 3 nodes executed
3. **Click "summarizer" node**: View state
4. **Inspect input**: Check if analysis was provided
5. **Check messages**: Look for errors
6. **Find issue**: Analysis field was empty
7. **Click "analyzer" node**: See it failed silently
8. **Fix code**: Add error handling
9. **Re-run**: Verify fix works

---

## Studio Features

### 1. Graph Visualization

**Node Types:**
- **Circle**: Regular node
- **Diamond**: Conditional routing
- **Rectangle**: Start/End

**Edge Types:**
- **Solid line**: Regular edge
- **Dashed line**: Conditional edge
- **Green highlight**: Current execution path
- **Red highlight**: Error path

**Interactive Features:**
- Zoom in/out with mouse wheel
- Pan by clicking and dragging
- Click nodes for details
- Hover for quick info

### 2. State Management

**View State:**
- Click any checkpoint to see full state
- Use JSON viewer for complex structures
- Filter by state keys

**Modify State:**
- Edit state in JSON editor
- Resume from modified state
- Test edge cases

**Export State:**
- Copy state as JSON
- Save to file
- Use in tests

### 3. Execution Control

**Playback Controls:**
- Play/Pause execution
- Step forward/backward
- Skip to checkpoint
- Restart from beginning

**Speed Control:**
- Slow motion for debugging
- Fast forward for demos
- Real-time for production testing

### 4. Multi-Workflow Support

**Switch between workflows:**
- Dropdown selector
- Keyboard shortcuts (Cmd+1, Cmd+2, etc.)
- Recent workflows list

**Compare workflows:**
- Side-by-side execution
- Diff state outputs
- Performance metrics

### 5. Collaboration

**Share sessions:**
- Copy session URL
- Team members can view live
- Export as recording

**Comments:**
- Annotate nodes
- Add debugging notes
- Link to issues

---

## Troubleshooting

### Common Issues

#### 1. Studio won't start

**Error**: `Cannot find langgraph.json`

**Solution**:
```bash
# Ensure you're in project root
cd /Volumes/JS-DEV/ai-lang-stuff
ls langgraph.json  # Should exist

# Verify JSON is valid
cat langgraph.json | python -m json.tool
```

#### 2. Workflows not loading

**Error**: `Graph 'research_agent' not found`

**Solution**:
```bash
# Check workflow file exists
ls workflows/research_agent.py

# Verify export name matches config
grep "research_agent =" workflows/research_agent.py
# Should see: research_agent = create_graph().compile(...)

# Test workflow standalone
python workflows/research_agent.py
```

#### 3. Ollama connection errors

**Error**: `Connection refused to localhost:11434`

**Solution**:
```bash
# Check Ollama is running
ollama list

# If not running, start it
ollama serve

# Verify connection
curl http://localhost:11434/api/tags
```

#### 4. Port already in use

**Error**: `Address already in use: 8123`

**Solution**:
```bash
# Find process using port
lsof -i :8123

# Kill the process
kill -9 <PID>

# Or change port in langgraph.json
# "api": { "port": 8124 }
```

#### 5. Python import errors

**Error**: `ModuleNotFoundError: No module named 'langchain_ollama'`

**Solution**:
```bash
# Install dependencies
uv sync

# Verify installation
python -c "import langchain_ollama"
```

### Debug Mode

Enable verbose logging:

```bash
# Set debug environment variable
export LANGGRAPH_DEBUG=true

# Run with verbose output
npx langgraph@latest dev --verbose
```

### Check Server Status

```bash
# Test API endpoint
curl http://localhost:8123/health

# List available graphs
curl http://localhost:8123/graphs
```

---

## Best Practices

### 1. Workflow Design

**Keep nodes focused:**
```python
# Good: Single responsibility
def researcher_node(state):
    """Only does research."""
    return {"research_findings": do_research()}

# Bad: Multiple responsibilities
def mega_node(state):
    """Does everything."""
    research = do_research()
    analysis = do_analysis(research)
    summary = do_summary(analysis)
    return {...}  # Too much in one node
```

**Use descriptive names:**
```python
# Good
workflow.add_node("security_scanner", security_scanner_node)

# Bad
workflow.add_node("node3", func3)
```

### 2. State Management

**Use TypedDict for state:**
```python
from typing import TypedDict

class WorkflowState(TypedDict):
    """Clear, typed state definition."""
    input: str
    output: str
    metadata: dict
```

**Initialize all fields:**
```python
# Good: All fields present
initial_state = {
    "input": "test",
    "output": "",
    "metadata": {}
}

# Bad: Missing fields
initial_state = {"input": "test"}
```

### 3. Error Handling

**Handle errors gracefully:**
```python
def node_with_error_handling(state):
    try:
        result = risky_operation()
        return {"result": result}
    except Exception as e:
        logger.error(f"Node failed: {e}")
        return {
            "errors": [str(e)],
            "result": None
        }
```

**Add retry logic:**
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def unreliable_node(state):
    # May fail, will retry
    return llm.invoke(prompt)
```

### 4. Checkpointing

**Save at critical points:**
```python
# Good: Checkpoint after expensive operations
workflow.add_edge("expensive_llm_call", "next_node")

# Checkpoint is automatic with SqliteSaver
```

**Use meaningful thread IDs:**
```python
# Good: Descriptive thread ID
thread_id = f"user_{user_id}_task_{task_id}_{timestamp}"

# Bad: Generic ID
thread_id = "thread1"
```

### 5. Performance

**Optimize node execution:**
```python
# Cache LLM instances
_llm_cache = {}

def get_llm(model: str):
    if model not in _llm_cache:
        _llm_cache[model] = ChatOllama(model=model)
    return _llm_cache[model]
```

**Batch operations:**
```python
# Good: Batch processing
def batch_process_node(state):
    results = llm.batch(state["items"])
    return {"results": results}

# Bad: Sequential
def sequential_process_node(state):
    results = [llm.invoke(item) for item in state["items"]]
    return {"results": results}
```

### 6. Testing

**Test workflows standalone:**
```python
if __name__ == "__main__":
    # Test execution
    result = workflow.invoke(test_state)
    assert result["output"] is not None
```

**Use small models for development:**
```python
# Development
llm = ChatOllama(model="gemma3:4b")  # Fast, small

# Production
llm = ChatOllama(model="qwen3:8b")  # Better quality
```

---

## Workflow Examples

### Research Agent

**Use case**: Automated research on any topic

**Flow**: Question → Research → Analysis → Summary

**Input**:
```json
{
  "question": "What is the future of AI?",
  "research_findings": "",
  "analysis": "",
  "summary": "",
  "messages": [],
  "iteration": 0
}
```

**Output**: Comprehensive summary with insights

**Visual**: Linear flow, 3 nodes, no branching

---

### Code Reviewer

**Use case**: Automated code quality checks

**Flow**: Syntax → Security → Style → Decision Gate → [Fix or Approve]

**Input**:
```json
{
  "code": "def unsafe_function(input):\n    exec(input)",
  "language": "python",
  "issues": [],
  "security_score": 0,
  "style_score": 0,
  "approved": false,
  "needs_rewrite": false,
  "fixed_code": "",
  "messages": [],
  "iteration": 0
}
```

**Output**: Review with scores, issues, and optionally fixed code

**Visual**: Conditional flow with cycles (re-review after fixes)

---

### RAG Pipeline

**Use case**: Question answering with document retrieval

**Flow**: Ingest Docs → Retrieve Relevant → Generate Answer

**Input**:
```json
{
  "query": "What are local LLMs?",
  "documents": ["Local LLMs run on your device...", "..."],
  "retrieved_docs": [],
  "context": "",
  "response": "",
  "sources": [],
  "messages": [],
  "iteration": 0
}
```

**Output**: Answer with sources

**Visual**: Linear flow with vector store interaction

---

## Advanced Features

### Custom Checkpointers

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Use PostgreSQL instead of SQLite
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)

graph = create_graph().compile(checkpointer=checkpointer)
```

### Streaming Responses

```python
# Stream execution in real-time
for chunk in graph.stream(initial_state, config):
    print(f"Node: {list(chunk.keys())[0]}")
    # Studio shows this live
```

### Human-in-the-Loop

```python
def approval_node(state):
    # Pause for human input
    if not state.get("approved"):
        # Studio will wait for user input
        return {"needs_approval": True}
    return {"approved": True}
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Play/Pause execution |
| `→` | Step forward |
| `←` | Step backward |
| `Cmd+1,2,3` | Switch workflow |
| `Cmd+R` | Restart workflow |
| `Cmd+E` | Export state |
| `Cmd+/` | Toggle logs |

---

## Resources

### Documentation

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Studio Guide**: https://langchain-ai.github.io/langgraph/studio/
- **API Reference**: https://langchain-ai.github.io/langgraph/reference/

### Examples

- **Templates**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Cookbooks**: https://langchain-ai.github.io/langgraph/tutorials/

### Community

- **Discord**: https://discord.gg/langchain
- **GitHub**: https://github.com/langchain-ai/langgraph
- **Issues**: https://github.com/langchain-ai/langgraph/issues

---

## Next Steps

1. **Run Example Workflows**: Test all 3 workflows in Studio
2. **Create Custom Workflow**: Build your own agent pipeline
3. **Add Debugging**: Use breakpoints and state inspection
4. **Optimize Performance**: Profile and improve bottlenecks
5. **Deploy**: Move to production with PostgreSQL checkpointer

---

## Support

**Need help?**

1. Check [Troubleshooting](#troubleshooting) section
2. Review example workflows in `workflows/`
3. Run standalone tests: `python workflows/<workflow>.py`
4. Check Ollama: `ollama list`
5. Open an issue or ask in Discord

**Happy debugging!**
