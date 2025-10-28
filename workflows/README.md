# LangGraph Studio Workflows

This directory contains workflow definitions for LangGraph Studio.

## Quick Start

```bash
# Start LangGraph Studio
npx langgraph@latest dev

# Or use npm script
npm run studio
```

Open browser to: http://localhost:8123/studio

---

## Available Workflows

### 1. Research Agent
**File**: `research_agent.py`
**Graph**: `research_agent`

**Description**: Sequential research pipeline with three agents:
- Researcher: Gathers information
- Analyzer: Extracts insights
- Summarizer: Creates final summary

**Use Case**: Automated research on any topic

**Example Input**:
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

**Test Standalone**:
```bash
python workflows/research_agent.py
```

---

### 2. Code Reviewer
**File**: `code_reviewer.py`
**Graph**: `code_reviewer`

**Description**: Code review pipeline with conditional routing:
- Syntax Checker: Finds syntax errors
- Security Scanner: Identifies vulnerabilities
- Style Reviewer: Checks code style
- Approval Gate: Makes decision
- Code Fixer: Attempts automatic fixes (if needed)

**Use Case**: Automated code quality checks

**Example Input**:
```json
{
  "code": "def process(user_input):\n    import os\n    os.system(user_input)",
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

**Test Standalone**:
```bash
python workflows/code_reviewer.py
```

---

### 3. RAG Pipeline
**File**: `rag_pipeline.py`
**Graph**: `rag_pipeline`

**Description**: Retrieval-augmented generation pipeline:
- Ingestion: Adds documents to vector store
- Retrieval: Finds relevant documents
- Generation: Generates answer using context

**Use Case**: Question answering with document retrieval

**Example Input**:
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

**Test Standalone**:
```bash
python workflows/rag_pipeline.py
```

---

## Creating New Workflows

### 1. Create Workflow File

Create `workflows/my_workflow.py`:

```python
"""My custom workflow."""

from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


class MyState(TypedDict):
    """Workflow state."""
    input: str
    output: str


def my_node(state: MyState) -> MyState:
    """Process input."""
    llm = ChatOllama(model="qwen3:8b")
    result = llm.invoke(state["input"])
    return {"output": result.content}


def create_graph():
    """Create workflow graph."""
    workflow = StateGraph(MyState)
    workflow.add_node("process", my_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    return workflow


# LangGraph Studio entry point
my_workflow = create_graph().compile(
    checkpointer=SqliteSaver.from_conn_string("./checkpoints/my_workflow.db")
)
```

### 2. Add to langgraph.json

Edit `langgraph.json`:

```json
{
  "graphs": {
    "my_workflow": {
      "path": "./workflows/my_workflow.py",
      "graph": "my_workflow",
      "description": "My custom workflow"
    }
  }
}
```

### 3. Test Standalone

```bash
python workflows/my_workflow.py
```

### 4. Run in Studio

```bash
npx langgraph@latest dev
# Open browser, select "my_workflow"
```

---

## Workflow Best Practices

### State Definition

Use TypedDict for clear state schema:

```python
from typing import TypedDict, Annotated, List
import operator

class MyState(TypedDict):
    """Well-documented state."""
    input: str
    output: str
    messages: Annotated[List, operator.add]  # Append messages
    iteration: int
```

### Node Functions

Keep nodes focused on single responsibility:

```python
def focused_node(state: MyState) -> MyState:
    """Does one thing well."""
    result = do_one_thing(state["input"])
    return {"output": result}
```

### Error Handling

Always handle errors gracefully:

```python
def safe_node(state: MyState) -> MyState:
    """Handles errors."""
    try:
        result = risky_operation()
        return {"output": result}
    except Exception as e:
        return {"errors": [str(e)], "output": ""}
```

### Conditional Routing

Use routing functions for complex flows:

```python
from typing import Literal

def route_function(state: MyState) -> Literal["path_a", "path_b"]:
    """Route based on state."""
    if state["score"] > 80:
        return "path_a"
    return "path_b"

workflow.add_conditional_edges(
    "decision_node",
    route_function,
    {"path_a": "node_a", "path_b": "node_b"}
)
```

---

## Testing Workflows

### Unit Tests

Test individual nodes:

```python
def test_my_node():
    state = {"input": "test"}
    result = my_node(state)
    assert "output" in result
    assert result["output"] != ""
```

### Integration Tests

Test full workflow:

```python
def test_workflow():
    initial_state = {"input": "test", "output": ""}
    config = {"configurable": {"thread_id": "test-001"}}

    result = my_workflow.invoke(initial_state, config)
    assert result["output"] != ""
```

### Studio Testing

1. Run workflow in Studio
2. Inspect state at each checkpoint
3. Verify output matches expectations
4. Check for errors in logs

---

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def my_node(state):
    logger.debug(f"Processing: {state['input']}")
    # ...
```

### Use Studio Breakpoints

1. Click node in Studio UI
2. Set breakpoint
3. Run workflow
4. Inspect state when paused
5. Continue or step forward

### Check Checkpoints

```python
# Get state at specific checkpoint
state = my_workflow.get_state(config)
print(state.values)

# List all checkpoints
# (Use Studio UI for visual inspection)
```

---

## Performance Optimization

### Cache LLM Instances

```python
_llm_cache = {}

def get_llm(model: str):
    if model not in _llm_cache:
        _llm_cache[model] = ChatOllama(model=model)
    return _llm_cache[model]
```

### Batch Operations

```python
def batch_node(state):
    llm = get_llm("qwen3:8b")
    results = llm.batch(state["items"])
    return {"results": results}
```

### Use Smaller Models in Dev

```python
import os

model = os.getenv("DEFAULT_MODEL", "gemma3:4b")
llm = ChatOllama(model=model)
```

---

## Resources

- **LangGraph Studio Guide**: `/docs/langgraph-studio-guide.md`
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Example Workflows**: `/examples/03-multi-agent/`

---

## Support

Need help? Check:

1. Documentation: `/docs/langgraph-studio-guide.md`
2. Example workflows in this directory
3. Project README: `/README.md`
4. LangGraph Discord: https://discord.gg/langchain
