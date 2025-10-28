# StateManager API Reference

LangGraph state management and persistence utilities with SQLite checkpointing.

## Overview

The `StateManager` class provides helper functions for managing LangGraph state persistence using SQLite checkpointers, creating state schemas, and managing checkpoint lifecycles for long-running workflows and multi-turn conversations.

**Module:** `utils.state_manager`

**Dependencies:**
- `langgraph` - Workflow framework
- `langchain-core` - Base message types
- `sqlite3` - Database storage
- Python 3.9+

---

## Class: StateManager

```python
class StateManager:
    """Manages LangGraph state persistence and checkpoint operations."""
```

All methods are static - no instantiation required.

---

## Checkpoint Management

### get_checkpointer()

```python
@staticmethod
def get_checkpointer(db_path: str = "./checkpoints.db") -> SqliteSaver
```

Create or connect to SQLite checkpointer for LangGraph persistence.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"./checkpoints.db"` | Path to SQLite database |

**Returns:**
- `SqliteSaver` - Configured SQLite checkpointer

**Raises:**
- `sqlite3.Error` - If database connection fails

**Example:**

```python
from utils.state_manager import StateManager
from langgraph.graph import StateGraph

# Create checkpointer
checkpointer = StateManager.get_checkpointer("./my_checkpoints.db")

# Use with LangGraph
graph = StateGraph(StateSchema)
# ... add nodes and edges
compiled = graph.compile(checkpointer=checkpointer)

# Now your workflow has automatic persistence
```

**Key Features:**
- Automatic database creation if doesn't exist
- Initializes checkpoint tables
- Thread-safe connections
- Automatic setup on first use

---

### load_checkpoint()

```python
@staticmethod
def load_checkpoint(
    thread_id: str,
    db_path: str = "./checkpoints.db"
) -> Optional[Dict[str, Any]]
```

Load a specific checkpoint by thread ID.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thread_id` | `str` | - | Unique thread identifier |
| `db_path` | `str` | `"./checkpoints.db"` | Path to database |

**Returns:**
- `Optional[Dict[str, Any]]` - Checkpoint state dictionary or `None` if not found

**Raises:**
- `sqlite3.Error` - If database query fails

**Example:**

```python
from utils.state_manager import StateManager

# Load checkpoint for specific thread
state = StateManager.load_checkpoint("thread-123")

if state:
    messages = state.get("messages", [])
    print(f"Found checkpoint with {len(messages)} messages")
else:
    print("No checkpoint found for this thread")
```

**Use Cases:**
- Resume interrupted workflows
- Recover from errors
- Load conversation history
- Debug workflow state

---

### list_checkpoints()

```python
@staticmethod
def list_checkpoints(
    db_path: str = "./checkpoints.db"
) -> List[Dict[str, Any]]
```

List all checkpoint thread IDs with metadata.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"./checkpoints.db"` | Path to database |

**Returns:**

List of checkpoint metadata dictionaries:

```python
[
    {
        "thread_id": str,        # Unique thread identifier
        "checkpoint_id": str,    # Checkpoint version identifier
        "timestamp": str         # Creation timestamp
    }
]
```

**Raises:**
- `sqlite3.Error` - If database query fails

**Example:**

```python
from utils.state_manager import StateManager

checkpoints = StateManager.list_checkpoints()

print(f"Found {len(checkpoints)} checkpoints:\n")
for cp in checkpoints:
    print(f"Thread: {cp['thread_id']}")
    print(f"  Checkpoint: {cp['checkpoint_id']}")
    print(f"  Created: {cp['timestamp']}\n")
```

**Sorting by Time:**

```python
checkpoints = StateManager.list_checkpoints()
# Results are already sorted by created_at DESC
recent = checkpoints[:5]  # Get 5 most recent
```

---

### clear_checkpoints()

```python
@staticmethod
def clear_checkpoints(
    db_path: str = "./checkpoints.db",
    thread_id: Optional[str] = None,
    confirm: bool = False
) -> int
```

Clear checkpoint data from database with safety confirmation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"./checkpoints.db"` | Path to database |
| `thread_id` | `Optional[str]` | `None` | Specific thread to clear (None = all) |
| `confirm` | `bool` | `False` | Must be `True` to delete |

**Returns:**
- `int` - Number of checkpoints deleted

**Raises:**
- `sqlite3.Error` - If database operation fails
- `ValueError` - If `confirm=False` (safety check)

**Example:**

```python
from utils.state_manager import StateManager

# Clear specific thread
deleted = StateManager.clear_checkpoints(
    thread_id="thread-123",
    confirm=True
)
print(f"Deleted {deleted} checkpoints")

# Clear all checkpoints (DANGEROUS)
deleted = StateManager.clear_checkpoints(confirm=True)
print(f"Deleted ALL {deleted} checkpoints")
```

**Safety Check:**

```python
# This will raise ValueError
try:
    StateManager.clear_checkpoints()  # confirm=False
except ValueError as e:
    print(f"Safety check: {e}")
```

---

## State Schema Creation

### create_state_schema()

```python
@staticmethod
def create_state_schema(
    fields: Dict[str, type],
    class_name: str = "AgentState"
) -> Type[TypedDict]
```

Create a TypedDict state schema for LangGraph workflows.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields` | `Dict[str, type]` | - | Field names to types mapping |
| `class_name` | `str` | `"AgentState"` | Name for generated class |

**Returns:**
- `Type[TypedDict]` - Generated state schema class

**Raises:**
- `ValueError` - If fields are invalid or empty
- `TypeError` - If field types are not valid type annotations

**Example:**

```python
from typing import Annotated, List
import operator
from langchain_core.messages import BaseMessage
from utils.state_manager import StateManager

# Create custom state schema
State = StateManager.create_state_schema({
    "messages": Annotated[List[BaseMessage], operator.add],
    "context": str,
    "iteration": int
})

# Use in LangGraph
from langgraph.graph import StateGraph

graph = StateGraph(State)

def my_node(state: State) -> State:
    messages = state["messages"]
    # ... process messages
    return {"messages": [response]}
```

**Field Type Patterns:**

```python
# Message accumulation (common for chat)
"messages": Annotated[List[BaseMessage], operator.add]

# List accumulation (sources, results)
"sources": Annotated[List[str], operator.add]

# Simple fields (replaced each update)
"query": str
"count": int
"approved": bool
```

---

## Utility Functions

### create_thread_id()

```python
def create_thread_id(prefix: str = "thread") -> str
```

Generate unique thread ID with timestamp.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | `"thread"` | Prefix for thread ID |

**Returns:**
- `str` - Unique thread ID in format `"{prefix}-{timestamp}"`

**Example:**

```python
from utils.state_manager import create_thread_id

# Default prefix
thread_id = create_thread_id()
print(thread_id)  # "thread-1698765432"

# Custom prefix
research_id = create_thread_id("research")
print(research_id)  # "research-1698765432"

chat_id = create_thread_id("chat")
print(chat_id)  # "chat-1698765432"
```

**Usage with LangGraph:**

```python
from utils.state_manager import create_thread_id

thread_id = create_thread_id("user-session")
config = {"configurable": {"thread_id": thread_id}}

# Use with compiled graph
result = graph.invoke(input_data, config=config)
```

---

### get_checkpoint_size()

```python
def get_checkpoint_size(
    db_path: str = "./checkpoints.db"
) -> Dict[str, Any]
```

Get checkpoint database statistics.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"./checkpoints.db"` | Path to database |

**Returns:**

Dictionary with statistics:

```python
{
    "file_size_mb": float,     # Database file size in MB
    "checkpoint_count": int,   # Total checkpoints
    "thread_count": int        # Unique threads
}
```

**Example:**

```python
from utils.state_manager import get_checkpoint_size

stats = get_checkpoint_size()
print(f"Database size: {stats['file_size_mb']:.2f} MB")
print(f"Total checkpoints: {stats['checkpoint_count']}")
print(f"Unique threads: {stats['thread_count']}")

# Monitor size
if stats['file_size_mb'] > 100:
    print("Warning: Checkpoint database is large")
    print("Consider clearing old checkpoints")
```

**Cleanup Trigger:**

```python
stats = get_checkpoint_size()
if stats['checkpoint_count'] > 1000:
    # Clear old checkpoints
    StateManager.clear_checkpoints(confirm=True)
```

---

## Pre-built State Schemas

### basic_agent_state()

```python
def basic_agent_state() -> Type[TypedDict]:
    """Create basic agent state with message history."""
```

Creates a simple state schema with message accumulation.

**Returns:**
- `Type[TypedDict]` - State with `messages` field

**Example:**

```python
from utils.state_manager import basic_agent_state
from langgraph.graph import StateGraph

State = basic_agent_state()

graph = StateGraph(State)

def agent_node(state: State) -> State:
    messages = state["messages"]
    # ... process messages
    response = llm.invoke(messages)
    return {"messages": [response]}

graph.add_node("agent", agent_node)
```

---

### research_state()

```python
def research_state() -> Type[TypedDict]:
    """Create research agent state with question, context, and sources."""
```

State schema optimized for research workflows.

**Fields:**
- `question: str` - Research question
- `context: Annotated[List[str], operator.add]` - Accumulated context
- `answer: str` - Final answer
- `sources: Annotated[List[str], operator.add]` - Source URLs

**Example:**

```python
from utils.state_manager import research_state
from langgraph.graph import StateGraph

State = research_state()

def research_node(state: State) -> State:
    question = state["question"]
    context = state.get("context", [])

    # Perform research
    new_info = search(question)

    return {
        "context": [new_info],
        "sources": [source_url]
    }

def answer_node(state: State) -> State:
    context = state["context"]
    question = state["question"]

    answer = generate_answer(question, context)
    return {"answer": answer}

graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("answer", answer_node)
```

---

### code_review_state()

```python
def code_review_state() -> Type[TypedDict]:
    """Create code review state with code, issues, and approval status."""
```

State schema for code review workflows.

**Fields:**
- `code: str` - Code to review
- `issues: Annotated[List[str], operator.add]` - Found issues
- `suggestions: Annotated[List[str], operator.add]` - Improvement suggestions
- `approved: bool` - Approval status

**Example:**

```python
from utils.state_manager import code_review_state
from langgraph.graph import StateGraph

State = code_review_state()

def review_node(state: State) -> State:
    code = state["code"]

    # Analyze code
    issues = find_issues(code)
    suggestions = generate_suggestions(issues)

    return {
        "issues": issues,
        "suggestions": suggestions,
        "approved": len(issues) == 0
    }

def fix_node(state: State) -> State:
    code = state["code"]
    suggestions = state["suggestions"]

    fixed_code = apply_fixes(code, suggestions)
    return {"code": fixed_code}

graph = StateGraph(State)
graph.add_node("review", review_node)
graph.add_node("fix", fix_node)
```

---

## Integration Examples

### Simple Chat Agent

```python
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from utils.state_manager import (
    StateManager,
    basic_agent_state,
    create_thread_id
)

# Create state schema
State = basic_agent_state()

# Create graph
graph = StateGraph(State)

# Add agent node
llm = ChatOllama(model="qwen3:8b")

def chat_node(state: State) -> State:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# Compile with persistence
checkpointer = StateManager.get_checkpointer()
app = graph.compile(checkpointer=checkpointer)

# Use with persistence
thread_id = create_thread_id("user-123")
config = {"configurable": {"thread_id": thread_id}}

# First message
result = app.invoke({"messages": [HumanMessage("Hello")]}, config=config)

# Second message - continues conversation
result = app.invoke({"messages": [HumanMessage("How are you?")]}, config=config)
```

---

### Research Agent with Recovery

```python
from utils.state_manager import (
    StateManager,
    research_state,
    create_thread_id
)
from langgraph.graph import StateGraph, END

State = research_state()
graph = StateGraph(State)

def research_node(state: State) -> State:
    question = state["question"]
    # ... research logic
    return {"context": [info], "sources": [url]}

def answer_node(state: State) -> State:
    # ... answer logic
    return {"answer": answer}

graph.add_node("research", research_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("research")
graph.add_edge("research", "answer")
graph.add_edge("answer", END)

# Compile with persistence
checkpointer = StateManager.get_checkpointer()
app = graph.compile(checkpointer=checkpointer)

# Execute with recovery
thread_id = create_thread_id("research")
config = {"configurable": {"thread_id": thread_id}}

try:
    result = app.invoke(
        {"question": "What are AI agents?"},
        config=config
    )
except Exception as e:
    print(f"Error: {e}")
    # Recover from checkpoint
    state = StateManager.load_checkpoint(thread_id)
    if state:
        print("Recovered state from checkpoint")
        # Resume or retry
```

---

### Multi-User Conversation Management

```python
from utils.state_manager import (
    StateManager,
    basic_agent_state,
    create_thread_id
)

State = basic_agent_state()
# ... create and compile graph with checkpointer

# Map users to thread IDs
user_threads = {}

def get_user_thread(user_id: str) -> str:
    if user_id not in user_threads:
        user_threads[user_id] = create_thread_id(f"user-{user_id}")
    return user_threads[user_id]

# User 1 conversation
user1_thread = get_user_thread("alice")
config1 = {"configurable": {"thread_id": user1_thread}}
app.invoke({"messages": [HumanMessage("Hello")]}, config=config1)

# User 2 conversation (separate)
user2_thread = get_user_thread("bob")
config2 = {"configurable": {"thread_id": user2_thread}}
app.invoke({"messages": [HumanMessage("Hi")]}, config=config2)

# Each user has isolated conversation history
```

---

### Checkpoint Maintenance

```python
from utils.state_manager import (
    StateManager,
    get_checkpoint_size
)
import schedule
import time

def cleanup_old_checkpoints():
    """Periodic cleanup of old checkpoints."""
    stats = get_checkpoint_size()

    print(f"Checkpoint stats:")
    print(f"  Size: {stats['file_size_mb']:.2f} MB")
    print(f"  Checkpoints: {stats['checkpoint_count']}")
    print(f"  Threads: {stats['thread_count']}")

    # Clear if too large
    if stats['file_size_mb'] > 100:
        print("Clearing old checkpoints...")
        deleted = StateManager.clear_checkpoints(confirm=True)
        print(f"Deleted {deleted} checkpoints")

# Run daily
schedule.every().day.at("02:00").do(cleanup_old_checkpoints)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Best Practices

### Thread ID Management

```python
# Good - descriptive prefixes
user_thread = create_thread_id(f"user-{user_id}")
session_thread = create_thread_id(f"session-{session_id}")
task_thread = create_thread_id(f"task-{task_name}")

# Avoid - generic prefixes
thread = create_thread_id()  # Less informative
```

---

### Error Recovery

```python
from utils.state_manager import StateManager, create_thread_id

def resilient_workflow(input_data, max_retries=3):
    thread_id = create_thread_id("workflow")
    config = {"configurable": {"thread_id": thread_id}}

    for attempt in range(max_retries):
        try:
            result = app.invoke(input_data, config=config)
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # Load checkpoint and retry
                state = StateManager.load_checkpoint(thread_id)
                if state:
                    print("Recovered from checkpoint")
            else:
                raise
```

---

### State Schema Design

```python
# Good - use Annotated for accumulation
from typing import Annotated, List
import operator

State = StateManager.create_state_schema({
    "messages": Annotated[List[BaseMessage], operator.add],  # Accumulates
    "result": str  # Replaces
})

# Avoid - plain lists without operator (will replace, not accumulate)
State = StateManager.create_state_schema({
    "messages": List[BaseMessage],  # Will replace, not accumulate
    "result": str
})
```

---

## Performance Considerations

### Database Size

Monitor and maintain checkpoint database:

```python
stats = get_checkpoint_size()
if stats['file_size_mb'] > 500:
    # Database too large
    # Options:
    # 1. Clear old checkpoints
    # 2. Archive to separate database
    # 3. Implement retention policy
    pass
```

---

### Checkpointing Frequency

LangGraph automatically checkpoints after each node execution. For expensive operations, minimize intermediate state:

```python
# Good - minimal state
State = StateManager.create_state_schema({
    "result": str,  # Only final result
    "metadata": dict
})

# Avoid - large intermediate state
State = StateManager.create_state_schema({
    "all_intermediate_results": Annotated[List[dict], operator.add],  # Large
    "debug_logs": Annotated[List[str], operator.add]  # Large
})
```

---

## See Also

- [Ollama Manager](./ollama_manager.md) - Model operations
- [Vector Store](./vector_store.md) - Document storage
- [Tool Registry](./tool_registry.md) - Tool management
- [Examples](../../examples/03-multi-agent/) - LangGraph examples

---

**Module Location:** `/Volumes/JS-DEV/ai-lang-stuff/utils/state_manager.py`

**LangGraph Docs:** https://langchain-ai.github.io/langgraph/
