# Multi-Agent Orchestration with LangGraph

## Overview

This directory contains examples demonstrating multi-agent orchestration using LangGraph, showcasing how to build sophisticated workflows where multiple specialized agents collaborate to solve complex tasks.

LangGraph enables you to create stateful, cyclical workflows with conditional routing, human-in-the-loop capabilities, and persistent memory - moving beyond simple sequential chains to sophisticated multi-agent systems.

---

## Table of Contents

1. [LangGraph Core Concepts](#langgraph-core-concepts)
2. [Orchestration Patterns](#orchestration-patterns)
3. [Prerequisites](#prerequisites)
4. [Available Examples](#available-examples)
5. [Usage Guide](#usage-guide)
6. [Workflow Visualizations](#workflow-visualizations)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## LangGraph Core Concepts

### What is LangGraph?

LangGraph is a library for building stateful, multi-agent workflows as directed graphs. Unlike simple chains that execute linearly, LangGraph allows you to:

- Define complex, cyclical workflows with loops and conditional routing
- Maintain persistent state across agent interactions
- Implement human-in-the-loop patterns
- Visualize and debug multi-agent systems
- Add checkpointing for fault tolerance and replay

### Key Components

#### 1. State

State is the core data structure that flows through your graph. It's defined as a `TypedDict` and can include message history, intermediate results, and control flags.

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Append-only message list
    intermediate_results: dict
    current_step: str
    is_complete: bool
```

**Key Features:**
- **Typed**: Uses `TypedDict` for clear schema definition
- **Annotations**: `Annotated[List, operator.add]` means lists are appended, not replaced
- **Passed between nodes**: Each node receives state, modifies it, and returns updates

#### 2. Nodes

Nodes are functions that process state and return updates. Each node represents a discrete step in your workflow.

```python
def researcher_node(state: AgentState) -> AgentState:
    """Research agent that gathers information"""
    llm = ChatOllama(model="qwen3:8b")

    # Process current state
    last_message = state["messages"][-1]
    response = llm.invoke(f"Research topic: {last_message.content}")

    # Return state updates (only changed fields)
    return {
        "messages": [AIMessage(content=response.content)],
        "intermediate_results": {"research": response.content},
        "current_step": "research_complete"
    }
```

**Best Practices:**
- Return only modified state fields
- Use descriptive function names
- Add docstrings explaining node purpose
- Keep nodes focused on single responsibilities

#### 3. Edges

Edges define the flow between nodes. LangGraph supports two types:

**Static Edges**: Direct connections between nodes
```python
workflow.add_edge("researcher", "analyst")  # Always go from researcher to analyst
```

**Conditional Edges**: Dynamic routing based on state
```python
def route_decision(state: AgentState) -> str:
    if state["is_complete"]:
        return "end"
    elif state["needs_revision"]:
        return "reviewer"
    else:
        return "writer"

workflow.add_conditional_edges(
    "analyst",
    route_decision,
    {
        "end": END,
        "reviewer": "reviewer",
        "writer": "writer"
    }
)
```

#### 4. Graph Compilation

Compile your workflow into an executable application:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

# Add edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({"messages": [HumanMessage(content="Research AI")]})
```

#### 5. Checkpointing

Enable state persistence with checkpointers for fault tolerance:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create checkpointer
conn = sqlite3.connect("./checkpoints.db")
checkpointer = SqliteSaver(conn)

# Compile with checkpointing
app = workflow.compile(checkpointer=checkpointer)

# Use with thread IDs for session management
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config)
```

---

## Orchestration Patterns

### Pattern 1: Sequential Pipeline

Multiple agents process information in sequence. Each agent specializes in one step.

**Use Case**: Research report generation, code review workflows

```
[Input] -> [Researcher] -> [Analyzer] -> [Writer] -> [Reviewer] -> [Output]
```

**Example**: Research pipeline where agents gather sources, analyze data, write report, and review quality.

**Key Feature**: Linear flow with specialized agents at each stage.

---

### Pattern 2: Conditional Routing

Agents route to different paths based on state conditions.

**Use Case**: Customer service, content moderation, quality control

```
                        / [Escalation Agent] -> [END]
[Input] -> [Classifier] - [Standard Response] -> [END]
                        \ [FAQ Response] -> [END]
```

**Example**: Customer service agent routes to different handlers based on query type.

**Key Feature**: Dynamic routing based on analysis results.

---

### Pattern 3: Iterative Refinement

Agents loop back to previous steps for quality improvement.

**Use Case**: Code generation with feedback, creative writing, optimization problems

```
[Input] -> [Generator] -> [Reviewer] -> [Generator] (loop until approved) -> [Output]
                            |                           ^
                            +---------------------------+
```

**Example**: Code generator creates solution, reviewer provides feedback, generator refines until approved.

**Key Feature**: Cycles allow iterative improvement until quality threshold met.

---

### Pattern 4: Parallel Processing

Multiple agents work simultaneously on different aspects.

**Use Case**: Multi-source research, distributed computation, parallel analysis

```
                    / [Agent A] \
[Input] -> [Split] -- [Agent B] -- [Merge] -> [Output]
                    \ [Agent C] /
```

**Example**: Three agents research different aspects simultaneously, results merged.

**Key Feature**: Concurrent execution for speed, results aggregated.

---

### Pattern 5: Human-in-the-Loop

Agent pauses execution to request human input.

**Use Case**: Approval workflows, creative collaboration, sensitive decisions

```
[Input] -> [Agent] -> [Request Human Input] -> [Wait] -> [Continue with Human Response] -> [Output]
```

**Example**: Agent drafts content, pauses for human review/edits, continues with feedback.

**Key Feature**: Interrupt nodes that pause execution and wait for external input.

---

### Pattern 6: Hierarchical Multi-Agent

Supervisor agent coordinates worker agents.

**Use Case**: Complex projects, task decomposition, resource allocation

```
                        / [Worker 1] \
[Input] -> [Supervisor] - [Worker 2] -- [Supervisor] (aggregate) -> [Output]
                        \ [Worker 3] /
```

**Example**: Supervisor breaks task into subtasks, delegates to workers, aggregates results.

**Key Feature**: Meta-agent manages workflow and delegates to specialized agents.

---

## Prerequisites

### 1. Ollama Setup

Ensure Ollama is running with required models:

```bash
# Start Ollama server
ollama serve

# Pull models for different agents
ollama pull qwen3:8b          # Fast, efficient agent
ollama pull qwen3:30b-a3b     # Powerful MoE for complex reasoning
ollama pull gemma3:12b        # Alternative multilingual agent
```

Verify models are available:
```bash
ollama list
```

### 2. Python Dependencies

Install LangGraph and dependencies:

```bash
# Using uv (recommended)
uv add langgraph langchain-ollama langchain-core

# Or using pip
pip install langgraph langchain-ollama langchain-core
```

### 3. Optional: LangGraph Studio

For visual workflow debugging and development:

```bash
# Launch LangGraph Studio
npx langgraph@latest dev

# Access at http://localhost:3000
```

### 4. Directory Setup

Create checkpoint storage directory:

```bash
mkdir -p /Volumes/JS-DEV/ai-lang-stuff/data/checkpoints
```

---

## Available Examples

### 1. `research_pipeline.py`

**Purpose**: Multi-agent research pipeline with sequential processing

**Agents**:
- **Researcher**: Gathers relevant sources and topics
- **Analyst**: Synthesizes information and identifies key insights
- **Writer**: Generates comprehensive report
- **Reviewer**: Quality checks and approves/rejects

**Pattern**: Sequential pipeline with conditional loop-back

**Run**:
```bash
uv run python examples/03-multi-agent/research_pipeline.py
```

**Expected Output**: Comprehensive research report on specified topic

---

### 2. `stateful_agent.py`

**Purpose**: Customer service agent with persistent conversation memory

**Features**:
- Session-based conversation tracking
- SQLite checkpoint persistence
- Multi-turn dialogue support
- Escalation routing

**Pattern**: Conditional routing with state persistence

**Run**:
```bash
uv run python examples/03-multi-agent/stateful_agent.py
```

**Expected Output**: Contextual responses across multiple conversation turns

---

### 3. `parallel_research.py`

**Purpose**: Parallel agent execution for multi-source research

**Agents**:
- Three specialized researchers working simultaneously
- Aggregator that merges findings

**Pattern**: Parallel processing with merge step

**Run**:
```bash
uv run python examples/03-multi-agent/parallel_research.py
```

**Expected Output**: Merged research findings from multiple sources

---

### 4. `code_review_workflow.py`

**Purpose**: Iterative code generation with automated review

**Agents**:
- **Coder**: Generates code solutions
- **Reviewer**: Analyzes code quality and suggests improvements
- **Refiner**: Applies feedback

**Pattern**: Iterative refinement with quality gates

**Run**:
```bash
uv run python examples/03-multi-agent/code_review_workflow.py
```

**Expected Output**: High-quality code that passes automated review

---

## Usage Guide

### Basic Usage

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define state schema
class MyState(TypedDict):
    messages: list
    result: str

# 2. Initialize models
llm = ChatOllama(model="qwen3:8b")

# 3. Define agent nodes
def agent_node(state: MyState) -> MyState:
    response = llm.invoke(state["messages"][-1].content)
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "result": response.content
    }

# 4. Build graph
workflow = StateGraph(MyState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 5. Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Hello!")],
    "result": ""
})

print(result["result"])
```

### Advanced Usage: Conditional Routing

```python
def should_continue(state: MyState) -> str:
    """Decide next node based on state"""
    if "approved" in state["result"].lower():
        return "end"
    elif "revise" in state["result"].lower():
        return "revise"
    else:
        return "review"

workflow.add_conditional_edges(
    "generator",
    should_continue,
    {
        "end": END,
        "revise": "generator",  # Loop back
        "review": "reviewer"
    }
)
```

### State Persistence Pattern

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Setup checkpointing
conn = sqlite3.connect("checkpoints.db")
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

# First conversation
config = {"configurable": {"thread_id": "conversation_1"}}
result1 = app.invoke({"messages": [HumanMessage(content="Hi")]}, config)

# Continue same conversation later
result2 = app.invoke({"messages": [HumanMessage(content="Continue")]}, config)
# Agent remembers context from result1
```

### Debugging Tips

**1. Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**2. Print state at each node:**
```python
def debug_node(state: MyState) -> MyState:
    print(f"Current state: {state}")
    # ... agent logic
    return updated_state
```

**3. Use LangSmith tracing:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="multi-agent-debug"
```

**4. Visualize with LangGraph Studio:**
```bash
npx langgraph@latest dev
# Load your workflow file
```

---

## Workflow Visualizations

### Sequential Pipeline Pattern

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  START  │────▶│Research │────▶│ Analyze │────▶│  Write  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                       │
                                                       ▼
                ┌─────────┐                      ┌─────────┐
                │   END   │◀────────────────────│ Review  │
                └─────────┘                      └─────────┘
```

### Iterative Refinement Pattern

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  START  │────▶│Generate │────▶│ Review  │
└─────────┘     └─────────┘     └─────────┘
                     ▲                │
                     │                │ (not approved)
                     └────────────────┘
                                      │
                                      │ (approved)
                                      ▼
                                 ┌─────────┐
                                 │   END   │
                                 └─────────┘
```

### Conditional Routing Pattern

```
                                ┌──────────────┐
                        ┌──────▶│  Escalate    │────┐
                        │       └──────────────┘    │
┌─────────┐     ┌──────────┐                       │     ┌─────────┐
│  START  │────▶│ Classify │──────────────────────────▶│   END   │
└─────────┘     └──────────┘                       │     └─────────┘
                        │       ┌──────────────┐    │
                        └──────▶│  Standard    │────┘
                                └──────────────┘
```

### Parallel Processing Pattern

```
                     ┌─────────────┐
              ┌─────▶│  Agent A    │─────┐
              │      └─────────────┘     │
┌─────────┐   │                          │      ┌─────────┐     ┌─────────┐
│  START  │───┼─────▶│  Agent B    │─────┼─────▶│  Merge  │────▶│   END   │
└─────────┘   │      └─────────────┘     │      └─────────┘     └─────────┘
              │                          │
              └─────▶│  Agent C    │─────┘
                     └─────────────┘
```

---

## Troubleshooting

### Issue: "Graph execution hangs indefinitely"

**Cause**: Node is waiting for response that never comes, or infinite loop in conditional routing.

**Solutions**:
1. Add timeout to LLM calls:
   ```python
   llm = ChatOllama(model="qwen3:8b", timeout=30)
   ```

2. Add max iterations to conditional edges:
   ```python
   workflow.add_conditional_edges(
       "node",
       routing_function,
       edges,
       max_iterations=10  # Prevent infinite loops
   )
   ```

3. Check routing logic for cycles:
   ```python
   def safe_routing(state: MyState) -> str:
       iteration_count = state.get("iteration_count", 0)
       if iteration_count > 5:
           return "end"  # Force exit after 5 iterations
       # ... normal routing logic
   ```

---

### Issue: "State not persisting between invocations"

**Cause**: Checkpointer not configured or using different thread IDs.

**Solutions**:
1. Verify checkpointer is passed to compile:
   ```python
   app = workflow.compile(checkpointer=checkpointer)  # Don't forget this!
   ```

2. Use consistent thread IDs:
   ```python
   config = {"configurable": {"thread_id": "session_123"}}
   result1 = app.invoke(state1, config)  # Same config for same session
   result2 = app.invoke(state2, config)
   ```

3. Check database file exists and is writable:
   ```bash
   ls -la checkpoints.db
   sqlite3 checkpoints.db "SELECT * FROM checkpoints;"
   ```

---

### Issue: "TypeError: state object is not subscriptable"

**Cause**: State type mismatch or incorrect state schema definition.

**Solutions**:
1. Use TypedDict for state:
   ```python
   from typing import TypedDict  # Not just dict!

   class MyState(TypedDict):
       field1: str
       field2: list
   ```

2. Return dict from nodes, not state object:
   ```python
   def my_node(state: MyState) -> MyState:
       # ❌ Wrong: return state
       # ✅ Right: return dict with updates
       return {"field1": "updated_value"}
   ```

3. Check annotations for list fields:
   ```python
   from typing import Annotated
   import operator

   class MyState(TypedDict):
       messages: Annotated[List[BaseMessage], operator.add]  # Required for lists!
   ```

---

### Issue: "Ollama connection refused"

**Cause**: Ollama server not running or wrong port.

**Solutions**:
1. Start Ollama server:
   ```bash
   ollama serve
   ```

2. Verify server is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. Check port in agent initialization:
   ```python
   llm = ChatOllama(
       model="qwen3:8b",
       base_url="http://localhost:11434"  # Verify correct port
   )
   ```

4. Check for port conflicts:
   ```bash
   lsof -i :11434
   ```

---

### Issue: "Model not found error"

**Cause**: Required model not pulled from Ollama.

**Solutions**:
1. List available models:
   ```bash
   ollama list
   ```

2. Pull missing model:
   ```bash
   ollama pull qwen3:8b
   ```

3. Use a different available model:
   ```python
   llm = ChatOllama(model="llama2")  # Use what you have
   ```

---

### Issue: "Agent responses are repetitive or low quality"

**Cause**: Poor prompts, wrong model, or insufficient context in state.

**Solutions**:
1. Improve system prompts:
   ```python
   from langchain_core.prompts import ChatPromptTemplate

   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a specialized researcher. Focus on finding credible sources and key insights."),
       ("human", "{input}")
   ])
   ```

2. Use more powerful model:
   ```python
   llm = ChatOllama(model="qwen3:30b-a3b")  # Larger, more capable
   ```

3. Pass more context in state:
   ```python
   def node_with_context(state: MyState) -> MyState:
       context = "\n".join([m.content for m in state["messages"][-5:]])
       prompt = f"Context: {context}\n\nTask: {state['task']}"
       # ... use prompt
   ```

4. Adjust temperature for variety:
   ```python
   llm = ChatOllama(model="qwen3:8b", temperature=0.8)  # Higher = more creative
   ```

---

### Issue: "LangGraph Studio not showing workflow"

**Cause**: File not in correct format or compilation error.

**Solutions**:
1. Ensure file exports compiled app:
   ```python
   # At end of file
   app = workflow.compile()

   if __name__ == "__main__":
       # For CLI usage
       result = app.invoke(...)
   ```

2. Check for syntax errors:
   ```bash
   python -m py_compile examples/03-multi-agent/your_workflow.py
   ```

3. Restart LangGraph Studio:
   ```bash
   # Kill existing process
   pkill -f "langgraph dev"

   # Restart
   npx langgraph@latest dev
   ```

---

## Next Steps

### Milestone 4: Advanced RAG and Vision

After mastering multi-agent orchestration, proceed to Milestone 4 to add:

1. **Retrieval-Augmented Generation (RAG)**
   - Vector stores for document embeddings
   - Semantic search and retrieval
   - Multi-document QA systems
   - Examples: `/examples/04-rag/`

2. **Vision Models**
   - Qwen3-VL for image understanding
   - Multi-modal agents (text + images)
   - Document analysis and OCR
   - Diagram interpretation

3. **Advanced Memory Patterns**
   - Long-term memory with vector stores
   - Episodic memory for conversations
   - Semantic memory for facts
   - Working memory for current tasks

4. **Production Patterns**
   - Error handling and retry logic
   - Monitoring and observability
   - Testing strategies for agents
   - Deployment configurations

### Recommended Learning Path

1. **Start Simple**: Run `research_pipeline.py` and understand the sequential pattern
2. **Add Complexity**: Modify examples to add new nodes or change routing
3. **Build Custom**: Create your own workflow for a specific use case
4. **Optimize**: Add checkpointing, improve prompts, tune model selection
5. **Scale**: Implement parallel processing and hierarchical patterns
6. **Deploy**: Move to production patterns with error handling and monitoring

### Additional Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangGraph Tutorials**: https://langchain-ai.github.io/langgraph/tutorials/
- **Example Repository**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **LangChain Discord**: Community support for questions
- **Project Wiki**: `/docs/` for architecture decisions and patterns

### Community Patterns

Explore community-contributed orchestration patterns:

- **Debate Agent**: Multiple agents argue different positions
- **Research Assistant**: Automated literature review and synthesis
- **Code Generation Pipeline**: Spec → Code → Test → Review
- **Content Creation**: Research → Outline → Write → Edit → Publish
- **Data Analysis**: Ingest → Clean → Analyze → Visualize → Report

---

## Contributing

Found a bug or want to add an example? Contributions welcome!

1. Create new example following existing patterns
2. Add comprehensive docstrings and comments
3. Include usage instructions in this README
4. Test with multiple models
5. Submit PR with clear description

---

**Happy Orchestrating!**

For questions or issues, refer to the main project documentation in `/docs/` or check the troubleshooting guide above.
