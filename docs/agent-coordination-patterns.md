# Agent Coordination Patterns with LangGraph

## Table of Contents
1. [Overview](#overview)
2. [Core Patterns](#core-patterns)
   - [Sequential Pipeline](#1-sequential-pipeline)
   - [Parallel Fan-Out/Fan-In](#2-parallel-fan-outfan-in)
   - [Conditional Routing](#3-conditional-routing)
   - [Human-in-the-Loop](#4-human-in-the-loop)
   - [Hierarchical Delegation](#5-hierarchical-delegation)
3. [State Management Best Practices](#state-management-best-practices)
4. [Pattern Selection Guide](#pattern-selection-guide)
5. [Performance Considerations](#performance-considerations)
6. [Real-World Examples](#real-world-examples)

---

## Overview

Multi-agent coordination enables complex AI workflows by combining specialized agents that work together toward a common goal. LangGraph provides a powerful framework for orchestrating these interactions through state-based graph workflows.

### Why Multi-Agent Systems?

- **Specialization**: Different models/prompts excel at different tasks
- **Modularity**: Agents can be developed, tested, and updated independently
- **Scalability**: Parallelize work across multiple agents
- **Reliability**: Isolate failures and implement targeted error handling
- **Maintainability**: Clear separation of concerns makes systems easier to understand

### LangGraph Fundamentals

LangGraph treats agent workflows as **stateful, directed graphs** where:
- **Nodes** represent agents or operations
- **Edges** represent transitions between agents
- **State** flows through the graph and accumulates information
- **Conditional edges** enable dynamic routing based on state

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state schema
class WorkflowState(TypedDict):
    messages: Annotated[list, operator.add]  # Accumulates messages
    result: str  # Single value, gets overwritten
    metadata: dict  # Complex data structure

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("agent_1", agent_1_function)
workflow.add_node("agent_2", agent_2_function)
workflow.add_edge("agent_1", "agent_2")
workflow.set_entry_point("agent_1")

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [], "result": "", "metadata": {}})
```

---

## Core Patterns

### 1. Sequential Pipeline

**When to Use**: Tasks that must be completed in a specific order, where each step depends on the previous step's output.

**Use Cases**:
- Research → Analysis → Writing → Review
- Data extraction → Transformation → Validation → Storage
- Plan generation → Execution → Verification
- Code generation → Testing → Documentation

#### Architecture Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent A   │────▶│   Agent B   │────▶│   Agent C   │
│  (Research) │     │  (Analyze)  │     │   (Write)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      └────── State flows forward ──────────────┘
```

#### Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
import operator

# Define state
class PipelineState(TypedDict):
    """State that flows through the pipeline."""
    topic: str
    raw_data: Annotated[List[str], operator.add]  # Accumulates
    analysis: str
    final_report: str
    errors: Annotated[List[str], operator.add]

# Initialize models
researcher_llm = ChatOllama(model="qwen3:8b")
analyst_llm = ChatOllama(model="qwen3:30b-a3b")
writer_llm = ChatOllama(model="gemma3:12b")

# Define agents
def research_agent(state: PipelineState) -> PipelineState:
    """Step 1: Gather information on the topic."""
    try:
        prompt = f"Research and list 5 key facts about: {state['topic']}"
        response = researcher_llm.invoke([HumanMessage(content=prompt)])

        # Parse response into list
        facts = [line.strip() for line in response.content.split('\n')
                 if line.strip() and not line.strip().startswith('#')]

        return {"raw_data": facts}
    except Exception as e:
        return {"errors": [f"Research failed: {str(e)}"]}

def analysis_agent(state: PipelineState) -> PipelineState:
    """Step 2: Analyze the gathered data."""
    if state.get("errors"):
        return {}  # Skip if previous step failed

    try:
        data_text = "\n".join(state["raw_data"])
        prompt = f"""Analyze these research findings about {state['topic']}:

{data_text}

Provide key insights, patterns, and implications."""

        response = analyst_llm.invoke([HumanMessage(content=prompt)])
        return {"analysis": response.content}
    except Exception as e:
        return {"errors": [f"Analysis failed: {str(e)}"]}

def writing_agent(state: PipelineState) -> PipelineState:
    """Step 3: Write comprehensive report."""
    if state.get("errors"):
        return {}

    try:
        prompt = f"""Write a comprehensive report on {state['topic']}.

Research Findings:
{chr(10).join(state['raw_data'])}

Analysis:
{state['analysis']}

Create a well-structured report with introduction, body, and conclusion."""

        response = writer_llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": response.content}
    except Exception as e:
        return {"errors": [f"Writing failed: {str(e)}"]}

# Build sequential graph
workflow = StateGraph(PipelineState)

# Add nodes in sequence
workflow.add_node("research", research_agent)
workflow.add_node("analyze", analysis_agent)
workflow.add_node("write", writing_agent)

# Connect in linear sequence
workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "write")
workflow.add_edge("write", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({
    "topic": "mechanistic interpretability in transformers",
    "raw_data": [],
    "analysis": "",
    "final_report": "",
    "errors": []
})

if result["errors"]:
    print("Pipeline failed with errors:", result["errors"])
else:
    print("Final Report:\n", result["final_report"])
```

#### Best Practices

- **Error Propagation**: Check for errors from previous steps before executing
- **State Validation**: Validate state at each step to ensure required data exists
- **Logging**: Log entry/exit from each agent for debugging
- **Idempotency**: Design agents to be safe to retry
- **Checkpointing**: Use `SqliteSaver` for long pipelines to recover from failures

---

### 2. Parallel Fan-Out/Fan-In

**When to Use**: Multiple independent operations can execute simultaneously, then results are combined.

**Use Cases**:
- Multi-source data gathering (APIs, databases, files)
- Parallel model inference for comparison
- Distributed task processing
- Multi-modal analysis (text + image + audio)

#### Architecture Diagram

```
                    ┌─────────────┐
              ┌────▶│   Agent A   │────┐
              │     │  (Source 1) │    │
┌──────────┐  │     └─────────────┘    │     ┌──────────┐
│  Router  │──┤                        ├────▶│ Combiner │
│ (Split)  │  │     ┌─────────────┐    │     │ (Join)   │
└──────────┘  ├────▶│   Agent B   │────┤     └──────────┘
              │     │  (Source 2) │    │
              │     └─────────────┘    │
              │                        │
              └────▶│   Agent C   │────┘
                    │  (Source 3) │
                    └─────────────┘

        Fan-Out              Fan-In
```

#### Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import operator
import asyncio

# Define state
class ParallelState(TypedDict):
    """State for parallel processing."""
    query: str
    results: Annotated[Dict[str, str], lambda x, y: {**x, **y}]  # Merge dicts
    combined_analysis: str
    errors: Annotated[List[str], operator.add]

# Initialize different models
fast_model = ChatOllama(model="qwen3:30b-a3b")  # MoE, fast
accurate_model = ChatOllama(model="qwen3:8b")   # Dense, accurate
creative_model = ChatOllama(model="gemma3:12b") # Good at creative tasks

# Define parallel agents
def agent_speed(state: ParallelState) -> ParallelState:
    """Fast model optimized for speed."""
    try:
        response = fast_model.invoke([HumanMessage(content=state["query"])])
        return {"results": {"speed": response.content}}
    except Exception as e:
        return {"errors": [f"Speed agent failed: {str(e)}"]}

def agent_accuracy(state: ParallelState) -> ParallelState:
    """Accurate model optimized for correctness."""
    try:
        response = accurate_model.invoke([HumanMessage(content=state["query"])])
        return {"results": {"accuracy": response.content}}
    except Exception as e:
        return {"errors": [f"Accuracy agent failed: {str(e)}"]}

def agent_creativity(state: ParallelState) -> ParallelState:
    """Creative model for novel solutions."""
    try:
        response = creative_model.invoke([HumanMessage(content=state["query"])])
        return {"results": {"creativity": response.content}}
    except Exception as e:
        return {"errors": [f"Creativity agent failed: {str(e)}"]}

def combiner(state: ParallelState) -> ParallelState:
    """Combine results from all parallel agents."""
    if not state.get("results"):
        return {"combined_analysis": "No results to combine"}

    # Synthesize responses
    results_text = "\n\n".join([
        f"**{name.upper()} Model Response:**\n{content}"
        for name, content in state["results"].items()
    ])

    prompt = f"""You have received responses from multiple AI models for this query:
"{state['query']}"

Responses:
{results_text}

Synthesize these responses into a single, comprehensive answer that leverages the
strengths of each model. Identify areas of agreement and disagreement."""

    try:
        response = accurate_model.invoke([HumanMessage(content=prompt)])
        return {"combined_analysis": response.content}
    except Exception as e:
        return {"errors": [f"Combiner failed: {str(e)}"]}

# Build parallel graph
workflow = StateGraph(ParallelState)

# Add all nodes
workflow.add_node("speed", agent_speed)
workflow.add_node("accuracy", agent_accuracy)
workflow.add_node("creativity", agent_creativity)
workflow.add_node("combine", combiner)

# Set multiple entry points (all parallel agents start simultaneously)
workflow.set_entry_point("speed")
workflow.set_entry_point("accuracy")
workflow.set_entry_point("creativity")

# All parallel agents feed into combiner
workflow.add_edge("speed", "combine")
workflow.add_edge("accuracy", "combine")
workflow.add_edge("creativity", "combine")
workflow.add_edge("combine", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({
    "query": "Explain quantum computing to a 10-year-old",
    "results": {},
    "combined_analysis": "",
    "errors": []
})

print("Combined Analysis:\n", result["combined_analysis"])
```

#### Best Practices

- **Independent Tasks**: Ensure agents don't depend on each other's results
- **Timeout Handling**: Set timeouts to prevent one slow agent from blocking
- **Partial Results**: Design combiner to handle missing results gracefully
- **Resource Limits**: Monitor memory/CPU when running many parallel agents
- **Result Ordering**: Don't assume any execution order

---

### 3. Conditional Routing

**When to Use**: Workflow path depends on runtime conditions, content analysis, or agent outputs.

**Use Cases**:
- Content classification → specialized handlers
- Quality checks → approve or retry
- Error handling → fallback strategies
- Dynamic workflow adaptation based on complexity

#### Architecture Diagram

```
┌──────────┐
│ Analyzer │
└─────┬────┘
      │
      ▼
  ┌───────┐
  │Router │
  └───┬───┘
      │
      ├────── condition A ────▶ ┌─────────┐
      │                         │ Agent A │
      │                         └─────────┘
      │
      ├────── condition B ────▶ ┌─────────┐
      │                         │ Agent B │
      │                         └─────────┘
      │
      └────── condition C ────▶ ┌─────────┐
                                │ Agent C │
                                └─────────┘
```

#### Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import operator

# Define state
class RoutingState(TypedDict):
    """State for conditional routing."""
    user_query: str
    query_type: str  # "code", "creative", "factual", "conversational"
    complexity: str  # "simple", "complex"
    response: str
    retries: int
    approved: bool

# Initialize models
classifier_llm = ChatOllama(model="qwen3:8b")
simple_llm = ChatOllama(model="gemma3:4b")    # Fast for simple queries
complex_llm = ChatOllama(model="qwen3:30b-a3b")  # Powerful for complex
code_llm = ChatOllama(model="qwen3:8b")       # Optimized for code

def classifier(state: RoutingState) -> RoutingState:
    """Classify the query to determine routing."""
    prompt = f"""Classify this query into exactly one category and complexity level.

Query: "{state['user_query']}"

Categories: code, creative, factual, conversational
Complexity: simple, complex

Respond ONLY with: category,complexity
Example: code,complex"""

    response = classifier_llm.invoke([HumanMessage(content=prompt)])

    # Parse response
    try:
        parts = response.content.strip().lower().split(',')
        query_type = parts[0].strip()
        complexity = parts[1].strip() if len(parts) > 1 else "simple"
    except:
        query_type = "conversational"
        complexity = "simple"

    return {
        "query_type": query_type,
        "complexity": complexity
    }

def route_query(state: RoutingState) -> str:
    """Determine which agent to route to based on classification."""
    query_type = state.get("query_type", "conversational")
    complexity = state.get("complexity", "simple")

    # Routing logic
    if query_type == "code":
        return "code_agent"
    elif complexity == "complex":
        return "complex_agent"
    else:
        return "simple_agent"

def simple_agent(state: RoutingState) -> RoutingState:
    """Handle simple queries with fast model."""
    response = simple_llm.invoke([HumanMessage(content=state["user_query"])])
    return {"response": response.content}

def complex_agent(state: RoutingState) -> RoutingState:
    """Handle complex queries with powerful model."""
    response = complex_llm.invoke([HumanMessage(content=state["user_query"])])
    return {"response": response.content}

def code_agent(state: RoutingState) -> RoutingState:
    """Handle code-related queries."""
    prompt = f"""You are a coding expert. Answer this query with code examples:

{state['user_query']}"""

    response = code_llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}

def quality_check(state: RoutingState) -> RoutingState:
    """Check if response meets quality standards."""
    prompt = f"""Evaluate this response quality:

Query: {state['user_query']}
Response: {state['response']}

Is this response comprehensive and accurate? Answer only: yes or no"""

    response = classifier_llm.invoke([HumanMessage(content=prompt)])
    approved = "yes" in response.content.lower()

    return {"approved": approved, "retries": state.get("retries", 0) + 1}

def route_after_check(state: RoutingState) -> str:
    """Route based on quality check results."""
    if state.get("approved"):
        return "end"
    elif state.get("retries", 0) >= 2:
        return "end"  # Max retries reached
    else:
        # Retry with more powerful model
        return "complex_agent"

# Build conditional graph
workflow = StateGraph(RoutingState)

# Add nodes
workflow.add_node("classify", classifier)
workflow.add_node("simple_agent", simple_agent)
workflow.add_node("complex_agent", complex_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("quality_check", quality_check)

# Set entry point
workflow.set_entry_point("classify")

# Conditional routing from classifier
workflow.add_conditional_edges(
    "classify",
    route_query,
    {
        "simple_agent": "simple_agent",
        "complex_agent": "complex_agent",
        "code_agent": "code_agent",
    }
)

# All agents go to quality check
workflow.add_edge("simple_agent", "quality_check")
workflow.add_edge("complex_agent", "quality_check")
workflow.add_edge("code_agent", "quality_check")

# Conditional routing after quality check
workflow.add_conditional_edges(
    "quality_check",
    route_after_check,
    {
        "complex_agent": "complex_agent",
        "end": END
    }
)

# Compile
app = workflow.compile()

# Test with different query types
queries = [
    "What's the weather like?",  # Simple conversational
    "Explain quantum entanglement and its implications for quantum computing",  # Complex factual
    "Write a Python function to implement quicksort",  # Code
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    result = app.invoke({
        "user_query": query,
        "query_type": "",
        "complexity": "",
        "response": "",
        "retries": 0,
        "approved": False
    })

    print(f"Classified as: {result['query_type']} ({result['complexity']})")
    print(f"Retries: {result['retries']}")
    print(f"Approved: {result['approved']}")
    print(f"\nResponse:\n{result['response'][:300]}...")
```

#### Best Practices

- **Clear Decision Criteria**: Make routing logic explicit and testable
- **Default Paths**: Always have a fallback route
- **Loop Detection**: Prevent infinite retry loops with max iteration counters
- **State Inspection**: Log routing decisions for debugging
- **Type Safety**: Use Literal types for route names

---

### 4. Human-in-the-Loop

**When to Use**: Human judgment required for critical decisions, quality validation, or task completion.

**Use Cases**:
- Content moderation before publication
- Approval workflows for financial/legal decisions
- Iterative refinement with user feedback
- Training data collection and validation

#### Architecture Diagram

```
┌─────────┐     ┌─────────┐     ┌──────────┐
│ Agent A │────▶│  Human  │────▶│ Agent B  │
│(Draft)  │     │ Review  │     │(Revise)  │
└─────────┘     └─────────┘     └──────────┘
                     │
                     ├─── Approve ───▶ END
                     │
                     └─── Reject ────▶ (restart)
```

#### Implementation

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import operator
import sqlite3

# Define state
class HITLState(TypedDict):
    """State for human-in-the-loop workflow."""
    task_description: str
    draft: str
    human_feedback: str
    revision_count: int
    final_output: str
    status: Literal["draft", "review", "approved", "revision"]

# Initialize
llm = ChatOllama(model="qwen3:8b")

def draft_agent(state: HITLState) -> HITLState:
    """Create initial draft."""
    prompt = f"""Create content based on this request:
{state['task_description']}

Make it comprehensive and professional."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft": response.content,
        "status": "review",
        "revision_count": state.get("revision_count", 0)
    }

def human_review(state: HITLState) -> HITLState:
    """Interrupt workflow for human review."""
    print("\n" + "="*60)
    print("HUMAN REVIEW REQUIRED")
    print("="*60)
    print(f"\nTask: {state['task_description']}")
    print(f"\nDraft (revision {state['revision_count']}):")
    print("-" * 60)
    print(state['draft'])
    print("-" * 60)

    # In production, this would pause and wait for human input via API/UI
    # For demo, we'll simulate input
    feedback = input("\nProvide feedback (or 'approve' to accept): ").strip()

    if feedback.lower() == 'approve':
        return {
            "status": "approved",
            "final_output": state['draft'],
            "human_feedback": "Approved"
        }
    else:
        return {
            "status": "revision",
            "human_feedback": feedback
        }

def revision_agent(state: HITLState) -> HITLState:
    """Revise based on human feedback."""
    prompt = f"""Revise this draft based on the feedback:

Original draft:
{state['draft']}

Human feedback:
{state['human_feedback']}

Create an improved version that addresses all feedback."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft": response.content,
        "status": "review",
        "revision_count": state.get("revision_count", 0) + 1
    }

def route_after_review(state: HITLState) -> str:
    """Route based on human decision."""
    status = state.get("status", "review")

    if status == "approved":
        return "end"
    elif state.get("revision_count", 0) >= 3:
        # Max revisions reached
        print("\nMax revisions reached. Finalizing current draft.")
        return "end"
    else:
        return "revise"

# Build graph with checkpoint for pausing
workflow = StateGraph(HITLState)

workflow.add_node("draft", draft_agent)
workflow.add_node("review", human_review)
workflow.add_node("revise", revision_agent)

workflow.set_entry_point("draft")
workflow.add_edge("draft", "review")

# Conditional routing after human review
workflow.add_conditional_edges(
    "review",
    route_after_review,
    {
        "revise": "revise",
        "end": END
    }
)

# Revisions loop back to review
workflow.add_edge("revise", "review")

# Compile with checkpointer for persistence
conn = sqlite3.connect("./data/hitl_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

# Execute with thread ID for persistence
thread_id = "content_review_001"
config = {"configurable": {"thread_id": thread_id}}

result = app.invoke({
    "task_description": "Write a blog post introduction about local-first AI development",
    "draft": "",
    "human_feedback": "",
    "revision_count": 0,
    "final_output": "",
    "status": "draft"
}, config)

print("\n" + "="*60)
print("FINAL OUTPUT")
print("="*60)
print(result["final_output"])
print(f"\nTotal revisions: {result['revision_count']}")
```

#### Advanced: Async HITL with Interrupts

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Build graph with interrupt_before
workflow = StateGraph(HITLState)
workflow.add_node("draft", draft_agent)
workflow.add_node("review", human_review)
workflow.add_node("revise", revision_agent)

# ... add edges ...

# Compile with interrupt before human review
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"]  # Pause before human review
)

# Initial invocation - will stop at review
config = {"configurable": {"thread_id": "async_review_001"}}
result = app.invoke(initial_state, config)

# ... human reviews in separate system, provides feedback ...

# Resume with human feedback
state_with_feedback = {**result, "human_feedback": "Make it more concise"}
result = app.invoke(state_with_feedback, config)
```

#### Best Practices

- **Clear Prompts**: Provide context and options to humans
- **Timeout Handling**: Set deadlines for human responses
- **State Persistence**: Use checkpointers to preserve state across sessions
- **Notification System**: Alert humans when their input is needed
- **Audit Trail**: Log all human decisions for compliance

---

### 5. Hierarchical Delegation

**When to Use**: Complex tasks that require breaking down into subtasks with specialized sub-agents.

**Use Cases**:
- Project management (planner → workers → integrator)
- Software development (architect → developers → testers)
- Research paper generation (outline → sections → editing)
- Multi-step problem solving with specialized experts

#### Architecture Diagram

```
                ┌─────────────┐
                │  Supervisor │
                │  (Planner)  │
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   Worker 1   │ │ Worker 2 │ │   Worker 3   │
│ (Specialist) │ │(Generalist)│(Specialist) │
└──────┬───────┘ └─────┬────┘ └──────┬───────┘
       │               │              │
       └───────────────┼──────────────┘
                       ▼
               ┌──────────────┐
               │  Integrator  │
               │  (Combiner)  │
               └──────────────┘
```

#### Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import operator

# Define hierarchical state
class HierarchicalState(TypedDict):
    """State for hierarchical delegation."""
    main_task: str
    subtasks: List[Dict[str, str]]  # [{"task": "...", "assigned_to": "..."}]
    completed_subtasks: Annotated[Dict[str, str], lambda x, y: {**x, **y}]
    final_result: str
    current_subtask_index: int

# Initialize models
supervisor_llm = ChatOllama(model="qwen3:30b-a3b")  # Best model for planning
researcher_llm = ChatOllama(model="qwen3:8b")
coder_llm = ChatOllama(model="qwen3:8b")
writer_llm = ChatOllama(model="gemma3:12b")

def supervisor(state: HierarchicalState) -> HierarchicalState:
    """Break down main task into subtasks and assign to workers."""
    prompt = f"""You are a project supervisor. Break down this task into 3-5 concrete subtasks:

Task: {state['main_task']}

For each subtask, specify:
1. Clear description
2. Which specialist should handle it: researcher, coder, or writer

Format as:
[researcher] Subtask description
[coder] Subtask description
[writer] Subtask description"""

    response = supervisor_llm.invoke([HumanMessage(content=prompt)])

    # Parse subtasks
    subtasks = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line.startswith('['):
            # Extract specialist and task
            try:
                end_bracket = line.index(']')
                specialist = line[1:end_bracket]
                task = line[end_bracket+1:].strip()
                subtasks.append({
                    "task": task,
                    "assigned_to": specialist
                })
            except:
                continue

    return {
        "subtasks": subtasks,
        "current_subtask_index": 0
    }

def researcher_worker(state: HierarchicalState) -> HierarchicalState:
    """Handle research subtasks."""
    idx = state["current_subtask_index"]
    subtask = state["subtasks"][idx]

    prompt = f"Research and provide detailed information: {subtask['task']}"
    response = researcher_llm.invoke([HumanMessage(content=prompt)])

    return {
        "completed_subtasks": {f"subtask_{idx}": response.content},
        "current_subtask_index": idx + 1
    }

def coder_worker(state: HierarchicalState) -> HierarchicalState:
    """Handle coding subtasks."""
    idx = state["current_subtask_index"]
    subtask = state["subtasks"][idx]

    prompt = f"Write code to accomplish: {subtask['task']}"
    response = coder_llm.invoke([HumanMessage(content=prompt)])

    return {
        "completed_subtasks": {f"subtask_{idx}": response.content},
        "current_subtask_index": idx + 1
    }

def writer_worker(state: HierarchicalState) -> HierarchicalState:
    """Handle writing subtasks."""
    idx = state["current_subtask_index"]
    subtask = state["subtasks"][idx]

    prompt = f"Write content for: {subtask['task']}"
    response = writer_llm.invoke([HumanMessage(content=prompt)])

    return {
        "completed_subtasks": {f"subtask_{idx}": response.content},
        "current_subtask_index": idx + 1
    }

def router(state: HierarchicalState) -> str:
    """Route to appropriate worker based on assignment."""
    idx = state["current_subtask_index"]

    # Check if all subtasks completed
    if idx >= len(state["subtasks"]):
        return "integrator"

    # Route to assigned worker
    assigned_to = state["subtasks"][idx]["assigned_to"]

    if "research" in assigned_to.lower():
        return "researcher"
    elif "cod" in assigned_to.lower():
        return "coder"
    elif "writ" in assigned_to.lower():
        return "writer"
    else:
        return "researcher"  # Default

def integrator(state: HierarchicalState) -> HierarchicalState:
    """Combine all subtask results into final output."""
    # Compile all subtask results
    results_text = "\n\n".join([
        f"**Subtask {i+1}: {state['subtasks'][i]['task']}**\n{result}"
        for i, result in enumerate(state["completed_subtasks"].values())
    ])

    prompt = f"""Integrate these subtask results into a cohesive final deliverable:

Main Task: {state['main_task']}

Subtask Results:
{results_text}

Create a well-structured, comprehensive final output."""

    response = supervisor_llm.invoke([HumanMessage(content=prompt)])

    return {"final_result": response.content}

# Build hierarchical graph
workflow = StateGraph(HierarchicalState)

# Add all nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_worker)
workflow.add_node("coder", coder_worker)
workflow.add_node("writer", writer_worker)
workflow.add_node("integrator", integrator)

# Entry point
workflow.set_entry_point("supervisor")

# Supervisor routes to workers
workflow.add_conditional_edges(
    "supervisor",
    lambda state: "researcher",  # Start with first subtask
)

# Workers route back through router to next worker or integrator
workflow.add_conditional_edges(
    "researcher",
    router,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "integrator": "integrator"
    }
)

workflow.add_conditional_edges(
    "coder",
    router,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "integrator": "integrator"
    }
)

workflow.add_conditional_edges(
    "writer",
    router,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "integrator": "integrator"
    }
)

# Integrator ends workflow
workflow.add_edge("integrator", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({
    "main_task": "Create a tutorial on building RAG systems with local models",
    "subtasks": [],
    "completed_subtasks": {},
    "final_result": "",
    "current_subtask_index": 0
})

print("Subtasks Created:")
for i, subtask in enumerate(result["subtasks"]):
    print(f"{i+1}. [{subtask['assigned_to']}] {subtask['task']}")

print("\n" + "="*60)
print("FINAL INTEGRATED RESULT")
print("="*60)
print(result["final_result"])
```

#### Best Practices

- **Clear Task Decomposition**: Supervisor must create well-defined, independent subtasks
- **Worker Specialization**: Design workers with specific expertise
- **Progress Tracking**: Monitor which subtasks are completed
- **Error Isolation**: One failing subtask shouldn't block others
- **Context Passing**: Integrator needs enough context from all subtasks

---

## State Management Best Practices

### State Schema Design

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class WellDesignedState(TypedDict):
    # Accumulating fields - use Annotated with operator
    messages: Annotated[List[BaseMessage], operator.add]
    errors: Annotated[List[str], operator.add]

    # Merging dicts
    metadata: Annotated[dict, lambda x, y: {**x, **y}]

    # Single values (last write wins)
    current_step: str
    result: str

    # Counters
    retry_count: int
    token_usage: int
```

### Key Principles

1. **Immutability**: Treat state updates as additions, not mutations
2. **Type Safety**: Use TypedDict for clear contracts
3. **Accumulation Strategy**: Choose appropriate reducer for each field
4. **Minimal State**: Only include data that flows between agents
5. **Validation**: Validate state shape at graph compilation

### Advanced State Patterns

#### Nested State with Subgraphs

```python
class ParentState(TypedDict):
    input: str
    subgraph_results: List[str]

class ChildState(TypedDict):
    input: str
    output: str

# Child graph
child_graph = StateGraph(ChildState)
# ... define child nodes ...
child_app = child_graph.compile()

# Parent graph uses child
def parent_node(state: ParentState):
    result = child_app.invoke({"input": state["input"]})
    return {"subgraph_results": [result["output"]]}
```

#### State Checkpointing

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Enable persistence
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = workflow.compile(checkpointer=checkpointer)

# Use with thread ID
config = {"configurable": {"thread_id": "session_123"}}
result = app.invoke(initial_state, config)

# Resume from checkpoint
resumed = app.invoke(None, config)  # Continues from last state
```

---

## Pattern Selection Guide

| Pattern | Complexity | Dependencies | Parallelizable | Best For |
|---------|-----------|--------------|----------------|----------|
| **Sequential Pipeline** | Low | Strong (each step needs previous) | No | Data transformation, multi-stage processing |
| **Parallel Fan-Out** | Medium | None (independent tasks) | Yes | Multi-source data gathering, model comparison |
| **Conditional Routing** | Medium | Conditional | Partially | Content classification, quality-based routing |
| **Human-in-the-Loop** | High | Blocks on human | No | Approval workflows, iterative refinement |
| **Hierarchical Delegation** | High | Moderate (subtask dependencies) | Partially | Complex projects, specialized expertise needed |

### Decision Tree

```
Start: Do tasks depend on each other?
│
├─ No → Are tasks independent and can run simultaneously?
│   │
│   ├─ Yes → Use PARALLEL FAN-OUT/FAN-IN
│   │
│   └─ No → Use CONDITIONAL ROUTING
│
└─ Yes → Does workflow path depend on runtime conditions?
    │
    ├─ Yes → Use CONDITIONAL ROUTING
    │
    └─ No → Is task breakdown required?
        │
        ├─ Yes → Use HIERARCHICAL DELEGATION
        │
        └─ No → Is human approval needed?
            │
            ├─ Yes → Use HUMAN-IN-THE-LOOP
            │
            └─ No → Use SEQUENTIAL PIPELINE
```

---

## Performance Considerations

### Optimization Strategies

#### 1. Model Selection by Stage

```python
# Use smaller/faster models for simple tasks
classifier_llm = ChatOllama(model="gemma3:4b")  # Fast classification

# Use powerful models only when needed
reasoning_llm = ChatOllama(model="qwen3:30b-a3b")  # Complex reasoning
```

#### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_agent_call(prompt: str) -> str:
    """Cache agent responses for repeated queries."""
    return llm.invoke([HumanMessage(content=prompt)]).content
```

#### 3. Streaming for Long Operations

```python
def streaming_agent(state):
    """Stream responses for better UX."""
    for chunk in llm.stream([HumanMessage(content=state["query"])]):
        print(chunk.content, end="", flush=True)
        # Accumulate chunks
    return {"response": full_response}
```

#### 4. Parallel Execution

```python
import asyncio
from langchain_core.runnables import RunnableParallel

# Execute multiple agents in parallel
parallel = RunnableParallel(
    agent_a=agent_a_chain,
    agent_b=agent_b_chain,
    agent_c=agent_c_chain
)

results = await parallel.ainvoke({"query": user_query})
```

### Monitoring and Debugging

#### Enable LangSmith Tracing

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-system"
```

#### Custom Logging

```python
import logging

logger = logging.getLogger(__name__)

def logged_agent(state):
    """Agent with comprehensive logging."""
    logger.info(f"Entering agent with state: {state.keys()}")

    try:
        result = llm.invoke([HumanMessage(content=state["input"])])
        logger.info(f"Agent succeeded, response length: {len(result.content)}")
        return {"output": result.content}
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        raise
```

---

## Real-World Examples

### Example 1: Code Review System (Sequential + Conditional)

**File**: `examples/03-multi-agent/code_review_system.py`

```python
# agents:
# 1. linter → static analysis
# 2. security_checker → vulnerability scan
# 3. reviewer → code quality review
# 4. router → approve or request changes based on scores
```

### Example 2: Multi-Source Research (Parallel Fan-Out)

**File**: `examples/03-multi-agent/research_aggregator.py`

```python
# Parallel agents:
# - web_researcher → search online sources
# - paper_researcher → search academic papers
# - database_researcher → query internal databases
# combiner → synthesize all findings
```

### Example 3: Content Moderation (HITL)

**File**: `examples/03-multi-agent/content_moderation.py`

```python
# workflow:
# 1. auto_moderator → flag suspicious content
# 2. human_review → review flagged items (interrupt)
# 3. action_executor → publish or reject based on decision
```

### Example 4: Software Development Team (Hierarchical)

**File**: `examples/03-multi-agent/dev_team_simulator.py`

```python
# hierarchy:
# supervisor (product manager) → breaks down features
# └─ workers:
#    ├─ architect → design system
#    ├─ backend_dev → implement APIs
#    ├─ frontend_dev → build UI
#    └─ tester → write tests
# integrator → combine all outputs into deployable system
```

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Multi-Agent Systems](https://python.langchain.com/docs/use_cases/multi_agent/)
- [LangGraph Tutorials](https://github.com/langchain-ai/langgraph/tree/main/examples)
- Project Research Plan: `plans/1-research-plan.md`
- Project Examples: `plans/3-kitchen-sink-plan.md`

---

## Next Steps

1. **Start Simple**: Begin with sequential pipeline pattern
2. **Add Complexity**: Introduce conditional routing as needed
3. **Parallelize**: Identify independent operations for fan-out
4. **Human Oversight**: Add HITL for critical decision points
5. **Scale Up**: Use hierarchical delegation for complex projects

All patterns can be combined. Real-world systems often use multiple patterns in different parts of the workflow.
