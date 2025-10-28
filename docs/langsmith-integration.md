# LangSmith Integration Guide

Complete guide to integrating LangSmith for tracing, debugging, and monitoring your local-first AI applications.

---

## Table of Contents
1. [Setup & Configuration](#setup--configuration)
2. [Tracing Integration](#tracing-integration)
3. [Debugging Workflows](#debugging-workflows)
4. [Local Alternatives](#local-alternatives)
5. [Best Practices](#best-practices)
6. [Code Examples](#code-examples)

---

## Setup & Configuration

### API Key Setup (Optional for Local)

LangSmith can run in two modes:
- **Cloud Mode**: Full features with LangSmith cloud backend
- **Local Mode**: Traces stored locally without API key

#### Cloud Mode Setup

1. Create account at [smith.langchain.com](https://smith.langchain.com)
2. Generate API key from settings
3. Configure environment variables:

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=ai-lang-stuff
```

4. Load environment variables:

```python
from dotenv import load_dotenv
load_dotenv()

# Verify configuration
import os
print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
```

#### Local Mode Setup

For completely local development without cloud dependencies:

```bash
# .env.local
LANGCHAIN_TRACING_V2=true
# Omit LANGCHAIN_API_KEY and LANGCHAIN_ENDPOINT
# Traces will be logged locally
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | Yes | `false` | Enable tracing |
| `LANGCHAIN_API_KEY` | No | None | LangSmith API key (cloud only) |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | API endpoint |
| `LANGCHAIN_PROJECT` | No | `default` | Project name for organizing traces |
| `LANGCHAIN_SESSION` | No | None | Session name for grouping runs |

### Project Configuration

Organize traces by project and environment:

```python
import os

# Development
os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-dev"

# Production
os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-prod"

# Feature-specific
os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-rag-experiments"
```

### Configuration File Pattern

```python
# config/tracing.py
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os

class TracingConfig(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = None
    endpoint: str = "https://api.smith.langchain.com"
    project: str = "ai-lang-stuff"
    session: Optional[str] = None

def load_tracing_config() -> TracingConfig:
    """Load tracing configuration from environment."""
    load_dotenv()

    return TracingConfig(
        enabled=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        project=os.getenv("LANGCHAIN_PROJECT", "ai-lang-stuff"),
        session=os.getenv("LANGCHAIN_SESSION")
    )

def configure_tracing(config: Optional[TracingConfig] = None):
    """Configure LangSmith tracing."""
    if config is None:
        config = load_tracing_config()

    os.environ["LANGCHAIN_TRACING_V2"] = str(config.enabled).lower()
    if config.api_key:
        os.environ["LANGCHAIN_API_KEY"] = config.api_key
    os.environ["LANGCHAIN_ENDPOINT"] = config.endpoint
    os.environ["LANGCHAIN_PROJECT"] = config.project
    if config.session:
        os.environ["LANGCHAIN_SESSION"] = config.session
```

---

## Tracing Integration

### Basic Tracing

Enable tracing for any LangChain component:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# All invocations are automatically traced
llm = ChatOllama(model="qwen3:8b")
response = llm.invoke([HumanMessage(content="Hello!")])
```

### Custom Run Names

Make traces easier to identify:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="customer_support_query",
    tags=["support", "high-priority"]
)

response = llm.invoke(
    [HumanMessage(content="Help with order #12345")],
    config=config
)
```

### Metadata Tagging

Add contextual metadata to traces:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="rag_query",
    tags=["rag", "production", "qwen3"],
    metadata={
        "user_id": "user_123",
        "query_type": "factual",
        "model_version": "qwen3:8b",
        "timestamp": "2025-10-26T10:30:00Z",
        "environment": "local"
    }
)

response = agent_executor.invoke(
    {"input": "What is RAG?"},
    config=config
)
```

### Session Tracking

Group related runs into sessions:

```python
import os
from datetime import datetime

# Set session name
session_name = f"user_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.environ["LANGCHAIN_SESSION"] = session_name

# All subsequent runs are grouped in this session
response1 = llm.invoke([HumanMessage(content="First question")])
response2 = llm.invoke([HumanMessage(content="Follow-up question")])
response3 = llm.invoke([HumanMessage(content="Final question")])
```

### Agent Tracing

Trace multi-step agent workflows:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the local database."""
    # Tool implementation
    return f"Results for: {query}"

llm = ChatOllama(model="qwen3:8b")
tools = [search_database]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Entire agent workflow is traced
config = RunnableConfig(
    run_name="multi_step_agent",
    tags=["agent", "tool-calling"],
    metadata={"agent_type": "tool_calling"}
)

result = agent_executor.invoke(
    {"input": "Search for local models"},
    config=config
)
```

### LangGraph State Tracing

Trace stateful graph workflows:

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str

def call_model(state: AgentState):
    """Call the language model."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Determine if we should continue."""
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

graph = workflow.compile()

# Trace entire graph execution
config = RunnableConfig(
    run_name="langgraph_workflow",
    tags=["graph", "stateful"],
    metadata={"graph_type": "agent_loop"}
)

result = graph.invoke(
    {"messages": [HumanMessage(content="Solve this problem")]},
    config=config
)
```

### Custom Tracing Context

Create custom tracing contexts for fine-grained control:

```python
from langsmith import traceable
from langchain_core.tracers.context import tracing_v2_enabled

@traceable(run_type="chain", name="custom_rag_chain")
def custom_rag_workflow(query: str) -> str:
    """Custom RAG workflow with detailed tracing."""

    # Step 1: Retrieve
    docs = retrieve_documents(query)

    # Step 2: Rerank
    ranked_docs = rerank_documents(docs, query)

    # Step 3: Generate
    response = generate_response(ranked_docs, query)

    return response

@traceable(run_type="retriever")
def retrieve_documents(query: str) -> list:
    """Retrieve documents from vector store."""
    # Implementation
    return ["doc1", "doc2", "doc3"]

@traceable(run_type="reranker")
def rerank_documents(docs: list, query: str) -> list:
    """Rerank documents by relevance."""
    # Implementation
    return docs[:2]

@traceable(run_type="llm")
def generate_response(docs: list, query: str) -> str:
    """Generate final response."""
    context = "\n".join(docs)
    prompt = f"Context: {context}\n\nQuery: {query}"
    return llm.invoke([HumanMessage(content=prompt)]).content

# Execute with tracing
with tracing_v2_enabled(project_name="rag-experiments"):
    result = custom_rag_workflow("What is machine learning?")
```

---

## Debugging Workflows

### Visualizing Agent Traces

Access traces via LangSmith UI or programmatically:

```python
from langsmith import Client

client = Client()

# Get recent runs for a project
runs = client.list_runs(
    project_name="ai-lang-stuff",
    execution_order=1,  # Only root runs
    limit=10
)

for run in runs:
    print(f"Run: {run.name}")
    print(f"Status: {run.status}")
    print(f"Duration: {run.end_time - run.start_time if run.end_time else 'N/A'}")
    print(f"Tokens: {run.total_tokens}")
    print("---")
```

### Identifying Bottlenecks

Analyze performance across workflow steps:

```python
from langsmith import Client
from datetime import timedelta

client = Client()

def analyze_run_performance(run_id: str):
    """Analyze performance of a specific run."""
    run = client.read_run(run_id)
    child_runs = client.list_runs(parent_run_id=run_id)

    print(f"Total Duration: {run.end_time - run.start_time}")
    print("\nStep Breakdown:")

    for child in child_runs:
        duration = child.end_time - child.start_time if child.end_time else None
        print(f"  {child.name}: {duration}")

        # Identify slow steps
        if duration and duration > timedelta(seconds=5):
            print(f"    ⚠️  SLOW STEP DETECTED")

# Aggregate statistics
def get_project_statistics(project_name: str, days: int = 7):
    """Get aggregate statistics for a project."""
    from datetime import datetime, timedelta

    start_time = datetime.now() - timedelta(days=days)
    runs = client.list_runs(
        project_name=project_name,
        start_time=start_time
    )

    total_runs = 0
    total_tokens = 0
    total_duration = timedelta()
    error_count = 0

    for run in runs:
        total_runs += 1
        total_tokens += run.total_tokens or 0
        if run.end_time and run.start_time:
            total_duration += (run.end_time - run.start_time)
        if run.error:
            error_count += 1

    print(f"Project: {project_name}")
    print(f"Total Runs: {total_runs}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Average Duration: {total_duration / total_runs if total_runs > 0 else 0}")
    print(f"Error Rate: {error_count / total_runs * 100 if total_runs > 0 else 0:.2f}%")

get_project_statistics("ai-lang-stuff")
```

### Error Analysis

Track and analyze errors:

```python
from langsmith import Client

client = Client()

def analyze_errors(project_name: str, limit: int = 50):
    """Analyze errors in recent runs."""
    runs = client.list_runs(
        project_name=project_name,
        filter='eq(error, true)',
        limit=limit
    )

    error_types = {}

    for run in runs:
        error_msg = run.error or "Unknown error"
        error_type = error_msg.split(":")[0] if ":" in error_msg else error_msg

        if error_type not in error_types:
            error_types[error_type] = []

        error_types[error_type].append({
            "run_id": run.id,
            "timestamp": run.start_time,
            "message": error_msg
        })

    print("Error Analysis:")
    for error_type, occurrences in sorted(
        error_types.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        print(f"\n{error_type}: {len(occurrences)} occurrences")
        print(f"  Latest: {occurrences[0]['timestamp']}")
        print(f"  Message: {occurrences[0]['message'][:100]}")

analyze_errors("ai-lang-stuff")
```

### Token Usage Tracking

Monitor token consumption:

```python
from langsmith import Client
from collections import defaultdict

client = Client()

def track_token_usage(project_name: str, days: int = 7):
    """Track token usage by model and run type."""
    from datetime import datetime, timedelta

    start_time = datetime.now() - timedelta(days=days)
    runs = client.list_runs(
        project_name=project_name,
        start_time=start_time
    )

    usage_by_model = defaultdict(lambda: {"count": 0, "tokens": 0})
    usage_by_type = defaultdict(lambda: {"count": 0, "tokens": 0})

    for run in runs:
        model = run.extra.get("metadata", {}).get("model_version", "unknown")
        run_type = run.run_type
        tokens = run.total_tokens or 0

        usage_by_model[model]["count"] += 1
        usage_by_model[model]["tokens"] += tokens

        usage_by_type[run_type]["count"] += 1
        usage_by_type[run_type]["tokens"] += tokens

    print("Token Usage by Model:")
    for model, stats in sorted(usage_by_model.items(), key=lambda x: x[1]["tokens"], reverse=True):
        print(f"  {model}:")
        print(f"    Runs: {stats['count']}")
        print(f"    Tokens: {stats['tokens']:,}")
        print(f"    Avg: {stats['tokens'] / stats['count'] if stats['count'] > 0 else 0:.0f}")

    print("\nToken Usage by Type:")
    for run_type, stats in sorted(usage_by_type.items(), key=lambda x: x[1]["tokens"], reverse=True):
        print(f"  {run_type}:")
        print(f"    Runs: {stats['count']}")
        print(f"    Tokens: {stats['tokens']:,}")

track_token_usage("ai-lang-stuff")
```

### Debugging Helper Script

```python
# scripts/debug_traces.py
"""
Debug LangSmith traces for a specific run.

Usage:
    python scripts/debug_traces.py <run_id>
"""

import sys
from langsmith import Client

def debug_run(run_id: str):
    """Print detailed debug information for a run."""
    client = Client()

    # Get main run
    run = client.read_run(run_id)

    print("=" * 80)
    print(f"Run Debug Information: {run.name}")
    print("=" * 80)

    print(f"\nStatus: {run.status}")
    print(f"Run Type: {run.run_type}")
    print(f"Start Time: {run.start_time}")
    print(f"End Time: {run.end_time}")

    if run.end_time and run.start_time:
        duration = run.end_time - run.start_time
        print(f"Duration: {duration}")

    print(f"\nInputs:")
    for key, value in (run.inputs or {}).items():
        print(f"  {key}: {str(value)[:200]}")

    print(f"\nOutputs:")
    for key, value in (run.outputs or {}).items():
        print(f"  {key}: {str(value)[:200]}")

    if run.error:
        print(f"\n❌ Error:")
        print(f"  {run.error}")

    print(f"\nMetadata:")
    for key, value in (run.extra.get("metadata", {}) or {}).items():
        print(f"  {key}: {value}")

    print(f"\nTags: {run.tags}")

    if run.total_tokens:
        print(f"\nTokens: {run.total_tokens:,}")

    # Get child runs
    child_runs = list(client.list_runs(parent_run_id=run_id))

    if child_runs:
        print(f"\n{'=' * 80}")
        print(f"Child Runs ({len(child_runs)}):")
        print("=" * 80)

        for i, child in enumerate(child_runs, 1):
            print(f"\n{i}. {child.name} ({child.run_type})")
            print(f"   Status: {child.status}")
            if child.end_time and child.start_time:
                duration = child.end_time - child.start_time
                print(f"   Duration: {duration}")
            if child.error:
                print(f"   ❌ Error: {child.error}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_traces.py <run_id>")
        sys.exit(1)

    debug_run(sys.argv[1])
```

---

## Local Alternatives

### LangSmith Local Mode

Run LangSmith without cloud dependency:

```python
import os

# Disable cloud endpoints
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Don't set LANGCHAIN_API_KEY or LANGCHAIN_ENDPOINT

# Traces are logged to console and local files
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b")
response = llm.invoke("Hello!")
# Output includes trace information
```

### OpenLLMetry Integration

Alternative observability platform:

```bash
# Install OpenLLMetry
uv add opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-langchain
```

```python
# config/openllmetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

def setup_openllmetry():
    """Configure OpenLLMetry for local tracing."""

    # Set up tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Use console exporter (or file/OTLP exporter)
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Instrument LangChain
    LangchainInstrumentor().instrument()

    print("OpenLLMetry configured for local tracing")

# Usage
setup_openllmetry()

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:8b")
response = llm.invoke("Hello!")
# Traces output to console
```

### Custom Logging Solution

Build custom local tracing:

```python
# utils/local_tracer.py
import json
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from contextlib import contextmanager

class LocalTracer:
    """Simple local tracing for LangChain operations."""

    def __init__(self, log_dir: str = "logs/traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace = None

    @contextmanager
    def trace_run(
        self,
        run_name: str,
        run_type: str = "chain",
        metadata: Optional[dict] = None
    ):
        """Context manager for tracing a run."""
        trace_id = f"{run_name}_{int(time.time())}"

        trace = {
            "trace_id": trace_id,
            "run_name": run_name,
            "run_type": run_type,
            "metadata": metadata or {},
            "start_time": datetime.now().isoformat(),
            "children": []
        }

        previous_trace = self.current_trace
        self.current_trace = trace

        try:
            yield trace
            trace["status"] = "success"
        except Exception as e:
            trace["status"] = "error"
            trace["error"] = str(e)
            raise
        finally:
            trace["end_time"] = datetime.now().isoformat()
            trace["duration"] = (
                datetime.fromisoformat(trace["end_time"]) -
                datetime.fromisoformat(trace["start_time"])
            ).total_seconds()

            # Save trace
            self._save_trace(trace)

            # Restore previous trace
            self.current_trace = previous_trace

    def _save_trace(self, trace: dict):
        """Save trace to JSON file."""
        filename = f"{trace['trace_id']}.json"
        filepath = self.log_dir / filename

        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)

        print(f"Trace saved: {filepath}")

    def add_event(self, event_name: str, data: Any = None):
        """Add event to current trace."""
        if self.current_trace:
            event = {
                "event": event_name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            self.current_trace.setdefault("events", []).append(event)

# Global tracer instance
tracer = LocalTracer()

# Usage example
with tracer.trace_run("rag_query", run_type="chain", metadata={"model": "qwen3:8b"}):
    tracer.add_event("retrieve_documents", {"query": "machine learning"})
    # ... perform retrieval

    tracer.add_event("generate_response", {"doc_count": 3})
    # ... generate response
```

### File-Based Trace Viewer

Simple HTML viewer for local traces:

```python
# scripts/view_traces.py
"""Generate HTML view of local traces."""

import json
from pathlib import Path
from datetime import datetime

def generate_trace_html(log_dir: str = "logs/traces") -> str:
    """Generate HTML view of all traces."""
    traces_path = Path(log_dir)
    trace_files = sorted(traces_path.glob("*.json"), reverse=True)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local Traces</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .trace { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }
            .success { border-left: 4px solid green; }
            .error { border-left: 4px solid red; }
            .metadata { color: #666; font-size: 0.9em; }
            .event { margin-left: 20px; padding: 5px; background: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>Local Traces</h1>
    """

    for trace_file in trace_files[:50]:  # Show last 50
        with open(trace_file) as f:
            trace = json.load(f)

        status_class = trace.get("status", "unknown")
        html += f"""
        <div class="trace {status_class}">
            <h3>{trace['run_name']} ({trace['run_type']})</h3>
            <div class="metadata">
                <strong>ID:</strong> {trace['trace_id']}<br>
                <strong>Duration:</strong> {trace.get('duration', 'N/A')}s<br>
                <strong>Status:</strong> {trace.get('status', 'unknown')}<br>
                <strong>Time:</strong> {trace['start_time']}<br>
        """

        if trace.get("metadata"):
            html += f"<strong>Metadata:</strong> {json.dumps(trace['metadata'])}<br>"

        if trace.get("error"):
            html += f"<strong>Error:</strong> {trace['error']}<br>"

        html += "</div>"

        if trace.get("events"):
            html += "<h4>Events:</h4>"
            for event in trace["events"]:
                html += f"""
                <div class="event">
                    <strong>{event['event']}</strong> @ {event['timestamp']}<br>
                    {json.dumps(event.get('data', {}), indent=2)}
                </div>
                """

        html += "</div>"

    html += "</body></html>"

    output_file = Path("logs/traces.html")
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"HTML viewer generated: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    generate_trace_html()
```

---

## Best Practices

### What to Trace

**Always Trace:**
- Production workflows
- Multi-step agent executions
- RAG pipelines
- Complex chains
- Error-prone operations
- Performance-critical paths

**Optionally Trace:**
- Development experiments
- Simple LLM calls
- Unit tests
- One-off scripts

**Never Trace:**
- Sensitive user data (use metadata filtering)
- API keys or credentials
- Personal information

### Performance Impact

Tracing has minimal overhead:

```python
import time
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b")

# Without tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
start = time.time()
for _ in range(10):
    llm.invoke("Hello")
no_trace_time = time.time() - start

# With tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
start = time.time()
for _ in range(10):
    llm.invoke("Hello")
trace_time = time.time() - start

overhead = ((trace_time - no_trace_time) / no_trace_time) * 100
print(f"Tracing overhead: {overhead:.2f}%")
# Typically < 5% for cloud, < 1% for local
```

**Optimization Tips:**
- Use sampling for high-volume operations
- Disable tracing for unit tests
- Use async tracing to reduce blocking
- Filter sensitive data before sending

### Cost Considerations

**Cloud Mode:**
- Free tier: 5K traces/month
- Paid tiers: Based on trace volume
- Consider cost for production use
- Local models = no inference cost

**Local Mode:**
- Zero cloud costs
- Storage costs (minimal, ~1KB per trace)
- No rate limits
- Full privacy

### Privacy Considerations

```python
from langchain_core.runnables import RunnableConfig

# Filter sensitive data
def sanitize_input(data: dict) -> dict:
    """Remove sensitive fields from trace data."""
    sensitive_keys = ["api_key", "password", "ssn", "credit_card"]
    return {
        k: v for k, v in data.items()
        if k.lower() not in sensitive_keys
    }

# Use metadata instead of inputs for sensitive data
config = RunnableConfig(
    metadata={
        "user_id_hash": "abc123",  # Hash instead of actual ID
        "query_type": "personal_info"  # Category, not content
    }
)

# Redact PII before tracing
def redact_pii(text: str) -> str:
    """Redact personally identifiable information."""
    import re
    # Redact emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Redact phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # Redact SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text

sanitized_query = redact_pii(user_query)
response = llm.invoke(sanitized_query, config=config)
```

### Development Workflow

```python
# config/environments.py
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

def configure_for_environment(env: Environment):
    """Configure tracing based on environment."""

    if env == Environment.DEVELOPMENT:
        # Local tracing only
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-dev"
        # No API key = local mode

    elif env == Environment.STAGING:
        # Cloud tracing with verbose metadata
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-staging"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_STAGING_KEY")

    elif env == Environment.PRODUCTION:
        # Cloud tracing with sampling
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-prod"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_PROD_KEY")
        # Enable sampling (trace 10% of requests)
        os.environ["LANGCHAIN_SAMPLING_RATE"] = "0.1"

# Usage
configure_for_environment(Environment.DEVELOPMENT)
```

---

## Code Examples

### Basic Tracing Setup

```python
# examples/02-mcp/langsmith_basic.py
"""
Basic LangSmith tracing example.

Demonstrates:
- Enabling tracing
- Simple LLM call
- Viewing trace data
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def main():
    # Load environment variables
    load_dotenv()

    # Enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "ai-lang-stuff-examples"

    print("LangSmith tracing enabled")
    print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")

    # Create LLM
    llm = ChatOllama(model="qwen3:8b", temperature=0.7)

    # Make call (automatically traced)
    response = llm.invoke([
        HumanMessage(content="Explain LangSmith in one sentence.")
    ])

    print(f"\nResponse: {response.content}")
    print("\nTrace available in LangSmith UI (if API key configured)")

if __name__ == "__main__":
    main()
```

### Advanced Configuration

```python
# examples/02-mcp/langsmith_advanced.py
"""
Advanced LangSmith tracing configuration.

Demonstrates:
- Custom run names
- Metadata tagging
- Session tracking
- Error handling
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

def main():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Create session
    session_name = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.environ["LANGCHAIN_SESSION"] = session_name

    llm = ChatOllama(model="qwen3:8b")

    # Example 1: Custom run name and tags
    print("Example 1: Custom metadata")
    config = RunnableConfig(
        run_name="sentiment_analysis",
        tags=["nlp", "sentiment", "production"],
        metadata={
            "model": "qwen3:8b",
            "task": "sentiment_analysis",
            "version": "1.0.0",
            "user_id": "demo_user"
        }
    )

    response = llm.invoke(
        [
            SystemMessage(content="Analyze sentiment as positive, negative, or neutral."),
            HumanMessage(content="I love this product!")
        ],
        config=config
    )
    print(f"Response: {response.content}\n")

    # Example 2: Error handling with tracing
    print("Example 2: Error handling")
    try:
        config = RunnableConfig(
            run_name="error_case",
            tags=["error", "debugging"],
            metadata={"expected_error": True}
        )

        # Intentional error for demonstration
        invalid_llm = ChatOllama(model="nonexistent-model")
        invalid_llm.invoke([HumanMessage(content="test")], config=config)

    except Exception as e:
        print(f"Error caught (trace preserved): {str(e)[:50]}...\n")

    # Example 3: Multi-turn conversation
    print("Example 3: Multi-turn conversation")
    conversation = [
        SystemMessage(content="You are a helpful math tutor.")
    ]

    for i, question in enumerate(["What is 2+2?", "What about 3*3?", "And 10/2?"]):
        conversation.append(HumanMessage(content=question))

        config = RunnableConfig(
            run_name=f"math_qa_turn_{i+1}",
            tags=["math", "education", "multi-turn"],
            metadata={"turn": i+1, "question": question}
        )

        response = llm.invoke(conversation, config=config)
        conversation.append(response)
        print(f"Q: {question}")
        print(f"A: {response.content}\n")

    print(f"All traces saved to session: {session_name}")
    print("View in LangSmith UI or use Client API to retrieve")

if __name__ == "__main__":
    main()
```

### Custom Evaluators

```python
# examples/02-mcp/langsmith_evaluators.py
"""
LangSmith custom evaluators for quality assessment.

Demonstrates:
- Creating custom evaluators
- Running evaluations
- Analyzing results
"""

from langchain_ollama import ChatOllama
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage

# Define test dataset
test_cases = [
    {
        "input": "What is the capital of France?",
        "expected": "Paris"
    },
    {
        "input": "What is 2+2?",
        "expected": "4"
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "expected": "Shakespeare"
    }
]

# Define evaluator
def correctness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Check if answer contains expected content."""
    answer = outputs.get("output", "").lower()
    expected = reference_outputs.get("expected", "").lower()

    is_correct = expected in answer

    return {
        "key": "correctness",
        "score": 1.0 if is_correct else 0.0,
        "comment": f"Expected '{expected}' in '{answer}'"
    }

def run_evaluation():
    """Run evaluation with LangSmith."""
    client = Client()
    llm = ChatOllama(model="qwen3:8b")

    # Create dataset
    dataset_name = "qa_test_set"

    # Check if dataset exists, create if not
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test dataset for QA evaluation"
        )

        for case in test_cases:
            client.create_example(
                inputs={"input": case["input"]},
                outputs={"expected": case["expected"]},
                dataset_id=dataset.id
            )

    # Define target function
    def qa_chain(inputs: dict) -> dict:
        """Simple QA chain."""
        response = llm.invoke([HumanMessage(content=inputs["input"])])
        return {"output": response.content}

    # Run evaluation
    print("Running evaluation...")
    results = evaluate(
        qa_chain,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="qa_eval"
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Dataset: {dataset_name}")
    print(f"Total runs: {len(test_cases)}")

    return results

if __name__ == "__main__":
    run_evaluation()
```

### Batch Operations

```python
# examples/02-mcp/langsmith_batch.py
"""
Batch operations with LangSmith tracing.

Demonstrates:
- Batch processing
- Performance comparison
- Aggregated tracing
"""

import time
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

def batch_process_sequential(prompts: List[str]) -> List[str]:
    """Process prompts sequentially."""
    llm = ChatOllama(model="qwen3:8b")
    results = []

    for i, prompt in enumerate(prompts):
        config = RunnableConfig(
            run_name=f"sequential_batch_item_{i}",
            tags=["batch", "sequential"],
            metadata={"batch_index": i, "batch_size": len(prompts)}
        )

        response = llm.invoke([HumanMessage(content=prompt)], config=config)
        results.append(response.content)

    return results

def batch_process_parallel(prompts: List[str]) -> List[str]:
    """Process prompts in parallel using batch."""
    llm = ChatOllama(model="qwen3:8b")

    config = RunnableConfig(
        run_name="parallel_batch_processing",
        tags=["batch", "parallel"],
        metadata={"batch_size": len(prompts)}
    )

    # Use batch method
    messages_list = [[HumanMessage(content=p)] for p in prompts]
    responses = llm.batch(messages_list, config=config)

    return [r.content for r in responses]

def main():
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Test prompts
    prompts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is reinforcement learning?",
        "What is supervised learning?",
        "What is unsupervised learning?"
    ]

    # Sequential processing
    print("Sequential Processing:")
    start = time.time()
    sequential_results = batch_process_sequential(prompts)
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")
    print(f"Results: {len(sequential_results)}")

    # Parallel processing
    print("\nParallel Processing:")
    start = time.time()
    parallel_results = batch_process_parallel(prompts)
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")
    print(f"Results: {len(parallel_results)}")

    # Comparison
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print("\nTraces available in LangSmith for performance analysis")

if __name__ == "__main__":
    main()
```

---

## Integration Checklist

Use this checklist when integrating LangSmith into new projects:

- [ ] Install dependencies (`langsmith`, `langchain-core`)
- [ ] Configure environment variables (`.env`)
- [ ] Set up project name and organization
- [ ] Add tracing to main workflows
- [ ] Add custom metadata for important runs
- [ ] Implement error tracking
- [ ] Set up session tracking for multi-turn conversations
- [ ] Configure sampling for production (if needed)
- [ ] Add privacy filters for sensitive data
- [ ] Create custom evaluators (if needed)
- [ ] Set up monitoring dashboards
- [ ] Document tracing patterns for team
- [ ] Test local-only mode (no API key)
- [ ] Verify traces are being captured
- [ ] Set up alerts for errors (if using cloud)

---

## Troubleshooting

### Common Issues

**Problem:** Traces not appearing in LangSmith UI

**Solutions:**
1. Verify `LANGCHAIN_TRACING_V2=true`
2. Check API key is valid
3. Verify network connectivity
4. Check project name is correct
5. Look for errors in console

**Problem:** High latency with tracing

**Solutions:**
1. Use async mode for tracing
2. Enable sampling
3. Switch to local-only mode
4. Reduce metadata size
5. Batch similar operations

**Problem:** Missing child runs in traces

**Solutions:**
1. Ensure all components are instrumented
2. Check for proper config propagation
3. Verify run context is maintained
3. Update to latest LangChain version

**Problem:** Sensitive data in traces

**Solutions:**
1. Implement PII redaction
2. Use metadata instead of inputs
3. Filter sensitive fields
4. Use local-only mode
5. Review privacy settings

---

## Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [LangChain Tracing Guide](https://python.langchain.com/docs/guides/debugging)
- [OpenTelemetry for LangChain](https://opentelemetry.io/)

---

## Summary

LangSmith provides powerful observability for local AI applications:

- **Flexible Setup**: Cloud or local-only modes
- **Comprehensive Tracing**: Full visibility into agent workflows
- **Performance Analysis**: Identify bottlenecks and optimize
- **Error Tracking**: Debug issues quickly
- **Privacy-First**: Local mode for sensitive data
- **Production-Ready**: Sampling, filtering, and monitoring

For this local-first project, use:
- **Development**: Local-only mode (no API key)
- **Production**: Cloud mode with sampling and PII filtering
- **Debugging**: Full tracing with custom metadata

Start with basic tracing, then add custom evaluators and monitoring as needed.
