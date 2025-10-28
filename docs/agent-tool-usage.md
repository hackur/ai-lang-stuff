# Agent Tool Usage Guide

## Overview

This guide documents how agents in the ai-lang-stuff project use the five core utilities to build local-first AI workflows. These utilities provide essential functionality for model management, protocol integration, knowledge retrieval, state persistence, and tool discovery.

## Available Utilities

The project provides five core utilities located in `/utils/`:

1. **ollama_manager.py** - Ollama server and model management
2. **mcp_client.py** - Model Context Protocol (MCP) integration
3. **vector_store.py** - Vector store creation and RAG operations
4. **state_manager.py** - LangGraph state persistence and checkpointing
5. **tool_registry.py** - Centralized tool registration and discovery

---

## 1. Ollama Manager

### Purpose
Manages Ollama server operations including health checks, model downloads, benchmarking, and intelligent model recommendations.

### When to Use
- Before starting any agent workflow (verify Ollama is running)
- When pulling models based on task requirements
- For model performance benchmarking
- When recommending models for specific use cases

### Core Capabilities

#### Health Checks
```python
from utils.ollama_manager import OllamaManager

manager = OllamaManager()

# Check if Ollama is running
if manager.check_ollama_running():
    print("Ollama is ready")
else:
    print("Please start Ollama: ollama serve")
```

#### Model Management
```python
# List installed models
models = manager.list_models()
print(f"Available models: {models}")

# Ensure model is available (pulls if needed)
if manager.ensure_model_available("qwen3:8b"):
    print("Model ready to use")

# Manually pull a model
manager.pull_model("gemma3:4b")

# Get detailed model information
info = manager.get_model_info("qwen3:8b")
print(f"Model parameters: {info.get('details', {}).get('parameter_size')}")
```

#### Model Recommendations
```python
# Get recommended model for specific tasks
fast_model = manager.recommend_model("fast")        # qwen3:30b-a3b (MoE)
quality_model = manager.recommend_model("quality")  # qwen3:30b
vision_model = manager.recommend_model("vision")    # qwen3-vl:8b
edge_model = manager.recommend_model("edge")        # gemma3:4b

# Task type mapping
# - "fast": Speed-optimized (qwen3:30b-a3b)
# - "balanced": Quality/speed tradeoff (qwen3:8b)
# - "quality": Best reasoning (qwen3:30b)
# - "embeddings": Text embeddings (nomic-embed-text)
# - "vision": Image understanding (qwen3-vl:8b)
# - "edge": Minimal resources (gemma3:4b)
# - "multilingual": 140+ languages (gemma3:12b)
# - "coding": Code generation (qwen3:30b-a3b)
```

#### Benchmarking
```python
# Benchmark model performance
results = manager.benchmark_model("qwen3:8b", "Explain quantum computing")
print(f"Latency: {results['latency']:.2f}s")
print(f"Throughput: {results['tokens_per_sec']:.1f} tokens/sec")
print(f"Response: {results['response'][:100]}...")
```

#### Convenience Functions
```python
from utils.ollama_manager import check_ollama, get_available_models, ensure_model

# Quick checks without instantiating manager
if check_ollama():
    models = get_available_models()
    ensure_model("qwen3:8b")
```

### Integration Pattern: Agent Initialization
```python
from langchain_ollama import ChatOllama
from utils.ollama_manager import OllamaManager

def initialize_agent(task_type: str = "balanced"):
    """Initialize agent with appropriate model for task."""
    manager = OllamaManager()

    # Verify Ollama is running
    if not manager.check_ollama_running():
        raise RuntimeError("Ollama server not running. Start with: ollama serve")

    # Get recommended model for task
    model_name = manager.recommend_model(task_type)

    # Ensure model is available
    if not manager.ensure_model_available(model_name):
        raise RuntimeError(f"Failed to load model: {model_name}")

    # Create LangChain LLM instance
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.7
    )

    return llm, model_name

# Usage
llm, model = initialize_agent("coding")  # Uses qwen3:30b-a3b for fast coding
```

### Error Handling
```python
from utils.ollama_manager import OllamaManager
from requests.exceptions import ConnectionError, Timeout

manager = OllamaManager()

try:
    manager.ensure_model_available("qwen3:8b")
except ConnectionError:
    print("Cannot connect to Ollama. Is it running?")
    print("Start with: ollama serve")
except Timeout:
    print("Request timed out. Check Ollama server status.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 2. MCP Client

### Purpose
Provides production-quality clients for interacting with Model Context Protocol (MCP) servers, enabling filesystem operations and web search capabilities with automatic retry logic.

### When to Use
- When agents need filesystem access (reading/writing files)
- For web search and URL fetching capabilities
- When integrating external MCP-compatible services
- To provide tool capabilities to LangChain agents

### Core Capabilities

#### Filesystem Operations
```python
from utils.mcp_client import FilesystemMCP, MCPConfig

async def filesystem_example():
    # Configure client with custom settings
    config = MCPConfig(host="localhost", port=8001, timeout=30)

    async with FilesystemMCP(config, base_path="/safe/directory") as fs:
        # Read file
        content = await fs.read_file("/safe/directory/document.txt")

        # Write file
        success = await fs.write_file(
            "/safe/directory/output.txt",
            "Processed content"
        )

        # List directory
        entries = await fs.list_directory("/safe/directory")
        for entry in entries:
            print(f"  {entry['name']} ({entry.get('type', 'file')})")

        # Search for files
        py_files = await fs.search_files("*.py", "/safe/directory")
        print(f"Found {len(py_files)} Python files")
```

#### Web Search Operations
```python
from utils.mcp_client import WebSearchMCP, MCPConfig

async def web_search_example():
    config = MCPConfig(host="localhost", port=8002)

    async with WebSearchMCP(config) as search:
        # Search the web
        results = await search.search("LangChain documentation", num_results=5)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet}")
            print(f"   Score: {result.score}")
            print()

        # Fetch specific URL
        content = await search.fetch_url("https://python.langchain.com/docs")
        print(f"Fetched {len(content)} characters")
```

#### LangChain Tool Integration
```python
from utils.mcp_client import create_filesystem_client, create_websearch_client
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

async def create_agent_with_mcp_tools():
    # Create MCP clients
    fs_client = create_filesystem_client(port=8001, base_path="/workspace")
    search_client = create_websearch_client(port=8002)

    # Connect clients
    await fs_client.connect()
    await search_client.connect()

    # Convert to LangChain tools
    fs_tools = fs_client.to_langchain_tools()
    search_tools = search_client.to_langchain_tools()
    all_tools = fs_tools + search_tools

    # Create agent with tools
    llm = ChatOllama(model="qwen3:8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with filesystem and web access."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools)

    # Use agent
    result = await executor.ainvoke({
        "input": "Search for Python tutorials and save the top 3 URLs to links.txt"
    })

    # Cleanup
    await fs_client.disconnect()
    await search_client.disconnect()

    return result
```

#### Path Validation and Security
```python
from utils.mcp_client import FilesystemMCP
from pathlib import Path

# Restrict filesystem access to specific directory
base_path = Path("/workspace/projects")
async with FilesystemMCP(base_path=base_path) as fs:
    # This will work (inside base_path)
    await fs.read_file("/workspace/projects/file.txt")

    # This will raise ValueError (path traversal attempt)
    try:
        await fs.read_file("/workspace/../etc/passwd")
    except ValueError as e:
        print(f"Security check prevented access: {e}")
```

### Integration Pattern: Research Agent with MCP
```python
from typing import TypedDict, List
from utils.mcp_client import create_websearch_client
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    question: str
    sources: List[str]
    answer: str

async def research_node(state: ResearchState) -> ResearchState:
    """Node that searches web and returns findings."""
    search_client = create_websearch_client()

    async with search_client:
        # Search for information
        results = await search_client.search(state["question"], num_results=3)

        # Extract sources
        sources = [r.url for r in results]

        # Fetch content from top result
        content = await search_client.fetch_url(results[0].url)

        # Process with LLM (simplified)
        answer = f"Based on {len(results)} sources: {content[:200]}..."

        return {
            "question": state["question"],
            "sources": sources,
            "answer": answer
        }
```

### Error Handling
```python
from utils.mcp_client import (
    MCPError,
    MCPConnectionError,
    MCPToolError,
    FilesystemMCP
)

async def safe_mcp_operations():
    try:
        async with FilesystemMCP() as fs:
            content = await fs.read_file("/path/to/file.txt")

    except MCPConnectionError:
        print("Cannot connect to MCP server. Is it running?")
    except MCPToolError as e:
        print(f"Tool execution failed: {e}")
    except MCPError as e:
        print(f"MCP error: {e}")
```

---

## 3. Vector Store Manager

### Purpose
Manages vector store operations with Chroma and FAISS backends for RAG (Retrieval-Augmented Generation) systems using local embeddings.

### When to Use
- Building RAG systems with local document retrieval
- Creating knowledge bases from documents
- Implementing semantic search
- Providing context to LLM prompts

### Core Capabilities

#### Creating Vector Stores
```python
from utils.vector_store import VectorStoreManager
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize manager with embedding model
manager = VectorStoreManager(
    embedding_model="qwen3-embedding",  # or "nomic-embed-text"
    base_url="http://localhost:11434"
)

# Load documents
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# Create vector store with automatic chunking
vectorstore = manager.create_from_documents(
    documents=documents,
    collection_name="project_docs",
    persist_dir="./data/vector_stores",
    store_type="chroma",  # or "faiss"
    chunk_size=1000,
    chunk_overlap=100
)

print(f"Created vector store with {len(documents)} documents")
```

#### Loading Existing Vector Stores
```python
# Load previously created vector store
vectorstore = manager.load_existing(
    collection_name="project_docs",
    persist_dir="./data/vector_stores",
    store_type="chroma"
)

# Perform similarity search
results = manager.similarity_search(
    vectorstore,
    query="How do I configure agents?",
    k=3
)

for doc in results:
    print(f"Content: {doc.page_content[:200]}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print()
```

#### Similarity Search with Scores
```python
# Get results with relevance scores (lower = more similar)
results = manager.similarity_search_with_score(
    vectorstore,
    query="LangGraph state management",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:150]}...")
    print()
```

#### Managing Collections
```python
# List all collections
collections = manager.list_collections("./data/vector_stores")
print(f"Chroma collections: {collections['chroma']}")
print(f"FAISS collections: {collections['faiss']}")

# Add documents to existing collection
new_docs = [
    Document(page_content="New information", metadata={"source": "update.md"})
]
manager.add_documents(vectorstore, new_docs)

# Delete a collection
manager.delete_collection(
    collection_name="old_docs",
    persist_dir="./data/vector_stores",
    store_type="chroma"
)
```

#### Convenience Functions
```python
from utils.vector_store import (
    create_chroma_store,
    create_faiss_store,
    load_vector_store
)

# Quick Chroma creation
chroma_store = create_chroma_store(
    documents=docs,
    collection_name="quick_docs",
    persist_dir="./data/vectors"
)

# Quick FAISS creation
faiss_store = create_faiss_store(
    documents=docs,
    collection_name="fast_docs",
    persist_dir="./data/vectors"
)

# Quick load
store = load_vector_store(
    collection_name="quick_docs",
    store_type="chroma"
)
```

### Integration Pattern: RAG Agent
```python
from utils.vector_store import VectorStoreManager
from utils.ollama_manager import OllamaManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def create_rag_agent(collection_name: str):
    """Create RAG agent with vector store retrieval."""
    # Initialize utilities
    vector_mgr = VectorStoreManager(embedding_model="qwen3-embedding")
    ollama_mgr = OllamaManager()

    # Ensure embedding model available
    ollama_mgr.ensure_model_available("qwen3-embedding")

    # Load vector store
    vectorstore = vector_mgr.load_existing(
        collection_name=collection_name,
        persist_dir="./data/vector_stores"
    )

    # Initialize LLM
    llm = ChatOllama(model="qwen3:8b")

    # Create RAG chain
    def rag_query(question: str) -> str:
        # Retrieve relevant documents
        docs = vector_mgr.similarity_search(
            vectorstore,
            query=question,
            k=3
        )

        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        # Generate answer
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return response.content

    return rag_query

# Usage
rag_agent = create_rag_agent("project_docs")
answer = rag_agent("How do I create a LangGraph workflow?")
print(answer)
```

### Error Handling
```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager()

try:
    # Load vector store
    vectorstore = manager.load_existing("my_collection", "./data/vectors")

except FileNotFoundError as e:
    print(f"Collection not found: {e}")
    # List available collections
    available = manager.list_collections("./data/vectors")
    print(f"Available: {available}")

except ConnectionError as e:
    print(f"Cannot connect to Ollama: {e}")
    print("Ensure Ollama is running and embedding model is available")
    print("Try: ollama pull qwen3-embedding")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 4. State Manager

### Purpose
Manages LangGraph state persistence using SQLite checkpointers, enabling resumable multi-step workflows and conversation history.

### When to Use
- Building stateful LangGraph workflows
- Implementing conversation memory
- Creating resumable agent workflows
- Managing multi-turn interactions

### Core Capabilities

#### Creating Checkpointers
```python
from utils.state_manager import StateManager

# Create SQLite checkpointer for persistence
checkpointer = StateManager.get_checkpointer("./data/checkpoints.db")

# Use with LangGraph
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)
# ... add nodes and edges
graph = graph_builder.compile(checkpointer=checkpointer)
```

#### Creating State Schemas
```python
from typing import Annotated, List
import operator
from langchain_core.messages import BaseMessage

# Dynamic state schema creation
AgentState = StateManager.create_state_schema(
    fields={
        "messages": Annotated[List[BaseMessage], operator.add],
        "context": str,
        "iteration": int,
        "final_answer": str
    },
    class_name="AgentState"
)

# Use in workflow
def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    iteration = state.get("iteration", 0)

    # Process...

    return {
        "messages": [new_message],
        "iteration": iteration + 1
    }
```

#### Predefined State Patterns
```python
from utils.state_manager import (
    basic_agent_state,
    research_state,
    code_review_state
)

# Basic agent with message history
BasicState = basic_agent_state()

# Research workflow state
ResearchState = research_state()
# Fields: question, context, answer, sources

# Code review workflow state
CodeReviewState = code_review_state()
# Fields: code, issues, suggestions, approved
```

#### Managing Checkpoints
```python
from utils.state_manager import StateManager

# Load specific checkpoint
state = StateManager.load_checkpoint(
    thread_id="thread-123",
    db_path="./data/checkpoints.db"
)
if state:
    print(f"Loaded state with {len(state['messages'])} messages")

# List all checkpoints
checkpoints = StateManager.list_checkpoints("./data/checkpoints.db")
for cp in checkpoints:
    print(f"Thread: {cp['thread_id']}")
    print(f"Time: {cp['timestamp']}")
    print()

# Clear specific thread
deleted = StateManager.clear_checkpoints(
    db_path="./data/checkpoints.db",
    thread_id="thread-123",
    confirm=True
)
print(f"Deleted {deleted} checkpoints")

# Clear all checkpoints (use with caution!)
deleted = StateManager.clear_checkpoints(
    db_path="./data/checkpoints.db",
    confirm=True
)
```

#### Utility Functions
```python
from utils.state_manager import create_thread_id, get_checkpoint_size

# Generate unique thread ID
thread_id = create_thread_id(prefix="research")
print(thread_id)  # "research-1698765432"

# Get database statistics
stats = get_checkpoint_size("./data/checkpoints.db")
print(f"Size: {stats['file_size_mb']:.2f} MB")
print(f"Checkpoints: {stats['checkpoint_count']}")
print(f"Threads: {stats['thread_count']}")
```

### Integration Pattern: Stateful LangGraph Agent
```python
from typing import Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from utils.state_manager import StateManager, create_thread_id

# Define state
AgentState = StateManager.create_state_schema({
    "messages": Annotated[List[BaseMessage], operator.add],
    "iteration": int
})

# Create nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    iteration = state.get("iteration", 0)

    # Process with LLM
    llm = ChatOllama(model="qwen3:8b")
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "iteration": iteration + 1
    }

def should_continue(state: AgentState) -> str:
    if state["iteration"] >= 5:
        return END
    return "agent"

# Build graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue)

# Compile with checkpointing
checkpointer = StateManager.get_checkpointer("./checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)

# Run with thread ID for persistence
thread_id = create_thread_id("conversation")
config = {"configurable": {"thread_id": thread_id}}

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello!")], "iteration": 0},
    config=config
)

# Continue conversation later
result = graph.invoke(
    {"messages": [HumanMessage(content="Tell me more")]},
    config=config  # Same thread_id
)
```

### Error Handling
```python
from utils.state_manager import StateManager
import sqlite3

try:
    checkpointer = StateManager.get_checkpointer("./checkpoints.db")

except sqlite3.Error as e:
    print(f"Database error: {e}")

try:
    schema = StateManager.create_state_schema(
        fields={"messages": List[BaseMessage]},  # Missing operator.add
        class_name="State"
    )
except (ValueError, TypeError) as e:
    print(f"Schema creation failed: {e}")
```

---

## 5. Tool Registry

### Purpose
Provides centralized tool registration, discovery, and management across the project with automatic LangChain conversion.

### When to Use
- Registering custom tools for agent use
- Auto-discovering tools from utility modules
- Converting tools to LangChain format
- Organizing tools by category

### Core Capabilities

#### Registering Tools
```python
from utils.tool_registry import get_registry

# Get singleton registry
registry = get_registry()

# Register custom tool
def search_documentation(query: str, max_results: int = 5) -> str:
    """Search project documentation for relevant information."""
    # Implementation...
    return f"Found {max_results} results for '{query}'"

registry.register_tool(
    name="search_docs",
    tool=search_documentation,
    description="Search project documentation for relevant information",
    category="database"
)

# Register with Pydantic schema
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results")

registry.register_tool(
    name="search_docs_typed",
    tool=search_documentation,
    description="Search documentation with type validation",
    category="database",
    args_schema=SearchArgs
)
```

#### Listing and Filtering Tools
```python
# List all tools
all_tools = registry.list_tools()
for tool in all_tools:
    print(f"{tool['name']}: {tool['description']} [{tool['category']}]")

# Filter by category
web_tools = registry.list_tools(category="web")
model_tools = registry.list_tools(category="models")

# Get specific tool
search_tool = registry.get_tool("search_docs")
result = search_tool("LangGraph", max_results=3)
```

#### Auto-Discovery
```python
# Auto-discover tools from utils/ directory
discovered = registry.auto_discover_utilities()
print(f"Discovered {discovered} tools")

# Tools are automatically categorized:
# - ollama_manager.* -> "models"
# - mcp_client.* -> "web" or "filesystem"
# - vector_store.* -> "database"
# - state_manager.* -> "workflow"
# - tool_registry.* -> "other"
```

#### LangChain Conversion
```python
# Convert all tools to LangChain format
lc_tools = registry.get_langchain_tools()

# Filter by categories
agent_tools = registry.get_langchain_tools(categories=["web", "filesystem"])

# Use in agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, lc_tools, prompt)
executor = AgentExecutor(agent=agent, tools=lc_tools)
```

#### Exporting Registry
```python
from pathlib import Path

# Export to JSON
json_str = registry.to_json()
print(json_str)

# Save to file
registry.to_json(filepath=Path("./data/tool_registry.json"))

# Inspect registry
print(repr(registry))
# Output: ToolRegistry(tools=15, categories=['web', 'filesystem', 'models', 'database', 'workflow'])
```

### Integration Pattern: Multi-Tool Agent
```python
from utils.tool_registry import get_registry
from utils.ollama_manager import OllamaManager
from utils.mcp_client import create_filesystem_client, create_websearch_client
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

async def create_multi_tool_agent():
    """Create agent with filesystem, web, and custom tools."""
    # Initialize registry
    registry = get_registry()
    registry.clear()  # Start fresh

    # Register custom tools
    def calculate(expression: str) -> float:
        """Safely evaluate mathematical expressions."""
        try:
            return float(eval(expression, {"__builtins__": {}}, {}))
        except:
            return 0.0

    registry.register_tool(
        name="calculate",
        tool=calculate,
        description="Evaluate mathematical expressions",
        category="other"
    )

    # Add MCP tools
    fs_client = create_filesystem_client(base_path="./workspace")
    search_client = create_websearch_client()

    await fs_client.connect()
    await search_client.connect()

    fs_tools = fs_client.to_langchain_tools()
    search_tools = search_client.to_langchain_tools()

    # Combine all tools
    custom_tools = registry.get_langchain_tools()
    all_tools = custom_tools + fs_tools + search_tools

    # Create agent
    llm_mgr = OllamaManager()
    llm_mgr.ensure_model_available("qwen3:8b")

    llm = ChatOllama(model="qwen3:8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to filesystem, web search, and calculator."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

    return executor, fs_client, search_client

# Usage
executor, fs, search = await create_multi_tool_agent()

result = await executor.ainvoke({
    "input": "Search for Python tutorials, calculate 15 * 23, and save the result to calc.txt"
})

# Cleanup
await fs.disconnect()
await search.disconnect()
```

### Error Handling
```python
from utils.tool_registry import get_registry

registry = get_registry()

try:
    # Register non-callable (fails)
    registry.register_tool("bad_tool", "not callable", "desc", "web")
except TypeError as e:
    print(f"Registration failed: {e}")

try:
    # Get non-existent tool
    tool = registry.get_tool("nonexistent")
except KeyError as e:
    print(f"Tool not found: {e}")
    available = registry.list_tools()
    print(f"Available tools: {[t['name'] for t in available]}")
```

---

## Integration Patterns

### Complete Agent Workflow
```python
from typing import Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from utils.ollama_manager import OllamaManager
from utils.vector_store import VectorStoreManager
from utils.state_manager import StateManager, create_thread_id
from utils.tool_registry import get_registry

async def create_complete_agent():
    """Create agent using all five utilities."""

    # 1. Ollama Manager - Ensure model available
    ollama_mgr = OllamaManager()
    if not ollama_mgr.check_ollama_running():
        raise RuntimeError("Ollama not running")

    model_name = ollama_mgr.recommend_model("balanced")
    ollama_mgr.ensure_model_available(model_name)
    ollama_mgr.ensure_model_available("qwen3-embedding")

    # 2. Vector Store - Load knowledge base
    vector_mgr = VectorStoreManager(embedding_model="qwen3-embedding")
    vectorstore = vector_mgr.load_existing(
        collection_name="docs",
        persist_dir="./data/vectors"
    )

    # 3. Tool Registry - Register tools
    registry = get_registry()

    def search_knowledge(query: str) -> str:
        """Search knowledge base for relevant information."""
        results = vector_mgr.similarity_search(vectorstore, query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    registry.register_tool(
        name="search_knowledge",
        tool=search_knowledge,
        description="Search project knowledge base",
        category="database"
    )

    tools = registry.get_langchain_tools()

    # 4. State Manager - Create stateful workflow
    AgentState = StateManager.create_state_schema({
        "messages": Annotated[List[BaseMessage], operator.add],
        "context": str,
        "iteration": int
    })

    # Define workflow nodes
    def retrieve_node(state: AgentState):
        """Retrieve relevant context."""
        last_message = state["messages"][-1]
        context = search_knowledge(last_message.content)
        return {"context": context}

    def generate_node(state: AgentState):
        """Generate response with context."""
        llm = ChatOllama(model=model_name)

        # Build contextualized prompt
        context = state.get("context", "")
        messages = state["messages"]

        system_msg = f"Use this context to answer:\n\n{context}"
        full_messages = [HumanMessage(content=system_msg)] + messages

        response = llm.invoke(full_messages)
        iteration = state.get("iteration", 0)

        return {"messages": [response], "iteration": iteration + 1}

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # Compile with checkpointing
    checkpointer = StateManager.get_checkpointer("./checkpoints.db")
    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Usage
graph = await create_complete_agent()

thread_id = create_thread_id("rag-session")
config = {"configurable": {"thread_id": thread_id}}

result = graph.invoke(
    {
        "messages": [HumanMessage(content="How do I use vector stores?")],
        "iteration": 0
    },
    config=config
)

print(result["messages"][-1].content)
```

---

## Best Practices

### 1. Always Verify Ollama Before Starting
```python
from utils.ollama_manager import check_ollama, ensure_model

if not check_ollama():
    raise RuntimeError("Start Ollama first: ollama serve")

ensure_model("qwen3:8b")
```

### 2. Use Context Managers for MCP Clients
```python
# Good: Automatic cleanup
async with FilesystemMCP() as fs:
    content = await fs.read_file("file.txt")

# Bad: Manual cleanup required
fs = FilesystemMCP()
await fs.connect()
content = await fs.read_file("file.txt")
await fs.disconnect()  # Easy to forget!
```

### 3. Persist Vector Stores for Reuse
```python
# Create once
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="persistent_docs",
    persist_dir="./data/vectors"
)

# Load many times
vectorstore = manager.load_existing("persistent_docs", "./data/vectors")
```

### 4. Use State Schemas for Type Safety
```python
# Good: Type-safe state
State = StateManager.create_state_schema({
    "messages": Annotated[List[BaseMessage], operator.add],
    "count": int
})

def node(state: State) -> State:
    # IDE autocomplete works!
    messages = state["messages"]
    return {"count": len(messages)}
```

### 5. Centralize Tools in Registry
```python
# Good: Single source of truth
registry = get_registry()
registry.register_tool(...)
tools = registry.get_langchain_tools()

# Bad: Tools scattered across codebase
tool1 = some_module.get_tool()
tool2 = other_module.create_tool()
tools = [tool1, tool2]  # Hard to track
```

### 6. Handle Errors Gracefully
```python
from utils.ollama_manager import OllamaManager

try:
    manager = OllamaManager()
    manager.ensure_model_available("qwen3:8b")
except ConnectionError:
    print("Ollama not running. Start with: ollama serve")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)
```

### 7. Use Appropriate Vector Store Backend
```python
# Chroma: Better for persistence and metadata filtering
chroma = manager.create_from_documents(..., store_type="chroma")

# FAISS: Better for speed and in-memory operations
faiss = manager.create_from_documents(..., store_type="faiss")
```

### 8. Clean Up Checkpoints Periodically
```python
from utils.state_manager import get_checkpoint_size, StateManager

stats = get_checkpoint_size()
if stats['file_size_mb'] > 100:  # 100 MB threshold
    print("Cleaning old checkpoints...")
    StateManager.clear_checkpoints(confirm=True)
```

---

## Error Handling Strategies

### Connection Errors
```python
from requests.exceptions import ConnectionError
from utils.ollama_manager import OllamaManager

try:
    manager = OllamaManager()
    manager.check_ollama_running()
except ConnectionError:
    print("ERROR: Cannot connect to Ollama")
    print("Solution: Run 'ollama serve' in a terminal")
    exit(1)
```

### Model Availability
```python
from utils.ollama_manager import OllamaManager

manager = OllamaManager()
models = manager.list_models()

if "qwen3:8b" not in models:
    print("Model not available. Pulling...")
    if not manager.pull_model("qwen3:8b"):
        print("ERROR: Failed to pull model")
        exit(1)
```

### MCP Server Errors
```python
from utils.mcp_client import MCPConnectionError, MCPToolError

try:
    async with FilesystemMCP() as fs:
        content = await fs.read_file("file.txt")
except MCPConnectionError:
    print("ERROR: MCP server not running")
    print("Solution: Start MCP server on port 8001")
except MCPToolError as e:
    print(f"ERROR: Tool execution failed: {e}")
```

### Vector Store Not Found
```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager()

try:
    store = manager.load_existing("my_docs", "./data/vectors")
except FileNotFoundError:
    print("Collection not found. Creating new one...")
    # Load documents and create
    store = manager.create_from_documents(...)
```

### State Management Errors
```python
from utils.state_manager import StateManager
import sqlite3

try:
    checkpointer = StateManager.get_checkpointer("./checkpoints.db")
except sqlite3.Error as e:
    print(f"ERROR: Database error: {e}")
    print("Solution: Check file permissions or disk space")
```

---

## Performance Considerations

### Model Selection Impact
```python
# Fast tasks (coding, simple Q&A)
fast_model = manager.recommend_model("fast")  # qwen3:30b-a3b (MoE)

# Quality tasks (reasoning, analysis)
quality_model = manager.recommend_model("quality")  # qwen3:30b

# Resource-constrained (edge devices)
edge_model = manager.recommend_model("edge")  # gemma3:4b
```

### Vector Store Backend Choice
```python
# FAISS: In-memory, faster search, no metadata filtering
faiss_store = manager.create_from_documents(..., store_type="faiss")

# Chroma: Persistent, metadata filtering, slightly slower
chroma_store = manager.create_from_documents(..., store_type="chroma")
```

### Checkpoint Database Maintenance
```python
# Monitor database size
stats = get_checkpoint_size("./checkpoints.db")
print(f"Size: {stats['file_size_mb']:.2f} MB")

# Clean old threads periodically
if stats['thread_count'] > 100:
    # Delete threads older than 30 days (implement custom logic)
    pass
```

### MCP Client Connection Pooling
```python
# Reuse clients instead of creating new ones
fs_client = create_filesystem_client()
await fs_client.connect()

# Use for multiple operations
await fs_client.read_file("file1.txt")
await fs_client.read_file("file2.txt")
await fs_client.write_file("output.txt", "data")

# Cleanup once
await fs_client.disconnect()
```

---

## Summary

The five core utilities provide essential building blocks for local-first AI agents:

1. **OllamaManager**: Model lifecycle management and recommendations
2. **MCPClient**: External service integration (filesystem, web)
3. **VectorStoreManager**: Knowledge retrieval and RAG systems
4. **StateManager**: Stateful workflows with persistence
5. **ToolRegistry**: Centralized tool organization and discovery

Use these utilities together to build sophisticated, production-ready agent workflows that run entirely on local infrastructure.
