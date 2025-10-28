# API Reference

Complete API documentation for the local-first AI toolkit utilities.

## Overview

This API reference provides comprehensive documentation for all core utilities in the toolkit. Each module is designed for production use with proper error handling, logging, and type safety.

## Core Utilities

### [OllamaManager](./ollama_manager.md)
Comprehensive Ollama server management, model operations, and intelligent recommendations.

**Key Features:**
- Health checks and server connectivity
- Model listing, pulling, and availability verification
- Performance benchmarking
- Task-based model recommendations

**Use Cases:**
- Verify Ollama is running before operations
- Automatically pull required models
- Benchmark model performance
- Select optimal models for specific tasks

---

### [MCP Client](./mcp_client.md)
Model Context Protocol clients for filesystem and web search operations.

**Key Features:**
- Async context managers with connection pooling
- Automatic retry logic with exponential backoff
- LangChain tool integration
- Path validation and security

**Use Cases:**
- File operations in agent workflows
- Web search and URL fetching
- MCP server integration
- Tool creation for LangChain agents

---

### [VectorStoreManager](./vector_store.md)
Unified interface for Chroma and FAISS vector stores with Ollama embeddings.

**Key Features:**
- Create, load, and manage vector stores
- Support for Chroma (persistence) and FAISS (performance)
- Document chunking and embedding
- Similarity search with metadata filtering

**Use Cases:**
- Building RAG systems
- Document indexing and search
- Semantic similarity matching
- Knowledge base management

---

### [StateManager](./state_manager.md)
LangGraph state persistence and checkpoint management with SQLite backend.

**Key Features:**
- SQLite checkpointing for workflow persistence
- State schema creation helpers
- Checkpoint loading and recovery
- Thread management and cleanup

**Use Cases:**
- Multi-turn conversation persistence
- Long-running agent workflows
- State recovery after errors
- Conversation history management

---

### [ToolRegistry](./tool_registry.md)
Centralized tool registry with auto-discovery and LangChain integration.

**Key Features:**
- Singleton registry pattern
- Auto-discovery of utility functions
- Category-based organization
- LangChain Tool conversion

**Use Cases:**
- Tool management across projects
- Dynamic tool discovery
- Agent tool provisioning
- Tool documentation and metadata

---

## Quick Navigation

### By Category

**Model Management:**
- [OllamaManager](./ollama_manager.md) - Ollama operations

**Data Storage:**
- [VectorStoreManager](./vector_store.md) - Vector stores for RAG
- [StateManager](./state_manager.md) - State persistence

**Integration:**
- [MCP Client](./mcp_client.md) - MCP protocol clients
- [ToolRegistry](./tool_registry.md) - Tool management

### By Use Case

**Starting New Project:**
1. [OllamaManager](./ollama_manager.md) - Verify Ollama setup
2. [ToolRegistry](./tool_registry.md) - Register available tools
3. [StateManager](./state_manager.md) - Set up persistence

**Building RAG System:**
1. [OllamaManager](./ollama_manager.md) - Get embedding model
2. [VectorStoreManager](./vector_store.md) - Create vector store
3. [MCP Client](./mcp_client.md) - Add filesystem tools

**Creating Agent:**
1. [ToolRegistry](./tool_registry.md) - Discover available tools
2. [MCP Client](./mcp_client.md) - Add external tools
3. [StateManager](./state_manager.md) - Enable conversation history

---

## Common Patterns

### Error Handling

All utilities use consistent error handling:

```python
try:
    result = manager.operation()
except ConnectionError as e:
    # Handle connection failures (Ollama, MCP servers)
    logger.error(f"Connection failed: {e}")
except ValueError as e:
    # Handle invalid inputs
    logger.error(f"Invalid input: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

### Logging

All utilities use Python's logging module:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Get module logger
logger = logging.getLogger(__name__)

# Utilities log at appropriate levels:
# - DEBUG: Detailed diagnostic info
# - INFO: General operations
# - WARNING: Important but non-critical issues
# - ERROR: Operation failures
```

### Type Safety

All utilities include comprehensive type hints:

```python
from typing import List, Optional, Dict, Any
from pathlib import Path

def operation(
    required: str,
    optional: Optional[int] = None,
    config: Dict[str, Any] = None
) -> List[str]:
    """Operation with full type annotations."""
    pass
```

### Context Managers

Async utilities use context managers for resource cleanup:

```python
async with FilesystemMCP(config) as client:
    result = await client.read_file("path/to/file.txt")
    # Connection automatically closed on exit
```

---

## Migration Guides

### From Direct Ollama API

**Before:**
```python
import requests

response = requests.get("http://localhost:11434/api/tags")
models = response.json()["models"]
```

**After:**
```python
from utils.ollama_manager import OllamaManager

manager = OllamaManager()
if manager.check_ollama_running():
    models = manager.list_models()
```

### From LangChain Chroma

**Before:**
```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./data")
```

**After:**
```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager(embedding_model="nomic-embed-text")
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="my_docs",
    persist_dir="./data"
)
```

### From Manual State Management

**Before:**
```python
from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
```

**After:**
```python
from utils.state_manager import basic_agent_state

State = basic_agent_state()
```

---

## Configuration

### Environment Variables

```bash
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-project
LANGCHAIN_API_KEY=your-key

# Logging level
LOG_LEVEL=INFO
```

### Default Paths

```python
# Vector stores
VECTOR_STORE_DIR = "./data/vector_stores"

# Checkpoints
CHECKPOINT_DB = "./checkpoints.db"

# MCP server defaults
FILESYSTEM_MCP_PORT = 8001
WEBSEARCH_MCP_PORT = 8002
```

---

## Testing

All utilities include comprehensive test coverage. See `tests/` directory.

```bash
# Run all tests
pytest tests/

# Run specific utility tests
pytest tests/test_ollama_manager.py
pytest tests/test_vector_store.py
pytest tests/test_state_manager.py

# Run with coverage
pytest --cov=utils tests/
```

---

## Performance Tips

### OllamaManager
- Use smaller models for simple tasks (gemma3:4b)
- Enable model caching for repeated operations
- Batch similar requests when possible

### VectorStoreManager
- Use Chroma for persistence, FAISS for speed
- Optimize chunk size (512-1024 tokens typical)
- Enable metadata filtering for faster searches

### MCP Client
- Reuse client instances with connection pooling
- Set appropriate timeouts for operations
- Handle retries gracefully

### StateManager
- Clear old checkpoints periodically
- Use thread IDs for conversation isolation
- Monitor checkpoint database size

### ToolRegistry
- Register tools at startup, not per-operation
- Use categories to filter tools efficiently
- Export registry to JSON for documentation

---

## Related Documentation

- [User Guide](../guides/) - Step-by-step tutorials
- [Examples](../../examples/) - Working code samples
- [Development Plan](../DEVELOPMENT-PLAN-20-POINTS.md) - Project roadmap

---

## Support

For issues, questions, or contributions:
- File issues on GitHub
- Check example implementations
- Review test cases for usage patterns

---

**Last Updated:** 2025-10-26
