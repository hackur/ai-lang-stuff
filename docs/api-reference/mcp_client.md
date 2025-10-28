# MCP Client API Reference

Model Context Protocol (MCP) client wrappers for filesystem and web search operations.

## Overview

The MCP client module provides production-quality clients for interacting with MCP servers, including filesystem operations and web search capabilities. All clients support async context managers, connection pooling, automatic retry logic, and LangChain tool integration.

**Module:** `utils.mcp_client`

**Dependencies:**
- `httpx` - Async HTTP client
- `langchain-core` - Tool integration
- `pydantic` - Configuration validation
- `tenacity` - Retry logic

---

## Configuration Classes

### MCPConfig

```python
class MCPConfig(BaseModel):
    """Configuration for MCP client connections."""
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | MCP server host |
| `port` | `int` | `8000` | MCP server port |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `base_delay` | `float` | `1.0` | Base delay for exponential backoff |
| `max_delay` | `float` | `10.0` | Maximum delay between retries |

**Properties:**

```python
@property
def base_url(self) -> str:
    """Get the base URL for the MCP server."""
    return f"http://{self.host}:{self.port}"
```

**Example:**

```python
from utils.mcp_client import MCPConfig

# Default configuration
config = MCPConfig()

# Custom configuration
config = MCPConfig(
    host="192.168.1.100",
    port=8001,
    timeout=60.0,
    max_retries=5
)

print(f"Server URL: {config.base_url}")
```

---

## Exception Classes

### MCPError

```python
class MCPError(Exception):
    """Base exception for MCP client errors."""
```

Base exception for all MCP-related errors.

---

### MCPConnectionError

```python
class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
```

Raised when unable to connect to or communicate with MCP server.

---

### MCPToolError

```python
class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""
```

Raised when tool execution fails or returns an error.

**Error Handling Example:**

```python
from utils.mcp_client import (
    FilesystemMCP,
    MCPConnectionError,
    MCPToolError
)

try:
    async with FilesystemMCP(config) as client:
        content = await client.read_file("/path/to/file.txt")
except MCPConnectionError as e:
    print(f"Cannot connect to MCP server: {e}")
except MCPToolError as e:
    print(f"Tool execution failed: {e}")
```

---

## Base Class: MCPClient

```python
class MCPClient(ABC):
    """Base class for MCP (Model Context Protocol) clients."""
```

Abstract base class providing connection management, retry logic, and error handling for MCP server interactions. All MCP clients inherit from this class.

### Constructor

```python
def __init__(
    self,
    config: Optional[MCPConfig] = None,
    client: Optional[httpx.AsyncClient] = None
) -> None
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | `MCPConfig` | No | `MCPConfig()` | MCP client configuration |
| `client` | `httpx.AsyncClient` | No | `None` | Pre-configured HTTP client |

**Example:**

```python
from utils.mcp_client import MCPConfig, FilesystemMCP

# Default configuration
client = FilesystemMCP()

# Custom configuration
config = MCPConfig(host="localhost", port=8001, timeout=60)
client = FilesystemMCP(config=config)

# Shared HTTP client (advanced)
import httpx
http_client = httpx.AsyncClient()
client = FilesystemMCP(client=http_client)
```

---

### Connection Methods

#### connect()

```python
async def connect(self) -> None
```

Establish connection to MCP server with health check.

**Raises:**
- `MCPConnectionError` - If connection fails after retries

**Example:**

```python
client = FilesystemMCP(config)
await client.connect()
print("Connected to MCP server")
```

---

#### disconnect()

```python
async def disconnect(self) -> None
```

Close connection to MCP server and cleanup resources.

**Example:**

```python
client = FilesystemMCP(config)
await client.connect()
# ... use client
await client.disconnect()
```

---

### Tool Execution Methods

#### call_tool()

```python
@retry(
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def call_tool(
    self,
    name: str,
    args: Dict[str, Any]
) -> Dict[str, Any]
```

Call a tool on the MCP server with automatic retry logic.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Tool name to execute |
| `args` | `Dict[str, Any]` | Yes | Tool arguments |

**Returns:**
- `Dict[str, Any]` - Tool execution result

**Raises:**
- `MCPConnectionError` - If not connected
- `MCPToolError` - If tool execution fails

**Example:**

```python
result = await client.call_tool(
    "read_file",
    {"path": "/path/to/file.txt"}
)
content = result.get("content", "")
```

---

### Context Manager Support

#### Async Context Manager

```python
async with MCPClient(config) as client:
    result = await client.call_tool("tool_name", {"arg": "value"})
```

**Features:**
- Automatic connection on entry
- Automatic disconnection on exit
- Exception-safe cleanup

**Example:**

```python
async def use_mcp():
    config = MCPConfig(port=8001)
    async with FilesystemMCP(config) as client:
        content = await client.read_file("/path/to/file.txt")
        print(content)
    # Connection automatically closed
```

---

### Abstract Methods

#### to_langchain_tools()

```python
@abstractmethod
def to_langchain_tools(self) -> List[Tool]:
    """Convert MCP client methods to LangChain tools."""
```

Must be implemented by subclasses to provide LangChain tool integration.

---

## FilesystemMCP Client

```python
class FilesystemMCP(MCPClient):
    """MCP client for filesystem operations."""
```

Provides file reading, writing, directory listing, and file search capabilities through an MCP server.

### Constructor

```python
def __init__(
    self,
    config: Optional[MCPConfig] = None,
    client: Optional[httpx.AsyncClient] = None,
    base_path: Optional[Union[str, Path]] = None
) -> None
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `MCPConfig` | No | MCP client configuration |
| `client` | `httpx.AsyncClient` | No | Pre-configured HTTP client |
| `base_path` | `Union[str, Path]` | No | Base path to restrict access |

**Example:**

```python
from utils.mcp_client import FilesystemMCP, MCPConfig

# Unrestricted access
fs = FilesystemMCP()

# Restricted to specific directory (security)
fs = FilesystemMCP(base_path="/safe/directory")

# Custom configuration
config = MCPConfig(port=8001)
fs = FilesystemMCP(config=config, base_path="/data")
```

---

### File Operations

#### read_file()

```python
async def read_file(self, path: Union[str, Path]) -> str
```

Read contents of a file.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `Union[str, Path]` | Yes | Path to file to read |

**Returns:**
- `str` - File contents

**Raises:**
- `MCPToolError` - If file read fails
- `ValueError` - If path is outside `base_path`

**Example:**

```python
async with FilesystemMCP(config) as fs:
    content = await fs.read_file("/path/to/file.txt")
    print(content)
```

---

#### write_file()

```python
async def write_file(
    self,
    path: Union[str, Path],
    content: str
) -> bool
```

Write content to a file.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `Union[str, Path]` | Yes | Path to file to write |
| `content` | `str` | Yes | Content to write |

**Returns:**
- `bool` - `True` if write successful

**Raises:**
- `MCPToolError` - If file write fails
- `ValueError` - If path is outside `base_path`

**Example:**

```python
async with FilesystemMCP(config) as fs:
    success = await fs.write_file(
        "/path/to/output.txt",
        "Hello, World!"
    )
    if success:
        print("File written successfully")
```

---

#### list_directory()

```python
async def list_directory(
    self,
    path: Union[str, Path]
) -> List[Dict[str, Any]]
```

List contents of a directory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `Union[str, Path]` | Yes | Path to directory |

**Returns:**
- `List[Dict[str, Any]]` - List of file/directory entries with metadata

**Entry Format:**
```python
{
    "name": str,        # File/directory name
    "type": str,        # "file" or "directory"
    "size": int,        # Size in bytes (files only)
    "modified": str     # Last modified timestamp
}
```

**Raises:**
- `MCPToolError` - If directory listing fails
- `ValueError` - If path is outside `base_path`

**Example:**

```python
async with FilesystemMCP(config) as fs:
    entries = await fs.list_directory("/path/to/dir")

    for entry in entries:
        name = entry["name"]
        entry_type = entry["type"]
        print(f"{name} ({entry_type})")
```

---

#### search_files()

```python
async def search_files(
    self,
    pattern: str,
    path: Optional[Union[str, Path]] = None
) -> List[str]
```

Search for files matching a glob pattern.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | `str` | Yes | Glob pattern (e.g., `"*.py"`, `"**/*.txt"`) |
| `path` | `Union[str, Path]` | No | Path to search in (defaults to `base_path`) |

**Returns:**
- `List[str]` - List of matching file paths

**Raises:**
- `MCPToolError` - If search fails

**Example:**

```python
async with FilesystemMCP(config) as fs:
    # Find all Python files
    py_files = await fs.search_files("*.py", "/project/src")

    # Find all text files recursively
    txt_files = await fs.search_files("**/*.txt", "/data")

    print(f"Found {len(py_files)} Python files")
```

---

### LangChain Integration

#### to_langchain_tools()

```python
def to_langchain_tools(self) -> List[Tool]
```

Convert filesystem operations to LangChain tools for agent use.

**Returns:**
- `List[Tool]` - LangChain Tool objects for:
  - `read_file` - Read file contents
  - `write_file` - Write content to file
  - `list_directory` - List directory contents
  - `search_files` - Search for files

**Example:**

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from utils.mcp_client import FilesystemMCP

# Create filesystem tools
fs = FilesystemMCP(base_path="/project")
tools = fs.to_langchain_tools()

# Create agent with filesystem tools
llm = ChatOllama(model="qwen3:8b")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent can now use filesystem operations
response = executor.invoke({
    "input": "Read the contents of README.md and summarize it"
})
```

---

## WebSearchMCP Client

```python
class WebSearchMCP(MCPClient):
    """MCP client for web search operations."""
```

Provides web search and URL fetching capabilities through an MCP server with structured result parsing.

### Search Results

#### SearchResult

```python
class SearchResult(BaseModel):
    """Structured search result from web search."""
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Search result title |
| `url` | `str` | Result URL |
| `snippet` | `str` | Text snippet from result |
| `score` | `Optional[float]` | Relevance score (if available) |

---

### Search Operations

#### search()

```python
async def search(
    self,
    query: str,
    num_results: int = 5
) -> List[SearchResult]
```

Perform web search with structured results.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query string |
| `num_results` | `int` | No | `5` | Maximum results to return |

**Returns:**
- `List[SearchResult]` - List of structured search results

**Raises:**
- `MCPToolError` - If search fails

**Example:**

```python
async with WebSearchMCP(config) as search:
    results = await search.search("LangChain documentation", num_results=5)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   {result.snippet}\n")
```

---

#### fetch_url()

```python
async def fetch_url(self, url: str) -> str
```

Fetch content from a URL.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | `str` | Yes | URL to fetch |

**Returns:**
- `str` - Page content as string

**Raises:**
- `MCPToolError` - If fetch fails

**Example:**

```python
async with WebSearchMCP(config) as search:
    content = await search.fetch_url("https://example.com")
    print(f"Fetched {len(content)} bytes")
```

---

### LangChain Integration

#### to_langchain_tools()

```python
def to_langchain_tools(self) -> List[Tool]
```

Convert web search operations to LangChain tools.

**Returns:**
- `List[Tool]` - LangChain Tool objects for:
  - `web_search` - Search the web
  - `fetch_url` - Fetch URL content

**Example:**

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from utils.mcp_client import WebSearchMCP

# Create web search tools
search = WebSearchMCP(MCPConfig(port=8002))
tools = search.to_langchain_tools()

# Create agent with search tools
llm = ChatOllama(model="qwen3:8b")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent can now search the web
response = executor.invoke({
    "input": "Search for the latest LangChain features and summarize"
})
```

---

## Convenience Factory Functions

### create_filesystem_client()

```python
def create_filesystem_client(
    host: str = "localhost",
    port: int = 8001,
    base_path: Optional[Union[str, Path]] = None
) -> FilesystemMCP
```

Create a filesystem MCP client with default configuration.

**Example:**

```python
from utils.mcp_client import create_filesystem_client

fs = create_filesystem_client(
    host="localhost",
    port=8001,
    base_path="/safe/directory"
)
```

---

### create_websearch_client()

```python
def create_websearch_client(
    host: str = "localhost",
    port: int = 8002
) -> WebSearchMCP
```

Create a web search MCP client with default configuration.

**Example:**

```python
from utils.mcp_client import create_websearch_client

search = create_websearch_client(host="localhost", port=8002)
```

---

## Security Considerations

### Path Validation

The `FilesystemMCP` client includes path traversal protection when `base_path` is set:

```python
def _validate_path(self, path: Union[str, Path]) -> Path:
    """Validate and resolve path to prevent path traversal attacks."""
```

**Example:**

```python
# Restrict access to /safe/directory
fs = FilesystemMCP(base_path="/safe/directory")

# This will work
await fs.read_file("/safe/directory/file.txt")

# This will raise ValueError
await fs.read_file("../../etc/passwd")
```

**Best Practices:**

1. Always set `base_path` when exposing to agents
2. Use absolute paths when possible
3. Validate user-provided paths
4. Log filesystem operations

---

## Performance Tips

### Connection Pooling

Reuse client instances to avoid connection overhead:

```python
# Good - reuse client
async with FilesystemMCP(config) as fs:
    for file_path in file_paths:
        content = await fs.read_file(file_path)

# Avoid - creates new connections
for file_path in file_paths:
    async with FilesystemMCP(config) as fs:
        content = await fs.read_file(file_path)
```

### Timeout Configuration

Adjust timeouts based on operation type:

```python
# Fast operations
quick_config = MCPConfig(timeout=10.0)

# Slow operations (large files, web searches)
slow_config = MCPConfig(timeout=120.0)
```

### Batch Operations

Group similar operations when possible:

```python
async with FilesystemMCP(config) as fs:
    # Batch reads
    contents = []
    for path in paths:
        content = await fs.read_file(path)
        contents.append(content)
```

---

## Integration Examples

### Research Agent

```python
from langchain_ollama import ChatOllama
from utils.mcp_client import WebSearchMCP, FilesystemMCP

async def research_agent(question: str):
    # Set up search client
    search = WebSearchMCP(MCPConfig(port=8002))
    fs = FilesystemMCP(MCPConfig(port=8001), base_path="./research")

    # Combine tools
    tools = search.to_langchain_tools() + fs.to_langchain_tools()

    # Create agent
    llm = ChatOllama(model="qwen3:8b")
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    # Execute research
    result = await executor.ainvoke({"input": question})
    return result
```

### File Processing Pipeline

```python
async def process_files(directory: str):
    async with FilesystemMCP() as fs:
        # Find all Python files
        files = await fs.search_files("*.py", directory)

        # Process each file
        for file_path in files:
            content = await fs.read_file(file_path)
            processed = process_content(content)
            await fs.write_file(f"{file_path}.processed", processed)
```

---

## See Also

- [Ollama Manager](./ollama_manager.md) - Model management
- [Tool Registry](./tool_registry.md) - Register MCP tools
- [State Manager](./state_manager.md) - Persistence for agents
- [Examples](../../examples/02-mcp/) - MCP integration examples

---

**Module Location:** `/Volumes/JS-DEV/ai-lang-stuff/utils/mcp_client.py`

**MCP Specification:** https://github.com/modelcontextprotocol
