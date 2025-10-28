# ToolRegistry API Reference

Centralized tool registry for managing and discovering tools across the project with auto-discovery and LangChain integration.

## Overview

The `ToolRegistry` class provides a singleton registry for maintaining a centralized collection of all available tools. It supports auto-discovery, categorization, metadata management, and seamless conversion to LangChain Tool objects.

**Module:** `utils.tool_registry`

**Dependencies:**
- `langchain-core` - Tool integration
- Python 3.9+

---

## Classes

### ToolMetadata

```python
class ToolMetadata:
    """Metadata container for a registered tool."""
```

Internal class for storing tool metadata.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the tool |
| `tool` | `Callable` | The callable function or tool object |
| `description` | `str` | Human-readable description |
| `category` | `str` | Organization category |
| `args_schema` | `Optional[type]` | Pydantic model for arguments |

**Methods:**

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert metadata to dictionary format (excludes callable)."""
```

---

## Class: ToolRegistry

```python
class ToolRegistry:
    """Singleton registry for managing tools across the project."""
```

### Singleton Pattern

```python
# All instances share the same registry
registry1 = ToolRegistry()
registry2 = ToolRegistry()
assert registry1 is registry2  # Same instance
```

**Alternative Access:**

```python
registry = ToolRegistry.get_instance()
```

---

## Registration Methods

### register_tool()

```python
def register_tool(
    self,
    name: str,
    tool: Callable,
    description: str,
    category: str,
    args_schema: Optional[type] = None
) -> None
```

Register a tool with metadata.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier |
| `tool` | `Callable` | Yes | Callable function or tool |
| `description` | `str` | Yes | What the tool does |
| `category` | `str` | Yes | Organization category |
| `args_schema` | `Optional[type]` | No | Pydantic model for arguments |

**Supported Categories:**
- `filesystem` - File and directory operations
- `web` - Web search, HTTP requests
- `models` - LLM and embedding operations
- `database` - Database and vector store operations
- `workflow` - Agent and workflow management
- `other` - Miscellaneous tools

**Raises:**
- `TypeError` - If tool is not callable
- Warns if tool name already exists (overwrites)

**Example:**

```python
from utils.tool_registry import ToolRegistry
from pydantic import BaseModel, Field

registry = ToolRegistry()

# Simple function
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

registry.register_tool(
    name="calculate_sum",
    tool=calculate_sum,
    description="Calculate sum of two numbers",
    category="other"
)

# With Pydantic schema
class SumArgs(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

registry.register_tool(
    name="calculate_sum_typed",
    tool=calculate_sum,
    description="Calculate sum with typed args",
    category="other",
    args_schema=SumArgs
)
```

---

## Retrieval Methods

### get_tool()

```python
def get_tool(self, name: str) -> Callable
```

Retrieve a tool by name.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Tool name |

**Returns:**
- `Callable` - The tool function

**Raises:**
- `KeyError` - If tool not found in registry

**Example:**

```python
registry = ToolRegistry()

# Get registered tool
calculate = registry.get_tool("calculate_sum")
result = calculate(10, 20)
print(f"Result: {result}")  # 30

# Handle missing tool
try:
    tool = registry.get_tool("nonexistent")
except KeyError as e:
    print(f"Tool not found: {e}")
```

---

### list_tools()

```python
def list_tools(
    self,
    category: Optional[str] = None
) -> List[Dict[str, Any]]
```

List all tools or filter by category.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | `Optional[str]` | No | Filter by category |

**Returns:**

List of dictionaries with tool metadata:

```python
[
    {
        "name": str,
        "description": str,
        "category": str,
        "signature": str
    }
]
```

**Example:**

```python
registry = ToolRegistry()

# List all tools
all_tools = registry.list_tools()
print(f"Total tools: {len(all_tools)}")

for tool in all_tools:
    print(f"- {tool['name']} ({tool['category']})")
    print(f"  {tool['description']}")

# Filter by category
fs_tools = registry.list_tools(category="filesystem")
print(f"\nFilesystem tools: {len(fs_tools)}")
```

---

## LangChain Integration

### get_langchain_tools()

```python
def get_langchain_tools(
    self,
    categories: Optional[List[str]] = None
) -> List[StructuredTool]
```

Convert registered tools to LangChain Tool objects.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `categories` | `Optional[List[str]]` | No | Filter by categories |

**Returns:**
- `List[StructuredTool]` - LangChain tools ready for agents

**Raises:**
- `ImportError` - If langchain-core not installed

**Example:**

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from utils.tool_registry import ToolRegistry

registry = ToolRegistry()

# Register tools
def read_file(path: str) -> str:
    """Read a file."""
    with open(path) as f:
        return f.read()

registry.register_tool(
    name="read_file",
    tool=read_file,
    description="Read contents of a file",
    category="filesystem"
)

# Get LangChain tools
tools = registry.get_langchain_tools()

# Create agent with registered tools
llm = ChatOllama(model="qwen3:8b")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent can now use registered tools
response = executor.invoke({"input": "Read README.md"})
```

**Filter by Category:**

```python
# Only filesystem tools
fs_tools = registry.get_langchain_tools(categories=["filesystem"])

# Multiple categories
tools = registry.get_langchain_tools(
    categories=["filesystem", "web"]
)
```

---

## Auto-Discovery

### auto_discover_utilities()

```python
def auto_discover_utilities(
    self,
    utils_dir: Optional[Path] = None
) -> int
```

Scan utils/ directory and auto-register functions with docstrings.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `utils_dir` | `Optional[Path]` | No | Path to utils directory |

**Returns:**
- `int` - Number of tools discovered and registered

**Discovery Rules:**
- Scans `*.py` files in utils directory
- Ignores files starting with `_`
- Ignores `tool_registry.py`
- Only registers functions with docstrings
- Automatically infers category from module name

**Example:**

```python
from utils.tool_registry import ToolRegistry

registry = ToolRegistry()

# Auto-discover from default utils/ directory
count = registry.auto_discover_utilities()
print(f"Discovered {count} tools")

# Custom directory
from pathlib import Path
count = registry.auto_discover_utilities(Path("./my_utils"))
```

**Category Inference:**

Module names are mapped to categories:

```python
# utils/file_operations.py -> "filesystem"
# utils/web_search.py -> "web"
# utils/model_manager.py -> "models"
# utils/database_helper.py -> "database"
# utils/workflow_builder.py -> "workflow"
```

---

## Export and Documentation

### to_json()

```python
def to_json(
    self,
    filepath: Optional[Path] = None
) -> str
```

Export registry to JSON format.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filepath` | `Optional[Path]` | No | Optional path to write JSON |

**Returns:**
- `str` - JSON string representation

**JSON Format:**

```json
{
  "tools": [
    {
      "name": "tool_name",
      "description": "What it does",
      "category": "filesystem",
      "signature": "(arg1: str, arg2: int) -> str"
    }
  ],
  "categories": {
    "filesystem": 3,
    "web": 2,
    "models": 1
  },
  "total_tools": 6
}
```

**Example:**

```python
registry = ToolRegistry()

# Get JSON string
json_str = registry.to_json()
print(json_str)

# Write to file
registry.to_json(Path("./docs/tools.json"))
```

**Generate Documentation:**

```python
import json
from pathlib import Path

registry = ToolRegistry()
registry.auto_discover_utilities()

# Export to JSON
registry.to_json(Path("./docs/api/tools.json"))

# Generate markdown
tools = registry.list_tools()
with open("./docs/api/tools.md", "w") as f:
    f.write("# Available Tools\n\n")
    for category in ["filesystem", "web", "models", "database", "workflow"]:
        cat_tools = [t for t in tools if t["category"] == category]
        if cat_tools:
            f.write(f"## {category.title()}\n\n")
            for tool in cat_tools:
                f.write(f"### {tool['name']}\n")
                f.write(f"{tool['description']}\n\n")
                f.write(f"**Signature:** `{tool['signature']}`\n\n")
```

---

## Utility Methods

### clear()

```python
def clear(self) -> None
```

Clear all registered tools (useful for testing).

**Example:**

```python
registry = ToolRegistry()
registry.register_tool(...)

# Clear for clean state
registry.clear()
assert len(registry) == 0
```

---

### Special Methods

```python
def __len__(self) -> int:
    """Return the number of registered tools."""

def __repr__(self) -> str:
    """Return string representation of registry."""
```

**Example:**

```python
registry = ToolRegistry()
registry.register_tool(...)

print(f"Registry has {len(registry)} tools")
print(registry)  # ToolRegistry(tools=5, categories=['filesystem', 'web'])
```

---

## Module-Level Functions

### get_registry()

```python
def get_registry() -> ToolRegistry:
    """Get the global ToolRegistry instance."""
```

Convenience function for accessing the singleton.

**Example:**

```python
from utils.tool_registry import get_registry

registry = get_registry()
registry.register_tool(...)
```

---

## Integration Examples

### Complete Agent Setup

```python
from utils.tool_registry import ToolRegistry
from utils.ollama_manager import OllamaManager
from utils.mcp_client import FilesystemMCP, WebSearchMCP
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Initialize registry
registry = ToolRegistry()

# Auto-discover utilities
registry.auto_discover_utilities()

# Register custom tools
def summarize_text(text: str) -> str:
    """Summarize text using LLM."""
    llm = ChatOllama(model="qwen3:8b")
    return llm.invoke(f"Summarize: {text}")

registry.register_tool(
    name="summarize",
    tool=summarize_text,
    description="Summarize text content",
    category="models"
)

# Add MCP tools
fs = FilesystemMCP()
for tool in fs.to_langchain_tools():
    registry.register_tool(
        name=f"fs_{tool.name}",
        tool=tool.func,
        description=tool.description,
        category="filesystem"
    )

# Get all tools for agent
tools = registry.get_langchain_tools()

# Create agent
llm = ChatOllama(model="qwen3:8b")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent has access to all registered tools
```

---

### Category-Based Tool Selection

```python
from utils.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.auto_discover_utilities()

# Create specialized agents with specific tool categories

# Filesystem agent
fs_tools = registry.get_langchain_tools(categories=["filesystem"])
fs_agent = create_agent(llm, fs_tools)

# Research agent
research_tools = registry.get_langchain_tools(categories=["web", "models"])
research_agent = create_agent(llm, research_tools)

# Data agent
data_tools = registry.get_langchain_tools(categories=["database", "filesystem"])
data_agent = create_agent(llm, data_tools)
```

---

### Dynamic Tool Loading

```python
from utils.tool_registry import ToolRegistry
from pathlib import Path
import importlib

registry = ToolRegistry()

def load_plugin_tools(plugin_dir: Path):
    """Dynamically load tools from plugins."""
    for plugin_file in plugin_dir.glob("*.py"):
        if plugin_file.stem.startswith("_"):
            continue

        # Import plugin module
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem,
            plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Register tools from plugin
        if hasattr(module, "register_tools"):
            module.register_tools(registry)

    return registry

# Load all plugins
load_plugin_tools(Path("./plugins"))

# Get all tools
tools = registry.get_langchain_tools()
```

---

### Tool Versioning

```python
from utils.tool_registry import ToolRegistry

registry = ToolRegistry()

def calculate_v1(a: int, b: int) -> int:
    """Calculate sum (v1)."""
    return a + b

def calculate_v2(a: int, b: int, c: int = 0) -> int:
    """Calculate sum with optional third number (v2)."""
    return a + b + c

# Register different versions
registry.register_tool(
    name="calculate_v1",
    tool=calculate_v1,
    description="Calculate sum (v1)",
    category="other"
)

registry.register_tool(
    name="calculate_v2",
    tool=calculate_v2,
    description="Calculate sum with optional third number (v2)",
    category="other"
)

# Use specific version
tool = registry.get_tool("calculate_v2")
```

---

## Best Practices

### Tool Naming

```python
# Good - descriptive, namespaced
"fs_read_file"
"web_search"
"db_query"

# Avoid - ambiguous
"read"
"search"
"query"
```

---

### Documentation

```python
def my_tool(arg1: str, arg2: int) -> str:
    """
    Short description of what the tool does.

    This is included as the tool description in the registry.
    Keep it concise and clear for LLM agents.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
    pass
```

---

### Category Organization

```python
# Group related tools in categories
registry.register_tool("fs_read", read_file, "Read file", "filesystem")
registry.register_tool("fs_write", write_file, "Write file", "filesystem")
registry.register_tool("fs_list", list_dir, "List directory", "filesystem")

# Then retrieve by category
fs_tools = registry.get_langchain_tools(categories=["filesystem"])
```

---

### Error Handling

```python
registry = ToolRegistry()

try:
    tool = registry.get_tool("my_tool")
except KeyError:
    # Tool not found, list available
    available = registry.list_tools()
    print(f"Available tools: {[t['name'] for t in available]}")
```

---

## Performance Tips

### Registration Timing

```python
# Good - register at startup
registry = ToolRegistry()
registry.auto_discover_utilities()
registry.register_tool(...)

# Then reuse
tools = registry.get_langchain_tools()

# Avoid - registering repeatedly
for i in range(100):
    registry = ToolRegistry()  # Singleton, but wasteful
    registry.register_tool(...)  # Re-registering same tools
```

---

### Selective Tool Loading

```python
# Load only needed categories
if task_requires_filesystem:
    tools = registry.get_langchain_tools(categories=["filesystem"])
else:
    tools = registry.get_langchain_tools(categories=["web", "models"])

# Fewer tools = faster agent reasoning
```

---

### Caching Tool Lists

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def get_tools_for_category(category: str):
    registry = ToolRegistry()
    return registry.get_langchain_tools(categories=[category])

# Cached, only computed once per category
fs_tools = get_tools_for_category("filesystem")
```

---

## Testing

```python
import pytest
from utils.tool_registry import ToolRegistry

@pytest.fixture
def clean_registry():
    """Provide clean registry for each test."""
    registry = ToolRegistry()
    registry.clear()
    yield registry
    registry.clear()

def test_register_tool(clean_registry):
    def my_tool(x: int) -> int:
        """My tool."""
        return x * 2

    clean_registry.register_tool(
        name="my_tool",
        tool=my_tool,
        description="Doubles a number",
        category="other"
    )

    assert len(clean_registry) == 1
    tool = clean_registry.get_tool("my_tool")
    assert tool(5) == 10

def test_auto_discovery(clean_registry):
    count = clean_registry.auto_discover_utilities()
    assert count > 0
    assert len(clean_registry) == count
```

---

## Troubleshooting

### Tool Not Found

```python
# List all registered tools
registry = ToolRegistry()
tools = registry.list_tools()
print("Available tools:")
for tool in tools:
    print(f"  - {tool['name']}")
```

---

### Auto-Discovery Not Finding Tools

```python
# Check requirements:
# 1. Function has docstring
def my_function():
    """This is required."""  # Must have docstring
    pass

# 2. Function not private
def _private():  # Ignored (starts with _)
    pass

def public():  # Discovered
    """Public function."""
    pass
```

---

### LangChain Conversion Fails

```python
# Ensure function signature is compatible
def good_tool(text: str) -> str:
    """Works with LangChain."""
    return text.upper()

# Complex signatures may need args_schema
from pydantic import BaseModel, Field

class ComplexArgs(BaseModel):
    param1: str = Field(description="First parameter")
    param2: int = Field(default=10, description="Second parameter")

def complex_tool(param1: str, param2: int = 10) -> str:
    """Use with args_schema."""
    return f"{param1} * {param2}"

registry.register_tool(
    name="complex",
    tool=complex_tool,
    description="Complex tool",
    category="other",
    args_schema=ComplexArgs  # Required for complex signatures
)
```

---

## See Also

- [MCP Client](./mcp_client.md) - External tool integration
- [Ollama Manager](./ollama_manager.md) - Model operations
- [State Manager](./state_manager.md) - Workflow persistence
- [Examples](../../examples/) - Usage examples

---

**Module Location:** `/Volumes/JS-DEV/ai-lang-stuff/utils/tool_registry.py`

**Example:** `/Volumes/JS-DEV/ai-lang-stuff/examples/tool_registry_demo.py`
