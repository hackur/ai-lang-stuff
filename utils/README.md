# Utils Package

Centralized utilities for the AI Lang Stuff project.

## ToolRegistry

The `ToolRegistry` class provides a centralized registry for managing and discovering tools across the project.

### Features

- **Singleton Pattern**: Ensures a single registry instance across the application
- **Tool Registration**: Register callable functions with metadata (name, description, category)
- **Category Management**: Organize tools by category (filesystem, web, models, database, workflow)
- **Auto-Discovery**: Automatically discover and register utilities from the utils directory
- **LangChain Integration**: Convert registered tools to LangChain Tool objects
- **JSON Export**: Export registry to JSON format for external consumption
- **Type Safety**: Full type hints and comprehensive error handling

### Quick Start

```python
from utils.tool_registry import get_registry

# Get the singleton instance
registry = get_registry()

# Register a tool
def my_tool(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y

registry.register_tool(
    name="add",
    tool=my_tool,
    description="Add two numbers",
    category="other"
)

# Retrieve and use the tool
tool = registry.get_tool("add")
result = tool(5, 3)  # Returns 8

# List all tools
all_tools = registry.list_tools()

# Filter by category
web_tools = registry.list_tools(category="web")

# Export to JSON
json_output = registry.to_json()
```

### Tool Categories

- `filesystem`: File operations
- `web`: Web search and fetch operations
- `models`: Ollama/LLM operations
- `database`: Vector store and database operations
- `workflow`: State management and orchestration
- `other`: Uncategorized utilities

### LangChain Integration

Convert registered tools to LangChain format for use in agents:

```python
from utils.tool_registry import get_registry

registry = get_registry()

# Register your tools...

# Convert to LangChain tools
lc_tools = registry.get_langchain_tools(categories=["web", "models"])

# Use with LangChain agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, lc_tools, prompt)
executor = AgentExecutor(agent=agent, tools=lc_tools)
```

### Auto-Discovery

Automatically discover and register utilities:

```python
from utils.tool_registry import get_registry

registry = get_registry()

# Auto-discover tools from utils/ directory
count = registry.auto_discover_utilities()
print(f"Discovered {count} tools")
```

### API Reference

#### ToolRegistry Methods

- `register_tool(name, tool, description, category, args_schema=None)`: Register a new tool
- `get_tool(name)`: Retrieve a tool by name
- `list_tools(category=None)`: List all tools or filter by category
- `get_langchain_tools(categories=None)`: Convert to LangChain Tool objects
- `auto_discover_utilities(utils_dir=None)`: Auto-discover and register utilities
- `to_json(filepath=None)`: Export registry to JSON
- `clear()`: Clear all registered tools (useful for testing)

#### Global Functions

- `get_registry()`: Get the singleton ToolRegistry instance

### Example

See `examples/tool_registry_demo.py` for a complete demonstration.

## Other Utilities

### MCPClient

MCP (Model Context Protocol) client for interacting with MCP servers.

### VectorStoreManager

Utilities for managing vector stores (ChromaDB, FAISS).

---

For more information, see the project documentation in `/docs`.
