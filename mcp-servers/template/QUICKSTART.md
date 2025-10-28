# MCP Server Template - Quick Start Guide

Get your custom MCP server up and running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- (Optional) Docker for containerized deployment

---

## 1. Copy Template

```bash
# Navigate to mcp-servers directory
cd /Volumes/JS-DEV/ai-lang-stuff/mcp-servers/

# Copy template to your new server
cp -r template my-custom-server
cd my-custom-server
```

---

## 2. Install Dependencies

```bash
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using uv (faster)
uv pip install -r requirements.txt
```

---

## 3. Test the Example Server

```bash
# Run the server
python server.py
```

You should see:

```
INFO - Initialized MCP server: example-server v1.0.0
INFO - Registered tool: greet
================================================================================
SERVER INFO
================================================================================
{
  "name": "example-server",
  "version": "1.0.0",
  ...
}
```

---

## 4. Create Your First Custom Tool

Edit `tools/example_tool.py` or create a new file `tools/my_tool.py`:

```python
from typing import Dict, Any
from pydantic import BaseModel, Field

class MyToolRequest(BaseModel):
    """Request model."""
    message: str = Field(..., description="Message to process")

def my_tool(message: str) -> Dict[str, Any]:
    """
    My custom tool.

    Args:
        message: Input message

    Returns:
        Processed message
    """
    # Validate
    request = MyToolRequest(message=message)

    # Process
    result = f"Processed: {request.message.upper()}"

    # Return
    return {"result": result}
```

---

## 5. Register Your Tool

Edit `server.py` main section:

```python
from tools.my_tool import my_tool

def main():
    server = MCPServer("my-server", version="1.0.0")

    # Register your tool
    server.register_tool(
        name="my_tool",
        function=my_tool,
        description="My custom tool",
        parameters=[
            ToolParameter(
                name="message",
                type="string",
                description="Message to process",
                required=True
            )
        ]
    )

    # Test it
    response = server.invoke_tool("my_tool", {"message": "hello"})
    print(response.result)
```

---

## 6. Test Your Tool

```bash
# Run the server
python server.py

# Expected output:
# {"result": "Processed: HELLO"}
```

---

## 7. Write Tests

Create `tests/test_my_tool.py`:

```python
from tools.my_tool import my_tool

def test_my_tool():
    """Test my custom tool."""
    result = my_tool("test")

    assert result["result"] == "Processed: TEST"
```

Run tests:

```bash
pytest tests/ -v
```

---

## 8. Configure Your Server

Edit `config.yaml`:

```yaml
name: "my-custom-server"
version: "1.0.0"
description: "My custom MCP server"

tools:
  - name: "my_tool"
    description: "My custom tool"
    enabled: true
    parameters:
      - name: "message"
        type: "string"
        description: "Message to process"
        required: true

settings:
  logging:
    level: "INFO"
```

Load config in server:

```python
server = MCPServer(name="my-server", config_path="config.yaml")
```

---

## 9. Deploy with Docker

```bash
# Build image
docker build -t my-custom-server .

# Run container
docker run -p 8000:8000 my-custom-server
```

Or use Docker Compose:

```bash
# Start server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
docker-compose down
```

---

## 10. Integrate with LangChain

```python
from langchain.tools import Tool
from server import MCPServer

# Create server
server = MCPServer("my-server")
# ... register tools ...

# Create LangChain tool
def langchain_tool_wrapper(tool_name: str, **kwargs):
    response = server.invoke_tool(tool_name, kwargs)
    if response.status == "success":
        return response.result
    else:
        raise Exception(response.error)

my_langchain_tool = Tool(
    name="my_tool",
    description="My custom tool",
    func=lambda message: langchain_tool_wrapper("my_tool", message=message)
)

# Use with agent
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3:8b")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [my_langchain_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[my_langchain_tool])

result = executor.invoke({"input": "Process the message 'hello world'"})
print(result)
```

---

## Common Patterns

### Pattern 1: File Processing Tool

```python
def process_file(file_path: str) -> Dict[str, Any]:
    """Process a file."""
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path) as f:
        content = f.read()

    return {
        "file": str(path),
        "lines": len(content.splitlines()),
        "words": len(content.split()),
        "chars": len(content)
    }
```

### Pattern 2: API Integration Tool

```python
import httpx

def fetch_api_data(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch data from API."""
    with httpx.Client() as client:
        response = client.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
```

### Pattern 3: Data Processing Tool

```python
def analyze_data(data: list) -> Dict[str, Any]:
    """Analyze list of numbers."""
    if not data:
        return {"error": "Empty data"}

    return {
        "count": len(data),
        "sum": sum(data),
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data)
    }
```

---

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore [tools/example_tool.py](tools/example_tool.py) for more patterns
3. Check [tests/test_server.py](tests/test_server.py) for testing examples
4. Review [config.yaml](config.yaml) for configuration options
5. See [Dockerfile](Dockerfile) for deployment options

---

## Troubleshooting

### Import Error

```bash
# Error: ModuleNotFoundError: No module named 'pydantic'
# Fix:
pip install -r requirements.txt
```

### Tool Not Found

```python
# Error: Tool 'my_tool' not found
# Fix: Verify tool is registered
print(server.list_tools())
```

### Configuration Error

```bash
# Error: Invalid configuration
# Fix: Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

---

## Resources

- **MCP Specification**: https://github.com/modelcontextprotocol
- **LangChain Docs**: https://python.langchain.com/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **pytest Docs**: https://docs.pytest.org/

---

## Support

For questions or issues:

1. Check the [README.md](README.md) troubleshooting section
2. Review example implementations
3. Check the test suite for usage patterns
4. Open an issue in the repository

Happy building!
