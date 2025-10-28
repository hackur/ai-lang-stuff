# MCP Server Template

A comprehensive template for building custom MCP (Model Context Protocol) servers with Python. This template provides a complete foundation including server implementation, tool registration, configuration, testing, and deployment.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Creating Custom Tools](#creating-custom-tools)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This template provides everything you need to build production-ready MCP servers:

- **Server Framework**: Complete MCP server implementation with request routing
- **Tool System**: Flexible tool registration and execution
- **Configuration**: YAML-based configuration with validation
- **Testing**: Comprehensive test suite with fixtures and mocks
- **Deployment**: Docker support with multi-stage builds
- **Documentation**: Detailed guides and examples

---

## Features

### Core Functionality

- **Tool Registration**: Dynamic tool registration with validation
- **Request Handling**: Support for JSON, dict, and object requests
- **Parameter Validation**: Automatic parameter validation using Pydantic
- **Error Handling**: Comprehensive error handling and logging
- **Configuration**: YAML configuration with environment overrides
- **Introspection**: Server and tool metadata endpoints

### Development Features

- **Type Safety**: Full type hints throughout
- **Testing**: pytest-based test suite with 90%+ coverage
- **Logging**: Structured logging with configurable levels
- **Documentation**: Extensive docstrings and examples
- **Docker**: Production-ready containerization

### Security Features

- **Path Validation**: Secure file access patterns
- **Input Validation**: Pydantic-based input validation
- **Rate Limiting**: Optional rate limiting support
- **Authentication**: Extensible auth system

---

## Quick Start

### 1. Copy Template

```bash
# Copy template to new server directory
cp -r mcp-servers/template mcp-servers/my-server
cd mcp-servers/my-server
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv pip install -r requirements.txt
```

### 3. Run Example Server

```bash
# Run the example server
python server.py
```

You should see output like:

```
INFO - Initialized MCP server: example-server v1.0.0
INFO - Registered tool: greet
================================================================================
SERVER INFO
================================================================================
{
  "name": "example-server",
  "version": "1.0.0",
  "description": "Example MCP server for demonstration",
  "tools": [...]
}
```

### 4. Test Server

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

## Project Structure

```
template/
├── server.py              # Main MCP server implementation
├── config.yaml            # Server configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── README.md             # This file
├── tools/                # Tool implementations
│   ├── __init__.py
│   └── example_tool.py   # Example tool with patterns
└── tests/                # Test suite
    ├── __init__.py
    └── test_server.py    # Server and tool tests
```

---

## Creating Custom Tools

### Step 1: Define Tool Function

Create a new file in `tools/` directory:

```python
# tools/my_tool.py

from typing import Dict, Any
from pydantic import BaseModel, Field

class MyToolRequest(BaseModel):
    """Input model for my tool."""
    input_param: str = Field(..., description="Input parameter")

def my_tool(input_param: str) -> Dict[str, Any]:
    """
    My custom tool implementation.

    Args:
        input_param: Description of parameter

    Returns:
        Dict with tool results

    Example:
        >>> result = my_tool("test")
        >>> print(result["output"])
    """
    # Validate input
    request = MyToolRequest(input_param=input_param)

    # Business logic
    output = f"Processed: {request.input_param}"

    # Return structured data
    return {
        "output": output,
        "metadata": {
            "input_length": len(request.input_param)
        }
    }
```

### Step 2: Register Tool

Add to `server.py` or create a registration function:

```python
from server import MCPServer, ToolParameter
from tools.my_tool import my_tool

server = MCPServer("my-server")

server.register_tool(
    name="my_tool",
    function=my_tool,
    description="My custom tool",
    parameters=[
        ToolParameter(
            name="input_param",
            type="string",
            description="Input parameter",
            required=True
        )
    ]
)
```

### Step 3: Test Tool

Create test in `tests/test_server.py`:

```python
def test_my_tool():
    """Test my custom tool."""
    from tools.my_tool import my_tool

    result = my_tool("test input")

    assert "output" in result
    assert result["output"] == "Processed: test input"
    assert result["metadata"]["input_length"] == 10
```

---

## Configuration

### Basic Configuration

Edit `config.yaml`:

```yaml
name: "my-mcp-server"
version: "1.0.0"
description: "My custom MCP server"

tools:
  - name: "my_tool"
    description: "My tool description"
    enabled: true
    parameters:
      - name: "input_param"
        type: "string"
        description: "Parameter description"
        required: true

settings:
  logging:
    level: "INFO"
  timeouts:
    default: 30
    maximum: 300
```

### Loading Configuration

```python
from server import MCPServer

# Load from config file
server = MCPServer(
    name="my-server",
    config_path="config.yaml"
)
```

### Environment Variables

Override configuration with environment variables:

```bash
export SERVER_NAME="production-server"
export LOG_LEVEL="WARNING"
python server.py
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_server.py -v

# Run specific test
pytest tests/test_server.py::TestMCPServer::test_server_initialization -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run only unit tests
pytest tests/ -v -m unit

# Run only integration tests
pytest tests/ -v -m integration
```

### Test Structure

The template includes comprehensive tests:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test complete workflows
- **Mock Patterns**: Examples of mocking external dependencies
- **Fixtures**: Reusable test data and setup

### Writing Tests

Example test:

```python
def test_my_tool():
    """Test my tool."""
    from tools.my_tool import my_tool

    # Test successful execution
    result = my_tool("test")
    assert result["output"] == "Processed: test"

    # Test error handling
    with pytest.raises(ValidationError):
        my_tool("")  # Empty input
```

---

## Deployment

### Docker Deployment

#### Build Image

```bash
# Production build
docker build --target production -t my-mcp-server:latest .

# Development build
docker build --target development -t my-mcp-server:dev .

# Run tests
docker build --target testing -t my-mcp-server:test .
```

#### Run Container

```bash
# Run production container
docker run -d \
  -p 8000:8000 \
  --name mcp-server \
  my-mcp-server:latest

# Run with custom config
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  --name mcp-server \
  my-mcp-server:latest

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e SERVER_NAME="prod-server" \
  -e LOG_LEVEL="WARNING" \
  --name mcp-server \
  my-mcp-server:latest
```

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/app/data
    environment:
      - SERVER_NAME=my-mcp-server
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run:

```bash
docker-compose up -d
docker-compose logs -f
```

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py

# Run in background
nohup python server.py > server.log 2>&1 &

# Check logs
tail -f server.log
```

---

## Best Practices

### Code Organization

1. **One tool per file**: Keep tools focused and modular
2. **Use type hints**: Always include type hints for parameters and returns
3. **Validate inputs**: Use Pydantic models for input validation
4. **Handle errors**: Wrap operations in try/except blocks
5. **Log operations**: Log important operations and errors

### Tool Development

1. **Clear naming**: Use descriptive tool and parameter names
2. **Detailed descriptions**: Provide comprehensive descriptions
3. **Include examples**: Add docstring examples showing usage
4. **Return structured data**: Return dicts or Pydantic models
5. **Test thoroughly**: Write tests for success and failure cases

### Security

1. **Validate all inputs**: Never trust user input
2. **Sanitize file paths**: Check paths are within allowed directories
3. **Limit resource usage**: Set timeouts and size limits
4. **Log security events**: Log authentication and authorization events
5. **Keep dependencies updated**: Regularly update dependencies

### Performance

1. **Cache results**: Cache expensive operations
2. **Use async**: Use async/await for I/O operations
3. **Batch operations**: Batch similar requests
4. **Set timeouts**: Configure appropriate timeouts
5. **Monitor metrics**: Track performance metrics

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Problem: ModuleNotFoundError
# Solution: Install dependencies
pip install -r requirements.txt
```

#### Configuration Errors

```bash
# Problem: Configuration validation failed
# Solution: Check config.yaml syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

#### Tool Not Found

```python
# Problem: Tool 'my_tool' not found
# Solution: Verify tool is registered
server.register_tool("my_tool", my_tool_function)
print(server.list_tools())
```

#### Permission Errors

```bash
# Problem: Permission denied accessing files
# Solution: Check allowed_paths configuration
# Ensure paths are within allowed directories
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Run server with debug mode:

```bash
export LOG_LEVEL=DEBUG
python server.py
```

Check server info:

```python
server = MCPServer("my-server")
info = server.get_server_info()
print(json.dumps(info, indent=2))
```

---

## Examples

### Example 1: Simple Calculator Tool

```python
def calculate(operation: str, a: float, b: float) -> Dict[str, Any]:
    """Simple calculator tool."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    result = operations[operation](a, b)

    return {
        "operation": operation,
        "inputs": [a, b],
        "result": result,
    }
```

### Example 2: File Processing Tool

```python
from pathlib import Path

def process_file(file_path: str, operation: str) -> Dict[str, Any]:
    """Process a file with specified operation."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if operation == "count_lines":
        with open(path) as f:
            count = len(f.readlines())
        return {"line_count": count}

    elif operation == "count_words":
        with open(path) as f:
            count = len(f.read().split())
        return {"word_count": count}

    else:
        raise ValueError(f"Unknown operation: {operation}")
```

### Example 3: API Integration Tool

```python
import httpx

def fetch_weather(city: str) -> Dict[str, Any]:
    """Fetch weather data for a city."""
    # Note: This is a template - add your API key
    url = f"https://api.weather.com/v1/weather?city={city}"

    with httpx.Client() as client:
        response = client.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        return {
            "city": city,
            "temperature": data.get("temperature"),
            "conditions": data.get("conditions"),
            "timestamp": data.get("timestamp"),
        }
```

---

## Advanced Topics

### Async Tools

```python
async def async_tool(param: str) -> Dict[str, Any]:
    """Example async tool."""
    import asyncio
    await asyncio.sleep(1)
    return {"result": param}
```

### Custom Validation

```python
from pydantic import validator

class CustomRequest(BaseModel):
    email: str

    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_tool(param: str) -> Dict[str, Any]:
    """Tool with caching."""
    # Expensive operation
    return {"result": param}
```

---

## Contributing

To contribute to this template:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

---

## License

This template is provided as-is for use in building MCP servers.

---

## Support

For issues, questions, or contributions:

- Check the troubleshooting section
- Review example implementations
- Consult the MCP specification
- Open an issue in the repository

---

## Changelog

### v1.0.0 (2025-10-26)

- Initial template release
- Core server implementation
- Example tools
- Comprehensive test suite
- Docker support
- Complete documentation

---

## Next Steps

1. **Customize Configuration**: Edit `config.yaml` for your needs
2. **Create Tools**: Add your custom tools in `tools/`
3. **Write Tests**: Add tests for your tools
4. **Deploy**: Use Docker or local deployment
5. **Monitor**: Set up logging and metrics
6. **Iterate**: Improve based on usage patterns

Happy building!
