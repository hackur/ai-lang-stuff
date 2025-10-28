# MCP Server Template - Setup Guide

Complete setup instructions for the MCP Server Template.

---

## Installation

### Step 1: Copy Template

```bash
# Navigate to mcp-servers directory
cd /Volumes/JS-DEV/ai-lang-stuff/mcp-servers/

# Copy template to your new server
cp -r template my-custom-server
cd my-custom-server
```

### Step 2: Install Dependencies

Before running the server, install required dependencies:

```bash
# Option A: Using pip
pip install pydantic pyyaml

# Option B: Using uv (recommended, faster)
uv pip install pydantic pyyaml

# Option C: Install all dependencies including optional ones
pip install -r requirements.txt

# Option D: Install in virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pydantic pyyaml
```

### Step 3: Verify Installation

```bash
# Test imports
python3 -c "import pydantic; import yaml; print('All dependencies installed!')"
```

---

## Running the Server

### Basic Usage

```bash
# Run the example server
python3 server.py
```

Expected output:

```
2025-10-26 10:00:00,000 - __main__ - INFO - Initialized MCP server: example-server v1.0.0
2025-10-26 10:00:00,001 - __main__ - INFO - Registered tool: greet
================================================================================
SERVER INFO
================================================================================
{
  "name": "example-server",
  "version": "1.0.0",
  "description": "Example MCP server for demonstration",
  "tools": [
    {
      "name": "greet",
      "description": "Greet a person with a custom greeting",
      ...
    }
  ]
}
```

### Running with Configuration

```bash
# Create your config file (or use the provided config.yaml)
# Then run:
python3 server.py

# The server will load config.yaml by default if using load_config()
```

### Running Tests

```bash
# Install test dependencies first
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test
pytest tests/test_server.py::TestMCPServer::test_server_initialization -v
```

---

## Development Setup

For active development, use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black ruff mypy

# Make changes...

# Run tests
pytest tests/ -v

# Format code
black .

# Lint code
ruff check .

# Type check
mypy server.py
```

---

## Docker Setup

### Build and Run

```bash
# Build Docker image
docker build -t my-mcp-server .

# Run container
docker run -p 8000:8000 my-mcp-server

# Run with custom config
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml my-mcp-server

# Run in background
docker run -d -p 8000:8000 --name mcp-server my-mcp-server

# View logs
docker logs -f mcp-server

# Stop container
docker stop mcp-server

# Remove container
docker rm mcp-server
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up -d --build
```

---

## Project Structure After Setup

```
my-custom-server/
├── server.py              # Main server implementation
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose config
├── README.md             # Full documentation
├── QUICKSTART.md         # Quick start guide
├── SETUP.md              # This file
├── __init__.py           # Package initialization
├── .gitignore            # Git ignore rules
├── .dockerignore         # Docker ignore rules
├── venv/                 # Virtual environment (if created)
├── tools/                # Tool implementations
│   ├── __init__.py
│   └── example_tool.py
└── tests/                # Test suite
    ├── __init__.py
    └── test_server.py
```

---

## Customization Checklist

After setup, customize these files:

- [ ] **config.yaml**: Update server name, version, and description
- [ ] **server.py**: Modify main() to register your tools
- [ ] **tools/**: Add your custom tool implementations
- [ ] **tests/**: Add tests for your tools
- [ ] **requirements.txt**: Add any additional dependencies
- [ ] **README.md**: Update with your server's documentation
- [ ] **Dockerfile**: Adjust if you need additional system dependencies
- [ ] **.gitignore**: Add any custom files to ignore

---

## Verification

After setup, verify everything works:

```bash
# 1. Verify imports
python3 -c "from server import MCPServer; print('Server imports OK')"

# 2. Verify tools
python3 -c "from tools.example_tool import greet; print(greet('Test'))"

# 3. Run server
python3 server.py

# 4. Run tests (if pytest installed)
pytest tests/ -v

# 5. Check Docker (if Docker installed)
docker build -t test-server . && echo "Docker build OK"
```

---

## Common Issues

### Issue 1: Module Not Found

```bash
# Error: ModuleNotFoundError: No module named 'pydantic'
# Solution:
pip install pydantic pyyaml
```

### Issue 2: Python Version

```bash
# Error: Python version incompatibility
# Solution: Use Python 3.10 or higher
python3 --version  # Should be 3.10+
```

### Issue 3: Permission Denied

```bash
# Error: Permission denied
# Solution: Use virtual environment or user install
pip install --user pydantic pyyaml
```

### Issue 4: Import Error After Install

```bash
# Error: Module installed but still can't import
# Solution: Ensure you're using the correct Python
which python3
python3 -m pip install pydantic pyyaml
```

---

## Next Steps

1. **Read QUICKSTART.md** for a 5-minute tutorial
2. **Read README.md** for complete documentation
3. **Explore tools/example_tool.py** for tool patterns
4. **Check tests/test_server.py** for testing examples
5. **Start building your custom tools!**

---

## Support

For help:

1. Check README.md troubleshooting section
2. Verify all dependencies are installed
3. Check Python version (must be 3.10+)
4. Review example implementations
5. Open an issue if problems persist

Happy building!
