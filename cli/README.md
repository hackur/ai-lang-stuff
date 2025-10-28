# ailang CLI

Comprehensive command-line interface for the AI Lang Stuff toolkit - a local-first AI development platform.

## Installation

```bash
# Install in development mode
cd cli
pip install -e .

# Or install from source
pip install .
```

## Quick Start

```bash
# Check installation
ailang --help

# List available models
ailang models list

# Show all examples
ailang examples list

# List MCP servers
ailang mcp list

# View RAG collections
ailang rag list
```

## Commands

### Model Management

Manage local LLM models with Ollama integration.

```bash
# List all local models
ailang models list
ailang models list --json-output

# Pull a new model
ailang models pull qwen3:8b
ailang models pull gemma3:12b --force

# Get model information
ailang models info qwen3:8b

# Benchmark model performance
ailang models benchmark qwen3:8b
ailang models benchmark qwen3:8b --iterations 10

# Get model recommendations
ailang models recommend coding
ailang models recommend reasoning --variant best
ailang models recommend vision
```

**Model Recommendations by Task:**

| Task | Recommended Models |
|------|-------------------|
| `coding` | qwen3:30b-a3b (fast), qwen3:8b (balanced), gemma3:4b (lightweight) |
| `reasoning` | qwen3:8b (best), gemma3:12b (fast) |
| `vision` | qwen3-vl:8b (best) |
| `multilingual` | gemma3:12b (best, 140+ languages) |
| `edge` | gemma3:4b (best), gemma3:3b (minimal) |

### Example Management

Run and validate example projects.

```bash
# List all examples
ailang examples list
ailang examples list --category 01-foundation
ailang examples list --json-output

# Run a specific example
ailang examples run 01-foundation/hello_ollama
ailang examples run 02-mcp/filesystem_integration
ailang examples run 03-multi-agent/langgraph_basics --args "--verbose"

# Validate all examples
ailang examples validate
ailang examples validate --category 02-mcp
ailang examples validate --fast
```

### MCP Server Management

Manage Model Context Protocol (MCP) servers.

```bash
# List all MCP servers
ailang mcp list
ailang mcp list --json-output

# Start an MCP server
ailang mcp start filesystem
ailang mcp start sqlite --port 8080
ailang mcp start custom-server --background

# Stop a running server
ailang mcp stop filesystem

# Test server functionality
ailang mcp test filesystem
ailang mcp test custom-server --timeout 30
```

**Background Mode:**
- Use `--background` / `-b` to run servers in background
- PID files stored in `~/.ailang/mcp-pids/`
- Use `ailang mcp list` to see running servers
- Use `ailang mcp stop <name>` to stop background servers

### RAG Utilities

Manage Retrieval-Augmented Generation (RAG) systems.

```bash
# List all collections
ailang rag list
ailang rag list --json-output

# Index documents from a directory
ailang rag index ./docs --name my-docs
ailang rag index ./code --name my-code --description "Project source code"
ailang rag index ./data --extensions .txt,.md,.py --exclude test,__pycache__

# Query a collection
ailang rag query my-docs "how to install"
ailang rag query my-code "authentication implementation" --top-k 10
ailang rag query my-docs "setup guide" --json-output

# Delete a collection
ailang rag delete my-docs
ailang rag delete old-collection --force
```

**RAG Features:**
- Indexes multiple file types (.txt, .md, .py, .js, .ts, etc.)
- Excludes patterns (tests, build artifacts, etc.)
- Simple keyword-based search (extensible to vector embeddings)
- Collections stored in `~/.ailang/rag/`

## Global Options

```bash
# Enable verbose output
ailang --verbose <command>
ailang -v <command>

# Quiet mode (suppress non-essential output)
ailang --quiet <command>
ailang -q <command>

# Show version
ailang --version

# Show help
ailang --help
ailang <command> --help
```

## Features

### Rich Terminal UI
- **Colored output** - Easy-to-read, color-coded messages
- **Progress bars** - Visual feedback for long operations
- **Spinner animations** - Loading indicators for async tasks
- **Tables** - Formatted data display
- **Tree views** - Hierarchical information

### Error Handling
- Graceful error messages
- Helpful troubleshooting tips
- Exit codes for scripting integration
- Verbose mode for debugging

### JSON Output
- All list commands support `--json-output`
- Easy integration with other tools
- Scriptable workflows

## Examples

### Complete Workflow

```bash
# 1. Check and install models
ailang models recommend coding --variant balanced
ailang models pull qwen3:8b

# 2. Benchmark the model
ailang models benchmark qwen3:8b --iterations 5

# 3. Run example projects
ailang examples list --category 01-foundation
ailang examples run 01-foundation/hello_ollama

# 4. Start MCP servers
ailang mcp start filesystem --background
ailang mcp start sqlite --background

# 5. Index documentation for RAG
ailang rag index ./docs --name project-docs
ailang rag query project-docs "API reference"

# 6. Validate everything works
ailang examples validate --fast
ailang mcp test filesystem
```

### Scripting Integration

```bash
#!/bin/bash

# Get list of models as JSON
models=$(ailang models list --json-output)

# Check if specific model exists
if echo "$models" | jq -e '.[] | select(.name == "qwen3:8b")' > /dev/null; then
    echo "Model exists"
else
    echo "Pulling model..."
    ailang models pull qwen3:8b
fi

# Run all examples in a category
ailang examples list --category 02-mcp --json-output | \
    jq -r '.[] | .name' | \
    while read example; do
        echo "Running: $example"
        ailang examples run "02-mcp/$example"
    done
```

## Directory Structure

```
cli/
├── ailang/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   └── commands/
│       ├── __init__.py
│       ├── models.py        # Model management
│       ├── examples.py      # Example runner
│       ├── mcp.py          # MCP server management
│       └── rag.py          # RAG utilities
├── setup.py                 # Installation script
└── README.md               # This file
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Type checking
mypy ailang/

# Linting
ruff check ailang/
```

## Troubleshooting

### Ollama Not Found

```bash
# Check if Ollama is installed
which ollama

# Install Ollama
# Visit: https://ollama.ai

# Start Ollama service
ollama serve
```

### Model Pull Fails

```bash
# Check internet connection
# Try specific model version
ailang models pull qwen3:8b

# Check Ollama logs
ollama logs
```

### MCP Server Won't Start

```bash
# Check if port is available
lsof -i :8080

# Check server logs (verbose mode)
ailang --verbose mcp start <server-name>

# Test server directly
ailang mcp test <server-name>
```

### Examples Fail Validation

```bash
# Run in verbose mode
ailang --verbose examples validate

# Run single example
ailang examples run <category>/<name>

# Check dependencies
pip list | grep langchain
```

## Configuration

The CLI uses these directories:

- **PID files**: `~/.ailang/mcp-pids/`
- **RAG collections**: `~/.ailang/rag/`
- **Cache**: `~/.cache/ailang/`

## Requirements

- Python 3.10+
- Ollama (for model management)
- Node.js (for JavaScript-based MCP servers)
- System dependencies per MCP server

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- Documentation: `/docs`
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Roadmap

- [ ] Vector embeddings for RAG
- [ ] LangSmith integration
- [ ] Model quantization tools
- [ ] Batch operations
- [ ] Configuration file support
- [ ] Plugin system
- [ ] Interactive TUI mode
- [ ] Docker support

## Credits

Built with:
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [Ollama](https://ollama.ai/) - Local LLM runtime

Part of the [AI Lang Stuff](https://github.com/yourusername/ai-lang-stuff) project.
