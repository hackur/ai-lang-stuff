# ailang CLI - Quick Start Guide

Get started with the ailang CLI in 5 minutes.

## Installation

```bash
# Navigate to cli directory
cd /Volumes/JS-DEV/ai-lang-stuff/cli

# Install in development mode
pip install -e .

# Or for production
pip install .
```

## Verify Installation

```bash
# Check version
ailang --version

# Show help
ailang --help

# Test with a simple command
ailang models recommend coding
```

## First Steps

### 1. Set Up Models

```bash
# See what models are recommended for your use case
ailang models recommend coding
ailang models recommend reasoning
ailang models recommend vision

# Pull a recommended model
ailang models pull qwen3:8b

# List your local models
ailang models list

# Get details about a model
ailang models info qwen3:8b
```

### 2. Run Examples

```bash
# See all available examples
ailang examples list

# Run a simple example
ailang examples run 01-foundation/hello_ollama

# Run with arguments
ailang examples run 02-mcp/filesystem_integration --args "--verbose"

# Validate all examples work
ailang examples validate --fast
```

### 3. Start MCP Servers

```bash
# List available MCP servers
ailang mcp list

# Start a server in background
ailang mcp start filesystem --background

# Check server status
ailang mcp list

# Test server functionality
ailang mcp test filesystem

# Stop when done
ailang mcp stop filesystem
```

### 4. Use RAG System

```bash
# Index your documentation
ailang rag index ./docs --name my-docs --description "Project documentation"

# List collections
ailang rag list

# Query the collection
ailang rag query my-docs "how to install"
ailang rag query my-docs "API reference" --top-k 10

# Clean up when done
ailang rag delete my-docs
```

## Common Workflows

### Development Workflow

```bash
# 1. Pull and benchmark models
ailang models pull qwen3:8b
ailang models benchmark qwen3:8b --iterations 5

# 2. Validate examples
ailang examples validate --category 01-foundation

# 3. Start MCP servers you need
ailang mcp start filesystem --background
ailang mcp start sqlite --background

# 4. Index your project docs
ailang rag index ./docs --name project-docs

# 5. Run your examples
ailang examples run 03-multi-agent/langgraph_basics
```

### Quick Testing

```bash
# Fast validation
ailang examples validate --fast

# Test specific example
ailang examples run 01-foundation/hello_ollama

# Check server status
ailang mcp list
```

### Production Setup

```bash
# Pull production models
ailang models pull qwen3:8b
ailang models pull gemma3:12b

# Start required MCP servers
ailang mcp start filesystem --background --port 8081
ailang mcp start sqlite --background --port 8082

# Index production documentation
ailang rag index /path/to/docs --name prod-docs

# Validate everything
ailang examples validate
ailang mcp test filesystem
ailang mcp test sqlite
```

## Tips & Tricks

### Use JSON Output for Scripting

```bash
# Get structured output
ailang models list --json-output | jq '.[0].name'
ailang examples list --json-output | jq '.[] | select(.category == "01-foundation")'
ailang rag list --json-output | jq '.[] | .doc_count'
```

### Enable Verbose Mode

```bash
# Get detailed output for debugging
ailang --verbose models pull qwen3:8b
ailang -v examples run test-example
```

### Background MCP Servers

```bash
# Start multiple servers
ailang mcp start filesystem --background
ailang mcp start sqlite --background
ailang mcp start custom-server --background

# Check all running
ailang mcp list

# Stop all
for server in $(ailang mcp list --json-output | jq -r '.[].name'); do
    ailang mcp stop "$server"
done
```

### Batch Operations

```bash
# Pull multiple models
for model in qwen3:8b gemma3:12b qwen3-vl:8b; do
    ailang models pull "$model"
done

# Validate by category
for category in 01-foundation 02-mcp 03-multi-agent; do
    ailang examples validate --category "$category"
done
```

## Troubleshooting

### Ollama Not Found

```bash
# Check if installed
which ollama

# Start Ollama service
ollama serve
```

### Permission Errors

```bash
# Use user install
pip install --user -e .

# Or create virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Import Errors

```bash
# Install missing dependencies
pip install click rich

# Or reinstall
pip install --force-reinstall -e .
```

### MCP Server Issues

```bash
# Check if server exists
ls -la ../mcp-servers/

# Test server manually
python ../mcp-servers/custom/your-server/server.py

# Check logs (verbose mode)
ailang --verbose mcp start your-server
```

## Next Steps

1. **Explore Examples**: Run through all example categories
2. **Create Custom MCP Servers**: Add your own servers in `mcp-servers/custom/`
3. **Build RAG Collections**: Index your codebases and documents
4. **Benchmark Models**: Find the best model for your use case
5. **Integrate with Projects**: Use ailang in your development workflow

## Getting Help

```bash
# General help
ailang --help

# Command-specific help
ailang models --help
ailang examples --help
ailang mcp --help
ailang rag --help

# Subcommand help
ailang models pull --help
ailang rag index --help
```

## Resources

- **Full Documentation**: See `README.md`
- **Examples**: Browse `../examples/`
- **MCP Servers**: Check `../mcp-servers/`
- **Project Docs**: Visit `../docs/`

## Success Checklist

- [ ] CLI installed successfully (`ailang --version`)
- [ ] At least one model pulled (`ailang models list`)
- [ ] Can run examples (`ailang examples list`)
- [ ] MCP server can start (`ailang mcp list`)
- [ ] Can create RAG collection (`ailang rag index ./docs`)
- [ ] Tests pass (`cd .. && make test`)

You're ready to build with ailang!
