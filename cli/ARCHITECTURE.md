# ailang CLI - Architecture Documentation

## Overview

The ailang CLI is built using Click framework with a modular command structure, designed for extensibility and maintainability.

## Project Structure

```
cli/
├── ailang/                      # Main package
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # CLI entry point and group registration
│   └── commands/               # Command modules
│       ├── __init__.py
│       ├── models.py           # Model management
│       ├── examples.py         # Example runner
│       ├── mcp.py             # MCP server management
│       └── rag.py             # RAG utilities
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_cli.py            # Basic CLI tests
│   └── test_commands.py       # Command-specific tests
├── setup.py                   # Installation (legacy)
├── pyproject.toml            # Modern Python packaging
├── Makefile                  # Development commands
├── install.sh                # Installation script
├── demo.sh                   # Feature demo script
├── README.md                 # User documentation
├── QUICKSTART.md            # Getting started guide
├── FEATURES.md              # Feature overview
└── ARCHITECTURE.md          # This file
```

## Design Principles

### 1. Modularity

Each command group is a separate module:
- **Separation of concerns** - Each module handles one domain
- **Independent testing** - Modules can be tested in isolation
- **Easy extension** - New commands add new modules

### 2. Local-First

All operations prioritize local execution:
- **Privacy** - No external API calls (except Ollama pulls)
- **Speed** - Local operations are fast
- **Reliability** - Works offline

### 3. User Experience

Rich terminal interface:
- **Progressive disclosure** - Help when needed
- **Visual feedback** - Progress bars, colors
- **Error recovery** - Helpful error messages

### 4. Automation-Friendly

Scriptable by design:
- **JSON output** - Machine-readable format
- **Exit codes** - Standard success/failure
- **Quiet mode** - Minimal output for scripts

## Command Architecture

### Entry Point (`main.py`)

```python
@click.group()
@click.version_option()
@click.option("--verbose")
@click.option("--quiet")
@click.pass_context
def cli(ctx, verbose, quiet):
    """Main entry point"""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["QUIET"] = quiet

# Register command groups
cli.add_command(models.models)
cli.add_command(examples.examples)
cli.add_command(mcp.mcp)
cli.add_command(rag.rag)
```

**Responsibilities:**
- Global option handling
- Context setup
- Command group registration
- Help text display

### Command Modules

Each module follows this pattern:

```python
@click.group()
def command_group():
    """Group description"""
    pass

@command_group.command()
@click.argument("name")
@click.option("--flag")
@click.pass_context
def subcommand(ctx, name, flag):
    """Subcommand implementation"""
    # Access global options
    verbose = ctx.obj.get("VERBOSE")

    # Use rich console
    console.print("[green]Success[/green]")

    # Handle errors gracefully
    try:
        operation()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
```

## Component Details

### Models Module (`commands/models.py`)

**Purpose:** Manage local LLM models via Ollama

**Key Components:**
- `MODEL_RECOMMENDATIONS` - Dictionary of task-based recommendations
- `list()` - Display available models
- `pull()` - Download models with progress
- `info()` - Show model metadata
- `benchmark()` - Performance testing
- `recommend()` - Smart recommendations

**Dependencies:**
- subprocess (Ollama CLI)
- rich (Terminal UI)
- click (CLI framework)

**Design Notes:**
- All Ollama calls wrapped in try/except
- User-friendly error messages
- Progress feedback for long operations
- JSON output for automation

### Examples Module (`commands/examples.py`)

**Purpose:** Run and validate example projects

**Key Components:**
- `EXAMPLES_DIR` - Path to examples directory
- `get_examples()` - Scan and catalog examples
- `list()` - Display available examples
- `run()` - Execute specific example
- `validate()` - Test all examples

**Dependencies:**
- pathlib (File operations)
- subprocess (Python execution)
- rich (Terminal UI)

**Design Notes:**
- Automatic docstring extraction
- Category-based organization
- Pass-through arguments to examples
- Validation with progress tracking

### MCP Module (`commands/mcp.py`)

**Purpose:** Manage MCP (Model Context Protocol) servers

**Key Components:**
- `MCP_DIR` - Path to MCP servers
- `PID_DIR` - Process ID tracking directory
- `get_mcp_servers()` - Discover available servers
- `get_server_pid()` - Check if server running
- `list()` - Show all servers with status
- `start()` - Launch server (fg or bg)
- `stop()` - Gracefully stop server
- `test()` - Run server test suite

**Dependencies:**
- subprocess (Process management)
- signal (Process signaling)
- os (Process operations)
- pathlib (File operations)
- rich (Terminal UI)

**Design Notes:**
- PID file management in `~/.ailang/mcp-pids/`
- Background process support
- Automatic server type detection (Python/Node)
- Health checking

### RAG Module (`commands/rag.py`)

**Purpose:** Retrieval-Augmented Generation utilities

**Key Components:**
- `RAG_DATA_DIR` - Collections storage (`~/.ailang/rag/`)
- `get_collections()` - List available collections
- `list()` - Display collections with metadata
- `index()` - Create collection from directory
- `query()` - Search collection
- `delete()` - Remove collection

**Dependencies:**
- json (Data serialization)
- pathlib (File operations)
- shutil (Directory operations)
- rich (Terminal UI)

**Design Notes:**
- JSON document storage
- Metadata tracking per collection
- Configurable file extensions
- Pattern exclusion support
- Simple keyword search (extensible to vectors)

## Data Flow

### Model Pull Operation

```
User: ailang models pull qwen3:8b
  ↓
main.cli() - Parse global options
  ↓
models.pull() - Validate arguments
  ↓
subprocess.Popen() - Start ollama pull
  ↓
Progress UI - Show download progress
  ↓
Validation - Check pull succeeded
  ↓
Success message - Confirm to user
```

### Example Run Operation

```
User: ailang examples run 01-foundation/hello_ollama
  ↓
main.cli() - Parse global options
  ↓
examples.run() - Validate example exists
  ↓
get_examples() - Find matching example
  ↓
subprocess.run() - Execute Python script
  ↓
Stream output - Show example output
  ↓
Exit code check - Report success/failure
```

### RAG Index Operation

```
User: ailang rag index ./docs --name my-docs
  ↓
main.cli() - Parse global options
  ↓
rag.index() - Validate directory
  ↓
Scan files - Find matching extensions
  ↓
Progress bar - Show indexing progress
  ↓
Read & parse - Extract content
  ↓
JSON storage - Save documents
  ↓
Metadata - Save collection info
  ↓
Success message - Show stats
```

## Error Handling Strategy

### Levels

1. **User Error** - Invalid input, missing files
   - Show helpful error message
   - Suggest correction
   - Exit with code 1

2. **System Error** - Missing dependencies, permission issues
   - Detect specific error
   - Show installation/fix instructions
   - Exit with code 1

3. **Runtime Error** - Unexpected failures
   - Show error details
   - In verbose mode, show traceback
   - Exit with code 1

### Example Patterns

```python
# User error
if not example_exists:
    console.print(f"[red]Example '{name}' not found.[/red]")
    console.print("\n[yellow]Available examples:[/yellow]")
    # Show suggestions
    return

# System error
except FileNotFoundError:
    console.print("[red]Error: Ollama not found.[/red]")
    console.print("Install from: https://ollama.ai")
    raise click.Abort()

# Runtime error
except subprocess.CalledProcessError as e:
    console.print(f"[red]Command failed: {e}[/red]")
    if verbose:
        console.print(f"[dim]{e.stderr}[/dim]")
    raise click.Abort()
```

## Testing Strategy

### Test Types

1. **Unit Tests** - Test individual functions
   - Model recommendations structure
   - File scanning logic
   - Data parsing

2. **Integration Tests** - Test command execution
   - CLI invocation
   - Help text generation
   - Option parsing

3. **Smoke Tests** - Test basic functionality
   - Commands run without error
   - Help text displays
   - JSON output formats

### Test Organization

```
tests/
├── test_cli.py           # Basic CLI tests
│   ├── test_cli_help()
│   ├── test_cli_version()
│   └── test_command_groups()
│
└── test_commands.py      # Command-specific tests
    ├── TestModelsCommand
    ├── TestExamplesCommand
    ├── TestMCPCommand
    └── TestRAGCommand
```

### Running Tests

```bash
# All tests with coverage
make test

# Fast tests without coverage
make test-fast

# Specific test file
pytest tests/test_cli.py

# Specific test
pytest tests/test_cli.py::test_cli_help

# With verbose output
pytest -v tests/
```

## Configuration Management

### Current Approach

Environment-based configuration:
- Global options via CLI flags
- Context passed through Click
- No config file (simplicity)

### Future Approach

Configuration file support:

```yaml
# ~/.ailang/config.yaml
models:
  default: qwen3:8b
  ollama_url: http://localhost:11434

mcp:
  default_port: 8080
  servers_dir: ~/.local/share/mcp-servers

rag:
  default_extensions: [.txt, .md, .py]
  exclude_patterns: [test, __pycache__]

ui:
  color: true
  progress: true
  json_indent: 2
```

## Extension Points

### Adding New Command Group

1. Create new module: `ailang/commands/newcmd.py`
2. Define command group:
   ```python
   @click.group()
   def newcmd():
       """New command description"""
       pass
   ```
3. Register in `main.py`:
   ```python
   from ailang.commands import newcmd
   cli.add_command(newcmd.newcmd)
   ```

### Adding New Subcommand

1. Add to existing command module
2. Use decorator pattern:
   ```python
   @command_group.command()
   @click.option("--flag")
   def subcommand(flag):
       """Subcommand description"""
       # Implementation
   ```

### Custom Output Formats

1. Add format option:
   ```python
   @click.option("--format", type=click.Choice(["table", "json", "yaml"]))
   ```
2. Implement format handlers
3. Use rich for formatting

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**
   - Import modules only when needed
   - Scan directories on demand
   - Cache results within command

2. **Parallel Operations**
   - Use subprocess.Popen for async
   - Background server management
   - Concurrent validation

3. **Efficient I/O**
   - Stream large files
   - Batch database operations
   - Minimize disk access

### Profiling

```bash
# Profile command execution
python -m cProfile -o profile.stats -m ailang.main models list

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## Security Considerations

### Input Validation

- Path traversal prevention
- Command injection prevention
- Argument sanitization

### File Operations

- Check permissions before operations
- Use safe path joining
- Validate file types

### Process Management

- Track PIDs properly
- Clean up on exit
- Handle signals gracefully

## Deployment

### Installation Methods

1. **Development Install**
   ```bash
   pip install -e .
   ```
   - Editable mode
   - Live code changes
   - Full tooling

2. **User Install**
   ```bash
   pip install .
   ```
   - System or user site-packages
   - Stable version
   - No dev dependencies

3. **Distribution**
   ```bash
   python -m build
   twine upload dist/*
   ```
   - PyPI publication
   - Version management
   - Dependency resolution

### System Integration

Entry point registration:
```python
# pyproject.toml
[project.scripts]
ailang = "ailang.main:cli"
```

Installed as system command:
- Available in PATH
- Tab completion (future)
- Man page (future)

## Monitoring & Logging

### Current Approach

- Console output via rich
- Error messages to stderr
- Verbose flag for details

### Future Approach

Logging framework:
```python
import logging

logger = logging.getLogger("ailang")

# File logging
handler = logging.FileHandler("~/.ailang/logs/ailang.log")
logger.addHandler(handler)

# Structured logging
logger.info("Command executed", extra={
    "command": "models list",
    "duration": 0.5,
    "result": "success"
})
```

## Maintenance

### Code Quality

Tools:
- **ruff** - Fast Python linter
- **mypy** - Type checking
- **pytest** - Testing framework
- **coverage** - Test coverage

Commands:
```bash
make lint    # Check code quality
make format  # Auto-format code
make test    # Run tests
make check   # Run all checks
```

### Documentation

Keep updated:
- README.md - User documentation
- ARCHITECTURE.md - This file
- Inline docstrings - API documentation
- CHANGELOG.md - Version history

### Versioning

Semantic versioning:
- MAJOR - Breaking changes
- MINOR - New features
- PATCH - Bug fixes

## Future Enhancements

### Planned Improvements

1. **Performance**
   - Result caching
   - Parallel operations
   - Async I/O

2. **Features**
   - Config file support
   - Plugin system
   - Interactive TUI
   - Shell completion

3. **Integration**
   - LangSmith tracing
   - Docker support
   - CI/CD templates
   - IDE extensions

4. **Quality**
   - Comprehensive tests
   - Performance benchmarks
   - Security audit
   - Accessibility

## References

- [Click Documentation](https://click.palletsprojects.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Python Packaging](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

## Contributing

See main project CONTRIBUTING.md for:
- Code style guide
- PR process
- Testing requirements
- Documentation standards
