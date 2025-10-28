# ailang CLI - Command Reference

Complete reference for all ailang CLI commands.

## Global Options

Available for all commands:

```bash
--verbose, -v    # Enable verbose output
--quiet, -q      # Suppress non-essential output
--version        # Show version and exit
--help          # Show help message
```

---

## Models Commands

Manage local LLM models with Ollama.

### `ailang models list`

List all available local models.

**Usage:**
```bash
ailang models list
ailang models list --json-output
```

**Options:**
- `--json-output` - Output as JSON for scripting

**Output:**
- Table view: Model name, ID, size, modified date
- JSON view: Structured array of model objects

**Example:**
```bash
$ ailang models list
Available Local Models
┌────────────────┬──────────────┬─────────┬──────────────┐
│ Model          │ ID           │ Size    │ Modified     │
├────────────────┼──────────────┼─────────┼──────────────┤
│ qwen3:8b       │ abc123       │ 4.7GB   │ 2 days ago   │
│ gemma3:12b     │ def456       │ 7.2GB   │ 1 week ago   │
└────────────────┴──────────────┴─────────┴──────────────┘
```

---

### `ailang models pull`

Pull a specific model from Ollama registry.

**Usage:**
```bash
ailang models pull <name>
ailang models pull <name> --force
```

**Arguments:**
- `name` - Model name with version (e.g., qwen3:8b)

**Options:**
- `--force` - Force pull even if model exists

**Features:**
- Progress bar during download
- Automatic verification
- Duplicate detection

**Example:**
```bash
$ ailang models pull qwen3:8b
Pulling model 'qwen3:8b'...
████████████████████ 100% Complete
✓ Successfully pulled model 'qwen3:8b'
```

---

### `ailang models info`

Show detailed information about a model.

**Usage:**
```bash
ailang models info <name>
```

**Arguments:**
- `name` - Model name (e.g., qwen3:8b)

**Output:**
- Model parameters
- Architecture details
- License information
- Template format

**Example:**
```bash
$ ailang models info qwen3:8b
Model Information: qwen3:8b
┌─────────────────┬──────────────────────────────┐
│ Property        │ Value                        │
├─────────────────┼──────────────────────────────┤
│ Parameters      │ 8B                           │
│ Format          │ GGUF                         │
│ Family          │ Qwen                         │
└─────────────────┴──────────────────────────────┘
```

---

### `ailang models benchmark`

Benchmark a model's performance.

**Usage:**
```bash
ailang models benchmark <name>
ailang models benchmark <name> --iterations 10
```

**Arguments:**
- `name` - Model name to benchmark

**Options:**
- `--iterations, -n` - Number of test iterations (default: 5)

**Output:**
- Average response time
- Min/max times
- Iteration count

**Example:**
```bash
$ ailang models benchmark qwen3:8b --iterations 5
Benchmarking model 'qwen3:8b' with 5 iterations...
████████████████████ 100% Complete

Benchmark Results: qwen3:8b
┌─────────────────┬──────────┐
│ Metric          │ Value    │
├─────────────────┼──────────┤
│ Average Time    │ 2.34s    │
│ Min Time        │ 2.15s    │
│ Max Time        │ 2.58s    │
│ Iterations      │ 5        │
└─────────────────┴──────────┘
```

---

### `ailang models recommend`

Get model recommendations for specific tasks.

**Usage:**
```bash
ailang models recommend <task>
ailang models recommend <task> --variant <variant>
```

**Arguments:**
- `task` - Task type (coding, reasoning, vision, multilingual, edge)

**Options:**
- `--variant` - Specific variant (fast, best, balanced, lightweight, minimal)

**Available Tasks:**
- `coding` - Code generation and analysis
- `reasoning` - Complex reasoning tasks
- `vision` - Vision and multimodal
- `multilingual` - Multiple languages
- `edge` - Edge/mobile devices

**Example:**
```bash
$ ailang models recommend coding
Recommended Models for: coding
┌─────────────────┬─────────────────┬────────────────┐
│ Variant         │ Model           │ Status         │
├─────────────────┼─────────────────┼────────────────┤
│ fast            │ qwen3:30b-a3b   │ ✗ Not installed│
│ balanced        │ qwen3:8b        │ ✓ Installed    │
│ lightweight     │ gemma3:4b       │ ✗ Not installed│
└─────────────────┴─────────────────┴────────────────┘

To install missing models:
  ailang models pull qwen3:30b-a3b
  ailang models pull gemma3:4b
```

---

## Examples Commands

Run and manage example projects.

### `ailang examples list`

Show all available examples.

**Usage:**
```bash
ailang examples list
ailang examples list --category <category>
ailang examples list --json-output
```

**Options:**
- `--category` - Filter by category (e.g., 01-foundation)
- `--json-output` - Output as JSON

**Output:**
- Tree view of examples by category
- Example name and description
- Total count

**Example:**
```bash
$ ailang examples list
Available Examples
├─ 01-foundation
│  ├─ hello_ollama: Basic Ollama interaction
│  └─ langchain_intro: Introduction to LangChain
├─ 02-mcp
│  ├─ filesystem_integration: MCP filesystem server
│  └─ sqlite_integration: MCP SQLite server
└─ 03-multi-agent
   └─ langgraph_basics: Basic LangGraph workflow

Total examples: 5
```

---

### `ailang examples run`

Run a specific example.

**Usage:**
```bash
ailang examples run <name>
ailang examples run <name> --args "<arguments>"
```

**Arguments:**
- `name` - Example name (category/name or just name)

**Options:**
- `--args` - Additional arguments to pass to example

**Example:**
```bash
$ ailang examples run 01-foundation/hello_ollama
Running example: 01-foundation/hello_ollama
Basic Ollama interaction

[Example output here...]

✓ Example completed successfully
```

---

### `ailang examples validate`

Validate all examples can be imported and run.

**Usage:**
```bash
ailang examples validate
ailang examples validate --category <category>
ailang examples validate --fast
```

**Options:**
- `--category` - Validate specific category only
- `--fast` - Skip long-running validations

**Output:**
- Pass/fail status for each example
- Error details for failures
- Overall pass rate

**Example:**
```bash
$ ailang examples validate --fast
Validating 5 examples...

Validation Results:

✓ Passed: 5
  01-foundation/hello_ollama
  01-foundation/langchain_intro
  02-mcp/filesystem_integration
  02-mcp/sqlite_integration
  03-multi-agent/langgraph_basics

Pass rate: 100.0% (5/5)
```

---

## MCP Commands

Manage Model Context Protocol (MCP) servers.

### `ailang mcp list`

List all available MCP servers.

**Usage:**
```bash
ailang mcp list
ailang mcp list --json-output
```

**Options:**
- `--json-output` - Output as JSON

**Output:**
- Server name, type, status, PID
- Running/stopped status
- Total server count

**Example:**
```bash
$ ailang mcp list
MCP Servers
┌────────────────┬──────────┬──────────┬────────┐
│ Name           │ Type     │ Status   │ PID    │
├────────────────┼──────────┼──────────┼────────┤
│ filesystem     │ official │ running  │ 12345  │
│ sqlite         │ official │ stopped  │ -      │
│ custom-server  │ custom   │ stopped  │ -      │
└────────────────┴──────────┴──────────┴────────┘

Total servers: 3
```

---

### `ailang mcp start`

Start an MCP server.

**Usage:**
```bash
ailang mcp start <name>
ailang mcp start <name> --port <port>
ailang mcp start <name> --background
```

**Arguments:**
- `name` - Server name

**Options:**
- `--port` - Port to run server on
- `--background, -b` - Run in background

**Features:**
- Foreground or background execution
- PID tracking for background processes
- Restart confirmation if already running
- Auto-detect Python/Node.js servers

**Example:**
```bash
$ ailang mcp start filesystem --background
Starting MCP server 'filesystem'...
✓ Server started in background (PID: 12345)
```

---

### `ailang mcp stop`

Stop a running MCP server.

**Usage:**
```bash
ailang mcp stop <name>
```

**Arguments:**
- `name` - Server name

**Features:**
- Graceful shutdown
- PID cleanup
- Process verification

**Example:**
```bash
$ ailang mcp stop filesystem
Stopping server 'filesystem' (PID: 12345)...
✓ Server stopped successfully
```

---

### `ailang mcp test`

Test an MCP server's functionality.

**Usage:**
```bash
ailang mcp test <name>
ailang mcp test <name> --timeout <seconds>
```

**Arguments:**
- `name` - Server name

**Options:**
- `--timeout` - Test timeout in seconds (default: 10)

**Requirements:**
- Server must have test.py file

**Example:**
```bash
$ ailang mcp test filesystem
Testing MCP server 'filesystem'...
✓ All tests passed
```

---

## RAG Commands

Manage Retrieval-Augmented Generation (RAG) systems.

### `ailang rag list`

List all RAG collections.

**Usage:**
```bash
ailang rag list
ailang rag list --json-output
```

**Options:**
- `--json-output` - Output as JSON

**Output:**
- Collection name, document count, created date
- Description
- Total collections

**Example:**
```bash
$ ailang rag list
RAG Collections
┌─────────────┬────────────┬──────────────┬─────────────────────┐
│ Name        │ Documents  │ Created      │ Description         │
├─────────────┼────────────┼──────────────┼─────────────────────┤
│ my-docs     │ 42         │ 2 days ago   │ Project docs        │
│ codebase    │ 128        │ 1 week ago   │ Source code index   │
└─────────────┴────────────┴──────────────┴─────────────────────┘

Total collections: 2
```

---

### `ailang rag index`

Index documents from a directory into a RAG collection.

**Usage:**
```bash
ailang rag index <directory>
ailang rag index <directory> --name <name>
ailang rag index <directory> --extensions <ext1,ext2>
ailang rag index <directory> --exclude <pattern1,pattern2>
```

**Arguments:**
- `directory` - Path to directory to index

**Options:**
- `--name` - Collection name (defaults to directory name)
- `--description` - Collection description
- `--extensions` - File extensions to index (default: .txt,.md,.py,.js,.ts)
- `--exclude` - Patterns to exclude (comma-separated)

**Features:**
- Progress bar during indexing
- Multiple file format support
- Pattern exclusion
- Metadata tracking

**Example:**
```bash
$ ailang rag index ./docs --name my-docs --exclude test,__pycache__
Indexing directory: /path/to/docs
Extensions: .txt, .md, .py
Excluding: test, __pycache__

Found 42 files to index

Indexing files... ████████████████████ 100%

✓ Successfully indexed 42 documents
Collection: my-docs
Location: ~/.ailang/rag/my-docs
```

---

### `ailang rag query`

Query a RAG collection.

**Usage:**
```bash
ailang rag query <collection> <query>
ailang rag query <collection> <query> --top-k <n>
ailang rag query <collection> <query> --json-output
```

**Arguments:**
- `collection` - Collection name
- `query` - Search query

**Options:**
- `--top-k` - Number of results to return (default: 5)
- `--json-output` - Output as JSON

**Features:**
- Keyword-based search (extensible to vectors)
- Ranked results
- Snippet preview

**Example:**
```bash
$ ailang rag query my-docs "installation guide" --top-k 3
Querying collection 'my-docs'...
Query: installation guide

Found 3 results:

1. docs/install.md
   Score: 5
   This guide walks through the installation process...

2. docs/quickstart.md
   Score: 3
   Quick installation: Run ./install.sh to begin...

3. docs/setup.md
   Score: 2
   Setup and installation instructions for new users...
```

---

### `ailang rag delete`

Delete a RAG collection.

**Usage:**
```bash
ailang rag delete <collection>
ailang rag delete <collection> --force
```

**Arguments:**
- `collection` - Collection name

**Options:**
- `--force` - Skip confirmation prompt

**Features:**
- Confirmation prompt (unless --force)
- Complete removal of collection data

**Example:**
```bash
$ ailang rag delete old-collection
This will delete collection 'old-collection' with 42 documents.
Are you sure? [y/N]: y
✓ Collection 'old-collection' deleted successfully
```

---

## Command Chaining

### Common Workflows

**Setup workflow:**
```bash
# Pull model, benchmark, and run example
ailang models pull qwen3:8b && \
ailang models benchmark qwen3:8b && \
ailang examples run 01-foundation/hello_ollama
```

**MCP workflow:**
```bash
# Start multiple servers
ailang mcp start filesystem --background && \
ailang mcp start sqlite --background && \
ailang mcp list
```

**RAG workflow:**
```bash
# Index and query
ailang rag index ./docs --name my-docs && \
ailang rag query my-docs "getting started"
```

### Scripting Examples

**Check and pull model:**
```bash
if ! ailang models list --json-output | grep -q "qwen3:8b"; then
    ailang models pull qwen3:8b
fi
```

**Run all examples in category:**
```bash
ailang examples list --category 01-foundation --json-output | \
    jq -r '.[] | .name' | \
    while read name; do
        ailang examples run "01-foundation/$name"
    done
```

**Start all MCP servers:**
```bash
for server in $(ailang mcp list --json-output | jq -r '.[].name'); do
    ailang mcp start "$server" --background
done
```

---

## Exit Codes

Standard exit codes for scripting:

- `0` - Success
- `1` - Error (user error, system error, or runtime error)
- `2` - Command line usage error (Click default)

---

## Environment Variables

Currently not used, but reserved for future:

- `AILANG_HOME` - Override home directory (~/.ailang)
- `AILANG_OLLAMA_URL` - Override Ollama endpoint
- `AILANG_NO_COLOR` - Disable colored output
- `AILANG_LOG_LEVEL` - Set logging level

---

## Tips & Best Practices

### Performance
- Use `--json-output` for scripting (faster parsing)
- Use `--quiet` to suppress progress bars in scripts
- Background MCP servers reduce command latency

### Debugging
- Use `--verbose` for detailed output
- Check `ailang mcp list` to verify server status
- Use `ailang examples validate --fast` for quick checks

### Automation
- Always use JSON output in scripts
- Check exit codes for error handling
- Use `--force` flags to skip confirmations

### Organization
- Use descriptive RAG collection names
- Group examples by category
- Keep background servers documented

---

## Getting Help

### Command Help
```bash
ailang --help                    # General help
ailang models --help             # Models help
ailang models pull --help        # Specific command help
```

### Documentation
- README.md - Full documentation
- QUICKSTART.md - Getting started
- FEATURES.md - Feature overview
- ARCHITECTURE.md - Technical details

### Resources
- Project repository: GitHub
- Issue tracker: GitHub Issues
- Discussions: GitHub Discussions

---

**Note:** This reference covers ailang CLI v0.1.0. For the latest updates, see the project repository.
