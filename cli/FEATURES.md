# ailang CLI - Feature Overview

Comprehensive feature list and capabilities of the ailang CLI tool.

## Core Features

### 1. Model Management (`ailang models`)

**Commands:**
- `list` - Display all local models with details
- `pull` - Download models from Ollama registry
- `info` - Show detailed model information
- `benchmark` - Performance testing with configurable iterations
- `recommend` - AI-powered model recommendations by task type

**Features:**
- ✅ JSON output for scripting
- ✅ Progress bars for downloads
- ✅ Automatic model verification
- ✅ Performance benchmarking
- ✅ Smart recommendations (6 task categories)
- ✅ Force pull option
- ✅ Detailed model metadata

**Recommendations by Task:**
| Task | Best Model | Fast Alternative | Lightweight |
|------|-----------|------------------|-------------|
| Coding | qwen3:8b | qwen3:30b-a3b | gemma3:4b |
| Reasoning | qwen3:8b | gemma3:12b | - |
| Vision | qwen3-vl:8b | - | - |
| Multilingual | gemma3:12b | - | - |
| Edge/Mobile | gemma3:4b | gemma3:3b | - |

### 2. Example Management (`ailang examples`)

**Commands:**
- `list` - Browse all examples with tree view
- `run` - Execute specific examples with arguments
- `validate` - Test all examples can run successfully

**Features:**
- ✅ Hierarchical tree display
- ✅ Category filtering
- ✅ Automatic docstring extraction
- ✅ Pass-through arguments
- ✅ Batch validation
- ✅ Fast validation mode
- ✅ Progress tracking
- ✅ Error reporting with context

**Example Categories:**
- `01-foundation` - Basic LLM usage
- `02-mcp` - MCP server integration
- `03-multi-agent` - LangGraph workflows
- `04-rag` - RAG and vision systems
- `05-interpretability` - TransformerLens analysis
- `06-production` - Production patterns

### 3. MCP Server Management (`ailang mcp`)

**Commands:**
- `list` - Show all servers with status
- `start` - Launch servers (foreground or background)
- `stop` - Gracefully stop running servers
- `test` - Run server test suites

**Features:**
- ✅ Background process management
- ✅ PID tracking
- ✅ Status monitoring
- ✅ Custom port configuration
- ✅ Python and Node.js support
- ✅ Automatic server discovery
- ✅ Health checks
- ✅ Log integration

**Server Types:**
- Official servers (from MCP repository)
- Custom servers (project-specific)

**Background Mode:**
- Servers run as daemon processes
- PID files in `~/.ailang/mcp-pids/`
- Automatic cleanup on stop
- Status tracking across sessions

### 4. RAG Utilities (`ailang rag`)

**Commands:**
- `list` - Show all collections with metadata
- `index` - Create collections from directories
- `query` - Search collections with ranking
- `delete` - Remove collections safely

**Features:**
- ✅ Multi-format indexing (.txt, .md, .py, .js, .ts, etc.)
- ✅ Pattern exclusion (tests, build artifacts)
- ✅ Metadata tracking
- ✅ Simple keyword search (extensible to vectors)
- ✅ Configurable result limit
- ✅ Collection management
- ✅ Progress bars for indexing
- ✅ Document counting

**Storage:**
- Location: `~/.ailang/rag/`
- Format: JSON documents
- Metadata: Collection info, timestamps, stats

**Indexing Options:**
- Custom extensions
- Exclusion patterns
- Collection naming
- Descriptions

## UI/UX Features

### Rich Terminal Output

**Tables:**
- Model listings
- Example catalogs
- Server status
- Collection info
- Benchmark results

**Progress Indicators:**
- Spinners for async operations
- Progress bars for downloads/indexing
- Percentage displays
- Time estimates

**Color Coding:**
- Cyan: Headers, names
- Green: Success, positive status
- Yellow: Warnings, pending
- Red: Errors, failures
- Magenta: PIDs, IDs
- Dim: Auxiliary info

**Tree Views:**
- Hierarchical examples
- Nested categories
- Clear indentation
- Visual organization

### Error Handling

**Features:**
- Graceful degradation
- Helpful error messages
- Troubleshooting hints
- Context preservation
- Exit code standards

**Error Types:**
- Missing dependencies
- Network failures
- File not found
- Permission errors
- Timeout errors

### Output Formats

**Human-Readable:**
- Rich tables
- Tree structures
- Colored text
- Progress bars
- Formatted lists

**Machine-Readable:**
- JSON output flag
- Structured data
- Script-friendly
- Parseable format

## Advanced Features

### Global Options

**Available:**
- `--verbose` / `-v` - Detailed output
- `--quiet` / `-q` - Minimal output
- `--version` - Show version
- `--help` - Context-sensitive help

### Context-Aware Help

**Features:**
- Command-level help
- Subcommand help
- Example usage
- Option descriptions
- Default values

### Scripting Support

**Features:**
- JSON output for all list commands
- Exit codes (0 = success, 1 = failure)
- Stderr for errors
- Stdout for data
- Quiet mode for scripts

**Example Scripts:**
```bash
# Check if model exists
if ailang models list --json-output | jq -e '.[] | select(.name == "qwen3:8b")'; then
    echo "Model exists"
fi

# Run all examples in category
ailang examples list --category 02-mcp --json-output | \
    jq -r '.[] | .name' | \
    while read name; do
        ailang examples run "02-mcp/$name"
    done

# Start all MCP servers
for server in filesystem sqlite; do
    ailang mcp start "$server" --background
done
```

### Validation & Testing

**Built-in Tests:**
- Example validation
- MCP server testing
- Model benchmarking
- Import verification

**CI/CD Integration:**
- Exit codes
- JSON reporting
- Fast mode
- Parallel execution

## Performance Features

### Optimizations

**Speed:**
- Parallel operations where possible
- Lazy loading
- Efficient file scanning
- Minimal dependencies

**Resource Usage:**
- Streaming for large files
- Incremental processing
- Memory-efficient indexing
- Background processing

### Caching

**Future Enhancement:**
- Model metadata cache
- Example scan cache
- RAG index cache
- MCP discovery cache

## Security Features

### Safe Operations

**Protections:**
- Path validation
- Input sanitization
- Confirmation prompts
- Force flags for destructive ops

**Privacy:**
- Local-first design
- No external calls (except Ollama)
- No telemetry
- User data control

## Extensibility

### Plugin Architecture

**Future:**
- Custom commands
- Third-party integrations
- Model providers
- RAG backends

### Configuration

**Future:**
- Config file support (~/.ailang/config.yaml)
- Environment variables
- Per-project settings
- Global preferences

## Platform Support

### Operating Systems

**Tested:**
- macOS (primary)
- Linux (compatible)
- Windows (with WSL)

**Requirements:**
- Python 3.10+
- Ollama (for model commands)
- Node.js (for JS-based MCP servers)

## Integration Points

### Works With

**Tools:**
- Ollama - Local model runtime
- LangChain - Framework integration
- LangGraph - Multi-agent workflows
- MCP - Server protocol
- Git - Version control

**Workflows:**
- Development pipelines
- CI/CD systems
- Testing frameworks
- Documentation generators

## Future Roadmap

### Planned Features

**Short Term (v0.2.0):**
- [ ] Vector embeddings for RAG
- [ ] LangSmith tracing integration
- [ ] Model quantization tools
- [ ] Batch operations
- [ ] Config file support

**Medium Term (v0.3.0):**
- [ ] Interactive TUI mode
- [ ] Plugin system
- [ ] Custom model providers
- [ ] Advanced RAG (hybrid search)
- [ ] Performance profiling

**Long Term (v1.0.0):**
- [ ] Docker support
- [ ] Cloud integration (optional)
- [ ] Team features
- [ ] Enterprise support
- [ ] GUI companion app

## Metrics & Analytics

### Available Data

**Usage:**
- Command frequency
- Error rates
- Performance metrics
- Resource usage

**Future:**
- Optional telemetry (opt-in)
- Usage analytics
- Error reporting
- Feature adoption

## Documentation

### Resources

**Included:**
- README.md - Full documentation
- QUICKSTART.md - Getting started
- FEATURES.md - This file
- Inline help - Command help text

**Online:**
- Project repository
- GitHub wiki
- Issue tracker
- Discussions

## Developer Features

### Testing

**Framework:**
- pytest integration
- Coverage reporting
- CI/CD ready
- Fast test mode

**Commands:**
```bash
make test          # Full test suite
make test-fast     # Quick tests
make lint          # Code quality
make format        # Auto-format
```

### Development Mode

**Features:**
- Editable install
- Live reloading
- Debug logging
- Verbose output

**Setup:**
```bash
pip install -e ".[dev]"
```

## API Stability

### Versioning

**Scheme:** Semantic versioning (MAJOR.MINOR.PATCH)

**Guarantees:**
- Stable command structure
- Backward-compatible options
- Deprecation warnings
- Migration guides

### Breaking Changes

**Policy:**
- Major version bumps
- Advance notice (1 version)
- Clear migration path
- Parallel support period

## Performance Benchmarks

### Target Metrics

**Command Response Times:**
- `models list` - < 100ms
- `examples list` - < 200ms
- `mcp list` - < 50ms
- `rag query` - < 500ms

**Operation Times:**
- Model pull - Network dependent
- RAG indexing - ~1000 docs/sec
- Example validation - ~5 examples/sec
- Server start - < 2 seconds

## Quality Standards

### Code Quality

**Metrics:**
- Type coverage: 100%
- Test coverage: >80%
- Linting: Zero errors
- Documentation: Complete

**Tools:**
- ruff - Linting
- mypy - Type checking
- pytest - Testing
- coverage - Metrics

### User Experience

**Standards:**
- Response time < 1s
- Clear error messages
- Helpful suggestions
- Consistent formatting

## Summary

The ailang CLI provides a comprehensive, production-ready interface for local AI development with:

- **4 command groups** (models, examples, mcp, rag)
- **15+ commands** total
- **Rich UI** with colors, progress, tables
- **JSON output** for automation
- **Background processing** for servers
- **Validation** and testing tools
- **Extensible architecture**
- **Local-first** design

Built for developers who want powerful, privacy-preserving AI tools that run entirely on their local machine.
