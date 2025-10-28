# Integration Tests Quick Start

Fast reference for running integration tests.

## TL;DR

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run all tests (fast - uses mocks)
pytest tests/integration/ -v

# Run with Ollama (requires Ollama running)
pytest tests/integration/ -m integration -v
```

## Common Commands

### Run All Tests
```bash
pytest tests/integration/ -v
```

### Run Specific File
```bash
pytest tests/integration/test_examples_run.py -v
pytest tests/integration/test_mcp_integration.py -v
pytest tests/integration/test_rag_pipeline.py -v
pytest tests/integration/test_multi_agent_workflows.py -v
```

### Run by Category
```bash
# Only integration tests (needs Ollama)
pytest tests/integration/ -m integration

# Skip slow tests
pytest tests/integration/ -m "not slow"

# Only MCP tests
pytest tests/integration/ -m mcp

# Only RAG tests
pytest tests/integration/ -m rag
```

### Run Specific Test
```bash
# By class
pytest tests/integration/test_examples_run.py::TestFoundationExamples -v

# By method
pytest tests/integration/test_rag_pipeline.py::TestDocumentIngestion::test_text_document_loading -v
```

### With Coverage
```bash
pytest tests/integration/ --cov=utils --cov=examples --cov-report=html
open htmlcov/index.html
```

## Setup Requirements

### Minimal Setup (for unmarked tests)
```bash
uv pip install -e ".[dev]"
```

### For Integration Tests
```bash
# Start Ollama
ollama serve

# Pull required models (in another terminal)
ollama pull qwen3:8b
ollama pull qwen3-embedding

# Run integration tests
pytest tests/integration/ -m integration
```

### For MCP Tests
```bash
# Start MCP servers (implementation-specific)
# Then run:
pytest tests/integration/ -m mcp
```

## Test Structure

```
tests/integration/
├── conftest.py                    # Shared fixtures
├── test_examples_run.py           # Example smoke tests (24 tests)
├── test_mcp_integration.py        # MCP integration (20 tests)
├── test_rag_pipeline.py           # RAG end-to-end (18 tests)
└── test_multi_agent_workflows.py  # Multi-agent (12 tests)
```

## Test Markers

- `integration` - Needs Ollama running
- `slow` - Takes >5 seconds
- `mcp` - Needs MCP servers
- `rag` - Tests RAG functionality

## Quick Examples

### Run Fast Tests Only
```bash
pytest tests/integration/ -m "not integration and not slow" -v
```

### Run Everything with Coverage
```bash
pytest tests/integration/ --cov=utils --cov=examples -v
```

### Debug Single Test
```bash
pytest tests/integration/test_examples_run.py::TestFoundationExamples::test_simple_chat_import -vvv -s
```

### List All Tests
```bash
pytest tests/integration/ --collect-only
```

## Troubleshooting

### "No module named pytest"
```bash
uv pip install -e ".[dev]"
```

### "Connection refused to localhost:11434"
```bash
ollama serve  # In separate terminal
```

### "Model not found"
```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding
```

### "MCP server connection failed"
Tests should gracefully skip if MCP servers aren't available.
MCP-marked tests require MCP servers running.

### "Database is locked"
Each test uses unique checkpoint files. If issues persist:
```bash
# Clean up checkpoint databases
rm tests/integration/checkpoints_*.db
rm -rf tests/integration/__pycache__
```

## File Descriptions

### conftest.py (438 lines)
- Shared fixtures for all tests
- Mock Ollama and MCP servers
- Temporary directories with auto-cleanup
- Test data and sample documents

### test_examples_run.py (447 lines)
- **Purpose**: Smoke tests for example scripts
- **Classes**: 7 test classes
- **Tests**: 24 test methods
- **Coverage**: Import validation, structure checks, output patterns

### test_mcp_integration.py (536 lines)
- **Purpose**: MCP server integration testing
- **Classes**: 6 test classes
- **Tests**: 20 test methods
- **Coverage**: Filesystem ops, tool integration, error handling

### test_rag_pipeline.py (665 lines)
- **Purpose**: End-to-end RAG workflows
- **Classes**: 7 test classes
- **Tests**: 18 test methods
- **Coverage**: Document ingestion, vector stores, QA chains

### test_multi_agent_workflows.py (667 lines)
- **Purpose**: LangGraph workflow testing
- **Classes**: 6 test classes
- **Tests**: 12 test methods
- **Coverage**: State persistence, parallel execution, error recovery

## Environment Variables

```bash
# Optional - override defaults
export OLLAMA_BASE_URL="http://localhost:11434"
export MCP_FILESYSTEM_PORT="8001"
export MCP_WEB_SEARCH_PORT="8002"
export LANGCHAIN_TRACING_V2="false"
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run integration tests
  run: |
    # Fast tests only (no Ollama)
    pytest tests/integration/ -m "not integration and not slow" -v
```

### With Ollama in CI
```yaml
- name: Install Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
    sleep 5
    ollama pull qwen3:8b

- name: Run integration tests
  run: |
    pytest tests/integration/ -v
```

## Test Statistics

- **Total Lines**: 2,753 lines
- **Test Files**: 5 files
- **Test Classes**: 26 classes
- **Test Methods**: 74 methods
- **Fixtures**: 20+ shared fixtures

## Quick Reference

| Task | Command |
|------|---------|
| All tests | `pytest tests/integration/ -v` |
| Fast only | `pytest tests/integration/ -m "not slow"` |
| With Ollama | `pytest tests/integration/ -m integration` |
| One file | `pytest tests/integration/test_examples_run.py` |
| One class | `pytest tests/integration/test_rag_pipeline.py::TestDocumentIngestion` |
| One test | `pytest ...::test_text_document_loading -v` |
| Coverage | `pytest tests/integration/ --cov=utils --cov=examples` |
| List tests | `pytest tests/integration/ --collect-only` |
| Debug | `pytest ... -vvv -s` |

## Next Steps

1. Run fast tests: `pytest tests/integration/ -m "not slow"`
2. If passing, run full suite: `pytest tests/integration/ -v`
3. Check coverage: `pytest tests/integration/ --cov=utils --cov-report=html`
4. For CI/CD, use selective markers to control execution time

## Documentation

- **README.md**: Full documentation with examples
- **TEST-SUMMARY.md**: Detailed statistics and design patterns
- **QUICKSTART.md**: This file

## Support

For issues or questions:
1. Check README.md troubleshooting section
2. Review test docstrings for expected behavior
3. Check conftest.py for available fixtures
4. Review example tests for patterns
