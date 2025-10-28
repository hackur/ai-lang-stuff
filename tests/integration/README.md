# Integration Tests

Comprehensive integration tests for the AI experimentation toolkit. These tests verify complete workflows and system integration.

## Test Structure

```
tests/integration/
├── conftest.py                     # Shared fixtures and utilities
├── test_examples_run.py            # Example script smoke tests
├── test_mcp_integration.py         # MCP server integration tests
├── test_rag_pipeline.py            # RAG pipeline end-to-end tests
└── test_multi_agent_workflows.py   # Multi-agent workflow tests
```

## Test Files

### conftest.py
Shared fixtures for all integration tests:
- **Directory fixtures**: `temp_dir`, `test_data_dir`, `vector_store_dir`, `checkpoint_dir`
- **Mock Ollama fixtures**: `mock_ollama_llm`, `mock_ollama_manager`, `mock_ollama_response`
- **Mock MCP fixtures**: `mock_mcp_filesystem`, `mock_mcp_web_search`
- **Test data fixtures**: `sample_documents`, `sample_pdf_content`, `mock_embeddings`
- **Cleanup fixtures**: Automatic cleanup of checkpoints and logging

### test_examples_run.py
Smoke tests for all example scripts:
- **Import validation**: Verifies examples can be imported without errors
- **Structure validation**: Checks for required components (main function, error handling)
- **Output patterns**: Validates expected output patterns and docstrings
- **Timeout handling**: Tests appropriate timeout behavior
- **Inventory checks**: Ensures expected examples exist

**Key Test Classes:**
- `TestFoundationExamples`: Tests for 01-foundation examples
- `TestMCPExamples`: Tests for 02-mcp examples
- `TestMultiAgentExamples`: Tests for 03-multi-agent examples
- `TestRAGExamples`: Tests for 04-rag examples
- `TestTimeoutHandling`: Timeout behavior tests
- `TestOutputPatterns`: Output format validation
- `TestExampleInventory`: Example availability checks

### test_mcp_integration.py
Full MCP server integration tests:
- **Filesystem operations**: List, read, write, search operations
- **Path validation**: Safety checks for path access
- **LangChain tool creation**: Converting MCP operations to LangChain tools
- **Agent integration**: Using MCP tools with LangChain agents
- **Combined tools**: Using multiple MCP servers together
- **Real file operations**: Actual filesystem operations with safety checks
- **Error handling**: Permission errors, missing files, invalid operations
- **Web search integration**: Web search tool testing (when available)

**Key Test Classes:**
- `TestFilesystemMCPIntegration`: Filesystem MCP client tests
- `TestMCPToolIntegration`: MCP tools with agents
- `TestCombinedMCPTools`: Multiple MCP servers together
- `TestRealFileOperations`: Actual file operations
- `TestMCPErrorHandling`: Error handling scenarios
- `TestWebSearchIntegration`: Web search functionality

### test_rag_pipeline.py
End-to-end RAG pipeline tests:
- **Document ingestion**: Loading and preprocessing documents
- **Chunking strategies**: Text splitting with overlap
- **Vector store operations**: Creation, persistence, retrieval
- **Query processing**: QA chain creation and execution
- **Source tracking**: Metadata preservation and source citations
- **Performance testing**: Chunk size impact and retrieval accuracy
- **Error handling**: Empty documents, missing collections

**Key Test Classes:**
- `TestDocumentIngestion`: Document loading and chunking
- `TestVectorStoreOperations`: Vector store CRUD operations
- `TestQueryProcessing`: QA chain and retrieval tests
- `TestSourceTracking`: Metadata and source preservation
- `TestEndToEndRAGPipeline`: Complete RAG workflows
- `TestRAGPerformance`: Performance characteristics
- `TestRAGErrorHandling`: Error scenarios

### test_multi_agent_workflows.py
Multi-agent workflow integration tests:
- **Workflow execution**: Sequential and conditional workflows
- **State persistence**: Checkpoint creation and recovery
- **Parallel execution**: Multiple agents running concurrently
- **Error recovery**: Handling errors within workflows
- **Multi-agent coordination**: Agent communication patterns
- **Checkpoint validation**: Database integrity checks

**Key Test Classes:**
- `TestWorkflowExecution`: Complete workflow execution
- `TestStatePersistence`: Checkpoint and recovery
- `TestParallelExecution`: Parallel agent patterns
- `TestErrorRecovery`: Error handling mechanisms
- `TestMultiAgentCoordination`: Agent coordination
- `TestCheckpointValidation`: Checkpoint data integrity

## Running Tests

### Run All Integration Tests
```bash
# From project root
pytest tests/integration/ -v

# With coverage
pytest tests/integration/ --cov=utils --cov=examples --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/integration/test_examples_run.py -v
pytest tests/integration/test_mcp_integration.py -v
pytest tests/integration/test_rag_pipeline.py -v
pytest tests/integration/test_multi_agent_workflows.py -v
```

### Run Tests by Marker
```bash
# Only integration tests (require Ollama)
pytest tests/integration/ -m integration

# Only slow tests
pytest tests/integration/ -m slow

# Only MCP tests
pytest tests/integration/ -m mcp

# Only RAG tests
pytest tests/integration/ -m rag

# Skip slow tests
pytest tests/integration/ -m "not slow"
```

### Run Specific Test Class or Method
```bash
# Run specific test class
pytest tests/integration/test_examples_run.py::TestFoundationExamples -v

# Run specific test method
pytest tests/integration/test_rag_pipeline.py::TestDocumentIngestion::test_text_document_loading -v
```

## Test Markers

Tests use pytest markers for categorization:

- **`@pytest.mark.integration`**: Requires Ollama server running
- **`@pytest.mark.slow`**: Takes significant time (>5 seconds)
- **`@pytest.mark.mcp`**: Requires MCP servers running
- **`@pytest.mark.rag`**: Tests RAG functionality

## Prerequisites

### For All Integration Tests
- Python 3.10+
- All dependencies installed: `uv pip install -e ".[dev]"`

### For Integration-Marked Tests
- Ollama running: `ollama serve`
- Required models pulled:
  ```bash
  ollama pull qwen3:8b
  ollama pull qwen3-embedding
  ```

### For MCP-Marked Tests
- MCP filesystem server running on port 8001
- MCP web search server running on port 8002 (if testing web search)

## Test Configuration

### pytest.ini Options
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires Ollama server",
    "slow: takes significant time",
    "mcp: requires MCP servers",
    "rag: tests RAG functionality"
]
```

### Environment Variables
Tests respect these environment variables:
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `MCP_FILESYSTEM_PORT`: MCP filesystem port (default: 8001)
- `MCP_WEB_SEARCH_PORT`: MCP web search port (default: 8002)
- `LANGCHAIN_TRACING_V2`: Disable in tests (set to "false")

## Fixtures

### Directory Fixtures
- **`temp_dir`**: Clean temporary directory, auto-cleanup
- **`test_data_dir`**: Pre-populated with sample files
- **`vector_store_dir`**: Directory for vector store persistence
- **`checkpoint_dir`**: Directory for LangGraph checkpoints

### Mock Fixtures
- **`mock_ollama_llm`**: Mock ChatOllama instance
- **`mock_ollama_manager`**: Mock OllamaManager
- **`mock_ollama_response`**: Mock AIMessage response
- **`mock_mcp_filesystem`**: Mock filesystem MCP server
- **`mock_mcp_web_search`**: Mock web search MCP server
- **`mock_embeddings`**: Mock embedding model

### Data Fixtures
- **`sample_documents`**: List of sample documents with metadata
- **`sample_pdf_content`**: Sample PDF-like text content
- **`test_data_dir`**: Directory with sample files (txt, json, nested)

### Utility Fixtures
- **`capture_logs`**: Configured log capturing
- **`mock_env_vars`**: Temporary environment variables
- **`cleanup_checkpoints`**: Auto-cleanup checkpoint files (autouse)
- **`reset_logging`**: Reset logging config (autouse)

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev]"

    - name: Install Ollama
      run: |
        curl -fsSL https://ollama.com/install.sh | sh
        ollama serve &
        sleep 5
        ollama pull qwen3:8b
        ollama pull qwen3-embedding

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v -m "not slow" --cov=utils --cov=examples

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Writing New Integration Tests

### Test Structure Template
```python
"""
Integration tests for [component].

Brief description of what this test file covers.
"""

from pathlib import Path
import pytest

# Import fixtures
from conftest import (
    temp_dir,
    mock_ollama_llm,
    # ... other fixtures
)


class Test[Component]:
    """Test suite for [component]."""

    def test_[specific_functionality](self, fixture1, fixture2):
        """Test [specific functionality].

        Args:
            fixture1: Description
            fixture2: Description
        """
        # Arrange
        # ... setup

        # Act
        # ... execute

        # Assert
        assert expected_condition
```

### Best Practices

1. **Use appropriate markers**:
   ```python
   @pytest.mark.integration
   @pytest.mark.slow
   def test_long_running_operation():
       pass
   ```

2. **Use fixtures for setup/cleanup**:
   ```python
   def test_with_temp_files(temp_dir, test_data_dir):
       # temp_dir is automatically cleaned up
       test_file = temp_dir / "test.txt"
       test_file.write_text("test")
   ```

3. **Mock external dependencies**:
   ```python
   def test_agent_execution(mock_ollama_llm):
       # Use mock instead of real Ollama
       agent = create_agent(mock_ollama_llm)
   ```

4. **Test error conditions**:
   ```python
   def test_handles_missing_file(temp_dir):
       with pytest.raises(FileNotFoundError):
           load_document(temp_dir / "missing.txt")
   ```

5. **Validate state completely**:
   ```python
   def test_workflow_execution(checkpoint_dir):
       # Execute workflow
       final_state = run_workflow()

       # Validate all aspects
       assert final_state["counter"] == expected_count
       assert len(final_state["messages"]) > 0
       assert final_state["result"] == expected_result
   ```

## Troubleshooting

### Ollama Connection Errors
```
Error: Connection refused to localhost:11434
```
**Solution**: Start Ollama: `ollama serve`

### Model Not Found Errors
```
Error: model 'qwen3:8b' not found
```
**Solution**: Pull model: `ollama pull qwen3:8b`

### MCP Server Errors
```
Error: MCP server connection failed
```
**Solution**:
- Check if MCP server is running
- Verify correct port configuration
- Tests should gracefully skip if MCP unavailable

### Checkpoint Database Locked
```
Error: database is locked
```
**Solution**:
- Ensure only one test accesses checkpoint at a time
- Use unique checkpoint files per test
- Auto-cleanup fixtures should prevent this

### Vector Store Errors
```
Error: Collection not found
```
**Solution**:
- Ensure vector store directory has write permissions
- Use unique collection names per test
- Check that embeddings fixture is properly configured

## Performance Considerations

### Test Execution Time
- **Fast tests** (<1s): Unit-style integration tests with mocks
- **Medium tests** (1-5s): Tests with real vector stores, small datasets
- **Slow tests** (>5s): End-to-end workflows with real models

Mark slow tests appropriately:
```python
@pytest.mark.slow
def test_complete_rag_pipeline():
    pass
```

### Resource Usage
- Tests use temporary directories that are automatically cleaned up
- Checkpoint databases are isolated per test
- Mock embeddings avoid real embedding computation
- Use smaller datasets for faster test execution

## Contributing

When adding new integration tests:
1. Place in appropriate test file based on component
2. Add appropriate pytest markers
3. Use existing fixtures when possible
4. Document test purpose in docstring
5. Include error handling tests
6. Ensure tests are isolated (no shared state)
7. Add to this README if introducing new patterns

## Related Documentation

- [Main README](../../README.md) - Project overview
- [Utils README](../../utils/README.md) - Utility module documentation
- [Examples](../../examples/) - Example scripts being tested
- [pytest Documentation](https://docs.pytest.org/) - pytest reference
