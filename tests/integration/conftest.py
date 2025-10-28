"""
Shared fixtures for integration tests.

Provides:
- Temporary directories and cleanup
- Mock MCP servers
- Test documents and data
- Ollama mocks
- Utility functions
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


# ============================================================================
# Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that's cleaned up after the test.

    Yields:
        Path: Temporary directory path
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    """Create a test data directory with sample files.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Test data directory with sample files
    """
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create sample text file
    (data_dir / "sample.txt").write_text("This is a sample text file for testing.")

    # Create sample JSON file
    import json
    (data_dir / "sample.json").write_text(
        json.dumps({"key": "value", "test": "data"})
    )

    # Create nested directory structure
    nested = data_dir / "nested" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "file.txt").write_text("Nested file content")

    return data_dir


@pytest.fixture
def vector_store_dir(temp_dir: Path) -> Path:
    """Create a directory for vector store persistence.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Vector store directory
    """
    vs_dir = temp_dir / "vector_stores"
    vs_dir.mkdir(parents=True, exist_ok=True)
    return vs_dir


@pytest.fixture
def checkpoint_dir(temp_dir: Path) -> Path:
    """Create a directory for LangGraph checkpoints.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Checkpoint directory
    """
    cp_dir = temp_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir


# ============================================================================
# Mock Ollama Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama_response() -> AIMessage:
    """Create a mock Ollama response.

    Returns:
        AIMessage: Mock AI response
    """
    return AIMessage(
        content="This is a mock response from the model.",
        response_metadata={"model": "qwen3:8b", "done": True}
    )


@pytest.fixture
def mock_ollama_llm(mock_ollama_response: AIMessage):
    """Create a mock Ollama LLM.

    Args:
        mock_ollama_response: Mock response fixture

    Returns:
        Mock: Mock LLM that returns predefined responses
    """
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_ollama_response
    mock_llm.batch.return_value = [mock_ollama_response]
    mock_llm.stream.return_value = iter(["Mock ", "streaming ", "response"])
    return mock_llm


@pytest.fixture
def mock_ollama_manager():
    """Create a mock OllamaManager.

    Returns:
        Mock: Mock OllamaManager with standard behavior
    """
    mock_manager = Mock()
    mock_manager.check_ollama_running.return_value = True
    mock_manager.ensure_model_available.return_value = True
    mock_manager.list_models.return_value = [
        {"name": "qwen3:8b", "size": "4.7GB"},
        {"name": "qwen3-embedding", "size": "274MB"},
    ]
    mock_manager.get_model_info.return_value = {
        "name": "qwen3:8b",
        "size": "4.7GB",
        "parameter_size": "8B",
    }
    return mock_manager


# ============================================================================
# Mock MCP Server Fixtures
# ============================================================================


@pytest.fixture
def mock_mcp_filesystem():
    """Create a mock MCP filesystem server.

    Returns:
        Mock: Mock filesystem MCP server
    """
    mock_fs = Mock()
    mock_fs.list_directory.return_value = ["file1.txt", "file2.txt", "subdir/"]
    mock_fs.read_file.return_value = "Mock file content"
    mock_fs.write_file.return_value = True
    mock_fs.search_files.return_value = ["match1.txt", "match2.py"]
    return mock_fs


@pytest.fixture
def mock_mcp_web_search():
    """Create a mock MCP web search server.

    Returns:
        Mock: Mock web search MCP server
    """
    mock_search = Mock()
    mock_search.search.return_value = {
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "snippet": "This is a test result snippet.",
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "snippet": "Another test result snippet.",
            },
        ],
        "total": 2,
    }
    return mock_search


# ============================================================================
# Test Document Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for RAG testing.

    Returns:
        List[Dict]: Sample documents with content and metadata
    """
    return [
        {
            "page_content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"source": "python_intro.txt", "page": 1},
        },
        {
            "page_content": "LangChain is a framework for developing applications powered by language models.",
            "metadata": {"source": "langchain_intro.txt", "page": 1},
        },
        {
            "page_content": "Ollama allows you to run large language models locally on your machine.",
            "metadata": {"source": "ollama_intro.txt", "page": 1},
        },
        {
            "page_content": "Vector stores enable semantic search by converting text into embeddings.",
            "metadata": {"source": "vector_stores.txt", "page": 1},
        },
    ]


@pytest.fixture
def sample_pdf_content() -> str:
    """Create sample PDF-like content for testing.

    Returns:
        str: Sample text content
    """
    return """
    Introduction to Local AI Development

    This document covers the fundamentals of developing AI applications
    using local language models. Local development offers several advantages:

    1. Privacy: Your data never leaves your machine
    2. Cost: No API fees or usage limits
    3. Control: Complete control over model behavior
    4. Speed: No network latency for inference

    Prerequisites
    - Python 3.10 or higher
    - Ollama installed and running
    - At least 8GB RAM for smaller models

    Getting Started
    The first step is to install Ollama and pull a model.
    We recommend starting with qwen3:8b for general use.
    """


# ============================================================================
# Mock Embedding Fixtures
# ============================================================================


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for vector store testing.

    Returns:
        Mock: Mock embeddings model
    """
    mock_embed = Mock()

    def embed_documents(texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        return [[0.1 * i] * 384 for i in range(len(texts))]

    def embed_query(text: str) -> List[float]:
        """Return fake embedding for query."""
        return [0.5] * 384

    mock_embed.embed_documents = embed_documents
    mock_embed.embed_query = embed_query

    return mock_embed


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def capture_logs(caplog):
    """Configure log capturing with appropriate level.

    Args:
        caplog: pytest log capture fixture

    Returns:
        caplog: Configured log capture
    """
    import logging
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def mock_env_vars() -> Generator[Dict[str, str], None, None]:
    """Mock environment variables for testing.

    Yields:
        Dict: Environment variables to use
    """
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "MCP_FILESYSTEM_PORT": "8001",
        "MCP_WEB_SEARCH_PORT": "8002",
        "LANGCHAIN_TRACING_V2": "false",  # Disable tracing in tests
    }

    os.environ.update(test_env)

    try:
        yield test_env
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def integration_markers():
    """Provide pytest markers for integration tests.

    Returns:
        Dict: Marker definitions
    """
    return {
        "integration": "Requires Ollama server running",
        "slow": "Test takes significant time to run",
        "mcp": "Requires MCP servers running",
        "rag": "Tests RAG functionality",
    }


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_agent_executor(response: str = "Mock agent response"):
    """Create a mock agent executor.

    Args:
        response: Response to return from invoke

    Returns:
        Mock: Mock AgentExecutor
    """
    mock_executor = Mock()
    mock_executor.invoke.return_value = {
        "input": "test input",
        "output": response,
    }
    return mock_executor


def create_mock_retriever(documents: List[Dict[str, Any]]):
    """Create a mock retriever for RAG testing.

    Args:
        documents: Documents to return from retrieval

    Returns:
        Mock: Mock retriever
    """
    from langchain_core.documents import Document

    mock_retriever = Mock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in documents
    ]
    return mock_retriever


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Automatically clean up checkpoint files after each test.

    Yields:
        None
    """
    yield

    # Clean up any checkpoint databases in the current directory
    for file in Path.cwd().glob("checkpoints_*.db"):
        try:
            file.unlink()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration after each test.

    Yields:
        None
    """
    import logging

    # Store original handlers
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    yield

    # Restore original configuration
    root.handlers = original_handlers
    root.level = original_level
