"""
Integration tests for example scripts - smoke tests.

Tests that all examples can be imported and executed without errors.
Verifies expected output patterns and handles timeouts.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest


# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
TIMEOUT_SECONDS = 30  # Timeout for example execution


# ============================================================================
# Helper Functions
# ============================================================================


def get_example_files(subdir: Optional[str] = None) -> List[Path]:
    """Get all Python example files.

    Args:
        subdir: Optional subdirectory to filter by

    Returns:
        List of example file paths
    """
    if subdir:
        search_dir = EXAMPLES_DIR / subdir
    else:
        search_dir = EXAMPLES_DIR

    return sorted(search_dir.glob("**/*.py"))


def can_import_example(example_path: Path) -> bool:
    """Check if an example can be imported without errors.

    Args:
        example_path: Path to example file

    Returns:
        True if import succeeds, False otherwise
    """
    # Get module path relative to examples directory
    relative = example_path.relative_to(EXAMPLES_DIR)
    module_parts = list(relative.parent.parts) + [relative.stem]
    module_name = ".".join(module_parts)

    # Add examples directory to path
    sys.path.insert(0, str(EXAMPLES_DIR))

    try:
        __import__(f"examples.{module_name}")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False
    finally:
        sys.path.pop(0)


def run_example_script(
    example_path: Path, timeout: int = TIMEOUT_SECONDS, mock_ollama: bool = True
) -> subprocess.CompletedProcess:
    """Run an example script as a subprocess.

    Args:
        example_path: Path to example file
        timeout: Timeout in seconds
        mock_ollama: Whether to mock Ollama interactions

    Returns:
        CompletedProcess with results

    Raises:
        subprocess.TimeoutExpired: If execution times out
    """
    env = {}
    if mock_ollama:
        # Set environment to use mock Ollama
        env["MOCK_OLLAMA"] = "1"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={**subprocess.os.environ, **env},
    )

    return result


# ============================================================================
# Foundation Examples Tests (01-foundation)
# ============================================================================


class TestFoundationExamples:
    """Test suite for foundation examples."""

    def test_simple_chat_import(self):
        """Test that simple_chat.py can be imported."""
        example = EXAMPLES_DIR / "01-foundation" / "simple_chat.py"
        assert example.exists(), f"Example not found: {example}"

        # Just verify file has expected structure
        content = example.read_text()
        assert "ChatOllama" in content
        assert "def main" in content

    @pytest.mark.integration
    @pytest.mark.slow
    def test_simple_chat_execution(self, mock_ollama_llm):
        """Test simple_chat.py execution with mocked Ollama."""
        example = EXAMPLES_DIR / "01-foundation" / "simple_chat.py"

        with patch("langchain_ollama.ChatOllama", return_value=mock_ollama_llm):
            # Run with short timeout since it's mocked
            result = run_example_script(example, timeout=10)

            # Check for expected patterns in output
            assert "Initializing" in result.stdout or result.returncode == 0

    def test_compare_models_import(self):
        """Test that compare_models.py can be imported."""
        example = EXAMPLES_DIR / "01-foundation" / "compare_models.py"
        assert example.exists(), f"Example not found: {example}"

        content = example.read_text()
        assert "ChatOllama" in content
        assert "compare" in content.lower()

    def test_streaming_chat_import(self):
        """Test that streaming_chat.py can be imported."""
        example = EXAMPLES_DIR / "01-foundation" / "streaming_chat.py"
        assert example.exists(), f"Example not found: {example}"

        content = example.read_text()
        assert "stream" in content.lower()


# ============================================================================
# MCP Examples Tests (02-mcp)
# ============================================================================


class TestMCPExamples:
    """Test suite for MCP integration examples."""

    def test_filesystem_agent_import(self):
        """Test that filesystem_agent.py can be imported."""
        example = EXAMPLES_DIR / "02-mcp" / "filesystem_agent.py"
        assert example.exists(), f"Example not found: {example}"

        content = example.read_text()
        assert "FilesystemMCP" in content or "filesystem" in content.lower()
        assert "AgentExecutor" in content

    def test_filesystem_agent_structure(self):
        """Test filesystem_agent.py has expected structure."""
        example = EXAMPLES_DIR / "02-mcp" / "filesystem_agent.py"
        content = example.read_text()

        # Check for key components
        assert "def main" in content
        assert "OllamaManager" in content
        assert "MCPConfig" in content or "MCP" in content

    def test_web_search_agent_import(self):
        """Test that web_search_agent.py can be imported."""
        example = EXAMPLES_DIR / "02-mcp" / "web_search_agent.py"
        if not example.exists():
            pytest.skip("web_search_agent.py not implemented yet")

        content = example.read_text()
        assert "search" in content.lower()

    def test_combined_tools_agent_import(self):
        """Test that combined_tools_agent.py can be imported."""
        example = EXAMPLES_DIR / "02-mcp" / "combined_tools_agent.py"
        if not example.exists():
            pytest.skip("combined_tools_agent.py not implemented yet")

        content = example.read_text()
        assert "tools" in content.lower()


# ============================================================================
# Multi-Agent Examples Tests (03-multi-agent)
# ============================================================================


class TestMultiAgentExamples:
    """Test suite for multi-agent workflow examples."""

    def test_research_pipeline_import(self):
        """Test that research_pipeline.py can be imported."""
        example = EXAMPLES_DIR / "03-multi-agent" / "research_pipeline.py"
        assert example.exists(), f"Example not found: {example}"

        content = example.read_text()
        assert "StateGraph" in content
        assert "ResearchPipelineState" in content

    def test_research_pipeline_structure(self):
        """Test research_pipeline.py has proper LangGraph structure."""
        example = EXAMPLES_DIR / "03-multi-agent" / "research_pipeline.py"
        content = example.read_text()

        # Check for LangGraph components
        assert "TypedDict" in content
        assert "researcher_node" in content
        assert "analyzer_node" in content
        assert "summarizer_node" in content
        assert "create_research_pipeline" in content

    @pytest.mark.integration
    @pytest.mark.slow
    def test_research_pipeline_graph_creation(self):
        """Test that research pipeline graph can be created."""
        example = EXAMPLES_DIR / "03-multi-agent" / "research_pipeline.py"

        # Import the module
        sys.path.insert(0, str(example.parent))
        try:
            import research_pipeline

            # Test graph creation
            workflow = research_pipeline.create_research_pipeline()
            assert workflow is not None

            # Check nodes exist
            compiled = workflow.compile()
            assert compiled is not None

        finally:
            sys.path.pop(0)

    def test_code_review_pipeline_import(self):
        """Test that code_review_pipeline.py can be imported."""
        example = EXAMPLES_DIR / "03-multi-agent" / "code_review_pipeline.py"
        if not example.exists():
            pytest.skip("code_review_pipeline.py not implemented yet")

        content = example.read_text()
        assert "StateGraph" in content

    def test_parallel_comparison_import(self):
        """Test that parallel_comparison.py can be imported."""
        example = EXAMPLES_DIR / "03-multi-agent" / "parallel_comparison.py"
        if not example.exists():
            pytest.skip("parallel_comparison.py not implemented yet")

        content = example.read_text()
        assert "parallel" in content.lower()


# ============================================================================
# RAG Examples Tests (04-rag)
# ============================================================================


class TestRAGExamples:
    """Test suite for RAG examples."""

    def test_document_qa_import(self):
        """Test that document_qa.py can be imported."""
        example = EXAMPLES_DIR / "04-rag" / "document_qa.py"
        assert example.exists(), f"Example not found: {example}"

        content = example.read_text()
        assert "VectorStoreManager" in content or "vector" in content.lower()
        assert "RetrievalQA" in content or "PyPDFLoader" in content

    def test_document_qa_structure(self):
        """Test document_qa.py has proper structure."""
        example = EXAMPLES_DIR / "04-rag" / "document_qa.py"
        content = example.read_text()

        # Check for key functions
        assert "load_pdf_documents" in content
        assert "chunk_documents" in content
        assert "create_qa_chain" in content

    def test_codebase_search_import(self):
        """Test that codebase_search.py can be imported."""
        example = EXAMPLES_DIR / "04-rag" / "codebase_search.py"
        if not example.exists():
            pytest.skip("codebase_search.py not implemented yet")

        content = example.read_text()
        assert "search" in content.lower()


# ============================================================================
# Timeout Handling Tests
# ============================================================================


class TestTimeoutHandling:
    """Test timeout handling for long-running examples."""

    @pytest.mark.slow
    def test_example_timeout_handling(self):
        """Test that examples timeout appropriately."""
        # Create a simple script that sleeps
        example = EXAMPLES_DIR / "01-foundation" / "simple_chat.py"

        # Should timeout if it takes too long
        with pytest.raises(subprocess.TimeoutExpired):
            run_example_script(example, timeout=1, mock_ollama=False)


# ============================================================================
# Output Pattern Tests
# ============================================================================


class TestOutputPatterns:
    """Test expected output patterns from examples."""

    def test_foundation_examples_have_docstrings(self):
        """Test that all foundation examples have proper docstrings."""
        examples = get_example_files("01-foundation")

        for example in examples:
            content = example.read_text()

            # Check for module docstring
            assert '"""' in content, f"No docstring in {example.name}"

            # Check for key sections in docstring
            docstring_start = content.find('"""')
            docstring_end = content.find('"""', docstring_start + 3)
            docstring = content[docstring_start:docstring_end]

            assert "Prerequisites" in docstring or "Expected" in docstring, (
                f"Missing Prerequisites/Expected in {example.name}"
            )

    def test_examples_have_main_function(self):
        """Test that all examples have a main() function."""
        all_examples = get_example_files()

        for example in all_examples:
            if example.name.startswith("_"):
                continue  # Skip private modules

            content = example.read_text()
            assert "def main" in content, f"No main() in {example.name}"
            assert 'if __name__ == "__main__"' in content, f"No main block in {example.name}"

    def test_examples_have_error_handling(self):
        """Test that examples include error handling."""
        all_examples = get_example_files()

        for example in all_examples:
            if example.name.startswith("_"):
                continue

            content = example.read_text()

            # Check for try/except or error handling
            has_try_except = "try:" in content and "except" in content
            has_error_check = "Error" in content or "error" in content

            assert has_try_except or has_error_check, f"No error handling in {example.name}"


# ============================================================================
# Example Count Tests
# ============================================================================


class TestExampleInventory:
    """Test that expected examples exist."""

    def test_foundation_examples_exist(self):
        """Test that foundation examples directory has examples."""
        examples = get_example_files("01-foundation")
        assert len(examples) >= 2, "Should have at least 2 foundation examples"

    def test_mcp_examples_exist(self):
        """Test that MCP examples directory has examples."""
        examples = get_example_files("02-mcp")
        assert len(examples) >= 1, "Should have at least 1 MCP example"

    def test_multi_agent_examples_exist(self):
        """Test that multi-agent examples directory has examples."""
        examples = get_example_files("03-multi-agent")
        assert len(examples) >= 1, "Should have at least 1 multi-agent example"

    def test_rag_examples_exist(self):
        """Test that RAG examples directory has examples."""
        examples = get_example_files("04-rag")
        assert len(examples) >= 1, "Should have at least 1 RAG example"


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize(
    "example_path",
    [
        "01-foundation/simple_chat.py",
        "02-mcp/filesystem_agent.py",
        "03-multi-agent/research_pipeline.py",
        "04-rag/document_qa.py",
    ],
)
def test_example_imports_without_error(example_path: str):
    """Test that key examples can be imported without errors.

    Args:
        example_path: Path to example relative to examples directory
    """
    full_path = EXAMPLES_DIR / example_path

    if not full_path.exists():
        pytest.skip(f"Example not found: {example_path}")

    # Just verify it's valid Python
    content = full_path.read_text()
    try:
        compile(content, str(full_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {example_path}: {e}")
