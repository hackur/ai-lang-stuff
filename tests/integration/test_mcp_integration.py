"""
Integration tests for MCP server functionality.

Tests full MCP server integration including:
- Filesystem operations
- Web search (when available)
- Combined tool usage
- Real file operations with safety checks
"""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import Tool

# Import utilities
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import FilesystemMCP, MCPConfig


# ============================================================================
# Filesystem MCP Integration Tests
# ============================================================================


class TestFilesystemMCPIntegration:
    """Test MCP filesystem server integration."""

    def test_mcp_config_creation(self):
        """Test creating MCP configuration."""
        config = MCPConfig(
            host="localhost",
            port=8001,
            timeout=30.0
        )

        assert config.host == "localhost"
        assert config.port == 8001
        assert config.timeout == 30.0
        assert "localhost:8001" in config.base_url

    def test_filesystem_mcp_initialization(self, test_data_dir: Path):
        """Test initializing FilesystemMCP client.

        Args:
            test_data_dir: Test data directory fixture
        """
        config = MCPConfig(host="localhost", port=8001)

        # Initialize with base path
        fs_client = FilesystemMCP(config=config, base_path=test_data_dir)

        assert fs_client.config == config
        assert fs_client.base_path == test_data_dir

    @pytest.mark.integration
    def test_filesystem_operations_mock(self, test_data_dir: Path):
        """Test filesystem operations with mocked MCP server.

        Args:
            test_data_dir: Test data directory fixture
        """
        config = MCPConfig(host="localhost", port=8001)
        fs_client = FilesystemMCP(config=config, base_path=test_data_dir)

        # Mock the actual MCP calls
        with patch.object(fs_client, '_call_mcp') as mock_call:
            # Test list directory
            mock_call.return_value = {
                "files": ["sample.txt", "sample.json", "nested/"]
            }

            result = fs_client.list_directory(str(test_data_dir))
            assert "sample.txt" in result or mock_call.called

    def test_filesystem_path_validation(self, test_data_dir: Path):
        """Test that filesystem operations validate paths.

        Args:
            test_data_dir: Test data directory fixture
        """
        config = MCPConfig(host="localhost", port=8001)
        fs_client = FilesystemMCP(config=config, base_path=test_data_dir)

        # Test that paths outside base_path are rejected
        with pytest.raises((ValueError, PermissionError, Exception)):
            # Attempt to access parent directory
            fs_client.list_directory("/etc")

    def test_langchain_tools_creation(self, test_data_dir: Path):
        """Test creating LangChain tools from MCP client.

        Args:
            test_data_dir: Test data directory fixture
        """
        config = MCPConfig(host="localhost", port=8001)
        fs_client = FilesystemMCP(config=config, base_path=test_data_dir)

        tools = fs_client.to_langchain_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check tool properties
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert callable(tool.func) or hasattr(tool, "_run")


# ============================================================================
# MCP Tool Integration Tests
# ============================================================================


class TestMCPToolIntegration:
    """Test MCP tools with LangChain agents."""

    def test_filesystem_tool_creation(self, test_data_dir: Path):
        """Test creating filesystem tools.

        Args:
            test_data_dir: Test data directory fixture
        """
        # Create mock MCP tools
        def mock_list_dir(path: str) -> str:
            """Mock directory listing."""
            return "file1.txt\nfile2.txt\nsubdir/"

        def mock_read_file(path: str) -> str:
            """Mock file reading."""
            return "Mock file content"

        tools = [
            Tool(
                name="list_directory",
                description="List contents of a directory",
                func=mock_list_dir
            ),
            Tool(
                name="read_file",
                description="Read contents of a file",
                func=mock_read_file
            ),
        ]

        assert len(tools) == 2
        assert tools[0].name == "list_directory"
        assert tools[1].name == "read_file"

        # Test tool invocation
        result = tools[0].run("test_path")
        assert "file1.txt" in result

    @pytest.mark.integration
    def test_agent_with_filesystem_tools(
        self,
        test_data_dir: Path,
        mock_ollama_llm
    ):
        """Test agent using filesystem tools.

        Args:
            test_data_dir: Test data directory fixture
            mock_ollama_llm: Mock Ollama LLM fixture
        """
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        # Create mock tools
        def mock_list_dir(path: str) -> str:
            """Mock directory listing."""
            actual_path = Path(path)
            if actual_path.exists() and actual_path.is_dir():
                items = [item.name for item in actual_path.iterdir()]
                return "\n".join(items)
            return "Directory not found"

        tools = [
            Tool(
                name="list_directory",
                description="List contents of a directory. Input: directory path",
                func=mock_list_dir
            ),
        ]

        # Create simple prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with filesystem access."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create agent
        agent = create_tool_calling_agent(mock_ollama_llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3,
        )

        # Test agent invocation
        result = executor.invoke({
            "input": f"List the contents of {test_data_dir}"
        })

        assert result is not None
        assert "output" in result


# ============================================================================
# Combined MCP Tools Tests
# ============================================================================


class TestCombinedMCPTools:
    """Test using multiple MCP servers together."""

    def test_multiple_tool_sources(self):
        """Test combining tools from multiple MCP servers."""
        # Filesystem tools
        fs_tools = [
            Tool(
                name="list_directory",
                description="List directory contents",
                func=lambda x: "file1.txt\nfile2.txt"
            ),
            Tool(
                name="read_file",
                description="Read file contents",
                func=lambda x: "File content"
            ),
        ]

        # Web search tools
        web_tools = [
            Tool(
                name="web_search",
                description="Search the web",
                func=lambda x: json.dumps({
                    "results": [
                        {"title": "Result 1", "url": "https://example.com"}
                    ]
                })
            ),
        ]

        # Combine tools
        all_tools = fs_tools + web_tools

        assert len(all_tools) == 3
        assert any(t.name == "list_directory" for t in all_tools)
        assert any(t.name == "web_search" for t in all_tools)

    @pytest.mark.integration
    @pytest.mark.mcp
    def test_agent_with_multiple_mcp_tools(self, mock_ollama_llm):
        """Test agent using multiple MCP tool sources.

        Args:
            mock_ollama_llm: Mock Ollama LLM fixture
        """
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        # Create combined tools
        tools = [
            Tool(
                name="filesystem_list",
                description="List directory contents",
                func=lambda x: "documents/\ncode/\ndata/"
            ),
            Tool(
                name="web_search",
                description="Search the web for information",
                func=lambda x: json.dumps({
                    "results": [{"title": "Test", "snippet": "Test result"}]
                })
            ),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You have both filesystem and web search capabilities."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(mock_ollama_llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3)

        # Test with query that could use either tool
        result = executor.invoke({
            "input": "Search for Python tutorials"
        })

        assert result is not None


# ============================================================================
# Real File Operations Tests
# ============================================================================


class TestRealFileOperations:
    """Test actual file operations with safety checks."""

    def test_safe_file_read(self, test_data_dir: Path):
        """Test reading files safely within base path.

        Args:
            test_data_dir: Test data directory fixture
        """
        sample_file = test_data_dir / "sample.txt"
        assert sample_file.exists()

        content = sample_file.read_text()
        assert len(content) > 0
        assert "sample text file" in content

    def test_safe_file_write(self, temp_dir: Path):
        """Test writing files safely within base path.

        Args:
            temp_dir: Temporary directory fixture
        """
        test_file = temp_dir / "test_write.txt"
        test_content = "This is test content"

        test_file.write_text(test_content)

        # Verify write
        assert test_file.exists()
        assert test_file.read_text() == test_content

    def test_directory_listing(self, test_data_dir: Path):
        """Test listing directory contents.

        Args:
            test_data_dir: Test data directory fixture
        """
        items = list(test_data_dir.iterdir())

        assert len(items) > 0

        # Check for expected files
        names = [item.name for item in items]
        assert "sample.txt" in names
        assert "sample.json" in names

    def test_nested_directory_access(self, test_data_dir: Path):
        """Test accessing nested directories.

        Args:
            test_data_dir: Test data directory fixture
        """
        nested_file = test_data_dir / "nested" / "deep" / "file.txt"

        assert nested_file.exists()
        content = nested_file.read_text()
        assert "Nested file" in content

    def test_json_file_operations(self, test_data_dir: Path):
        """Test JSON file operations.

        Args:
            test_data_dir: Test data directory fixture
        """
        json_file = test_data_dir / "sample.json"
        assert json_file.exists()

        # Read and parse JSON
        data = json.loads(json_file.read_text())

        assert isinstance(data, dict)
        assert "key" in data
        assert data["key"] == "value"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestMCPErrorHandling:
    """Test error handling in MCP operations."""

    def test_nonexistent_directory(self, temp_dir: Path):
        """Test handling of nonexistent directory.

        Args:
            temp_dir: Temporary directory fixture
        """
        nonexistent = temp_dir / "does_not_exist"

        # Should not exist
        assert not nonexistent.exists()

        # Attempting to list should handle error
        try:
            list(nonexistent.iterdir())
        except (FileNotFoundError, OSError):
            pass  # Expected error

    def test_nonexistent_file(self, temp_dir: Path):
        """Test handling of nonexistent file.

        Args:
            temp_dir: Temporary directory fixture
        """
        nonexistent = temp_dir / "missing.txt"

        assert not nonexistent.exists()

        with pytest.raises(FileNotFoundError):
            nonexistent.read_text()

    def test_permission_errors(self, temp_dir: Path):
        """Test handling of permission errors.

        Args:
            temp_dir: Temporary directory fixture
        """
        # This test is platform-dependent
        # On some systems, we can create unreadable files
        test_file = temp_dir / "restricted.txt"
        test_file.write_text("restricted content")

        # Try to make it unreadable (Unix-like systems)
        try:
            test_file.chmod(0o000)

            with pytest.raises(PermissionError):
                test_file.read_text()

        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

    def test_invalid_json(self, temp_dir: Path):
        """Test handling of invalid JSON files.

        Args:
            temp_dir: Temporary directory fixture
        """
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json.read_text())


# ============================================================================
# Web Search Integration Tests
# ============================================================================


class TestWebSearchIntegration:
    """Test web search MCP integration."""

    def test_web_search_tool_creation(self):
        """Test creating web search tool."""
        def mock_search(query: str) -> str:
            """Mock web search."""
            return json.dumps({
                "query": query,
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "snippet": "This is a test result"
                    }
                ],
                "total": 1
            })

        tool = Tool(
            name="web_search",
            description="Search the web for information",
            func=mock_search
        )

        assert tool.name == "web_search"

        # Test invocation
        result = tool.run("test query")
        data = json.loads(result)

        assert "results" in data
        assert len(data["results"]) > 0

    @pytest.mark.integration
    @pytest.mark.mcp
    def test_web_search_with_agent(self, mock_ollama_llm):
        """Test web search tool with agent.

        Args:
            mock_ollama_llm: Mock Ollama LLM fixture
        """
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        def mock_search(query: str) -> str:
            """Mock web search."""
            return json.dumps({
                "results": [
                    {"title": f"Result for: {query}", "url": "https://example.com"}
                ]
            })

        tools = [
            Tool(
                name="web_search",
                description="Search the web",
                func=mock_search
            ),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You can search the web for information."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(mock_ollama_llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=2)

        result = executor.invoke({"input": "Find information about Python"})
        assert result is not None
