"""
Comprehensive test suite for utils/mcp_client.py.

Tests MCP client connection management, retry logic, error handling,
and specific implementations for filesystem and web search operations.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.tools import Tool

from utils.mcp_client import (
    MCPClient,
    MCPConfig,
    MCPConnectionError,
    MCPError,
    MCPToolError,
    FilesystemMCP,
    SearchResult,
    WebSearchMCP,
    create_filesystem_client,
    create_websearch_client,
)


# Fixtures


@pytest.fixture
def mcp_config() -> MCPConfig:
    """Create default MCP configuration for testing."""
    return MCPConfig(host="localhost", port=8000, timeout=10.0, max_retries=3)


@pytest.fixture
def filesystem_config() -> MCPConfig:
    """Create MCP configuration for filesystem client."""
    return MCPConfig(host="localhost", port=8001)


@pytest.fixture
def websearch_config() -> MCPConfig:
    """Create MCP configuration for web search client."""
    return MCPConfig(host="localhost", port=8002)


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Create mock httpx AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def temp_base_path(tmp_path: Path) -> Path:
    """Create temporary base path for filesystem testing."""
    return tmp_path


# MCPConfig Tests


class TestMCPConfig:
    """Test MCP configuration model."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = MCPConfig()
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = MCPConfig(
            host="192.168.1.1",
            port=9000,
            timeout=60.0,
            max_retries=5,
            base_delay=2.0,
            max_delay=20.0,
        )
        assert config.host == "192.168.1.1"
        assert config.port == 9000
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 20.0

    def test_base_url_property(self, mcp_config: MCPConfig):
        """Test base_url property construction."""
        assert mcp_config.base_url == "http://localhost:8000"

    def test_config_immutability(self, mcp_config: MCPConfig):
        """Test that config is frozen/immutable."""
        with pytest.raises(Exception):
            mcp_config.host = "newhost"


# MCPClient Base Class Tests


class ConcreteMCPClient(MCPClient):
    """Concrete implementation of MCPClient for testing."""

    def to_langchain_tools(self):
        """Minimal implementation for testing."""
        return []


class TestMCPClient:
    """Test base MCPClient functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, mcp_config: MCPConfig):
        """Test client initialization with config."""
        client = ConcreteMCPClient(config=mcp_config)
        assert client.config == mcp_config
        assert client._client is None
        assert client._owned_client is True
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_client_lazy_creation(self, mcp_config: MCPConfig):
        """Test lazy HTTP client creation."""
        client = ConcreteMCPClient(config=mcp_config)
        http_client = client.client
        assert isinstance(http_client, httpx.AsyncClient)
        assert client._client is not None

    @pytest.mark.asyncio
    async def test_client_with_provided_httpx_client(
        self, mcp_config: MCPConfig, mock_httpx_client: AsyncMock
    ):
        """Test initialization with pre-configured httpx client."""
        client = ConcreteMCPClient(config=mcp_config, client=mock_httpx_client)
        assert client._client == mock_httpx_client
        assert client._owned_client is False

    @pytest.mark.asyncio
    async def test_connect_success(self, mcp_config: MCPConfig):
        """Test successful connection to MCP server."""
        with patch.object(ConcreteMCPClient, "_health_check") as mock_health:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_health.return_value = mock_response

            client = ConcreteMCPClient(config=mcp_config)
            await client.connect()

            assert client._connected is True
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mcp_config: MCPConfig):
        """Test connect when already connected skips health check."""
        with patch.object(ConcreteMCPClient, "_health_check") as mock_health:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_health.return_value = mock_response

            client = ConcreteMCPClient(config=mcp_config)
            await client.connect()
            await client.connect()  # Second call should be no-op

            mock_health.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    async def test_connect_health_check_fails(self, mcp_config: MCPConfig):
        """Test connection failure when health check returns non-200."""
        with patch.object(ConcreteMCPClient, "_health_check") as mock_health:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_health.return_value = mock_response

            client = ConcreteMCPClient(config=mcp_config)
            with pytest.raises(MCPConnectionError, match="Health check failed"):
                await client.connect()

            assert client._connected is False

    @pytest.mark.asyncio
    async def test_connect_network_error(self, mcp_config: MCPConfig):
        """Test connection failure due to network error."""
        with patch.object(
            ConcreteMCPClient, "_health_check"
        ) as mock_health:
            mock_health.side_effect = httpx.RequestError("Connection refused")

            client = ConcreteMCPClient(config=mcp_config)
            with pytest.raises(MCPConnectionError, match="Failed to connect"):
                await client.connect()

            assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, mcp_config: MCPConfig, mock_httpx_client: AsyncMock):
        """Test disconnection closes owned client."""
        client = ConcreteMCPClient(config=mcp_config, client=mock_httpx_client)
        client._owned_client = True
        client._connected = True

        await client.disconnect()

        assert client._connected is False
        assert client._client is None
        mock_httpx_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_not_owned_client(
        self, mcp_config: MCPConfig, mock_httpx_client: AsyncMock
    ):
        """Test disconnection does not close non-owned client."""
        client = ConcreteMCPClient(config=mcp_config, client=mock_httpx_client)
        client._owned_client = False
        client._connected = True

        await client.disconnect()

        assert client._connected is False
        mock_httpx_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mcp_config: MCPConfig):
        """Test successful tool call."""
        with patch.object(ConcreteMCPClient, "connect") as mock_connect, \
             patch.object(ConcreteMCPClient, "client") as mock_client:

            mock_response = MagicMock()
            mock_response.json.return_value = {"result": {"data": "test_data"}}
            mock_response.raise_for_status = MagicMock()

            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = True

            result = await client.call_tool("test_tool", {"arg": "value"})

            assert result == {"data": "test_data"}
            mock_http_client.post.assert_called_once_with(
                "/tools/call",
                json={"name": "test_tool", "arguments": {"arg": "value"}},
            )

    @pytest.mark.asyncio
    async def test_call_tool_auto_connect(self, mcp_config: MCPConfig):
        """Test tool call automatically connects if not connected."""
        with patch.object(ConcreteMCPClient, "connect") as mock_connect, \
             patch.object(ConcreteMCPClient, "client") as mock_client:

            mock_response = MagicMock()
            mock_response.json.return_value = {"result": {}}
            mock_response.raise_for_status = MagicMock()

            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = False

            await client.call_tool("test_tool", {})

            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self, mcp_config: MCPConfig):
        """Test tool call with error in response."""
        with patch.object(ConcreteMCPClient, "client") as mock_client:

            mock_response = MagicMock()
            mock_response.json.return_value = {"error": "Tool execution failed"}
            mock_response.raise_for_status = MagicMock()

            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = True

            with pytest.raises(MCPToolError, match="Tool execution failed"):
                await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_http_error(self, mcp_config: MCPConfig):
        """Test tool call with HTTP status error."""
        with patch.object(ConcreteMCPClient, "client") as mock_client:

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=mock_response
            )

            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = True

            with pytest.raises(MCPToolError, match="Tool call failed with status"):
                await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_request_error(self, mcp_config: MCPConfig):
        """Test tool call with request error."""
        with patch.object(ConcreteMCPClient, "client") as mock_client:

            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(
                side_effect=httpx.RequestError("Connection error")
            )
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = True

            with pytest.raises(MCPToolError, match="Tool call request failed"):
                await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mcp_config: MCPConfig):
        """Test async context manager protocol."""
        with patch.object(ConcreteMCPClient, "connect") as mock_connect, \
             patch.object(ConcreteMCPClient, "disconnect") as mock_disconnect:

            async with ConcreteMCPClient(config=mcp_config) as client:
                assert isinstance(client, ConcreteMCPClient)

            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()

    def test_sync_context_manager_not_supported(self, mcp_config: MCPConfig):
        """Test sync context manager raises NotImplementedError."""
        client = ConcreteMCPClient(config=mcp_config)
        with pytest.raises(NotImplementedError, match="Use async context manager"):
            with client:
                pass


# FilesystemMCP Tests


class TestFilesystemMCP:
    """Test FilesystemMCP client functionality."""

    @pytest.mark.asyncio
    async def test_filesystem_initialization(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test filesystem client initialization."""
        client = FilesystemMCP(config=filesystem_config, base_path=temp_base_path)
        assert client.config == filesystem_config
        assert client.base_path == temp_base_path

    @pytest.mark.asyncio
    async def test_filesystem_initialization_no_base_path(
        self, filesystem_config: MCPConfig
    ):
        """Test filesystem client initialization without base path."""
        client = FilesystemMCP(config=filesystem_config)
        assert client.base_path is None

    def test_validate_path_no_base_path(self, filesystem_config: MCPConfig):
        """Test path validation without base path constraint."""
        client = FilesystemMCP(config=filesystem_config)
        path = client._validate_path("/tmp/test.txt")
        assert isinstance(path, Path)
        assert path.is_absolute()

    def test_validate_path_with_base_path(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test path validation with base path constraint."""
        client = FilesystemMCP(config=filesystem_config, base_path=temp_base_path)
        test_file = temp_base_path / "test.txt"
        validated = client._validate_path(test_file)
        assert validated == test_file.resolve()

    def test_validate_path_traversal_attack(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test path validation prevents path traversal attacks."""
        client = FilesystemMCP(config=filesystem_config, base_path=temp_base_path)
        malicious_path = temp_base_path / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="outside allowed base path"):
            client._validate_path(malicious_path)

    @pytest.mark.asyncio
    async def test_read_file_success(self, filesystem_config: MCPConfig):
        """Test successful file read operation."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"content": "file contents"}

            client = FilesystemMCP(config=filesystem_config)
            content = await client.read_file("/tmp/test.txt")

            assert content == "file contents"
            mock_call_tool.assert_called_once()
            args = mock_call_tool.call_args[0]
            assert args[0] == "read_file"
            assert "path" in args[1]

    @pytest.mark.asyncio
    async def test_read_file_validates_path(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test read_file validates path before calling tool."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"content": "test"}

            client = FilesystemMCP(config=filesystem_config, base_path=temp_base_path)
            malicious_path = "../../../etc/passwd"

            with pytest.raises(ValueError, match="outside allowed base path"):
                await client.read_file(malicious_path)

    @pytest.mark.asyncio
    async def test_write_file_success(self, filesystem_config: MCPConfig):
        """Test successful file write operation."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"success": True}

            client = FilesystemMCP(config=filesystem_config)
            result = await client.write_file("/tmp/test.txt", "content")

            assert result is True
            mock_call_tool.assert_called_once()
            args = mock_call_tool.call_args[0]
            assert args[0] == "write_file"
            assert args[1]["content"] == "content"

    @pytest.mark.asyncio
    async def test_write_file_failure(self, filesystem_config: MCPConfig):
        """Test file write operation failure."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"success": False}

            client = FilesystemMCP(config=filesystem_config)
            result = await client.write_file("/tmp/test.txt", "content")

            assert result is False

    @pytest.mark.asyncio
    async def test_list_directory_success(self, filesystem_config: MCPConfig):
        """Test successful directory listing."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "entries": [
                    {"name": "file1.txt", "type": "file", "size": 100},
                    {"name": "dir1", "type": "directory"},
                ]
            }

            client = FilesystemMCP(config=filesystem_config)
            entries = await client.list_directory("/tmp")

            assert len(entries) == 2
            assert entries[0]["name"] == "file1.txt"
            assert entries[1]["name"] == "dir1"

    @pytest.mark.asyncio
    async def test_list_directory_empty(self, filesystem_config: MCPConfig):
        """Test directory listing with no entries."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"entries": []}

            client = FilesystemMCP(config=filesystem_config)
            entries = await client.list_directory("/tmp/empty")

            assert entries == []

    @pytest.mark.asyncio
    async def test_search_files_with_pattern(self, filesystem_config: MCPConfig):
        """Test file search with glob pattern."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "files": ["/tmp/file1.py", "/tmp/dir/file2.py"]
            }

            client = FilesystemMCP(config=filesystem_config)
            files = await client.search_files("*.py", "/tmp")

            assert len(files) == 2
            assert "/tmp/file1.py" in files
            mock_call_tool.assert_called_once()
            args = mock_call_tool.call_args[0]
            assert args[0] == "search_files"
            assert args[1]["pattern"] == "*.py"

    @pytest.mark.asyncio
    async def test_search_files_default_path(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test file search uses base_path as default."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"files": []}

            client = FilesystemMCP(config=filesystem_config, base_path=temp_base_path)
            await client.search_files("*.txt")

            args = mock_call_tool.call_args[0]
            assert str(temp_base_path.resolve()) in args[1]["path"]

    @pytest.mark.asyncio
    async def test_to_langchain_tools(self, filesystem_config: MCPConfig):
        """Test conversion to LangChain tools."""
        client = FilesystemMCP(config=filesystem_config)
        tools = client.to_langchain_tools()

        assert len(tools) == 4
        assert all(isinstance(tool, Tool) for tool in tools)

        tool_names = [tool.name for tool in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names
        assert "search_files" in tool_names

    @pytest.mark.asyncio
    async def test_langchain_tool_read_file(self, filesystem_config: MCPConfig):
        """Test LangChain read_file tool execution."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"content": "test content"}

            client = FilesystemMCP(config=filesystem_config)
            tools = client.to_langchain_tools()
            read_tool = next(t for t in tools if t.name == "read_file")

            result = read_tool.func("/tmp/test.txt")
            assert result == "test content"


# WebSearchMCP Tests


class TestWebSearchMCP:
    """Test WebSearchMCP client functionality."""

    @pytest.mark.asyncio
    async def test_websearch_initialization(self, websearch_config: MCPConfig):
        """Test web search client initialization."""
        client = WebSearchMCP(config=websearch_config)
        assert client.config == websearch_config

    @pytest.mark.asyncio
    async def test_search_success(self, websearch_config: MCPConfig):
        """Test successful web search."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "snippet": "First result snippet",
                        "score": 0.95,
                    },
                    {
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "snippet": "Second result snippet",
                        "score": 0.87,
                    },
                ]
            }

            client = WebSearchMCP(config=websearch_config)
            results = await client.search("test query", num_results=2)

            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)
            assert results[0].title == "Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].score == 0.95
            assert results[1].title == "Result 2"

            mock_call_tool.assert_called_once_with(
                "search", {"query": "test query", "num_results": 2}
            )

    @pytest.mark.asyncio
    async def test_search_no_results(self, websearch_config: MCPConfig):
        """Test web search with no results."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"results": []}

            client = WebSearchMCP(config=websearch_config)
            results = await client.search("obscure query")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_result_missing_fields(self, websearch_config: MCPConfig):
        """Test search result parsing with missing optional fields."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "snippet": "Snippet",
                        # score is optional
                    }
                ]
            }

            client = WebSearchMCP(config=websearch_config)
            results = await client.search("query")

            assert len(results) == 1
            assert results[0].score is None

    @pytest.mark.asyncio
    async def test_fetch_url_success(self, websearch_config: MCPConfig):
        """Test successful URL fetch."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "content": "<html><body>Page content</body></html>"
            }

            client = WebSearchMCP(config=websearch_config)
            content = await client.fetch_url("https://example.com")

            assert content == "<html><body>Page content</body></html>"
            mock_call_tool.assert_called_once_with(
                "fetch_url", {"url": "https://example.com"}
            )

    @pytest.mark.asyncio
    async def test_fetch_url_empty_content(self, websearch_config: MCPConfig):
        """Test URL fetch with empty content."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"content": ""}

            client = WebSearchMCP(config=websearch_config)
            content = await client.fetch_url("https://example.com")

            assert content == ""

    @pytest.mark.asyncio
    async def test_to_langchain_tools(self, websearch_config: MCPConfig):
        """Test conversion to LangChain tools."""
        client = WebSearchMCP(config=websearch_config)
        tools = client.to_langchain_tools()

        assert len(tools) == 2
        assert all(isinstance(tool, Tool) for tool in tools)

        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "fetch_url" in tool_names

    @pytest.mark.asyncio
    async def test_langchain_tool_search(self, websearch_config: MCPConfig):
        """Test LangChain web_search tool execution."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {
                "results": [
                    {
                        "title": "Test",
                        "url": "https://test.com",
                        "snippet": "Test snippet",
                    }
                ]
            }

            client = WebSearchMCP(config=websearch_config)
            tools = client.to_langchain_tools()
            search_tool = next(t for t in tools if t.name == "web_search")

            result = search_tool.func("test query")
            assert "Test" in result
            assert "https://test.com" in result
            assert "Test snippet" in result

    @pytest.mark.asyncio
    async def test_langchain_tool_fetch_url(self, websearch_config: MCPConfig):
        """Test LangChain fetch_url tool execution."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.return_value = {"content": "Page content"}

            client = WebSearchMCP(config=websearch_config)
            tools = client.to_langchain_tools()
            fetch_tool = next(t for t in tools if t.name == "fetch_url")

            result = fetch_tool.func("https://example.com")
            assert result == "Page content"


# SearchResult Model Tests


class TestSearchResult:
    """Test SearchResult Pydantic model."""

    def test_search_result_creation(self):
        """Test creating search result with all fields."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            score=0.95,
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.score == 0.95

    def test_search_result_optional_score(self):
        """Test search result with optional score field."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
        )
        assert result.score is None


# Factory Functions Tests


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_filesystem_client_defaults(self):
        """Test filesystem client creation with defaults."""
        client = create_filesystem_client()
        assert isinstance(client, FilesystemMCP)
        assert client.config.host == "localhost"
        assert client.config.port == 8001
        assert client.base_path is None

    def test_create_filesystem_client_custom(self, temp_base_path: Path):
        """Test filesystem client creation with custom values."""
        client = create_filesystem_client(
            host="192.168.1.1",
            port=9001,
            base_path=temp_base_path,
        )
        assert client.config.host == "192.168.1.1"
        assert client.config.port == 9001
        assert client.base_path == temp_base_path

    def test_create_websearch_client_defaults(self):
        """Test web search client creation with defaults."""
        client = create_websearch_client()
        assert isinstance(client, WebSearchMCP)
        assert client.config.host == "localhost"
        assert client.config.port == 8002

    def test_create_websearch_client_custom(self):
        """Test web search client creation with custom values."""
        client = create_websearch_client(
            host="192.168.1.1",
            port=9002,
        )
        assert client.config.host == "192.168.1.1"
        assert client.config.port == 9002


# Error Handling Tests


class TestErrorHandling:
    """Test error scenarios and exception handling."""

    @pytest.mark.asyncio
    async def test_mcp_connection_error_inheritance(self):
        """Test MCPConnectionError is subclass of MCPError."""
        assert issubclass(MCPConnectionError, MCPError)

    @pytest.mark.asyncio
    async def test_mcp_tool_error_inheritance(self):
        """Test MCPToolError is subclass of MCPError."""
        assert issubclass(MCPToolError, MCPError)

    @pytest.mark.asyncio
    async def test_connection_timeout(self, mcp_config: MCPConfig):
        """Test connection timeout handling."""
        with patch.object(ConcreteMCPClient, "_health_check") as mock_health:
            mock_health.side_effect = httpx.TimeoutException("Connection timeout")

            client = ConcreteMCPClient(config=mcp_config)
            with pytest.raises(httpx.TimeoutException):
                await client.connect()

    @pytest.mark.asyncio
    async def test_tool_call_retry_logic(self, mcp_config: MCPConfig):
        """Test retry logic for tool calls on transient errors."""
        # Note: The retry decorator will retry on RequestError and TimeoutException
        # This test verifies the decorator is applied
        with patch.object(ConcreteMCPClient, "client") as mock_client:
            mock_http_client = AsyncMock()
            # First two calls fail, third succeeds
            mock_response_success = MagicMock()
            mock_response_success.json.return_value = {"result": {"data": "success"}}
            mock_response_success.raise_for_status = MagicMock()

            mock_http_client.post = AsyncMock(
                side_effect=[
                    httpx.TimeoutException("Timeout"),
                    httpx.TimeoutException("Timeout"),
                    mock_response_success,
                ]
            )
            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            client = ConcreteMCPClient(config=mcp_config)
            client._connected = True

            result = await client.call_tool("test_tool", {})
            assert result == {"data": "success"}
            assert mock_http_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_filesystem_tool_error_propagation(
        self, filesystem_config: MCPConfig
    ):
        """Test filesystem operations propagate MCPToolError."""
        with patch.object(FilesystemMCP, "call_tool") as mock_call_tool:
            mock_call_tool.side_effect = MCPToolError("File not found")

            client = FilesystemMCP(config=filesystem_config)
            with pytest.raises(MCPToolError, match="File not found"):
                await client.read_file("/nonexistent.txt")

    @pytest.mark.asyncio
    async def test_websearch_tool_error_propagation(self, websearch_config: MCPConfig):
        """Test web search operations propagate MCPToolError."""
        with patch.object(WebSearchMCP, "call_tool") as mock_call_tool:
            mock_call_tool.side_effect = MCPToolError("Search service unavailable")

            client = WebSearchMCP(config=websearch_config)
            with pytest.raises(MCPToolError, match="Search service unavailable"):
                await client.search("query")


# Integration-style Tests


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_filesystem_complete_workflow(
        self, filesystem_config: MCPConfig, temp_base_path: Path
    ):
        """Test complete filesystem workflow: connect, write, read, list, disconnect."""
        with patch.object(FilesystemMCP, "_health_check") as mock_health, \
             patch.object(FilesystemMCP, "client") as mock_client:

            # Mock health check
            mock_response_health = MagicMock()
            mock_response_health.status_code = 200
            mock_health.return_value = mock_response_health

            # Mock HTTP client
            mock_http_client = AsyncMock()

            # Setup mock responses for different tool calls
            def create_response(result_data):
                response = MagicMock()
                response.json.return_value = {"result": result_data}
                response.raise_for_status = MagicMock()
                return response

            mock_http_client.post = AsyncMock(side_effect=[
                create_response({"success": True}),  # write
                create_response({"content": "test content"}),  # read
                create_response({"entries": [{"name": "test.txt"}]}),  # list
            ])

            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            async with FilesystemMCP(
                config=filesystem_config, base_path=temp_base_path
            ) as fs:
                # Write file
                test_file = temp_base_path / "test.txt"
                success = await fs.write_file(test_file, "test content")
                assert success is True

                # Read file
                content = await fs.read_file(test_file)
                assert content == "test content"

                # List directory
                entries = await fs.list_directory(temp_base_path)
                assert len(entries) == 1
                assert entries[0]["name"] == "test.txt"

    @pytest.mark.asyncio
    async def test_websearch_complete_workflow(self, websearch_config: MCPConfig):
        """Test complete web search workflow: connect, search, fetch, disconnect."""
        with patch.object(WebSearchMCP, "_health_check") as mock_health, \
             patch.object(WebSearchMCP, "client") as mock_client:

            # Mock health check
            mock_response_health = MagicMock()
            mock_response_health.status_code = 200
            mock_health.return_value = mock_response_health

            # Mock HTTP client
            mock_http_client = AsyncMock()

            def create_response(result_data):
                response = MagicMock()
                response.json.return_value = {"result": result_data}
                response.raise_for_status = MagicMock()
                return response

            mock_http_client.post = AsyncMock(side_effect=[
                create_response({
                    "results": [
                        {
                            "title": "Result",
                            "url": "https://example.com",
                            "snippet": "Snippet",
                        }
                    ]
                }),  # search
                create_response({"content": "Page content"}),  # fetch
            ])

            mock_client.__get__ = MagicMock(return_value=mock_http_client)

            async with WebSearchMCP(config=websearch_config) as search:
                # Search
                results = await search.search("query")
                assert len(results) == 1
                assert results[0].url == "https://example.com"

                # Fetch URL
                content = await search.fetch_url(results[0].url)
                assert content == "Page content"
