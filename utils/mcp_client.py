"""
MCP Client wrappers for Model Context Protocol integration.

This module provides production-quality clients for interacting with MCP servers,
including filesystem operations and web search capabilities. All clients support
connection pooling, retry logic, and LangChain tool integration.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, TypeAlias

import httpx
from langchain_core.tools import Tool
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Type aliases
ToolArgs: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = dict[str, Any]
FileEntry: TypeAlias = dict[str, Any]


class MCPError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    pass


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""

    pass


class MCPConfig(BaseModel):
    """Configuration for MCP client connections."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="localhost", description="MCP server host")
    port: int = Field(default=8000, description="MCP server port")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")
    max_delay: float = Field(default=10.0, description="Maximum delay for exponential backoff")

    @property
    def base_url(self) -> str:
        """Get the base URL for the MCP server."""
        return f"http://{self.host}:{self.port}"


class MCPClient(ABC):
    """
    Base class for MCP (Model Context Protocol) clients.

    Provides connection management, retry logic, and error handling for MCP server
    interactions. Subclasses should implement specific tool methods.

    Args:
        config: MCP client configuration
        client: Optional pre-configured httpx client

    Example:
        ```python
        config = MCPConfig(host="localhost", port=8000)
        async with MCPClient(config) as client:
            result = await client.call_tool("tool_name", {"arg": "value"})
        ```
    """

    def __init__(
        self,
        config: MCPConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize MCP client with configuration."""
        self.config = config or MCPConfig()
        self._client = client
        self._owned_client = client is None
        self._connected = False

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                follow_redirects=True,
            )
        return self._client

    async def connect(self) -> None:
        """
        Establish connection to MCP server.

        Raises:
            MCPConnectionError: If connection fails after retries
        """
        if self._connected:
            return

        try:
            response = await self._health_check()
            if response.status_code != 200:
                raise MCPConnectionError(f"Health check failed with status {response.status_code}")
            self._connected = True
            logger.info(f"Connected to MCP server at {self.config.base_url}")
        except httpx.RequestError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        if self._client and self._owned_client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("Disconnected from MCP server")

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _health_check(self) -> httpx.Response:
        """Perform health check on MCP server."""
        return await self.client.get("/health")

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def call_tool(self, name: str, args: ToolArgs) -> ToolResult:
        """
        Call a tool on the MCP server.

        Args:
            name: Tool name to execute
            args: Tool arguments as dictionary

        Returns:
            Tool execution result as dictionary

        Raises:
            MCPConnectionError: If not connected to server
            MCPToolError: If tool execution fails
        """
        if not self._connected:
            await self.connect()

        try:
            response = await self.client.post(
                "/tools/call",
                json={"name": name, "arguments": args},
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise MCPToolError(f"Tool execution failed: {result['error']}")

            logger.debug(f"Tool {name} executed successfully")
            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            raise MCPToolError(f"Tool call failed with status {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise MCPToolError(f"Tool call request failed: {e}") from e

    @asynccontextmanager
    async def __aenter__(self) -> AsyncIterator["MCPClient"]:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @contextmanager
    def __enter__(self) -> None:
        """Sync context manager entry (not recommended, use async)."""
        raise NotImplementedError("Use async context manager (async with) instead")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Sync context manager exit."""
        pass

    @abstractmethod
    def to_langchain_tools(self) -> list[Tool]:
        """
        Convert MCP client methods to LangChain tools.

        Returns:
            List of LangChain Tool objects
        """
        pass


class FilesystemMCP(MCPClient):
    """
    MCP client for filesystem operations.

    Provides file reading, writing, directory listing, and file search capabilities
    through an MCP server. Integrates with LangChain for agent usage.

    Example:
        ```python
        config = MCPConfig(port=8001)
        async with FilesystemMCP(config) as fs:
            content = await fs.read_file("/path/to/file.txt")
            await fs.write_file("/path/to/output.txt", "Hello, World!")
            files = await fs.list_directory("/path/to/dir")
            results = await fs.search_files("*.py", "/path/to/search")
        ```
    """

    def __init__(
        self,
        config: MCPConfig | None = None,
        client: httpx.AsyncClient | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        """
        Initialize filesystem MCP client.

        Args:
            config: MCP client configuration
            client: Optional pre-configured httpx client
            base_path: Base path to restrict filesystem access (security)
        """
        super().__init__(config, client)
        self.base_path = Path(base_path) if base_path else None

    def _validate_path(self, path: str | Path) -> Path:
        """
        Validate and resolve path to prevent path traversal attacks.

        Args:
            path: Path to validate

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is outside base_path
        """
        resolved = Path(path).resolve()
        if self.base_path:
            base_resolved = self.base_path.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(f"Path {path} is outside allowed base path {self.base_path}")
        return resolved

    async def read_file(self, path: str | Path) -> str:
        """
        Read contents of a file.

        Args:
            path: Path to file to read

        Returns:
            File contents as string

        Raises:
            MCPToolError: If file read fails
        """
        validated_path = self._validate_path(path)
        result = await self.call_tool("read_file", {"path": str(validated_path)})
        return result.get("content", "")

    async def write_file(self, path: str | Path, content: str) -> bool:
        """
        Write content to a file.

        Args:
            path: Path to file to write
            content: Content to write

        Returns:
            True if write successful

        Raises:
            MCPToolError: If file write fails
        """
        validated_path = self._validate_path(path)
        result = await self.call_tool(
            "write_file",
            {"path": str(validated_path), "content": content},
        )
        return result.get("success", False)

    async def list_directory(self, path: str | Path) -> list[FileEntry]:
        """
        List contents of a directory.

        Args:
            path: Path to directory to list

        Returns:
            List of file/directory entries with metadata

        Raises:
            MCPToolError: If directory listing fails
        """
        validated_path = self._validate_path(path)
        result = await self.call_tool("list_directory", {"path": str(validated_path)})
        return result.get("entries", [])

    async def search_files(
        self,
        pattern: str,
        path: str | Path | None = None,
    ) -> list[str]:
        """
        Search for files matching a pattern.

        Args:
            pattern: Glob pattern to match (e.g., "*.py", "**/*.txt")
            path: Path to search in (defaults to base_path or current directory)

        Returns:
            List of matching file paths

        Raises:
            MCPToolError: If search fails
        """
        search_path = path or self.base_path or Path.cwd()
        validated_path = self._validate_path(search_path)
        result = await self.call_tool(
            "search_files",
            {"pattern": pattern, "path": str(validated_path)},
        )
        return result.get("files", [])

    def to_langchain_tools(self) -> list[Tool]:
        """
        Convert filesystem operations to LangChain tools.

        Returns:
            List of LangChain Tool objects for read_file, write_file,
            list_directory, and search_files operations
        """
        import asyncio

        def read_file_sync(path: str) -> str:
            """Read file contents."""
            return asyncio.run(self.read_file(path))

        def write_file_sync(path: str, content: str) -> str:
            """Write content to file."""
            success = asyncio.run(self.write_file(path, content))
            return "Success" if success else "Failed"

        def list_directory_sync(path: str) -> str:
            """List directory contents."""
            entries = asyncio.run(self.list_directory(path))
            return "\n".join([e.get("name", "") for e in entries])

        def search_files_sync(pattern: str, path: str = ".") -> str:
            """Search for files matching pattern."""
            files = asyncio.run(self.search_files(pattern, path))
            return "\n".join(files)

        return [
            Tool(
                name="read_file",
                description="Read contents of a file. Input: file path as string.",
                func=read_file_sync,
            ),
            Tool(
                name="write_file",
                description="Write content to a file. Input: JSON with 'path' and 'content' keys.",
                func=lambda x: write_file_sync(x["path"], x["content"]),
            ),
            Tool(
                name="list_directory",
                description="List contents of a directory. Input: directory path as string.",
                func=list_directory_sync,
            ),
            Tool(
                name="search_files",
                description="Search for files matching a pattern. Input: JSON with 'pattern' and optional 'path' keys.",
                func=lambda x: search_files_sync(x["pattern"], x.get("path", ".")),
            ),
        ]


class SearchResult(BaseModel):
    """Structured search result from web search."""

    title: str = Field(description="Search result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Text snippet from result")
    score: float | None = Field(default=None, description="Relevance score")


class WebSearchMCP(MCPClient):
    """
    MCP client for web search operations.

    Provides web search and URL fetching capabilities through an MCP server.
    Results are parsed into structured format for easy consumption.

    Example:
        ```python
        config = MCPConfig(port=8002)
        async with WebSearchMCP(config) as search:
            results = await search.search("LangChain documentation", num_results=5)
            content = await search.fetch_url("https://example.com")
        ```
    """

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """
        Perform web search.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (default: 5)

        Returns:
            List of structured search results

        Raises:
            MCPToolError: If search fails
        """
        result = await self.call_tool(
            "search",
            {"query": query, "num_results": num_results},
        )

        raw_results = result.get("results", [])
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                score=r.get("score"),
            )
            for r in raw_results
        ]

    async def fetch_url(self, url: str) -> str:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Page content as string

        Raises:
            MCPToolError: If fetch fails
        """
        result = await self.call_tool("fetch_url", {"url": url})
        return result.get("content", "")

    def to_langchain_tools(self) -> list[Tool]:
        """
        Convert web search operations to LangChain tools.

        Returns:
            List of LangChain Tool objects for search and fetch_url operations
        """
        import asyncio

        def search_sync(query: str, num_results: int = 5) -> str:
            """Search the web and return formatted results."""
            results = asyncio.run(self.search(query, num_results))
            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. {result.title}\n   URL: {result.url}\n   {result.snippet}\n"
                )
            return "\n".join(formatted)

        def fetch_url_sync(url: str) -> str:
            """Fetch content from a URL."""
            return asyncio.run(self.fetch_url(url))

        return [
            Tool(
                name="web_search",
                description="Search the web for information. Input: JSON with 'query' and optional 'num_results' (default: 5).",
                func=lambda x: search_sync(
                    x if isinstance(x, str) else x["query"],
                    x.get("num_results", 5) if isinstance(x, dict) else 5,
                ),
            ),
            Tool(
                name="fetch_url",
                description="Fetch content from a URL. Input: URL string.",
                func=fetch_url_sync,
            ),
        ]


# Convenience factory functions


def create_filesystem_client(
    host: str = "localhost",
    port: int = 8001,
    base_path: str | Path | None = None,
) -> FilesystemMCP:
    """
    Create a filesystem MCP client with default configuration.

    Args:
        host: MCP server host
        port: MCP server port
        base_path: Base path to restrict filesystem access

    Returns:
        Configured FilesystemMCP client
    """
    config = MCPConfig(host=host, port=port)
    return FilesystemMCP(config=config, base_path=base_path)


def create_websearch_client(
    host: str = "localhost",
    port: int = 8002,
) -> WebSearchMCP:
    """
    Create a web search MCP client with default configuration.

    Args:
        host: MCP server host
        port: MCP server port

    Returns:
        Configured WebSearchMCP client
    """
    config = MCPConfig(host=host, port=port)
    return WebSearchMCP(config=config)
