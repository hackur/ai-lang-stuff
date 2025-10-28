"""
MCP Server Template Package

A comprehensive template for building custom MCP servers.

Example:
    >>> from server import MCPServer
    >>> from tools import register_example_tools
    >>>
    >>> server = MCPServer("my-server")
    >>> register_example_tools(server)
    >>> response = server.invoke_tool("greet", {"name": "Alice"})
    >>> print(response.result)
"""

__version__ = "1.0.0"
__author__ = "MCP Server Template"

from .server import MCPServer, ToolDefinition, ToolParameter, ToolRequest, ToolResponse

__all__ = [
    "MCPServer",
    "ToolDefinition",
    "ToolParameter",
    "ToolRequest",
    "ToolResponse",
]
