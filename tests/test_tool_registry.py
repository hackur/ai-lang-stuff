"""
Tests for the ToolRegistry class.
"""

import json

import pytest

from utils.tool_registry import ToolRegistry, get_registry


def sample_tool(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


def another_tool(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


class TestToolRegistry:
    """Test suite for ToolRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()

    def test_singleton_pattern(self):
        """Test that ToolRegistry follows singleton pattern."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        registry3 = get_registry()

        assert registry1 is registry2
        assert registry2 is registry3

    def test_register_tool(self):
        """Test registering a tool."""
        registry = get_registry()

        registry.register_tool(
            name="add",
            tool=sample_tool,
            description="Add two numbers",
            category="other",
        )

        assert len(registry) == 1
        assert registry.get_tool("add") == sample_tool

    def test_register_tool_not_callable(self):
        """Test that registering a non-callable raises TypeError."""
        registry = get_registry()

        with pytest.raises(TypeError, match="must be callable"):
            registry.register_tool(
                name="invalid",
                tool="not a function",  # type: ignore
                description="Invalid tool",
                category="other",
            )

    def test_get_tool_not_found(self):
        """Test that getting a non-existent tool raises KeyError."""
        registry = get_registry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get_tool("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )
        registry.register_tool(
            name="upper",
            tool=another_tool,
            description="Uppercase text",
            category="other",
        )

        tools = registry.list_tools()
        assert len(tools) == 2
        assert all("name" in t for t in tools)
        assert all("description" in t for t in tools)
        assert all("category" in t for t in tools)

    def test_list_tools_filtered(self):
        """Test filtering tools by category."""
        registry = get_registry()

        registry.register_tool(
            name="file_read",
            tool=sample_tool,
            description="Read file",
            category="filesystem",
        )
        registry.register_tool(
            name="web_fetch",
            tool=another_tool,
            description="Fetch URL",
            category="web",
        )

        filesystem_tools = registry.list_tools(category="filesystem")
        assert len(filesystem_tools) == 1
        assert filesystem_tools[0]["name"] == "file_read"

        web_tools = registry.list_tools(category="web")
        assert len(web_tools) == 1
        assert web_tools[0]["name"] == "web_fetch"

    def test_get_langchain_tools(self):
        """Test converting tools to LangChain format."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )

        lc_tools = registry.get_langchain_tools()
        assert len(lc_tools) == 1
        assert lc_tools[0].name == "add"
        assert "Add numbers" in lc_tools[0].description

    def test_get_langchain_tools_filtered(self):
        """Test filtering LangChain tools by categories."""
        registry = get_registry()

        registry.register_tool(
            name="file_read",
            tool=sample_tool,
            description="Read file",
            category="filesystem",
        )
        registry.register_tool(
            name="web_fetch",
            tool=another_tool,
            description="Fetch URL",
            category="web",
        )

        filesystem_tools = registry.get_langchain_tools(categories=["filesystem"])
        assert len(filesystem_tools) == 1
        assert filesystem_tools[0].name == "file_read"

    def test_to_json(self):
        """Test exporting registry to JSON."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )

        json_str = registry.to_json()
        data = json.loads(json_str)

        assert "tools" in data
        assert "categories" in data
        assert "total_tools" in data
        assert data["total_tools"] == 1

    def test_to_json_with_file(self, tmp_path):
        """Test exporting registry to JSON file."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )

        output_file = tmp_path / "registry.json"
        registry.to_json(filepath=output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["total_tools"] == 1

    def test_clear(self):
        """Test clearing the registry."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )
        assert len(registry) == 1

        registry.clear()
        assert len(registry) == 0

    def test_repr(self):
        """Test string representation."""
        registry = get_registry()

        registry.register_tool(
            name="add", tool=sample_tool, description="Add numbers", category="other"
        )

        repr_str = repr(registry)
        assert "ToolRegistry" in repr_str
        assert "tools=1" in repr_str
