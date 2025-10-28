"""
Test suite for MCP server template.

This file demonstrates testing patterns for MCP servers:
- Unit tests for individual tools
- Integration tests for server functionality
- Mock patterns for external dependencies
- Fixture usage for test data

Run with: pytest tests/test_server.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# Import server components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    MCPServer,
    ServerConfig,
    ToolDefinition,
    ToolParameter,
    ToolRequest,
    ToolResponse,
)
from tools.example_tool import analyze_text, fetch_data, greet


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server():
    """Create a basic MCP server for testing."""
    server = MCPServer(
        name="test-server",
        version="1.0.0",
        description="Test server",
    )
    return server


@pytest.fixture
def configured_server(tmp_path):
    """Create an MCP server with configuration file."""
    # Create test config
    config_file = tmp_path / "config.yaml"
    config_content = """
name: "test-server"
version: "1.0.0"
description: "Test server with config"
tools:
  - name: "test_tool"
    description: "Test tool"
    enabled: true
    parameters:
      - name: "input"
        type: "string"
        description: "Test input"
        required: true
"""
    config_file.write_text(config_content)

    # Load server with config
    server = MCPServer(name="test", config_path=str(config_file))
    return server


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Hello world. This is a test. Testing is important."


# ============================================================================
# Unit Tests - Server Core Functionality
# ============================================================================


class TestMCPServer:
    """Test core MCP server functionality."""

    def test_server_initialization(self):
        """Test server initializes with correct properties."""
        server = MCPServer(
            name="test-server",
            version="2.0.0",
            description="Test description",
        )

        assert server.name == "test-server"
        assert server.version == "2.0.0"
        assert server.description == "Test description"
        assert len(server.tools) == 0

    def test_server_config_loading(self, configured_server):
        """Test server loads configuration correctly."""
        assert configured_server.name == "test-server"
        assert configured_server.version == "1.0.0"
        assert "test_tool" in configured_server.tool_definitions

    def test_tool_registration(self, server):
        """Test registering a tool."""

        def test_func(param1: str) -> str:
            return f"Result: {param1}"

        server.register_tool(
            name="test_tool",
            function=test_func,
            description="Test function",
        )

        assert "test_tool" in server.tools
        assert "test_tool" in server.tool_definitions

    def test_duplicate_tool_registration_raises_error(self, server):
        """Test that registering duplicate tool raises ValueError."""

        def test_func():
            return "test"

        server.register_tool("test_tool", test_func)

        with pytest.raises(ValueError, match="already registered"):
            server.register_tool("test_tool", test_func)

    def test_tool_unregistration(self, server):
        """Test unregistering a tool."""

        def test_func():
            return "test"

        server.register_tool("test_tool", test_func)
        assert "test_tool" in server.tools

        server.unregister_tool("test_tool")
        assert "test_tool" not in server.tools

    def test_unregister_nonexistent_tool_raises_error(self, server):
        """Test unregistering non-existent tool raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            server.unregister_tool("nonexistent_tool")

    def test_list_tools(self, server):
        """Test listing all registered tools."""

        def func1():
            return "1"

        def func2():
            return "2"

        server.register_tool("tool1", func1, "First tool")
        server.register_tool("tool2", func2, "Second tool")

        tools = server.list_tools()
        assert len(tools) == 2
        assert any(t.name == "tool1" for t in tools)
        assert any(t.name == "tool2" for t in tools)

    def test_get_tool_definition(self, server):
        """Test getting tool definition."""

        def test_func(param: str) -> str:
            return param

        server.register_tool(
            "test_tool",
            test_func,
            "Test description",
            parameters=[
                ToolParameter(
                    name="param",
                    type="string",
                    description="Test param",
                    required=True,
                )
            ],
        )

        definition = server.get_tool_definition("test_tool")
        assert definition is not None
        assert definition.name == "test_tool"
        assert len(definition.parameters) == 1

    def test_get_server_info(self, server):
        """Test getting server information."""

        def test_func(param: str) -> str:
            return param

        server.register_tool("test_tool", test_func, "Test tool")

        info = server.get_server_info()
        assert info["name"] == "test-server"
        assert info["version"] == "1.0.0"
        assert len(info["tools"]) == 1
        assert info["tools"][0]["name"] == "test_tool"


# ============================================================================
# Unit Tests - Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_valid_parameters(self, server):
        """Test validation passes for valid parameters."""

        def test_func(param1: str, param2: int = 10) -> str:
            return f"{param1}-{param2}"

        server.register_tool(
            "test_tool",
            test_func,
            parameters=[
                ToolParameter(name="param1", type="string", description="P1", required=True),
                ToolParameter(
                    name="param2", type="int", description="P2", required=False, default=10
                ),
            ],
        )

        is_valid, error = server.validate_parameters("test_tool", {"param1": "test"})
        assert is_valid
        assert error is None

    def test_missing_required_parameter(self, server):
        """Test validation fails for missing required parameter."""

        def test_func(param1: str) -> str:
            return param1

        server.register_tool(
            "test_tool",
            test_func,
            parameters=[
                ToolParameter(name="param1", type="string", description="P1", required=True)
            ],
        )

        is_valid, error = server.validate_parameters("test_tool", {})
        assert not is_valid
        assert "Missing required parameter" in error

    def test_unknown_parameter(self, server):
        """Test validation fails for unknown parameter."""

        def test_func(param1: str) -> str:
            return param1

        server.register_tool(
            "test_tool",
            test_func,
            parameters=[
                ToolParameter(name="param1", type="string", description="P1", required=True)
            ],
        )

        is_valid, error = server.validate_parameters(
            "test_tool", {"param1": "test", "unknown": "value"}
        )
        assert not is_valid
        assert "Unknown parameters" in error

    def test_nonexistent_tool_validation(self, server):
        """Test validation fails for non-existent tool."""
        is_valid, error = server.validate_parameters("nonexistent", {})
        assert not is_valid
        assert "not found" in error


# ============================================================================
# Unit Tests - Tool Invocation
# ============================================================================


class TestToolInvocation:
    """Test tool invocation."""

    def test_successful_invocation(self, server):
        """Test successful tool invocation."""

        def add(a: int, b: int) -> int:
            return a + b

        server.register_tool(
            "add",
            add,
            parameters=[
                ToolParameter(name="a", type="int", description="First number", required=True),
                ToolParameter(name="b", type="int", description="Second number", required=True),
            ],
        )

        response = server.invoke_tool("add", {"a": 5, "b": 3})

        assert response.status == "success"
        assert response.tool == "add"
        assert response.result == 8
        assert response.error is None

    def test_invocation_with_error(self, server):
        """Test tool invocation with error."""

        def divide(a: int, b: int) -> float:
            return a / b

        server.register_tool(
            "divide",
            divide,
            parameters=[
                ToolParameter(name="a", type="int", description="Numerator", required=True),
                ToolParameter(name="b", type="int", description="Denominator", required=True),
            ],
        )

        response = server.invoke_tool("divide", {"a": 10, "b": 0})

        assert response.status == "error"
        assert response.tool == "divide"
        assert "division by zero" in response.error.lower()

    def test_invoke_nonexistent_tool(self, server):
        """Test invoking non-existent tool returns error."""
        response = server.invoke_tool("nonexistent", {})

        assert response.status == "error"
        assert "not found" in response.error

    def test_invocation_with_invalid_parameters(self, server):
        """Test invocation with invalid parameters returns error."""

        def test_func(param: str) -> str:
            return param

        server.register_tool(
            "test_tool",
            test_func,
            parameters=[
                ToolParameter(name="param", type="string", description="Test", required=True)
            ],
        )

        # Missing required parameter
        response = server.invoke_tool("test_tool", {})
        assert response.status == "error"
        assert "validation failed" in response.error.lower()


# ============================================================================
# Unit Tests - Request Handling
# ============================================================================


class TestRequestHandling:
    """Test request handling."""

    def test_handle_dict_request(self, server):
        """Test handling request as dict."""

        def echo(message: str) -> str:
            return message

        server.register_tool(
            "echo",
            echo,
            parameters=[
                ToolParameter(name="message", type="string", description="Message", required=True)
            ],
        )

        request = {"tool": "echo", "parameters": {"message": "Hello"}}
        response = server.handle_request(request)

        assert response.status == "success"
        assert response.result == "Hello"

    def test_handle_json_string_request(self, server):
        """Test handling request as JSON string."""

        def echo(message: str) -> str:
            return message

        server.register_tool(
            "echo",
            echo,
            parameters=[
                ToolParameter(name="message", type="string", description="Message", required=True)
            ],
        )

        request = json.dumps({"tool": "echo", "parameters": {"message": "Hello"}})
        response = server.handle_request(request)

        assert response.status == "success"
        assert response.result == "Hello"

    def test_handle_toolrequest_object(self, server):
        """Test handling ToolRequest object."""

        def echo(message: str) -> str:
            return message

        server.register_tool(
            "echo",
            echo,
            parameters=[
                ToolParameter(name="message", type="string", description="Message", required=True)
            ],
        )

        request = ToolRequest(tool="echo", parameters={"message": "Hello"})
        response = server.handle_request(request)

        assert response.status == "success"
        assert response.result == "Hello"

    def test_handle_invalid_json_request(self, server):
        """Test handling invalid JSON request."""
        response = server.handle_request("invalid json{")

        assert response.status == "error"
        assert "Invalid JSON" in response.error

    def test_handle_invalid_request_format(self, server):
        """Test handling request with invalid format."""
        response = server.handle_request({"invalid": "format"})

        assert response.status == "error"
        assert "Invalid request format" in response.error


# ============================================================================
# Unit Tests - Example Tools
# ============================================================================


class TestGreetTool:
    """Test greet tool."""

    def test_basic_greeting(self):
        """Test basic greeting."""
        result = greet("Alice")
        assert "Alice" in result["message"]
        assert result["name"] == "Alice"
        assert not result["formal"]

    def test_custom_greeting(self):
        """Test custom greeting."""
        result = greet("Bob", greeting="Hi")
        assert "Hi, Bob" in result["message"]

    def test_formal_greeting(self):
        """Test formal greeting."""
        result = greet("Charlie", formal=True)
        assert result["formal"]
        assert "Good day" in result["message"]

    def test_invalid_name_raises_error(self):
        """Test invalid name raises error."""
        with pytest.raises(ValidationError):
            greet("Alice123")  # Contains numbers

    def test_empty_greeting_raises_error(self):
        """Test empty greeting raises error."""
        with pytest.raises(ValidationError):
            greet("Alice", greeting="")


class TestAnalyzeTextTool:
    """Test analyze_text tool."""

    def test_basic_analysis(self, sample_text):
        """Test basic text analysis."""
        result = analyze_text(sample_text)

        assert result["character_count"] > 0
        assert result["word_count"] > 0
        assert result["sentence_count"] > 0
        assert result["average_word_length"] > 0

    def test_word_analysis(self, sample_text):
        """Test word frequency analysis."""
        result = analyze_text(sample_text, include_words=True)

        assert result["most_common_words"] is not None
        assert len(result["most_common_words"]) > 0

    def test_sentence_analysis(self, sample_text):
        """Test sentence analysis."""
        result = analyze_text(sample_text, include_sentences=True)

        assert result["longest_sentence"] is not None
        assert len(result["longest_sentence"]) > 0

    def test_analysis_without_optional_features(self, sample_text):
        """Test analysis with optional features disabled."""
        result = analyze_text(sample_text, include_words=False, include_sentences=False)

        assert result["most_common_words"] is None
        assert result["longest_sentence"] is None


class TestFetchDataTool:
    """Test fetch_data tool."""

    def test_valid_url(self):
        """Test fetching with valid URL."""
        result = fetch_data("https://api.example.com/data")

        assert result["status_code"] == 200
        assert result["url"] == "https://api.example.com/data"

    def test_custom_timeout(self):
        """Test custom timeout parameter."""
        result = fetch_data("https://api.example.com/data", timeout=60)

        assert result is not None

    def test_invalid_url_raises_error(self):
        """Test invalid URL raises error."""
        with pytest.raises(ValidationError):
            fetch_data("invalid-url")  # Not http/https

    def test_timeout_out_of_range_raises_error(self):
        """Test timeout out of range raises error."""
        with pytest.raises(ValidationError):
            fetch_data("https://api.example.com", timeout=500)  # > 300


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_tools(self, server):
        """Test complete workflow: register, configure, invoke."""
        from tools.example_tool import register_example_tools

        # Register tools
        register_example_tools(server)

        # Verify tools registered
        tools = server.list_tools()
        assert len(tools) == 3

        # Test each tool
        response1 = server.invoke_tool("greet", {"name": "Alice"})
        assert response1.status == "success"

        response2 = server.invoke_tool("analyze_text", {"text": "Hello world"})
        assert response2.status == "success"

        response3 = server.invoke_tool("fetch_data", {"url": "https://example.com"})
        assert response3.status == "success"

    def test_server_info_includes_all_tools(self, server):
        """Test server info includes all registered tools."""
        from tools.example_tool import register_example_tools

        register_example_tools(server)

        info = server.get_server_info()
        assert len(info["tools"]) == 3
        assert any(t["name"] == "greet" for t in info["tools"])
        assert any(t["name"] == "analyze_text" for t in info["tools"])
        assert any(t["name"] == "fetch_data" for t in info["tools"])


# ============================================================================
# Mock Patterns
# ============================================================================


class TestMockPatterns:
    """Demonstrate mocking patterns for testing."""

    @patch("tools.example_tool.logger")
    def test_mock_logger(self, mock_logger, server):
        """Test with mocked logger."""

        def test_func():
            return "test"

        server.register_tool("test_tool", test_func)
        server.invoke_tool("test_tool", {})

        # Verify logger was called
        assert mock_logger.info.called

    def test_mock_external_dependency(self):
        """Test mocking external HTTP requests."""
        # In real implementation, you would mock httpx or requests
        # Example:
        # with patch("httpx.get") as mock_get:
        #     mock_get.return_value = MagicMock(status_code=200, json=lambda: {"data": "test"})
        #     result = fetch_data("https://example.com")
        #     assert result["status_code"] == 200
        pass


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
