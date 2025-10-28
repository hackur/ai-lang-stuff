"""
MCP Server Template

This is a template for building custom MCP (Model Context Protocol) servers.
Copy this directory and customize for your specific use case.

Architecture:
- Server class handles MCP protocol and request routing
- Individual tools are registered and handle specific operations
- Configuration loaded from config.yaml
- Comprehensive error handling and logging
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class ToolParameter(BaseModel):
    """Model for tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, int, bool, etc.)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")


class ToolDefinition(BaseModel):
    """Model for tool definition."""

    name: str = Field(..., description="Tool name (must be unique)")
    description: str = Field(..., description="Tool description")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    enabled: bool = Field(default=True, description="Whether tool is enabled")


class ServerConfig(BaseModel):
    """Model for server configuration."""

    name: str = Field(..., description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    description: str = Field(..., description="Server description")
    tools: List[ToolDefinition] = Field(default_factory=list, description="Registered tools")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")


class ToolRequest(BaseModel):
    """Model for tool invocation request."""

    tool: str = Field(..., description="Tool name to invoke")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolResponse(BaseModel):
    """Model for tool invocation response."""

    status: str = Field(..., description="Response status (success, error)")
    tool: str = Field(..., description="Tool name that was invoked")
    result: Optional[Any] = Field(default=None, description="Tool result data")
    error: Optional[str] = Field(default=None, description="Error message if status=error")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# MCP Server Implementation
# ============================================================================


class MCPServer:
    """
    MCP Server base class.

    Handles tool registration, request routing, and response formatting.
    Extend this class to create custom MCP servers.

    Example:
        >>> server = MCPServer("my-server")
        >>> server.register_tool("greet", greet_function, "Greet a user")
        >>> result = server.invoke_tool("greet", {"name": "Alice"})
        >>> print(result.result)
        Hello, Alice!
    """

    def __init__(
        self,
        name: str,
        config_path: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
    ):
        """
        Initialize MCP server.

        Args:
            name: Server name
            config_path: Path to YAML configuration file
            version: Server version
            description: Server description
        """
        self.name = name
        self.version = version
        self.description = description
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}

        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

        logger.info(f"Initialized MCP server: {self.name} v{self.version}")

    def load_config(self, config_path: str) -> None:
        """
        Load server configuration from YAML file.

        Args:
            config_path: Path to config.yaml file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValidationError: If config doesn't match schema
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        # Validate and load configuration
        try:
            config = ServerConfig(**config_data)
            self.name = config.name
            self.version = config.version
            self.description = config.description

            # Store tool definitions for introspection
            for tool_def in config.tools:
                self.tool_definitions[tool_def.name] = tool_def

            logger.info(f"Loaded config from {config_path}: {len(config.tools)} tools defined")

        except ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
    ) -> None:
        """
        Register a tool with the server.

        Args:
            name: Tool name (must be unique)
            function: Python function to execute
            description: Tool description
            parameters: List of parameter definitions

        Raises:
            ValueError: If tool name already registered

        Example:
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>> server.register_tool("add", add, "Add two numbers")
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self.tools[name] = function

        # Create tool definition if not already defined
        if name not in self.tool_definitions:
            self.tool_definitions[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or [],
            )

        logger.info(f"Registered tool: {name}")

    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool from the server.

        Args:
            name: Tool name to unregister

        Raises:
            KeyError: If tool not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")

        del self.tools[name]
        if name in self.tool_definitions:
            del self.tool_definitions[name]

        logger.info(f"Unregistered tool: {name}")

    def list_tools(self) -> List[ToolDefinition]:
        """
        Get list of all registered tools.

        Returns:
            List of tool definitions

        Example:
            >>> tools = server.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        return list(self.tool_definitions.values())

    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """
        Get definition for a specific tool.

        Args:
            name: Tool name

        Returns:
            Tool definition or None if not found
        """
        return self.tool_definitions.get(name)

    def validate_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against tool definition.

        Args:
            tool_name: Name of tool to validate for
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        tool_def = self.tool_definitions.get(tool_name)

        if not tool_def:
            return False, f"Tool '{tool_name}' not found"

        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in parameters:
                return False, f"Missing required parameter: {param.name}"

        # Check for unknown parameters
        defined_params = {p.name for p in tool_def.parameters}
        provided_params = set(parameters.keys())
        unknown_params = provided_params - defined_params

        if unknown_params:
            return False, f"Unknown parameters: {', '.join(unknown_params)}"

        return True, None

    def invoke_tool(
        self, tool_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> ToolResponse:
        """
        Invoke a registered tool.

        Args:
            tool_name: Name of tool to invoke
            parameters: Tool parameters

        Returns:
            ToolResponse with result or error

        Example:
            >>> response = server.invoke_tool("greet", {"name": "Bob"})
            >>> if response.status == "success":
            ...     print(response.result)
        """
        parameters = parameters or {}

        # Check if tool exists
        if tool_name not in self.tools:
            return ToolResponse(
                status="error",
                tool=tool_name,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}",
            )

        # Validate parameters
        is_valid, error_msg = self.validate_parameters(tool_name, parameters)
        if not is_valid:
            return ToolResponse(
                status="error",
                tool=tool_name,
                error=f"Parameter validation failed: {error_msg}",
            )

        # Execute tool
        try:
            logger.info(f"Invoking tool: {tool_name} with parameters: {parameters}")
            result = self.tools[tool_name](**parameters)

            return ToolResponse(
                status="success",
                tool=tool_name,
                result=result,
                metadata={"parameters": parameters},
            )

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return ToolResponse(
                status="error",
                tool=tool_name,
                error=f"Execution failed: {str(e)}",
                metadata={"parameters": parameters},
            )

    def handle_request(self, request: Union[str, Dict[str, Any], ToolRequest]) -> ToolResponse:
        """
        Handle incoming MCP request.

        Args:
            request: Request as JSON string, dict, or ToolRequest object

        Returns:
            ToolResponse

        Example:
            >>> request = {"tool": "greet", "parameters": {"name": "Charlie"}}
            >>> response = server.handle_request(request)
        """
        try:
            # Parse request
            if isinstance(request, str):
                request_data = json.loads(request)
                request_obj = ToolRequest(**request_data)
            elif isinstance(request, dict):
                request_obj = ToolRequest(**request)
            elif isinstance(request, ToolRequest):
                request_obj = request
            else:
                return ToolResponse(
                    status="error",
                    tool="unknown",
                    error=f"Invalid request type: {type(request)}",
                )

            # Invoke tool
            return self.invoke_tool(request_obj.tool, request_obj.parameters)

        except json.JSONDecodeError as e:
            return ToolResponse(
                status="error",
                tool="unknown",
                error=f"Invalid JSON: {str(e)}",
            )
        except ValidationError as e:
            return ToolResponse(
                status="error",
                tool="unknown",
                error=f"Invalid request format: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}", exc_info=True)
            return ToolResponse(
                status="error",
                tool="unknown",
                error=f"Internal server error: {str(e)}",
            )

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information.

        Returns:
            Dict with server name, version, description, and available tools
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            "default": p.default,
                        }
                        for p in tool.parameters
                    ],
                }
                for tool in self.list_tools()
            ],
        }


# ============================================================================
# Example Usage
# ============================================================================


def main():
    """Example usage of MCP server template."""
    # Create server
    server = MCPServer(
        name="example-server",
        version="1.0.0",
        description="Example MCP server for demonstration",
    )

    # Define example tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet a person."""
        return f"{greeting}, {name}!"

    # Register tool
    server.register_tool(
        name="greet",
        function=greet,
        description="Greet a person with a custom greeting",
        parameters=[
            ToolParameter(
                name="name",
                type="string",
                description="Name of person to greet",
                required=True,
            ),
            ToolParameter(
                name="greeting",
                type="string",
                description="Greeting to use",
                required=False,
                default="Hello",
            ),
        ],
    )

    # Get server info
    print("\n" + "=" * 80)
    print("SERVER INFO")
    print("=" * 80)
    print(json.dumps(server.get_server_info(), indent=2))

    # Invoke tool
    print("\n" + "=" * 80)
    print("INVOKING TOOL")
    print("=" * 80)

    response = server.invoke_tool("greet", {"name": "Alice", "greeting": "Hi"})
    print(json.dumps(response.model_dump(), indent=2))

    # Handle request
    print("\n" + "=" * 80)
    print("HANDLING REQUEST")
    print("=" * 80)

    request = {"tool": "greet", "parameters": {"name": "Bob"}}
    response = server.handle_request(request)
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
