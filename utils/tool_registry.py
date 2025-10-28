"""
Centralized tool registry for managing and discovering tools across the project.

This module provides a singleton ToolRegistry class that maintains a centralized
registry of all available tools, with support for auto-discovery, categorization,
and conversion to LangChain Tool objects.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolMetadata:
    """Metadata container for a registered tool."""

    def __init__(
        self,
        name: str,
        tool: Callable,
        description: str,
        category: str,
        args_schema: Optional[type] = None,
    ):
        """
        Initialize tool metadata.

        Args:
            name: Unique identifier for the tool
            tool: Callable function or tool object
            description: Human-readable description of what the tool does
            category: Category for organizing tools (filesystem, web, models, etc.)
            args_schema: Optional Pydantic model for tool arguments
        """
        self.name = name
        self.tool = tool
        self.description = description
        self.category = category
        self.args_schema = args_schema

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary format.

        Returns:
            Dictionary with tool metadata (excluding the callable)
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "signature": str(inspect.signature(self.tool)),
        }


class ToolRegistry:
    """
    Singleton registry for managing tools across the project.

    Provides centralized tool registration, discovery, and conversion
    to LangChain Tool objects for use in agents.
    """

    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ToolRegistry":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry if not already initialized."""
        if not self._initialized:
            self._tools: Dict[str, ToolMetadata] = {}
            self._initialized = True
            logger.info("ToolRegistry initialized")

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """
        Get the singleton instance of ToolRegistry.

        Returns:
            The singleton ToolRegistry instance
        """
        return cls()

    def register_tool(
        self,
        name: str,
        tool: Callable,
        description: str,
        category: str,
        args_schema: Optional[type] = None,
    ) -> None:
        """
        Register a tool with metadata.

        Args:
            name: Unique identifier for the tool
            tool: Callable function or tool object
            description: Human-readable description of what the tool does
            category: Category for organizing tools (filesystem, web, models, database, workflow)
            args_schema: Optional Pydantic model for tool arguments

        Raises:
            ValueError: If tool is not callable or name is already registered
            TypeError: If tool is not a valid callable
        """
        if not callable(tool):
            raise TypeError(f"Tool '{name}' must be callable, got {type(tool)}")

        if name in self._tools:
            logger.warning(f"Tool '{name}' is already registered, overwriting")

        if category not in ["filesystem", "web", "models", "database", "workflow", "other"]:
            logger.warning(f"Unusual category '{category}' for tool '{name}'")

        metadata = ToolMetadata(
            name=name,
            tool=tool,
            description=description,
            category=category,
            args_schema=args_schema,
        )

        self._tools[name] = metadata
        logger.info(f"Registered tool: {name} (category: {category})")

    def get_tool(self, name: str) -> Callable:
        """
        Retrieve a tool by name.

        Args:
            name: The name of the tool to retrieve

        Returns:
            The callable tool function

        Raises:
            KeyError: If tool is not found in registry
        """
        if name not in self._tools:
            available = ", ".join(self._tools.keys())
            raise KeyError(f"Tool '{name}' not found in registry. Available tools: {available}")

        return self._tools[name].tool

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all tools or filter by category.

        Args:
            category: Optional category to filter by (filesystem, web, models, database, workflow)

        Returns:
            List of dictionaries containing tool metadata
        """
        tools = self._tools.values()

        if category:
            tools = [t for t in tools if t.category == category]
            logger.debug(f"Filtered tools by category '{category}': {len(tools)} tools")

        return [t.to_dict() for t in tools]

    def get_langchain_tools(self, categories: Optional[List[str]] = None) -> List[Any]:
        """
        Convert registered tools to LangChain Tool objects.

        Args:
            categories: Optional list of categories to filter by

        Returns:
            List of LangChain StructuredTool objects ready for agent use

        Raises:
            ImportError: If langchain_core is not installed
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain_core is required to use get_langchain_tools(). "
                "Install it with: pip install langchain-core"
            )

        tools = self._tools.values()

        if categories:
            tools = [t for t in tools if t.category in categories]
            logger.debug(f"Filtered tools by categories {categories}: {len(tools)} tools")

        langchain_tools = []
        for metadata in tools:
            try:
                # Create StructuredTool from registered tool
                lc_tool = StructuredTool.from_function(
                    func=metadata.tool,
                    name=metadata.name,
                    description=metadata.description,
                    args_schema=metadata.args_schema,
                )
                langchain_tools.append(lc_tool)
                logger.debug(f"Converted tool '{metadata.name}' to LangChain Tool")
            except Exception as e:
                logger.error(f"Failed to convert tool '{metadata.name}' to LangChain Tool: {e}")

        logger.info(f"Generated {len(langchain_tools)} LangChain tools")
        return langchain_tools

    def auto_discover_utilities(self, utils_dir: Optional[Path] = None) -> int:
        """
        Scan utils/ directory and auto-register common utility functions.

        Discovers Python modules in the utils directory and automatically
        registers functions that have proper docstrings and type hints.

        Args:
            utils_dir: Optional path to utils directory (defaults to project utils/)

        Returns:
            Number of tools discovered and registered
        """
        if utils_dir is None:
            # Default to project utils directory
            project_root = Path(__file__).parent.parent
            utils_dir = project_root / "utils"

        if not utils_dir.exists():
            logger.warning(f"Utils directory not found: {utils_dir}")
            return 0

        discovered_count = 0

        # Scan Python files in utils directory
        for py_file in utils_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "tool_registry.py":
                continue

            try:
                # Import module dynamically
                module_name = py_file.stem
                spec = __import__(f"utils.{module_name}")
                module = getattr(spec, module_name)

                # Inspect functions in module
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if name.startswith("_"):
                        continue

                    # Check for docstring
                    if not obj.__doc__:
                        logger.debug(f"Skipping {module_name}.{name}: no docstring")
                        continue

                    # Extract description from docstring
                    description = obj.__doc__.strip().split("\n")[0]

                    # Infer category from module name
                    category = self._infer_category(module_name)

                    # Register the tool
                    tool_name = f"{module_name}.{name}"
                    self.register_tool(
                        name=tool_name,
                        tool=obj,
                        description=description,
                        category=category,
                    )
                    discovered_count += 1

            except Exception as e:
                logger.error(f"Failed to discover tools from {py_file}: {e}")

        logger.info(f"Auto-discovered {discovered_count} tools from {utils_dir}")
        return discovered_count

    def _infer_category(self, module_name: str) -> str:
        """
        Infer tool category from module name.

        Args:
            module_name: Name of the Python module

        Returns:
            Inferred category string
        """
        category_map = {
            "file": "filesystem",
            "fs": "filesystem",
            "filesystem": "filesystem",
            "web": "web",
            "http": "web",
            "search": "web",
            "model": "models",
            "ollama": "models",
            "llm": "models",
            "db": "database",
            "database": "database",
            "vector": "database",
            "workflow": "workflow",
            "state": "workflow",
            "agent": "workflow",
        }

        for key, category in category_map.items():
            if key in module_name.lower():
                return category

        return "other"

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Export registry to JSON format.

        Args:
            filepath: Optional path to write JSON file

        Returns:
            JSON string representation of the registry
        """
        registry_data = {
            "tools": self.list_tools(),
            "categories": self._get_category_summary(),
            "total_tools": len(self._tools),
        }

        json_str = json.dumps(registry_data, indent=2)

        if filepath:
            filepath = Path(filepath)
            filepath.write_text(json_str)
            logger.info(f"Exported registry to {filepath}")

        return json_str

    def _get_category_summary(self) -> Dict[str, int]:
        """
        Get summary of tools by category.

        Returns:
            Dictionary mapping category names to tool counts
        """
        summary: Dict[str, int] = {}
        for metadata in self._tools.values():
            summary[metadata.category] = summary.get(metadata.category, 0) + 1
        return summary

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._tools.clear()
        logger.info("Cleared all registered tools")

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return f"ToolRegistry(tools={len(self._tools)}, categories={list(self._get_category_summary().keys())})"


# Convenience function for global access
def get_registry() -> ToolRegistry:
    """
    Get the global ToolRegistry instance.

    Returns:
        The singleton ToolRegistry instance
    """
    return ToolRegistry.get_instance()
