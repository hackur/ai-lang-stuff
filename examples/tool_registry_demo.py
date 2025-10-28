"""
Demonstration of ToolRegistry usage.

This example shows how to use the centralized ToolRegistry to:
1. Register custom tools
2. List and filter tools by category
3. Export registry to JSON
4. Convert tools to LangChain format (when dependencies available)

Prerequisites:
- Python 3.11+
- No dependencies required for basic usage
- langchain-core required for LangChain tool conversion

Expected output:
- Tool registration confirmations
- Tool listings by category
- JSON export of registry
"""

import json
from pathlib import Path

from utils.tool_registry import get_registry


# Example tools to register
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return f"Searching for '{query}' (max {max_results} results)..."


def read_file(filepath: str) -> str:
    """Read contents of a file."""
    return f"Reading file: {filepath}"


def query_vector_db(query: str, top_k: int = 3) -> list:
    """Query a vector database."""
    return [f"Result {i} for '{query}'" for i in range(1, top_k + 1)]


def run_ollama_model(model: str, prompt: str) -> str:
    """Run inference with an Ollama model."""
    return f"Running {model} with prompt: {prompt}"


def main():
    """Demonstrate ToolRegistry usage."""
    print("=" * 60)
    print("ToolRegistry Demonstration")
    print("=" * 60)
    print()

    # Get the singleton registry instance
    registry = get_registry()
    registry.clear()  # Start fresh for demo

    print("1. Registering tools...")
    print("-" * 60)

    # Register tools with different categories
    registry.register_tool(
        name="search_web",
        tool=search_web,
        description="Search the web for information",
        category="web",
    )

    registry.register_tool(
        name="read_file",
        tool=read_file,
        description="Read contents of a file",
        category="filesystem",
    )

    registry.register_tool(
        name="query_vector_db",
        tool=query_vector_db,
        description="Query a vector database",
        category="database",
    )

    registry.register_tool(
        name="run_ollama_model",
        tool=run_ollama_model,
        description="Run inference with an Ollama model",
        category="models",
    )

    print(f"Registered {len(registry)} tools")
    print()

    print("2. Listing all tools...")
    print("-" * 60)
    all_tools = registry.list_tools()
    for tool in all_tools:
        print(f"  - {tool['name']}: {tool['description']} [{tool['category']}]")
    print()

    print("3. Filtering by category...")
    print("-" * 60)
    categories = ["web", "filesystem", "database", "models"]
    for category in categories:
        tools = registry.list_tools(category=category)
        print(f"  {category}: {len(tools)} tool(s)")
        for tool in tools:
            print(f"    - {tool['name']}")
    print()

    print("4. Testing tool execution...")
    print("-" * 60)
    search_tool = registry.get_tool("search_web")
    result = search_tool("Python tutorials", max_results=3)
    print(f"  Result: {result}")
    print()

    print("5. Exporting to JSON...")
    print("-" * 60)
    json_output = registry.to_json()
    data = json.loads(json_output)
    print(f"  Total tools: {data['total_tools']}")
    print(f"  Categories: {list(data['categories'].keys())}")
    print("  Category breakdown:")
    for cat, count in data["categories"].items():
        print(f"    - {cat}: {count} tool(s)")
    print()

    # Optionally save to file
    output_path = Path("tool_registry.json")
    registry.to_json(filepath=output_path)
    print(f"  Exported to: {output_path}")
    print()

    print("6. Converting to LangChain tools (if available)...")
    print("-" * 60)
    try:
        # This requires langchain-core to be installed
        lc_tools = registry.get_langchain_tools(categories=["web", "models"])
        print(f"  Converted {len(lc_tools)} tools to LangChain format")
        for tool in lc_tools:
            print(f"    - {tool.name}: {tool.description}")
    except ImportError as e:
        print(f"  Skipped: {e}")
    print()

    print("=" * 60)
    print(f"Registry state: {repr(registry)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
