"""
Filesystem Agent using MCP Server Integration.

This example demonstrates how to create an AI agent with filesystem access through
the MCP (Model Context Protocol) filesystem server. The agent can read files,
write files, list directories, and search for files using natural language commands.

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`
- MCP filesystem server available (uses async client)

Expected output:
The agent will execute filesystem operations based on natural language requests,
such as reading a file, listing directory contents, or searching for files.
You'll see the agent's reasoning process and the results of filesystem operations.

Usage:
    uv run python examples/02-mcp/filesystem_agent.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from utils import OllamaManager, FilesystemMCP, MCPConfig


def main():
    """Main execution function."""
    print("=" * 70)
    print("Filesystem Agent with MCP Integration")
    print("=" * 70)
    print()

    # Step 1: Check Ollama prerequisites
    print("Step 1/4: Checking prerequisites...")
    print("-" * 70)

    manager = OllamaManager()
    if not manager.check_ollama_running():
        print("✗ Ollama is not running")
        print("\nPlease start Ollama with: ollama serve")
        return
    print("✓ Ollama is running")

    # Ensure model is available
    model_name = "qwen3:8b"
    if not manager.ensure_model_available(model_name):
        print(f"✗ Model {model_name} not available")
        print(f"\nPlease download with: ollama pull {model_name}")
        return
    print(f"✓ Model {model_name} is available")
    print()

    # Step 2: Initialize MCP filesystem client
    print("Step 2/4: Setting up MCP filesystem client...")
    print("-" * 70)

    try:
        # Configure MCP client - restrict to examples directory for safety
        config = MCPConfig(
            host="localhost",
            port=8001,  # Filesystem MCP server port
            timeout=30.0,
        )

        # Note: This example shows the synchronous pattern using async client
        # In production, you'd typically run this in an async context

        # Create filesystem client with base path restriction
        base_path = Path(__file__).parent.parent  # Restrict to examples/ directory
        fs_client = FilesystemMCP(config=config, base_path=base_path)

        print("✓ MCP filesystem client initialized")
        print(f"  Base path: {base_path}")
        print(f"  Server: {config.base_url}")
    except Exception as e:
        print(f"✗ Failed to initialize MCP client: {e}")
        print("\nNote: This example requires an MCP filesystem server running.")
        print("For demonstration, we'll continue with simulated tools.")
        print()
        # In a real scenario, you might want to exit here
        # For demo purposes, we'll create mock tools
        fs_client = None
    print()

    # Step 3: Create agent with filesystem tools
    print("Step 3/4: Creating agent with filesystem capabilities...")
    print("-" * 70)

    # Initialize the chat model
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.3,  # Lower temperature for more focused file operations
    )

    # Create LangChain tools from MCP client
    if fs_client:
        tools = fs_client.to_langchain_tools()
        print(f"✓ Created {len(tools)} filesystem tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    else:
        # Create mock tools for demonstration
        from langchain_core.tools import Tool

        def mock_list_dir(path: str) -> str:
            """Mock directory listing."""
            return "01-foundation/\n02-mcp/\n03-multi-agent/\n04-rag/"

        tools = [
            Tool(
                name="list_directory",
                description="List contents of a directory. Input: directory path as string.",
                func=mock_list_dir,
            )
        ]
        print(f"✓ Created {len(tools)} mock filesystem tools (for demo)")

    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant with filesystem access capabilities.

You can:
- Read file contents using read_file
- Write content to files using write_file
- List directory contents using list_directory
- Search for files using search_files

When given a task:
1. Break it down into clear steps
2. Use the appropriate tools
3. Provide clear explanations of what you're doing
4. Show results in a user-friendly format

Be concise but thorough in your responses.""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Show reasoning process
        max_iterations=5,
        handle_parsing_errors=True,
    )

    print("✓ Agent created and ready")
    print()

    # Step 4: Execute filesystem tasks
    print("Step 4/4: Executing filesystem tasks...")
    print("-" * 70)
    print()

    # Example task: List the examples directory
    task = "List all directories in the current examples folder and tell me what you find."

    print(f"Task: {task}")
    print()
    print("Agent execution:")
    print("=" * 70)

    try:
        result = agent_executor.invoke({"input": task})

        print("=" * 70)
        print()
        print("Final Result:")
        print("-" * 70)
        print(result["output"])
        print("-" * 70)

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        print("\nThis may be due to MCP server not running.")

    print()
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("- Try the web_search_agent.py example for web search capabilities")
    print("- Try the combined_tools_agent.py for using multiple MCP servers")
    print("- Modify the task to explore different filesystem operations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Verify model is installed: ollama pull qwen3:8b")
        print("3. Check if MCP server is running on port 8001")
        print("4. Verify base path permissions")
        sys.exit(1)
