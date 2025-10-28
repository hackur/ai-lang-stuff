"""
Combined Tools Agent using Multiple MCP Servers.

This example demonstrates how to create an AI agent that uses multiple MCP servers
simultaneously - both filesystem and web search capabilities. The agent can research
information online and save results to files, or read local files and search for
related information on the web.

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`
- MCP filesystem server available on port 8001
- MCP web search server available on port 8002

Expected output:
The agent will combine web research and filesystem operations to complete complex
tasks, such as researching a topic and saving the findings to a file, or reading
a file and searching for related information online.

Usage:
    uv run python examples/02-mcp/combined_tools_agent.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from utils import FilesystemMCP, MCPConfig, OllamaManager, WebSearchMCP


def main():
    """Main execution function."""
    print("=" * 70)
    print("Combined Tools Agent with Multiple MCP Servers")
    print("=" * 70)
    print()

    # Step 1: Check Ollama prerequisites
    print("Step 1/5: Checking prerequisites...")
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
    print("Step 2/5: Setting up MCP filesystem client...")
    print("-" * 70)

    fs_tools = []
    try:
        # Configure filesystem MCP client
        fs_config = MCPConfig(host="localhost", port=8001, timeout=30.0)

        # Create filesystem client with base path restriction
        base_path = Path(__file__).parent.parent  # Restrict to examples/ directory
        fs_client = FilesystemMCP(config=fs_config, base_path=base_path)

        # Get filesystem tools
        fs_tools = fs_client.to_langchain_tools()

        print("✓ Filesystem MCP client initialized")
        print(f"  Base path: {base_path}")
        print(f"  Tools: {len(fs_tools)}")

    except Exception as e:
        print(f"⚠ Filesystem MCP client unavailable: {e}")
        print("  Continuing with mock tools...")

        # Create mock filesystem tools
        from langchain_core.tools import Tool

        def mock_list_dir(path: str) -> str:
            return "01-foundation/\n02-mcp/\n03-multi-agent/"

        def mock_write_file(args: dict) -> str:
            return f"Mock: Would write to {args.get('path', 'unknown')}"

        fs_tools = [
            Tool(
                name="list_directory",
                description="List contents of a directory. Input: directory path as string.",
                func=mock_list_dir,
            ),
            Tool(
                name="write_file",
                description="Write content to a file. Input: JSON with 'path' and 'content' keys.",
                func=mock_write_file,
            ),
        ]
        print(f"✓ Created {len(fs_tools)} mock filesystem tools")
    print()

    # Step 3: Initialize MCP web search client
    print("Step 3/5: Setting up MCP web search client...")
    print("-" * 70)

    search_tools = []
    try:
        # Configure web search MCP client
        search_config = MCPConfig(host="localhost", port=8002, timeout=60.0, max_retries=3)

        # Create web search client
        search_client = WebSearchMCP(config=search_config)

        # Get search tools
        search_tools = search_client.to_langchain_tools()

        print("✓ Web search MCP client initialized")
        print(f"  Server: {search_config.base_url}")
        print(f"  Tools: {len(search_tools)}")

    except Exception as e:
        print(f"⚠ Web search MCP client unavailable: {e}")
        print("  Continuing with mock tools...")

        # Create mock search tools
        from langchain_core.tools import Tool

        def mock_search(query: str) -> str:
            return f"Mock search results for: {query}\n1. Result 1\n2. Result 2"

        search_tools = [
            Tool(
                name="web_search",
                description="Search the web for information. Input: search query string.",
                func=mock_search,
            )
        ]
        print(f"✓ Created {len(search_tools)} mock search tools")
    print()

    # Step 4: Create agent with all tools
    print("Step 4/5: Creating agent with combined capabilities...")
    print("-" * 70)

    # Combine all tools
    all_tools = fs_tools + search_tools

    print(f"✓ Combined {len(all_tools)} tools total:")
    print("\n  Filesystem capabilities:")
    for tool in fs_tools:
        print(f"    - {tool.name}")
    print("\n  Web search capabilities:")
    for tool in search_tools:
        print(f"    - {tool.name}")
    print()

    # Initialize the chat model
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.4,  # Balanced for research and file operations
    )

    # Create comprehensive agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a powerful AI assistant with both filesystem and web search capabilities.

Your capabilities:

FILESYSTEM OPERATIONS:
- read_file: Read contents of files
- write_file: Write content to files (saves research, notes, etc.)
- list_directory: List directory contents
- search_files: Find files matching patterns

WEB RESEARCH:
- web_search: Search the web for information
- fetch_url: Fetch content from specific URLs

You excel at complex workflows that combine research and documentation:

Example workflows:
1. Research a topic → Write findings to file
2. Read a file → Search web for related updates
3. Search files for patterns → Research related topics → Update files
4. Gather information from multiple sources → Compile into document

Guidelines:
- Break complex tasks into clear steps
- Use web search for current information
- Use filesystem for persistence and documentation
- Cite sources when researching
- Organize output in clear, readable formats
- Verify paths before writing files
- Provide progress updates for multi-step tasks

Be thorough, organized, and user-focused.""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,  # Show full reasoning process
        max_iterations=8,  # Allow more iterations for complex workflows
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    print("✓ Agent created and ready")
    print()

    # Step 5: Execute combined workflow task
    print("Step 5/5: Executing combined workflow...")
    print("-" * 70)
    print()

    # Example task that requires both web search and filesystem
    task = """Research the latest best practices for using local LLMs with LangChain,
    focusing on Ollama integration. Then, list the example directories we have
    and tell me which ones would benefit from updates based on your findings."""

    print("Complex Task:")
    print(f"{task}")
    print()
    print("Agent execution:")
    print("=" * 70)

    try:
        result = agent_executor.invoke({"input": task})

        print("=" * 70)
        print()
        print("Final Results:")
        print("-" * 70)
        print(result["output"])
        print("-" * 70)

        # Show workflow steps
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print()
            print("Workflow Breakdown:")
            print("-" * 70)
            for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                print(f"\nStep {i}: Used {action.tool}")
                print(
                    f"Purpose: {action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)[:100]}"
                )
                # Show abbreviated observation
                obs_preview = observation[:150] + "..." if len(observation) > 150 else observation
                print(f"Result: {obs_preview}")
            print("-" * 70)

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        print("\nThis may be due to MCP servers not running.")

    print()
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("-" * 70)
    print("1. Multiple MCP servers can be combined seamlessly")
    print("2. Agents can orchestrate complex multi-tool workflows")
    print("3. Web research + filesystem = powerful documentation pipeline")
    print("4. Tool composition enables sophisticated automation")
    print()
    print("Next steps:")
    print("- Experiment with different task combinations")
    print("- Try saving research results to actual files")
    print("- Create a workflow that monitors files and searches for updates")
    print("- Build a documentation system that auto-updates from web research")


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
        print("3. Check filesystem MCP server on port 8001")
        print("4. Check web search MCP server on port 8002")
        print("5. Verify network connectivity for web searches")
        print("6. Check file permissions for base_path directory")
        sys.exit(1)
