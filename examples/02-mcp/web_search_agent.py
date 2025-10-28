"""
Web Search Agent using MCP Server Integration.

This example demonstrates how to create an AI agent with web search and URL fetching
capabilities through the MCP web search server. The agent can search the web,
fetch URL content, and synthesize information from multiple sources.

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`
- MCP web search server available (uses async client)

Expected output:
The agent will search the web based on natural language queries, fetch relevant
URLs, and provide synthesized answers. You'll see search results and the agent's
analysis of the information found.

Usage:
    uv run python examples/02-mcp/web_search_agent.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from utils import MCPConfig, OllamaManager, WebSearchMCP


def main():
    """Main execution function."""
    print("=" * 70)
    print("Web Search Agent with MCP Integration")
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

    # Step 2: Initialize MCP web search client
    print("Step 2/4: Setting up MCP web search client...")
    print("-" * 70)

    try:
        # Configure MCP client for web search
        config = MCPConfig(
            host="localhost",
            port=8002,  # Web search MCP server port
            timeout=60.0,  # Longer timeout for web operations
            max_retries=3,
        )

        # Create web search client
        search_client = WebSearchMCP(config=config)

        print("✓ MCP web search client initialized")
        print(f"  Server: {config.base_url}")
        print(f"  Timeout: {config.timeout}s")
    except Exception as e:
        print(f"✗ Failed to initialize MCP client: {e}")
        print("\nNote: This example requires an MCP web search server running.")
        print("For demonstration, we'll continue with simulated tools.")
        print()
        search_client = None
    print()

    # Step 3: Create agent with web search tools
    print("Step 3/4: Creating agent with web search capabilities...")
    print("-" * 70)

    # Initialize the chat model
    # Using qwen3:8b for balanced performance and quality
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.5,  # Moderate temperature for creative but focused research
    )

    # Create LangChain tools from MCP client
    if search_client:
        tools = search_client.to_langchain_tools()
        print(f"✓ Created {len(tools)} web search tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    else:
        # Create mock tools for demonstration
        from langchain_core.tools import Tool

        def mock_search(query: str) -> str:
            """Mock web search."""
            return """1. LangChain Documentation
   URL: https://python.langchain.com/
   Comprehensive documentation for LangChain Python library...

2. LangChain GitHub Repository
   URL: https://github.com/langchain-ai/langchain
   Official GitHub repository with source code and examples..."""

        tools = [
            Tool(
                name="web_search",
                description="Search the web for information. Input: search query string.",
                func=mock_search,
            )
        ]
        print(f"✓ Created {len(tools)} mock web search tools (for demo)")

    # Create agent prompt with research focus
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI research assistant with web search capabilities.

You can:
- Search the web using web_search
- Fetch content from URLs using fetch_url

When researching a topic:
1. Break down the query into specific search terms
2. Search for relevant information
3. Fetch detailed content from promising URLs if needed
4. Synthesize findings into a clear, comprehensive answer
5. Cite your sources with URLs

Guidelines:
- Search for multiple perspectives when appropriate
- Verify information across multiple sources
- Provide URLs for users to verify information
- Be clear about the recency of information
- Acknowledge if information might be outdated

Be thorough but concise in your responses.""",
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
        verbose=True,  # Show reasoning and search process
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # Return search results
    )

    print("✓ Agent created and ready")
    print()

    # Step 4: Execute web search tasks
    print("Step 4/4: Executing web search tasks...")
    print("-" * 70)
    print()

    # Example research task
    research_query = "What are the latest features in LangChain for local LLM integration?"

    print(f"Research Query: {research_query}")
    print()
    print("Agent execution:")
    print("=" * 70)

    try:
        result = agent_executor.invoke({"input": research_query})

        print("=" * 70)
        print()
        print("Research Results:")
        print("-" * 70)
        print(result["output"])
        print("-" * 70)

        # Show intermediate steps if available
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print()
            print("Search Process:")
            print("-" * 70)
            for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                print(f"\nStep {i}: {action.tool}")
                print(f"Input: {action.tool_input}")
                print(
                    f"Result: {observation[:200]}..."
                    if len(observation) > 200
                    else f"Result: {observation}"
                )
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
    print("- Try the filesystem_agent.py example for file operations")
    print("- Try the combined_tools_agent.py for using multiple capabilities")
    print("- Modify the research query to explore different topics")
    print("- Experiment with different temperature settings for varied responses")


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
        print("3. Check if MCP web search server is running on port 8002")
        print("4. Verify internet connectivity for web searches")
        print("5. Check MCP server logs for detailed error messages")
        sys.exit(1)
