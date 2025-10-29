"""
Example: Basic Chat Interaction with Local LLM

Purpose:
    Demonstrates the simplest possible interaction with a local LLM using Ollama,
    serving as the foundation for understanding local model communication and
    LangChain message formatting.

Prerequisites:
    - Ollama running with qwen3:8b model
    - Python packages: langchain-ollama
    - Estimated 30 seconds runtime

Expected Output:
    A clear explanation of Python list comprehensions with examples, demonstrating
    successful Ollama connection and basic inference with system and user message
    formatting.

Usage:
    uv run python examples/01-foundation/simple_chat.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


def main():
    print("Initializing local model...")

    # Initialize chat model
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    # Create messages
    messages = [
        SystemMessage(content="You are a helpful Python programming assistant."),
        HumanMessage(content="Explain list comprehensions in Python with an example."),
    ]

    print("Sending request to local model...")
    print()

    # Get response
    response = llm.invoke(messages)

    # Print response
    print("Response from qwen3:8b:")
    print("-" * 60)
    print(response.content)
    print("-" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Is the model installed? Try: ollama pull qwen3:8b")
        print("3. Check server: curl http://localhost:11434/api/tags")
