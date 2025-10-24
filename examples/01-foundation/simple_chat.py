"""
Simple chat example using Ollama and LangChain.

This demonstrates the most basic usage of a local LLM.

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`

Expected output:
Clear explanation of list comprehensions with examples.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


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
