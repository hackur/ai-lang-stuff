"""
Streaming chat example showing token-by-token response.

This demonstrates how to get streaming responses from local models,
which provides a better user experience for long responses.

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`

Expected output:
Tokens appearing one at a time in real-time.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def main():
    print("Initializing local model for streaming...")

    # Initialize chat model
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.8,  # Higher temperature for more creative haiku
    )

    # Create message
    message = HumanMessage(content="Write a haiku about artificial intelligence.")

    print("Streaming response from qwen3:8b:")
    print("-" * 60)

    # Stream response token by token
    for chunk in llm.stream([message]):
        print(chunk.content, end="", flush=True)

    print()
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
