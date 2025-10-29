"""
Example: Systematic Model Comparison and Benchmarking

Purpose:
    Compares multiple local LLMs on standardized tasks to help select the optimal model
    for specific use cases, measuring response quality, speed, and consistency across
    different model sizes and architectures.

Prerequisites:
    - Ollama running with models: qwen3:8b, qwen3:30b-a3b, gemma3:4b
    - Python packages: langchain-ollama
    - Estimated 3-5 minutes runtime (depends on models)

Expected Output:
    Performance comparison showing response time, response length, and quality preview
    for each model on a coding task. Summary includes fastest model and most detailed
    responses to inform model selection decisions.

Usage:
    uv run python examples/01-foundation/compare_models.py
"""

import time

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def test_model(model_name: str, prompt: str) -> tuple[str, float]:
    """Test a model and return response with timing."""
    print(f"\nTesting {model_name}...")

    llm = ChatOllama(model=model_name, base_url="http://localhost:11434")

    start_time = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed_time = time.time() - start_time

    return response.content, elapsed_time


def main():
    # Test prompt
    prompt = "Write a Python function to calculate the fibonacci sequence using memoization."

    # Models to test
    models = ["qwen3:8b", "qwen3:30b-a3b", "gemma3:4b"]

    print("Comparing models on coding task:")
    print("=" * 80)

    results = []

    for model in models:
        try:
            response, elapsed = test_model(model, prompt)
            results.append((model, response, elapsed))
        except Exception as e:
            print(f"Error with {model}: {e}")
            print(f"Make sure model is installed: ollama pull {model}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for model, response, elapsed in results:
        print(f"\nModel: {model}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...")
        print("-" * 80)

    # Summary
    if results:
        print("\nSUMMARY")
        print("-" * 80)
        fastest = min(results, key=lambda x: x[2])
        print(f"Fastest: {fastest[0]} ({fastest[2]:.2f}s)")

        longest_response = max(results, key=lambda x: len(x[1]))
        print(f"Most detailed: {longest_response[0]} ({len(longest_response[1])} chars)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Are models installed? Try: ollama list")
        print("3. Pull missing models: ollama pull <model-name>")
