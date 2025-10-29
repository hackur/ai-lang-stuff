"""
Example: Batch Inference with Ollama

Purpose:
    Demonstrates batch processing capabilities for LLM inference using the
    OllamaManager. Shows significant performance improvements (3-5x) when
    processing multiple prompts concurrently versus sequential processing.

Prerequisites:
    - Ollama running with qwen3:8b model
    - For benchmark comparison: gemma3:4b (optional but recommended)

Expected Output:
    - Batch generation results for 10+ prompts
    - Performance comparison: sequential vs concurrent
    - Multi-model benchmark comparison
    - Timing statistics showing speedup

Usage:
    uv run python examples/01-foundation/batch_inference.py
"""

import asyncio
import logging
import time

from utils.ollama_manager import OllamaManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_batch_generation():
    """Demonstrate batch generation with concurrent processing."""
    print("\n" + "=" * 80)
    print("DEMO 1: Batch Generation with Concurrent Processing")
    print("=" * 80)

    manager = OllamaManager()

    if not manager.check_ollama_running():
        print("\nError: Ollama is not running. Please start Ollama first:")
        print("  ollama serve")
        return

    model = "qwen3:8b"
    if not manager.ensure_model_available(model):
        print(f"\nError: Could not ensure model '{model}' is available")
        return

    prompts = [
        "What is Python?",
        "Explain asyncio in one sentence.",
        "What is machine learning?",
        "Define artificial intelligence.",
        "What is a neural network?",
        "Explain deep learning briefly.",
        "What is natural language processing?",
        "Define computer vision.",
        "What is reinforcement learning?",
        "Explain supervised learning.",
        "What is unsupervised learning?",
        "Define transfer learning.",
    ]

    print(f"\nProcessing {len(prompts)} prompts using model: {model}")
    print("Max concurrent requests: 5")

    start_time = time.time()
    responses = await manager.batch_generate(
        prompts=prompts, model=model, max_concurrent=5, rate_limit_delay=0.05
    )
    batch_duration = time.time() - start_time

    print(f"\nBatch processing completed in {batch_duration:.2f} seconds")
    print(f"Average time per prompt: {batch_duration / len(prompts):.2f} seconds")

    successful = sum(1 for r in responses if r.get("success", False))
    print(f"Success rate: {successful}/{len(prompts)} ({successful/len(prompts)*100:.1f}%)")

    print("\n" + "-" * 80)
    print("Sample Responses (first 3):")
    print("-" * 80)
    for i, resp in enumerate(responses[:3]):
        if resp["success"]:
            print(f"\n[{i+1}] Prompt: {resp['prompt']}")
            print(f"Response: {resp['response'][:200]}...")
        else:
            print(f"\n[{i+1}] Prompt: {resp['prompt']}")
            print(f"Error: {resp.get('error', 'Unknown error')}")

    return batch_duration, len(prompts)


async def demo_sequential_vs_concurrent():
    """Compare sequential vs concurrent processing performance."""
    print("\n" + "=" * 80)
    print("DEMO 2: Sequential vs Concurrent Performance Comparison")
    print("=" * 80)

    manager = OllamaManager()

    model = "qwen3:8b"
    test_prompts = [
        "Count from 1 to 5.",
        "Name three colors.",
        "What is 2 + 2?",
        "Say hello in Spanish.",
        "Name a programming language.",
        "What day comes after Monday?",
        "How many legs does a cat have?",
        "What is the capital of France?",
    ]

    print(f"\nTesting with {len(test_prompts)} prompts")

    print("\n1. Sequential Processing...")
    sequential_start = time.time()

    sequential_responses = []
    for prompt in test_prompts:
        responses = await manager.batch_generate(prompts=[prompt], model=model, max_concurrent=1)
        sequential_responses.extend(responses)

    sequential_duration = time.time() - sequential_start

    print(f"   Sequential time: {sequential_duration:.2f} seconds")
    print(f"   Average per prompt: {sequential_duration / len(test_prompts):.2f} seconds")

    print("\n2. Concurrent Processing...")
    concurrent_start = time.time()

    concurrent_responses = await manager.batch_generate(
        prompts=test_prompts, model=model, max_concurrent=5
    )

    concurrent_duration = time.time() - concurrent_start

    print(f"   Concurrent time: {concurrent_duration:.2f} seconds")
    print(f"   Average per prompt: {concurrent_duration / len(test_prompts):.2f} seconds")

    speedup = sequential_duration / concurrent_duration if concurrent_duration > 0 else 0
    print(f"\n   Speedup: {speedup:.2f}x faster")

    if speedup >= 3.0:
        print("   Excellent! Achieved 3x+ speedup target")
    elif speedup >= 2.0:
        print("   Good speedup achieved")
    else:
        print("   Note: Speedup may vary based on model size and hardware")

    return speedup


async def demo_batch_benchmark():
    """Demonstrate multi-model benchmarking."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multi-Model Batch Benchmarking")
    print("=" * 80)

    manager = OllamaManager()

    available_models = manager.list_models()
    print(f"\nAvailable models: {available_models}")

    test_models = []
    for model in ["qwen3:8b", "gemma3:4b", "qwen3:30b-a3b"]:
        if model in available_models:
            test_models.append(model)

    if not test_models:
        print("\nNote: No test models available for benchmarking")
        print("Install recommended models:")
        print("  ollama pull qwen3:8b")
        print("  ollama pull gemma3:4b")
        return

    print(f"\nBenchmarking {len(test_models)} models: {test_models}")
    print("Running 3 iterations per model for averaging...")

    test_prompt = "Explain what a neural network is in one sentence."

    start_time = time.time()
    results = await manager.batch_benchmark(models=test_models, prompt=test_prompt, num_runs=3)
    benchmark_duration = time.time() - start_time

    print(f"\nBenchmark completed in {benchmark_duration:.2f} seconds")

    print("\n" + "-" * 80)
    print("Benchmark Results:")
    print("-" * 80)
    print(f"{'Model':<20} {'Latency (s)':<15} {'Tokens/sec':<15} {'Status'}")
    print("-" * 80)

    for model_name, result in sorted(results.items(), key=lambda x: x[1].get("latency", 999)):
        if "error" in result:
            print(f"{model_name:<20} {'N/A':<15} {'N/A':<15} Error: {result['error'][:30]}")
        else:
            latency = result["latency"]
            tokens_per_sec = result["tokens_per_sec"]
            print(f"{model_name:<20} {latency:<15.2f} {tokens_per_sec:<15.1f} OK")

    print("-" * 80)

    if len(results) >= 2:
        sorted_results = sorted(results.items(), key=lambda x: x[1].get("latency", 999))
        fastest = sorted_results[0]
        print(f"\nFastest model: {fastest[0]} ({fastest[1]['latency']:.2f}s)")

        if fastest[1].get("tokens_per_sec", 0) > 0:
            print(f"Throughput: {fastest[1]['tokens_per_sec']:.1f} tokens/second")


async def main():
    """Run all batch processing demonstrations."""
    print("\n" + "=" * 80)
    print("Batch Inference Demonstration")
    print("=" * 80)

    try:
        batch_duration, num_prompts = await demo_batch_generation()

        speedup = await demo_sequential_vs_concurrent()

        await demo_batch_benchmark()

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Successfully processed {num_prompts} prompts in batch")
        print(f"Concurrent processing achieved {speedup:.2f}x speedup")
        print("\nKey Takeaways:")
        print("- Batch processing significantly improves throughput")
        print("- Concurrent requests reduce total processing time")
        print("- Rate limiting prevents overwhelming the server")
        print("- Error handling ensures partial failures don't stop processing")

    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Please ensure Ollama is running and models are available")


if __name__ == "__main__":
    asyncio.run(main())
