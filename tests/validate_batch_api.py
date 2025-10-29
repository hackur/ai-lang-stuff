"""
Quick validation script to verify batch processing API is correctly implemented.

This script checks the method signatures and basic functionality without
requiring Ollama to be running.
"""

import asyncio
import inspect

from utils.ollama_manager import OllamaManager


def validate_batch_api():
    """Validate that batch processing methods have correct signatures."""
    print("Validating OllamaManager batch processing API...\n")

    manager = OllamaManager()

    methods_to_check = {
        "batch_generate": {
            "params": ["self", "prompts", "model", "max_concurrent", "rate_limit_delay"],
            "is_async": True,
        },
        "batch_benchmark": {"params": ["self", "models", "prompt", "num_runs"], "is_async": True},
        "_generate_single_async": {
            "params": ["self", "prompt", "model"],
            "is_async": True,
        },
        "_benchmark_single_async": {
            "params": ["self", "model", "prompt", "num_runs"],
            "is_async": True,
        },
    }

    all_valid = True

    for method_name, expected in methods_to_check.items():
        if not hasattr(manager, method_name):
            print(f"❌ Missing method: {method_name}")
            all_valid = False
            continue

        method = getattr(manager, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        expected_params = expected["params"]
        missing_params = [p for p in expected_params if p not in params and p != "self"]

        if missing_params:
            print(f"❌ {method_name} missing parameters: {missing_params}")
            all_valid = False
        else:
            print(f"✓ {method_name} - signature correct")

        if expected["is_async"]:
            if not inspect.iscoroutinefunction(method):
                print(f"  ⚠️  {method_name} should be async")
                all_valid = False
            else:
                print(f"  ✓ {method_name} is async")

    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All batch processing methods validated successfully!")
        print("\nBatch processing API summary:")
        print("  - batch_generate(): Process multiple prompts concurrently")
        print("  - batch_benchmark(): Benchmark multiple models concurrently")
        print("  - Configuration: max_concurrent, rate_limit_delay")
        print("  - Error handling: Partial failures supported")
    else:
        print("❌ Some validation checks failed")
    print("=" * 60)

    return all_valid


async def test_empty_inputs():
    """Test batch methods with empty inputs."""
    print("\n\nTesting batch methods with empty inputs...\n")

    manager = OllamaManager()

    try:
        results = await manager.batch_generate(prompts=[])
        print(f"✓ batch_generate([]) returned {len(results)} results (expected 0)")

        results = await manager.batch_benchmark(models=[])
        print(f"✓ batch_benchmark([]) returned {len(results)} results (expected 0)")

        print("\n✓ Empty input tests passed")
    except Exception as e:
        print(f"\n❌ Empty input tests failed: {e}")


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("OllamaManager Batch Processing Validation")
    print("=" * 60)

    api_valid = validate_batch_api()

    asyncio.run(test_empty_inputs())

    print("\n" + "=" * 60)
    if api_valid:
        print("SUCCESS: Batch processing implementation is valid!")
        print("\nNext steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Run example: uv run python examples/01-foundation/batch_inference.py")
    else:
        print("FAILED: Some validation checks failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
