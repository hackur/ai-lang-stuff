# Batch Processing Quick Reference

## Quick Start

```python
import asyncio
from utils.ollama_manager import OllamaManager

async def main():
    manager = OllamaManager()

    # Batch generate
    results = await manager.batch_generate(
        prompts=["Question 1", "Question 2", "Question 3"],
        model="qwen3:8b",
        max_concurrent=5
    )

    # Batch benchmark
    benchmarks = await manager.batch_benchmark(
        models=["qwen3:8b", "gemma3:4b"],
        prompt="Test prompt",
        num_runs=3
    )

asyncio.run(main())
```

## Method Signatures

### batch_generate

```python
async def batch_generate(
    prompts: list[str],              # List of prompts to process
    model: str = "qwen3:8b",         # Model to use
    max_concurrent: int = 5,         # Max concurrent requests
    rate_limit_delay: float = 0.1,   # Delay between batches (seconds)
    **generate_kwargs: Any           # Extra args for Ollama API
) -> list[BatchResponse]
```

Returns list of dicts with:
- `prompt`: Original prompt
- `response`: Model response
- `model`: Model name
- `success`: True/False
- `error`: Error message (if failed)

### batch_benchmark

```python
async def batch_benchmark(
    models: list[str],                      # Models to benchmark
    prompt: str = "Hello, how are you?",    # Test prompt
    num_runs: int = 3                       # Runs per model
) -> dict[str, BenchmarkResult]
```

Returns dict of model -> stats:
- `latency`: Average response time (seconds)
- `tokens_per_sec`: Throughput
- `response`: Sample response
- `num_runs`: Number of runs

## Common Patterns

### Process multiple prompts
```python
prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
results = await manager.batch_generate(prompts)

for r in results:
    if r["success"]:
        print(f"{r['prompt']}: {r['response']}")
```

### Compare model performance
```python
models = ["qwen3:8b", "gemma3:4b"]
results = await manager.batch_benchmark(models)

for model, stats in results.items():
    print(f"{model}: {stats['tokens_per_sec']:.1f} tok/s")
```

### Handle failures
```python
results = await manager.batch_generate(prompts)

successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"Success: {len(successful)}/{len(results)}")
```

### Process in chunks
```python
chunk_size = 10
for i in range(0, len(prompts), chunk_size):
    chunk = prompts[i:i+chunk_size]
    results = await manager.batch_generate(chunk)
    # Process results...
```

## Configuration Guide

### Concurrency Settings

| Use Case | max_concurrent | rate_limit_delay |
|----------|----------------|------------------|
| Small models (<8B) | 8-10 | 0.05 |
| Medium models (8B-30B) | 5 | 0.1 |
| Large models (>30B) | 2-3 | 0.2 |
| Resource-constrained | 2 | 0.5 |

### Performance Tuning

**Speed priority:**
```python
results = await manager.batch_generate(
    prompts,
    model="qwen3:30b-a3b",  # Fast MoE model
    max_concurrent=8,
    rate_limit_delay=0.05
)
```

**Stability priority:**
```python
results = await manager.batch_generate(
    prompts,
    model="qwen3:8b",
    max_concurrent=3,
    rate_limit_delay=0.2
)
```

## Error Handling

### Basic error handling
```python
results = await manager.batch_generate(prompts)

for result in results:
    if not result["success"]:
        print(f"Error: {result['error']}")
        print(f"Failed prompt: {result['prompt']}")
```

### Retry failed prompts
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
async def batch_with_retry(prompts):
    return await manager.batch_generate(prompts)
```

## Performance Metrics

### Expected speedup
- 10 prompts: 4-5x faster
- 50 prompts: 5x faster
- 100 prompts: 5x faster

### Resource usage (5 concurrent)
- Memory: +50-100MB
- CPU: 80-120%
- Network: Rate-limited

## Common Issues

### Slow performance
```python
# Increase concurrency
max_concurrent=8

# Reduce rate limit
rate_limit_delay=0.05

# Use faster model
model="qwen3:30b-a3b"
```

### High memory usage
```python
# Reduce concurrency
max_concurrent=2

# Process in chunks
chunk_size = 10
```

### Connection errors
```python
# Increase timeout
manager = OllamaManager(timeout=60)

# Check Ollama is running
if not manager.check_ollama_running():
    print("Start Ollama: ollama serve")
```

## Testing

```bash
# Run tests
uv run pytest tests/test_batch_processing.py -v

# Validate API
uv run python tests/validate_batch_api.py

# Run example
uv run python examples/01-foundation/batch_inference.py
```

## See Also

- Full documentation: `docs/BATCH-PROCESSING.md`
- Example code: `examples/01-foundation/batch_inference.py`
- Unit tests: `tests/test_batch_processing.py`
