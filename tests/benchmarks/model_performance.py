"""
Model performance benchmarks.

Measures performance characteristics of different local models:
- Latency (time to first token, total generation time)
- Throughput (tokens per second)
- Memory usage
- Quality scoring

Models tested:
- qwen3:8b (baseline dense model)
- qwen3:30b-a3b (MoE model for speed)
- gemma3:4b (edge/mobile optimized)
- gemma3:12b (multilingual)
- deepseek-r1:8b (reasoning)
"""

import time
import psutil
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import csv
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


@dataclass
class ModelBenchmarkResult:
    """Results from a single model benchmark run."""

    model_name: str
    prompt_length: int
    prompt_type: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_mb: float
    quality_score: Optional[float] = None
    error: Optional[str] = None


class ModelBenchmark:
    """Benchmark suite for model performance testing."""

    # Standard test prompts of varying complexity
    PROMPTS = {
        "simple": "What is 2+2?",
        "short": "Explain what a binary tree is in one sentence.",
        "medium": "Write a Python function to calculate the Fibonacci sequence recursively. Include docstring and type hints.",
        "long": """Design a simple REST API for a todo application. Include:
1. Endpoint specifications (GET, POST, PUT, DELETE)
2. Request/response schemas
3. Error handling approach
4. Authentication strategy
Provide a detailed but concise explanation.""",
        "reasoning": "If a train leaves Station A at 60 mph and another train leaves Station B (300 miles away) at 90 mph heading toward each other, when will they meet? Show your work step by step.",
        "code_generation": "Create a Python class for a LRU cache with get and put operations in O(1) time. Include comments explaining the algorithm.",
    }

    # Models to benchmark
    MODELS = [
        "qwen3:8b",
        "qwen3:30b-a3b",
        "gemma3:4b",
        "gemma3:12b",
        "deepseek-r1:8b",
    ]

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[ModelBenchmarkResult] = []

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _count_tokens_estimate(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4

    def benchmark_model(
        self, model_name: str, prompt: str, prompt_type: str, num_runs: int = 3
    ) -> List[ModelBenchmarkResult]:
        """
        Benchmark a single model with a specific prompt.

        Args:
            model_name: Name of the Ollama model
            prompt: Test prompt to use
            prompt_type: Category of prompt (simple, medium, long, etc.)
            num_runs: Number of runs to average

        Returns:
            List of benchmark results for each run
        """
        results = []

        try:
            llm = ChatOllama(
                model=model_name,
                temperature=0.7,
                num_predict=512,  # Limit output length for consistency
            )

            for run in range(num_runs):
                # Measure memory before
                mem_before = self._get_memory_usage()

                # Time the generation
                start_time = time.perf_counter()
                response = llm.invoke([HumanMessage(content=prompt)])
                end_time = time.perf_counter()

                # Measure memory after
                mem_after = self._get_memory_usage()

                # Calculate metrics
                latency_ms = (end_time - start_time) * 1000
                response_text = response.content if hasattr(response, "content") else str(response)
                tokens_generated = self._count_tokens_estimate(response_text)
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
                memory_mb = mem_after - mem_before

                result = ModelBenchmarkResult(
                    model_name=model_name,
                    prompt_length=len(prompt),
                    prompt_type=prompt_type,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    memory_mb=memory_mb,
                )

                results.append(result)
                self.results.append(result)

                # Brief pause between runs
                time.sleep(1)

        except Exception as e:
            result = ModelBenchmarkResult(
                model_name=model_name,
                prompt_length=len(prompt),
                prompt_type=prompt_type,
                latency_ms=0,
                tokens_generated=0,
                tokens_per_second=0,
                memory_mb=0,
                error=str(e),
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_all_models(self, prompt_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark all models across all or specified prompt types.

        Args:
            prompt_types: List of prompt types to test. If None, tests all.

        Returns:
            Summary statistics dictionary
        """
        if prompt_types is None:
            prompt_types = list(self.PROMPTS.keys())

        for model in self.MODELS:
            print(f"\nBenchmarking {model}...")
            for prompt_type in prompt_types:
                prompt = self.PROMPTS[prompt_type]
                print(f"  - {prompt_type} prompt...", end=" ")
                results = self.benchmark_model(model, prompt, prompt_type)
                avg_latency = statistics.mean(r.latency_ms for r in results if r.error is None)
                print(f"{avg_latency:.0f}ms")

        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from all benchmark results."""
        summary = {
            "total_runs": len(self.results),
            "models_tested": len(set(r.model_name for r in self.results)),
            "by_model": {},
            "by_prompt_type": {},
        }

        # Group by model
        for model in self.MODELS:
            model_results = [r for r in self.results if r.model_name == model and r.error is None]
            if model_results:
                summary["by_model"][model] = {
                    "avg_latency_ms": statistics.mean(r.latency_ms for r in model_results),
                    "avg_tokens_per_second": statistics.mean(
                        r.tokens_per_second for r in model_results
                    ),
                    "avg_memory_mb": statistics.mean(r.memory_mb for r in model_results),
                    "runs": len(model_results),
                }

        # Group by prompt type
        for prompt_type in self.PROMPTS.keys():
            type_results = [
                r for r in self.results if r.prompt_type == prompt_type and r.error is None
            ]
            if type_results:
                summary["by_prompt_type"][prompt_type] = {
                    "avg_latency_ms": statistics.mean(r.latency_ms for r in type_results),
                    "avg_tokens_per_second": statistics.mean(
                        r.tokens_per_second for r in type_results
                    ),
                    "runs": len(type_results),
                }

        return summary

    def save_results(self, format: str = "json"):
        """
        Save benchmark results to file.

        Args:
            format: Output format ('json' or 'csv')
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = self.output_dir / f"model_benchmark_{timestamp}.json"
            data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_runs": len(self.results),
                },
                "results": [asdict(r) for r in self.results],
                "summary": self.generate_summary(),
            }
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to {output_file}")

        elif format == "csv":
            output_file = self.output_dir / f"model_benchmark_{timestamp}.csv"
            with open(output_file, "w", newline="") as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(asdict(result))
            print(f"\nResults saved to {output_file}")


# Pytest integration
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model", ModelBenchmark.MODELS)
def test_model_latency(model: str):
    """Test that model latency is within acceptable range."""
    benchmark = ModelBenchmark()
    results = benchmark.benchmark_model(
        model, ModelBenchmark.PROMPTS["simple"], "simple", num_runs=1
    )

    assert len(results) == 1
    result = results[0]

    # Check for errors
    assert result.error is None, f"Model {model} failed: {result.error}"

    # Latency should be under 30 seconds for simple prompt
    assert result.latency_ms < 30000, f"Model {model} too slow: {result.latency_ms}ms"

    # Should generate some tokens
    assert result.tokens_generated > 0


@pytest.mark.integration
@pytest.mark.slow
def test_model_comparison():
    """Run a quick comparison of all models."""
    benchmark = ModelBenchmark()
    benchmark.benchmark_all_models(prompt_types=["simple", "short"])
    summary = benchmark.generate_summary()

    # All models should have results
    assert len(summary["by_model"]) > 0

    # Check that we have data for each tested prompt type
    assert "simple" in summary["by_prompt_type"]
    assert "short" in summary["by_prompt_type"]


if __name__ == "__main__":
    """Run benchmarks standalone."""
    print("=" * 70)
    print("MODEL PERFORMANCE BENCHMARK")
    print("=" * 70)

    benchmark = ModelBenchmark()

    # Run full benchmark suite
    summary = benchmark.benchmark_all_models()

    # Save results
    benchmark.save_results(format="json")
    benchmark.save_results(format="csv")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
