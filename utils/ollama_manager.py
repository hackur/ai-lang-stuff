"""Ollama management utilities for local-first AI toolkit.

This module provides comprehensive utilities for managing Ollama server
operations, including health checks, model management, benchmarking, and
intelligent model recommendations.

Example:
    >>> manager = OllamaManager()
    >>> if manager.check_ollama_running():
    ...     models = manager.list_models()
    ...     manager.ensure_model_available("qwen3:8b")
"""

import asyncio
import logging
import time
from typing import Any, TypeAlias

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

# Type aliases
ModelDict: TypeAlias = dict[str, str | int | float | bool | list | dict]
BenchmarkResult: TypeAlias = dict[str, str | float]
BatchResponse: TypeAlias = dict[str, Any]

# Configure logging
logger = logging.getLogger(__name__)


class OllamaManager:
    """Manage Ollama server and model operations.

    This class provides methods for checking server status, managing models,
    benchmarking performance, and recommending models based on task requirements.

    Attributes:
        base_url: The base URL of the Ollama server (default: http://localhost:11434)
        timeout: Request timeout in seconds (default: 30)
    """

    # Model recommendations mapping
    MODEL_RECOMMENDATIONS: dict[str, str] = {
        "fast": "qwen3:30b-a3b",  # MoE optimized for speed
        "balanced": "qwen3:8b",  # Dense, reliable for most tasks
        "quality": "qwen3:30b",  # Best quality reasoning
        "embeddings": "nomic-embed-text",  # Optimized for embeddings
        "vision": "qwen3-vl:8b",  # Best local vision model
        "edge": "gemma3:4b",  # Minimal resource usage
        "multilingual": "gemma3:12b",  # 140+ languages
        "coding": "qwen3:30b-a3b",  # Fast code generation
    }

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """Initialize the OllamaManager.

        Args:
            base_url: The base URL of the Ollama server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.info(f"Initialized OllamaManager with base_url={self.base_url}")

    def check_ollama_running(self) -> bool:
        """Test connection to Ollama server.

        Attempts to connect to the Ollama server and verify it's responsive.

        Returns:
            True if Ollama server is running and responsive, False otherwise.

        Example:
            >>> manager = OllamaManager()
            >>> if manager.check_ollama_running():
            ...     print("Ollama is ready")
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,  # Short timeout for health check
            )
            response.raise_for_status()
            logger.info("Ollama server is running")
            return True
        except ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )
            return False
        except Timeout:
            logger.warning(f"Timeout connecting to Ollama server at {self.base_url}")
            return False
        except RequestException as e:
            logger.error(f"Error checking Ollama server: {e}")
            return False

    def list_models(self) -> list[str]:
        """Get list of installed models.

        Calls the Ollama API to retrieve all installed models.

        Returns:
            List of model names (e.g., ["qwen3:8b", "gemma3:4b"]).
            Returns empty list if no models installed or on error.

        Example:
            >>> manager = OllamaManager()
            >>> models = manager.list_models()
            >>> print(f"Found {len(models)} models")
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            models = [model.get("name", "") for model in data.get("models", [])]
            models = [m for m in models if m]  # Filter out empty strings

            logger.info(f"Found {len(models)} installed models")
            logger.debug(f"Installed models: {models}")

            return models
        except RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []

    def ensure_model_available(self, model: str) -> bool:
        """Ensure a model is available, pulling if necessary.

        Checks if the model is installed. If not, attempts to pull it.

        Args:
            model: Model name (e.g., "qwen3:8b")

        Returns:
            True if model is available after check/pull, False otherwise.

        Example:
            >>> manager = OllamaManager()
            >>> if manager.ensure_model_available("qwen3:8b"):
            ...     print("Model ready to use")
        """
        logger.info(f"Ensuring model '{model}' is available")

        # First check if model is already installed
        installed_models = self.list_models()
        if model in installed_models:
            logger.info(f"Model '{model}' is already installed")
            return True

        # Model not found, attempt to pull
        logger.info(f"Model '{model}' not found. Attempting to pull...")
        return self.pull_model(model)

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Downloads and installs a model, streaming progress to stdout.

        Args:
            model: Model name to pull (e.g., "qwen3:8b")

        Returns:
            True on success, False on failure.

        Example:
            >>> manager = OllamaManager()
            >>> success = manager.pull_model("qwen3:8b")
        """
        try:
            print(f"\nPulling model '{model}'...")
            logger.info(f"Starting pull for model '{model}'")

            response = requests.post(
                f"{self.base_url}/api/pull", json={"name": model}, stream=True, timeout=self.timeout
            )
            response.raise_for_status()

            # Stream progress updates
            for line in response.iter_lines():
                if line:
                    try:
                        import json

                        data = json.loads(line)
                        status = data.get("status", "")

                        if "total" in data and "completed" in data:
                            total = data["total"]
                            completed = data["completed"]
                            percentage = (completed / total * 100) if total > 0 else 0
                            print(f"\r{status}: {percentage:.1f}%", end="", flush=True)
                        else:
                            print(f"\r{status}", end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            print()  # New line after progress
            logger.info(f"Successfully pulled model '{model}'")
            return True

        except ConnectionError:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?")
            return False
        except Timeout:
            logger.error(f"Timeout while pulling model '{model}'")
            return False
        except RequestException as e:
            logger.error(f"Error pulling model '{model}': {e}")
            return False

    def get_model_info(self, model: str) -> ModelDict:
        """Get detailed information about a model.

        Retrieves model metadata including size, family, parameters, etc.

        Args:
            model: Model name (e.g., "qwen3:8b")

        Returns:
            Dictionary with model information. Empty dict if model not found.
            Keys may include: modelfile, parameters, template, details, etc.

        Example:
            >>> manager = OllamaManager()
            >>> info = manager.get_model_info("qwen3:8b")
            >>> if info:
            ...     print(f"Model size: {info.get('details', {}).get('parameter_size')}")
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show", json={"name": model}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"Retrieved info for model '{model}'")
            logger.debug(f"Model info: {data}")

            return data
        except RequestException as e:
            logger.error(f"Error getting info for model '{model}': {e}")
            return {}

    def benchmark_model(self, model: str, prompt: str = "Hello, how are you?") -> BenchmarkResult:
        """Benchmark a model's performance.

        Sends a prompt to the model and measures response time and throughput.

        Args:
            model: Model name to benchmark
            prompt: Test prompt to send (default: "Hello, how are you?")

        Returns:
            Dictionary with benchmarking results:
                - latency: Total response time in seconds
                - tokens_per_sec: Throughput (if available)
                - prompt: The prompt used
                - response: The model's response
                - error: Error message (if failed)

        Example:
            >>> manager = OllamaManager()
            >>> results = manager.benchmark_model("qwen3:8b")
            >>> print(f"Latency: {results['latency']:.2f}s")
        """
        logger.info(f"Benchmarking model '{model}' with prompt: '{prompt[:50]}...'")

        try:
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()

            end_time = time.time()
            latency = end_time - start_time

            data = response.json()
            response_text = data.get("response", "")

            # Calculate tokens per second if eval_count is available
            tokens_per_sec = 0.0
            if "eval_count" in data and "eval_duration" in data:
                eval_count = data["eval_count"]
                eval_duration_ns = data["eval_duration"]
                if eval_duration_ns > 0:
                    tokens_per_sec = eval_count / (eval_duration_ns / 1e9)

            results = {
                "latency": latency,
                "tokens_per_sec": tokens_per_sec,
                "prompt": prompt,
                "response": response_text,
                "model": model,
            }

            logger.info(
                f"Benchmark complete: latency={latency:.2f}s, tokens/sec={tokens_per_sec:.1f}"
            )

            return results

        except RequestException as e:
            logger.error(f"Error benchmarking model '{model}': {e}")
            return {
                "latency": 0.0,
                "tokens_per_sec": 0.0,
                "prompt": prompt,
                "response": "",
                "model": model,
                "error": str(e),
            }

    def recommend_model(self, task_type: str) -> str:
        """Recommend a model based on task type.

        Provides intelligent model recommendations based on task requirements.

        Args:
            task_type: Type of task. Supported types:
                - "fast": Speed-optimized (qwen3:30b-a3b)
                - "balanced": Good quality/speed tradeoff (qwen3:8b)
                - "quality": Best reasoning (qwen3:30b)
                - "embeddings": Text embeddings (nomic-embed-text)
                - "vision": Image understanding (qwen3-vl:8b)
                - "edge": Minimal resources (gemma3:4b)
                - "multilingual": 140+ languages (gemma3:12b)
                - "coding": Code generation (qwen3:30b-a3b)

        Returns:
            Recommended model name. Returns "qwen3:8b" if task_type not recognized.

        Example:
            >>> manager = OllamaManager()
            >>> model = manager.recommend_model("vision")
            >>> print(f"Recommended: {model}")
        """
        task_type = task_type.lower()

        if task_type in self.MODEL_RECOMMENDATIONS:
            recommended = self.MODEL_RECOMMENDATIONS[task_type]
            logger.info(f"Recommended model for '{task_type}': {recommended}")
            return recommended

        logger.warning(
            f"Unknown task type '{task_type}'. "
            f"Supported types: {list(self.MODEL_RECOMMENDATIONS.keys())}. "
            "Defaulting to 'balanced'."
        )
        return self.MODEL_RECOMMENDATIONS["balanced"]

    def get_running_models(self) -> list[ModelDict]:
        """Get list of currently running models.

        Returns:
            List of dictionaries with running model information.
            Each dict contains: name, size, digest, details, etc.
            Returns empty list on error.

        Example:
            >>> manager = OllamaManager()
            >>> running = manager.get_running_models()
            >>> for model in running:
            ...     print(f"Running: {model['name']}")
        """
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            logger.info(f"Found {len(models)} running models")

            return models
        except RequestException as e:
            logger.error(f"Error getting running models: {e}")
            return []

    async def _generate_single_async(
        self, prompt: str, model: str, **generate_kwargs: Any
    ) -> BatchResponse:
        """Async wrapper for single generate request.

        Args:
            prompt: The prompt to send to the model
            model: Model name to use
            **generate_kwargs: Additional arguments for generate API

        Returns:
            Dictionary with response data or error information
        """
        loop = asyncio.get_event_loop()

        def _sync_generate() -> BatchResponse:
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False, **generate_kwargs},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                return {
                    "prompt": prompt,
                    "response": data.get("response", ""),
                    "model": model,
                    "success": True,
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                }
            except RequestException as e:
                logger.error(f"Error generating response for prompt '{prompt[:50]}...': {e}")
                return {
                    "prompt": prompt,
                    "response": "",
                    "model": model,
                    "success": False,
                    "error": str(e),
                }

        return await loop.run_in_executor(None, _sync_generate)

    async def batch_generate(
        self,
        prompts: list[str],
        model: str = "qwen3:8b",
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1,
        **generate_kwargs: Any,
    ) -> list[BatchResponse]:
        """Generate responses for multiple prompts concurrently.

        Uses asyncio to process multiple prompts concurrently while respecting
        rate limits and maximum concurrent request constraints.

        Args:
            prompts: List of prompts to process
            model: Model to use for all prompts (default: qwen3:8b)
            max_concurrent: Maximum concurrent requests (default: 5)
            rate_limit_delay: Delay in seconds between batch starts (default: 0.1)
            **generate_kwargs: Additional arguments for generate API

        Returns:
            List of responses in same order as input prompts. Each response is a dict with:
                - prompt: Original prompt
                - response: Model's response text
                - model: Model name used
                - success: True if successful, False if error
                - error: Error message (if success=False)

        Example:
            >>> import asyncio
            >>> manager = OllamaManager()
            >>> prompts = ["Hello", "Hi there", "Good morning"]
            >>> responses = asyncio.run(manager.batch_generate(prompts))
            >>> print(len(responses))
            3
            >>> for resp in responses:
            ...     if resp["success"]:
            ...         print(f"Prompt: {resp['prompt']} -> Response: {resp['response'][:50]}")
        """
        logger.info(
            f"Starting batch generation for {len(prompts)} prompts using model '{model}' "
            f"(max_concurrent={max_concurrent})"
        )

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(prompt: str, index: int) -> tuple[int, BatchResponse]:
            async with semaphore:
                # Add delay to respect rate limits
                if index > 0 and rate_limit_delay > 0:
                    await asyncio.sleep(rate_limit_delay)

                result = await self._generate_single_async(prompt, model, **generate_kwargs)
                return index, result

        # Create tasks for all prompts
        tasks = [_generate_with_semaphore(prompt, idx) for idx, prompt in enumerate(prompts)]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by original index to maintain order and handle exceptions
        sorted_results: list[tuple[int, BatchResponse]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                sorted_results.append(
                    (
                        i,
                        {
                            "prompt": prompts[i] if i < len(prompts) else "",
                            "response": "",
                            "model": model,
                            "success": False,
                            "error": str(result),
                        },
                    )
                )
            else:
                sorted_results.append(result)

        sorted_results.sort(key=lambda x: x[0])
        responses = [resp for _, resp in sorted_results]

        # Log summary
        successful = sum(1 for r in responses if r.get("success", False))
        logger.info(f"Batch generation complete: {successful}/{len(prompts)} successful")

        return responses

    async def _benchmark_single_async(
        self, model: str, prompt: str, num_runs: int
    ) -> tuple[str, BenchmarkResult]:
        """Async wrapper for benchmarking a single model.

        Args:
            model: Model name to benchmark
            prompt: Test prompt
            num_runs: Number of runs to average

        Returns:
            Tuple of (model_name, benchmark_results)
        """
        loop = asyncio.get_event_loop()

        def _sync_benchmark() -> BenchmarkResult:
            latencies: list[float] = []
            tokens_per_secs: list[float] = []
            response_text = ""

            for run in range(num_runs):
                logger.debug(f"Benchmarking {model} - run {run + 1}/{num_runs}")
                result = self.benchmark_model(model, prompt)

                if "error" not in result:
                    latencies.append(result["latency"])
                    if result["tokens_per_sec"] > 0:
                        tokens_per_secs.append(result["tokens_per_sec"])
                    response_text = result.get("response", "")

            if not latencies:
                return {
                    "model": model,
                    "latency": 0.0,
                    "tokens_per_sec": 0.0,
                    "prompt": prompt,
                    "response": "",
                    "error": "All benchmark runs failed",
                }

            avg_latency = sum(latencies) / len(latencies)
            avg_tokens_per_sec = (
                sum(tokens_per_secs) / len(tokens_per_secs) if tokens_per_secs else 0.0
            )

            return {
                "model": model,
                "latency": avg_latency,
                "tokens_per_sec": avg_tokens_per_sec,
                "prompt": prompt,
                "response": response_text,
                "num_runs": num_runs,
            }

        result = await loop.run_in_executor(None, _sync_benchmark)
        return model, result

    async def batch_benchmark(
        self,
        models: list[str],
        prompt: str = "Hello, how are you?",
        num_runs: int = 3,
    ) -> dict[str, BenchmarkResult]:
        """Benchmark multiple models concurrently.

        Runs the same prompt on multiple models and collects performance metrics.
        Results are averaged over multiple runs for accuracy.

        Args:
            models: List of model names to benchmark
            prompt: Test prompt (same for all models, default: "Hello, how are you?")
            num_runs: Number of runs per model to average (default: 3)

        Returns:
            Dictionary mapping model name to benchmark results. Each result contains:
                - model: Model name
                - latency: Average response time in seconds
                - tokens_per_sec: Average throughput (tokens/second)
                - prompt: The prompt used
                - response: Sample response text
                - num_runs: Number of runs averaged
                - error: Error message (if failed)

        Example:
            >>> import asyncio
            >>> manager = OllamaManager()
            >>> results = asyncio.run(manager.batch_benchmark(
            ...     ["qwen3:8b", "gemma3:4b"],
            ...     prompt="Test",
            ...     num_runs=2
            ... ))
            >>> for model, result in results.items():
            ...     print(f"{model}: {result['tokens_per_sec']:.2f} tok/s")
        """
        logger.info(
            f"Starting batch benchmark for {len(models)} models " f"with {num_runs} runs each"
        )

        # Create tasks for all models
        tasks = [self._benchmark_single_async(model, prompt, num_runs) for model in models]

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary
        results: dict[str, BenchmarkResult] = {}
        for result in results_list:
            if isinstance(result, Exception):
                logger.error(f"Benchmark task failed with exception: {result}")
                continue

            model_name, benchmark_result = result
            results[model_name] = benchmark_result

        # Log summary
        logger.info(f"Batch benchmark complete for {len(results)}/{len(models)} models")

        return results


# Convenience functions for common operations
def check_ollama() -> bool:
    """Quick check if Ollama is running.

    Returns:
        True if Ollama server is responsive.
    """
    manager = OllamaManager()
    return manager.check_ollama_running()


def get_available_models() -> list[str]:
    """Quick function to get list of installed models.

    Returns:
        List of installed model names.
    """
    manager = OllamaManager()
    return manager.list_models()


def ensure_model(model: str) -> bool:
    """Quick function to ensure a model is available.

    Args:
        model: Model name to ensure is available

    Returns:
        True if model is available.
    """
    manager = OllamaManager()
    return manager.ensure_model_available(model)
