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

import logging
import time
from typing import Dict, List
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout

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
    MODEL_RECOMMENDATIONS = {
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

    def list_models(self) -> List[str]:
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

    def get_model_info(self, model: str) -> Dict:
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

    def benchmark_model(self, model: str, prompt: str = "Hello, how are you?") -> Dict:
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

    def get_running_models(self) -> List[Dict]:
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


# Convenience functions for common operations
def check_ollama() -> bool:
    """Quick check if Ollama is running.

    Returns:
        True if Ollama server is responsive.
    """
    manager = OllamaManager()
    return manager.check_ollama_running()


def get_available_models() -> List[str]:
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
