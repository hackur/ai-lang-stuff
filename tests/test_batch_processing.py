"""
Unit tests for batch processing capabilities in OllamaManager.

Tests the new batch_generate and batch_benchmark methods with mocked responses.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from utils.ollama_manager import OllamaManager


class TestBatchProcessing:
    """Test batch processing methods in OllamaManager."""

    @pytest.fixture
    def manager(self):
        """Create OllamaManager instance for testing."""
        return OllamaManager()

    def test_manager_has_batch_methods(self, manager):
        """Test that batch methods exist."""
        assert hasattr(manager, "batch_generate")
        assert hasattr(manager, "batch_benchmark")
        assert hasattr(manager, "_generate_single_async")
        assert hasattr(manager, "_benchmark_single_async")

    @pytest.mark.asyncio
    async def test_batch_generate_empty_list(self, manager):
        """Test batch_generate with empty prompt list."""
        results = await manager.batch_generate(prompts=[])
        assert results == []

    @pytest.mark.asyncio
    @patch("utils.ollama_manager.requests.post")
    async def test_batch_generate_success(self, mock_post, manager):
        """Test successful batch generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "eval_count": 10,
            "eval_duration": 1000000000,
        }
        mock_post.return_value = mock_response

        prompts = ["Hello", "Hi there", "Good morning"]
        results = await manager.batch_generate(prompts=prompts, model="qwen3:8b", max_concurrent=2)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["prompt"] == prompts[i]
            assert result["success"] is True
            assert result["model"] == "qwen3:8b"
            assert "response" in result

    @pytest.mark.asyncio
    @patch("utils.ollama_manager.requests.post")
    async def test_batch_generate_maintains_order(self, mock_post, manager):
        """Test that batch_generate maintains prompt order."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Response",
            "eval_count": 5,
            "eval_duration": 500000000,
        }
        mock_post.return_value = mock_response

        prompts = ["First", "Second", "Third", "Fourth", "Fifth"]
        results = await manager.batch_generate(prompts=prompts, model="test-model")

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["prompt"] == prompts[i]

    @pytest.mark.asyncio
    @patch("utils.ollama_manager.requests.post")
    async def test_batch_generate_partial_failure(self, mock_post, manager):
        """Test batch_generate handles partial failures gracefully."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Network error")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Success",
                "eval_count": 5,
                "eval_duration": 500000000,
            }
            return mock_response

        mock_post.side_effect = side_effect

        prompts = ["First", "Second", "Third"]
        results = await manager.batch_generate(prompts=prompts)

        assert len(results) == 3

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        assert len(successful) == 2
        assert len(failed) == 1

    @pytest.mark.asyncio
    @patch("utils.ollama_manager.OllamaManager.benchmark_model")
    async def test_batch_benchmark_success(self, mock_benchmark, manager):
        """Test successful batch benchmarking."""
        mock_benchmark.return_value = {
            "model": "test-model",
            "latency": 1.5,
            "tokens_per_sec": 25.0,
            "prompt": "Test",
            "response": "Response",
        }

        models = ["qwen3:8b", "gemma3:4b"]
        results = await manager.batch_benchmark(models=models, num_runs=2)

        assert len(results) == 2
        assert "qwen3:8b" in results
        assert "gemma3:4b" in results

        for model_name, result in results.items():
            assert result["model"] == model_name
            assert "latency" in result
            assert "tokens_per_sec" in result

    @pytest.mark.asyncio
    async def test_batch_benchmark_empty_list(self, manager):
        """Test batch_benchmark with empty model list."""
        results = await manager.batch_benchmark(models=[])
        assert results == {}

    @pytest.mark.asyncio
    @patch("utils.ollama_manager.requests.post")
    async def test_concurrent_limit_respected(self, mock_post, manager):
        """Test that max_concurrent limit is respected."""
        import threading

        concurrent_calls = []
        lock = threading.Lock()

        def track_concurrent(*args, **kwargs):
            with lock:
                concurrent_calls.append(threading.current_thread().ident)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Response",
                "eval_count": 5,
                "eval_duration": 500000000,
            }
            return mock_response

        mock_post.side_effect = track_concurrent

        prompts = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
        await manager.batch_generate(prompts=prompts, max_concurrent=3)

        assert len(concurrent_calls) == len(prompts)

    def test_batch_response_type_alias(self):
        """Test that BatchResponse type alias is imported."""
        from utils.ollama_manager import BatchResponse

        assert BatchResponse is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
