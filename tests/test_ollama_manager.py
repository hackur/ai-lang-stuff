"""Comprehensive test suite for utils/ollama_manager.py.

This module tests all OllamaManager methods including HTTP interactions,
error handling, and convenience functions using mocked requests.
"""

import pytest
from unittest.mock import Mock, patch
from requests.exceptions import ConnectionError, Timeout, RequestException
import json

from utils.ollama_manager import (
    OllamaManager,
    check_ollama,
    get_available_models,
    ensure_model,
)


@pytest.fixture
def ollama_manager():
    """Provide an OllamaManager instance for testing.

    Returns:
        OllamaManager: Manager instance with default settings
    """
    return OllamaManager()


@pytest.fixture
def custom_ollama_manager():
    """Provide an OllamaManager instance with custom settings.

    Returns:
        OllamaManager: Manager instance with custom URL and timeout
    """
    return OllamaManager(base_url="http://custom:8080", timeout=60)


class TestOllamaManagerInit:
    """Test OllamaManager initialization."""

    def test_default_initialization(self, ollama_manager):
        """Test manager initializes with default values."""
        assert ollama_manager.base_url == "http://localhost:11434"
        assert ollama_manager.timeout == 30

    def test_custom_initialization(self, custom_ollama_manager):
        """Test manager initializes with custom values."""
        assert custom_ollama_manager.base_url == "http://custom:8080"
        assert custom_ollama_manager.timeout == 60

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from base_url."""
        manager = OllamaManager(base_url="http://localhost:11434/")
        assert manager.base_url == "http://localhost:11434"


class TestCheckOllamaRunning:
    """Test check_ollama_running method."""

    @patch("utils.ollama_manager.requests.get")
    def test_check_ollama_running_success(self, mock_get, ollama_manager):
        """Test successful Ollama server check."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = ollama_manager.check_ollama_running()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    @patch("utils.ollama_manager.requests.get")
    def test_check_ollama_running_connection_error(self, mock_get, ollama_manager):
        """Test Ollama server check with connection error."""
        mock_get.side_effect = ConnectionError("Connection refused")

        result = ollama_manager.check_ollama_running()

        assert result is False

    @patch("utils.ollama_manager.requests.get")
    def test_check_ollama_running_timeout(self, mock_get, ollama_manager):
        """Test Ollama server check with timeout."""
        mock_get.side_effect = Timeout("Request timeout")

        result = ollama_manager.check_ollama_running()

        assert result is False

    @patch("utils.ollama_manager.requests.get")
    def test_check_ollama_running_request_exception(self, mock_get, ollama_manager):
        """Test Ollama server check with generic request exception."""
        mock_get.side_effect = RequestException("Generic error")

        result = ollama_manager.check_ollama_running()

        assert result is False


class TestListModels:
    """Test list_models method."""

    @patch("utils.ollama_manager.requests.get")
    def test_list_models_success(self, mock_get, ollama_manager):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}, {"name": "qwen3:30b-a3b"}]
        }
        mock_get.return_value = mock_response

        result = ollama_manager.list_models()

        assert result == ["qwen3:8b", "gemma3:4b", "qwen3:30b-a3b"]
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=30)

    @patch("utils.ollama_manager.requests.get")
    def test_list_models_empty(self, mock_get, ollama_manager):
        """Test listing models when none are installed."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        result = ollama_manager.list_models()

        assert result == []

    @patch("utils.ollama_manager.requests.get")
    def test_list_models_filters_empty_names(self, mock_get, ollama_manager):
        """Test that empty model names are filtered out."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:8b"},
                {"name": ""},  # Empty name
                {"other_field": "value"},  # Missing name
            ]
        }
        mock_get.return_value = mock_response

        result = ollama_manager.list_models()

        assert result == ["qwen3:8b"]

    @patch("utils.ollama_manager.requests.get")
    def test_list_models_request_exception(self, mock_get, ollama_manager):
        """Test listing models with request exception."""
        mock_get.side_effect = RequestException("Connection failed")

        result = ollama_manager.list_models()

        assert result == []


class TestEnsureModelAvailable:
    """Test ensure_model_available method."""

    @patch.object(OllamaManager, "list_models")
    def test_ensure_model_available_already_installed(self, mock_list, ollama_manager):
        """Test ensuring a model that's already installed."""
        mock_list.return_value = ["qwen3:8b", "gemma3:4b"]

        result = ollama_manager.ensure_model_available("qwen3:8b")

        assert result is True
        mock_list.assert_called_once()

    @patch.object(OllamaManager, "pull_model")
    @patch.object(OllamaManager, "list_models")
    def test_ensure_model_available_needs_pull(self, mock_list, mock_pull, ollama_manager):
        """Test ensuring a model that needs to be pulled."""
        mock_list.return_value = ["gemma3:4b"]
        mock_pull.return_value = True

        result = ollama_manager.ensure_model_available("qwen3:8b")

        assert result is True
        mock_list.assert_called_once()
        mock_pull.assert_called_once_with("qwen3:8b")

    @patch.object(OllamaManager, "pull_model")
    @patch.object(OllamaManager, "list_models")
    def test_ensure_model_available_pull_fails(self, mock_list, mock_pull, ollama_manager):
        """Test ensuring a model when pull fails."""
        mock_list.return_value = []
        mock_pull.return_value = False

        result = ollama_manager.ensure_model_available("qwen3:8b")

        assert result is False
        mock_pull.assert_called_once_with("qwen3:8b")


class TestPullModel:
    """Test pull_model method."""

    @patch("utils.ollama_manager.requests.post")
    @patch("builtins.print")
    def test_pull_model_success(self, mock_print, mock_post, ollama_manager):
        """Test successful model pull."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        # Simulate streaming response
        progress_lines = [
            json.dumps({"status": "downloading", "total": 1000, "completed": 500}).encode(),
            json.dumps({"status": "downloading", "total": 1000, "completed": 1000}).encode(),
            json.dumps({"status": "complete"}).encode(),
        ]
        mock_response.iter_lines.return_value = progress_lines
        mock_post.return_value = mock_response

        result = ollama_manager.pull_model("qwen3:8b")

        assert result is True
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull", json={"name": "qwen3:8b"}, stream=True, timeout=30
        )

    @patch("utils.ollama_manager.requests.post")
    def test_pull_model_connection_error(self, mock_post, ollama_manager):
        """Test model pull with connection error."""
        mock_post.side_effect = ConnectionError("Cannot connect")

        result = ollama_manager.pull_model("qwen3:8b")

        assert result is False

    @patch("utils.ollama_manager.requests.post")
    def test_pull_model_timeout(self, mock_post, ollama_manager):
        """Test model pull with timeout."""
        mock_post.side_effect = Timeout("Request timeout")

        result = ollama_manager.pull_model("qwen3:8b")

        assert result is False

    @patch("utils.ollama_manager.requests.post")
    def test_pull_model_request_exception(self, mock_post, ollama_manager):
        """Test model pull with generic request exception."""
        mock_post.side_effect = RequestException("Generic error")

        result = ollama_manager.pull_model("qwen3:8b")

        assert result is False

    @patch("utils.ollama_manager.requests.post")
    @patch("builtins.print")
    def test_pull_model_handles_invalid_json(self, mock_print, mock_post, ollama_manager):
        """Test that pull_model handles invalid JSON lines gracefully."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b"invalid json",
            json.dumps({"status": "success"}).encode(),
        ]
        mock_post.return_value = mock_response

        result = ollama_manager.pull_model("qwen3:8b")

        assert result is True


class TestGetModelInfo:
    """Test get_model_info method."""

    @patch("utils.ollama_manager.requests.post")
    def test_get_model_info_success(self, mock_post, ollama_manager):
        """Test successful model info retrieval."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "modelfile": "FROM qwen3:8b",
            "parameters": {"temperature": 0.7},
            "details": {"parameter_size": "8B", "quantization_level": "Q4_K_M"},
        }
        mock_post.return_value = mock_response

        result = ollama_manager.get_model_info("qwen3:8b")

        assert "modelfile" in result
        assert "parameters" in result
        assert "details" in result
        assert result["details"]["parameter_size"] == "8B"
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/show", json={"name": "qwen3:8b"}, timeout=30
        )

    @patch("utils.ollama_manager.requests.post")
    def test_get_model_info_not_found(self, mock_post, ollama_manager):
        """Test getting info for non-existent model."""
        mock_post.side_effect = RequestException("Model not found")

        result = ollama_manager.get_model_info("nonexistent:model")

        assert result == {}

    @patch("utils.ollama_manager.requests.post")
    def test_get_model_info_connection_error(self, mock_post, ollama_manager):
        """Test getting model info with connection error."""
        mock_post.side_effect = ConnectionError("Cannot connect")

        result = ollama_manager.get_model_info("qwen3:8b")

        assert result == {}


class TestBenchmarkModel:
    """Test benchmark_model method."""

    @patch("utils.ollama_manager.time.time")
    @patch("utils.ollama_manager.requests.post")
    def test_benchmark_model_success(self, mock_post, mock_time, ollama_manager):
        """Test successful model benchmarking."""
        # Mock time progression
        mock_time.side_effect = [100.0, 102.5]  # 2.5 second elapsed

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": "Hello! I'm doing well, thank you.",
            "eval_count": 250,
            "eval_duration": 2000000000,  # 2 seconds in nanoseconds
        }
        mock_post.return_value = mock_response

        result = ollama_manager.benchmark_model("qwen3:8b", "Hello, how are you?")

        assert result["latency"] == 2.5
        assert result["tokens_per_sec"] == 125.0  # 250 tokens / 2 seconds
        assert result["prompt"] == "Hello, how are you?"
        assert result["response"] == "Hello! I'm doing well, thank you."
        assert result["model"] == "qwen3:8b"
        assert "error" not in result

    @patch("utils.ollama_manager.time.time")
    @patch("utils.ollama_manager.requests.post")
    def test_benchmark_model_without_eval_info(self, mock_post, mock_time, ollama_manager):
        """Test benchmarking when eval_count/eval_duration not provided."""
        mock_time.side_effect = [100.0, 101.5]

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response

        result = ollama_manager.benchmark_model("qwen3:8b")

        assert result["latency"] == 1.5
        assert result["tokens_per_sec"] == 0.0
        assert "error" not in result

    @patch("utils.ollama_manager.requests.post")
    def test_benchmark_model_request_exception(self, mock_post, ollama_manager):
        """Test benchmarking with request exception."""
        mock_post.side_effect = RequestException("Connection failed")

        result = ollama_manager.benchmark_model("qwen3:8b")

        assert result["latency"] == 0.0
        assert result["tokens_per_sec"] == 0.0
        assert result["model"] == "qwen3:8b"
        assert "error" in result
        assert "Connection failed" in result["error"]

    @patch("utils.ollama_manager.time.time")
    @patch("utils.ollama_manager.requests.post")
    def test_benchmark_model_custom_prompt(self, mock_post, mock_time, ollama_manager):
        """Test benchmarking with custom prompt."""
        mock_time.side_effect = [100.0, 101.0]

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": "Custom response",
            "eval_count": 100,
            "eval_duration": 1000000000,
        }
        mock_post.return_value = mock_response

        custom_prompt = "Write a poem about AI"
        result = ollama_manager.benchmark_model("qwen3:8b", custom_prompt)

        assert result["prompt"] == custom_prompt
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3:8b", "prompt": custom_prompt, "stream": False},
            timeout=30,
        )


class TestRecommendModel:
    """Test recommend_model method."""

    def test_recommend_model_fast(self, ollama_manager):
        """Test recommendation for fast task type."""
        result = ollama_manager.recommend_model("fast")
        assert result == "qwen3:30b-a3b"

    def test_recommend_model_balanced(self, ollama_manager):
        """Test recommendation for balanced task type."""
        result = ollama_manager.recommend_model("balanced")
        assert result == "qwen3:8b"

    def test_recommend_model_quality(self, ollama_manager):
        """Test recommendation for quality task type."""
        result = ollama_manager.recommend_model("quality")
        assert result == "qwen3:30b"

    def test_recommend_model_embeddings(self, ollama_manager):
        """Test recommendation for embeddings task type."""
        result = ollama_manager.recommend_model("embeddings")
        assert result == "nomic-embed-text"

    def test_recommend_model_vision(self, ollama_manager):
        """Test recommendation for vision task type."""
        result = ollama_manager.recommend_model("vision")
        assert result == "qwen3-vl:8b"

    def test_recommend_model_edge(self, ollama_manager):
        """Test recommendation for edge task type."""
        result = ollama_manager.recommend_model("edge")
        assert result == "gemma3:4b"

    def test_recommend_model_multilingual(self, ollama_manager):
        """Test recommendation for multilingual task type."""
        result = ollama_manager.recommend_model("multilingual")
        assert result == "gemma3:12b"

    def test_recommend_model_coding(self, ollama_manager):
        """Test recommendation for coding task type."""
        result = ollama_manager.recommend_model("coding")
        assert result == "qwen3:30b-a3b"

    def test_recommend_model_case_insensitive(self, ollama_manager):
        """Test that task type is case insensitive."""
        result = ollama_manager.recommend_model("VISION")
        assert result == "qwen3-vl:8b"

    def test_recommend_model_unknown_type(self, ollama_manager):
        """Test recommendation for unknown task type defaults to balanced."""
        result = ollama_manager.recommend_model("unknown_task")
        assert result == "qwen3:8b"  # Default balanced model


class TestGetRunningModels:
    """Test get_running_models method."""

    @patch("utils.ollama_manager.requests.get")
    def test_get_running_models_success(self, mock_get, ollama_manager):
        """Test successful retrieval of running models."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:8b", "size": 8000000000},
                {"name": "gemma3:4b", "size": 4000000000},
            ]
        }
        mock_get.return_value = mock_response

        result = ollama_manager.get_running_models()

        assert len(result) == 2
        assert result[0]["name"] == "qwen3:8b"
        assert result[1]["name"] == "gemma3:4b"
        mock_get.assert_called_once_with("http://localhost:11434/api/ps", timeout=30)

    @patch("utils.ollama_manager.requests.get")
    def test_get_running_models_empty(self, mock_get, ollama_manager):
        """Test getting running models when none are running."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        result = ollama_manager.get_running_models()

        assert result == []

    @patch("utils.ollama_manager.requests.get")
    def test_get_running_models_request_exception(self, mock_get, ollama_manager):
        """Test getting running models with request exception."""
        mock_get.side_effect = RequestException("Connection failed")

        result = ollama_manager.get_running_models()

        assert result == []


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch.object(OllamaManager, "check_ollama_running")
    def test_check_ollama_success(self, mock_check):
        """Test check_ollama convenience function."""
        mock_check.return_value = True

        result = check_ollama()

        assert result is True
        mock_check.assert_called_once()

    @patch.object(OllamaManager, "check_ollama_running")
    def test_check_ollama_failure(self, mock_check):
        """Test check_ollama convenience function with failure."""
        mock_check.return_value = False

        result = check_ollama()

        assert result is False

    @patch.object(OllamaManager, "list_models")
    def test_get_available_models(self, mock_list):
        """Test get_available_models convenience function."""
        mock_list.return_value = ["qwen3:8b", "gemma3:4b"]

        result = get_available_models()

        assert result == ["qwen3:8b", "gemma3:4b"]
        mock_list.assert_called_once()

    @patch.object(OllamaManager, "ensure_model_available")
    def test_ensure_model_success(self, mock_ensure):
        """Test ensure_model convenience function."""
        mock_ensure.return_value = True

        result = ensure_model("qwen3:8b")

        assert result is True
        mock_ensure.assert_called_once_with("qwen3:8b")

    @patch.object(OllamaManager, "ensure_model_available")
    def test_ensure_model_failure(self, mock_ensure):
        """Test ensure_model convenience function with failure."""
        mock_ensure.return_value = False

        result = ensure_model("qwen3:8b")

        assert result is False


class TestModelRecommendations:
    """Test MODEL_RECOMMENDATIONS class attribute."""

    def test_model_recommendations_contains_all_types(self, ollama_manager):
        """Test that all expected task types are in recommendations."""
        expected_types = [
            "fast",
            "balanced",
            "quality",
            "embeddings",
            "vision",
            "edge",
            "multilingual",
            "coding",
        ]

        for task_type in expected_types:
            assert task_type in ollama_manager.MODEL_RECOMMENDATIONS

    def test_model_recommendations_values_are_strings(self, ollama_manager):
        """Test that all recommendation values are strings."""
        for model in ollama_manager.MODEL_RECOMMENDATIONS.values():
            assert isinstance(model, str)
            assert len(model) > 0


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch("utils.ollama_manager.requests.get")
    def test_list_models_http_error(self, mock_get, ollama_manager):
        """Test list_models handling HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = RequestException("404 Not Found")
        mock_get.return_value = mock_response

        result = ollama_manager.list_models()

        assert result == []

    @patch("utils.ollama_manager.requests.post")
    def test_get_model_info_timeout(self, mock_post, ollama_manager):
        """Test get_model_info handling timeouts."""
        mock_post.side_effect = Timeout("Request timeout")

        result = ollama_manager.get_model_info("qwen3:8b")

        assert result == {}

    @patch("utils.ollama_manager.requests.get")
    def test_get_running_models_connection_error(self, mock_get, ollama_manager):
        """Test get_running_models handling connection errors."""
        mock_get.side_effect = ConnectionError("Cannot connect")

        result = ollama_manager.get_running_models()

        assert result == []
