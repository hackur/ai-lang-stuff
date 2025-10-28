"""Test suite for error recovery patterns.

Tests for retry strategies, error classification, graceful degradation,
health checks, circuit breakers, and recovery orchestration.
"""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from utils.error_recovery import (
    CircuitBreaker,
    CircuitState,
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    GracefulDegradation,
    HealthCheck,
    RecoveryManager,
    RetryStrategy,
    with_circuit_breaker,
    with_retry,
)


# ============================================================================
# RetryStrategy Tests
# ============================================================================


class TestRetryStrategy:
    """Tests for RetryStrategy class."""

    def test_successful_execution(self):
        """Test successful execution without retries."""
        retry = RetryStrategy(max_retries=3)

        def success_func():
            return "success"

        result = retry.execute(success_func)
        assert result == "success"

    def test_retry_with_eventual_success(self):
        """Test retry mechanism with eventual success."""
        retry = RetryStrategy(max_retries=3, base_delay=0.1, jitter=False)

        attempt = {"count": 0}

        def flaky_func():
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = retry.execute(flaky_func)
        assert result == "success"
        assert attempt["count"] == 3

    def test_retry_exhaustion(self):
        """Test all retries are exhausted."""
        retry = RetryStrategy(max_retries=2, base_delay=0.1)

        def always_fails():
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            retry.execute(always_fails)

    def test_fallback_execution(self):
        """Test fallback is executed after all retries fail."""
        retry = RetryStrategy(max_retries=2, base_delay=0.1)

        def fails():
            raise TimeoutError("Always fails")

        def fallback():
            return "fallback_result"

        result = retry.execute(fails, fallback=fallback)
        assert result == "fallback_result"

    def test_backoff_calculation(self):
        """Test exponential backoff calculation."""
        retry = RetryStrategy(base_delay=1.0, backoff_factor=2.0, jitter=False)

        assert retry._calculate_delay(0) == 1.0
        assert retry._calculate_delay(1) == 2.0
        assert retry._calculate_delay(2) == 4.0
        assert retry._calculate_delay(3) == 8.0

    def test_backoff_max_delay(self):
        """Test delay is capped at max_delay."""
        retry = RetryStrategy(base_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=False)

        assert retry._calculate_delay(10) == 5.0  # Should be capped

    def test_jitter_adds_variation(self):
        """Test jitter adds random variation to delays."""
        retry = RetryStrategy(base_delay=1.0, backoff_factor=2.0, jitter=True)

        delays = [retry._calculate_delay(0) for _ in range(10)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1

        # All delays should be in expected range (0.8 to 1.2)
        for delay in delays:
            assert 0.8 <= delay <= 1.2

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test async execution with retries."""
        retry = RetryStrategy(max_retries=3, base_delay=0.1)

        attempt = {"count": 0}

        async def async_func():
            attempt["count"] += 1
            if attempt["count"] < 2:
                raise ConnectionError("Temporary failure")
            return "async_success"

        result = await retry.execute_async(async_func)
        assert result == "async_success"
        assert attempt["count"] == 2


# ============================================================================
# CircuitBreaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        def fails():
            raise ConnectionError("Failure")

        # Fail 3 times to reach threshold
        for _ in range(3):
            with pytest.raises(ConnectionError):
                breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        """Test circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        def fails():
            raise ConnectionError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(fails)

        # Next call should be rejected
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            breaker.call(fails)

    def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

        def fails():
            raise ConnectionError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        # Should transition to half-open
        with pytest.raises(ConnectionError):
            breaker.call(fails)

        assert breaker.state == CircuitState.OPEN  # Back to open after failure

    def test_closes_after_successful_recovery(self):
        """Test circuit closes after successful recovery."""
        breaker = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.5, success_threshold=2
        )

        attempt = {"count": 0}

        def sometimes_works():
            attempt["count"] += 1
            if attempt["count"] <= 2:
                raise ConnectionError("Failure")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(sometimes_works)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        # Successful calls should close circuit
        result1 = breaker.call(sometimes_works)
        result2 = breaker.call(sometimes_works)

        assert result1 == "success"
        assert result2 == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_manual_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker(failure_threshold=2)

        def fails():
            raise ConnectionError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


# ============================================================================
# ErrorClassifier Tests
# ============================================================================


class TestErrorClassifier:
    """Tests for ErrorClassifier class."""

    def test_classify_connection_error(self):
        """Test classification of connection errors."""
        classifier = ErrorClassifier()
        error_info = classifier.classify(ConnectionError("Connection refused"))

        assert error_info.category == ErrorCategory.CONNECTION
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.retryable is True

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        classifier = ErrorClassifier()
        error_info = classifier.classify(TimeoutError("Request timeout"))

        assert error_info.category == ErrorCategory.TIMEOUT
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.retryable is True

    def test_classify_memory_error(self):
        """Test classification of memory errors."""
        classifier = ErrorClassifier()
        error_info = classifier.classify(MemoryError("Out of memory"))

        assert error_info.category == ErrorCategory.RESOURCE
        assert error_info.severity == ErrorSeverity.CRITICAL
        assert error_info.retryable is False

    def test_classify_value_error(self):
        """Test classification of value errors."""
        classifier = ErrorClassifier()
        error_info = classifier.classify(ValueError("Invalid value"))

        assert error_info.category == ErrorCategory.PERMANENT
        assert error_info.severity == ErrorSeverity.LOW
        assert error_info.retryable is False

    def test_error_history_tracking(self):
        """Test error history is tracked."""
        classifier = ErrorClassifier(max_history=5)

        for i in range(10):
            classifier.classify(ConnectionError(f"Error {i}"))

        # Should only keep last 5
        assert len(classifier.error_history) == 5

    def test_error_patterns(self):
        """Test error pattern analysis."""
        classifier = ErrorClassifier()

        classifier.classify(ConnectionError("Error 1"))
        classifier.classify(ConnectionError("Error 2"))
        classifier.classify(TimeoutError("Error 3"))

        patterns = classifier.get_error_patterns()

        assert patterns["connection"] == 2
        assert patterns["timeout"] == 1

    def test_clear_history(self):
        """Test clearing error history."""
        classifier = ErrorClassifier()

        classifier.classify(ConnectionError("Error"))
        assert len(classifier.error_history) == 1

        classifier.clear_history()
        assert len(classifier.error_history) == 0


# ============================================================================
# GracefulDegradation Tests
# ============================================================================


class TestGracefulDegradation:
    """Tests for GracefulDegradation class."""

    def test_get_fallback_model(self):
        """Test getting next fallback model."""
        degradation = GracefulDegradation(fallback_models=["model1", "model2", "model3"])

        assert degradation.get_fallback_model("model1") == "model2"
        assert degradation.get_fallback_model("model2") == "model3"
        assert degradation.get_fallback_model("model3") is None

    def test_cache_response(self):
        """Test caching responses."""
        degradation = GracefulDegradation()

        degradation.cache_response("key1", "value1")
        cached = degradation.get_cached_response("key1")

        assert cached == "value1"

    def test_cache_expiration(self):
        """Test cached responses expire."""
        degradation = GracefulDegradation()

        degradation.cache_response("key1", "value1")

        # Should be expired after max_age
        cached = degradation.get_cached_response("key1", max_age=0.001)
        time.sleep(0.01)

        cached = degradation.get_cached_response("key1", max_age=0.001)
        assert cached is None

    def test_clear_cache(self):
        """Test clearing cache."""
        degradation = GracefulDegradation()

        degradation.cache_response("key1", "value1")
        degradation.cache_response("key2", "value2")

        degradation.clear_cache()

        assert degradation.get_cached_response("key1") is None
        assert degradation.get_cached_response("key2") is None

    def test_simplify_workflow(self):
        """Test workflow simplification."""
        degradation = GracefulDegradation()

        workflow = {
            "max_iterations": 10,
            "tools": ["tool1", "tool2", "tool3", "tool4"],
            "temperature": 0.7,
        }

        simplified = degradation.simplify_workflow(workflow)

        assert simplified["max_iterations"] == 3
        assert len(simplified["tools"]) == 3
        assert simplified["temperature"] == 0.3


# ============================================================================
# HealthCheck Tests
# ============================================================================


class TestHealthCheck:
    """Tests for HealthCheck class."""

    @patch("utils.error_recovery.requests.get")
    def test_check_ollama_success(self, mock_get):
        """Test successful Ollama health check."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        health = HealthCheck()
        status = health.check_ollama()

        assert status.healthy is True
        assert status.component == "ollama"

    @patch("utils.error_recovery.requests.get")
    def test_check_ollama_failure(self, mock_get):
        """Test failed Ollama health check."""
        mock_get.side_effect = ConnectionError("Connection refused")

        health = HealthCheck()
        status = health.check_ollama()

        assert status.healthy is False
        assert status.component == "ollama"

    @patch("utils.error_recovery.requests.get")
    def test_check_model_available(self, mock_get):
        """Test checking model availability."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        mock_get.return_value = mock_response

        health = HealthCheck()
        status = health.check_model_available("qwen3:8b")

        assert status.healthy is True
        assert status.component == "model:qwen3:8b"

    @patch("utils.error_recovery.requests.get")
    def test_check_model_not_found(self, mock_get):
        """Test checking unavailable model."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        mock_get.return_value = mock_response

        health = HealthCheck()
        status = health.check_model_available("nonexistent:model")

        assert status.healthy is False
        assert status.component == "model:nonexistent:model"

    def test_check_system_resources(self):
        """Test system resource check."""
        health = HealthCheck()
        status = health.check_system_resources()

        assert status.component == "system_resources"
        assert "memory_available_gb" in status.details
        assert "memory_percent" in status.details
        assert "disk_available_gb" in status.details
        assert "disk_percent" in status.details
        assert "cpu_percent" in status.details

    @patch("utils.error_recovery.requests.get")
    def test_check_all(self, mock_get):
        """Test comprehensive health check."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        mock_get.return_value = mock_response

        health = HealthCheck()
        statuses = health.check_all(models=["qwen3:8b"])

        assert "ollama" in statuses
        assert "model:qwen3:8b" in statuses
        assert "system_resources" in statuses


# ============================================================================
# RecoveryManager Tests
# ============================================================================


class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    def test_initialization(self):
        """Test recovery manager initialization."""
        manager = RecoveryManager()

        assert manager.retry_strategy is not None
        assert manager.classifier is not None
        assert manager.degradation is not None
        assert manager.health_check is not None

    def test_execute_with_recovery_success(self):
        """Test successful execution with recovery."""
        manager = RecoveryManager()

        def success_func(value):
            return f"result: {value}"

        result = manager.execute_with_recovery(success_func, "test")
        assert result == "result: test"

    def test_execute_with_recovery_fallback(self):
        """Test execution with fallback."""
        manager = RecoveryManager()

        def fails(value):
            raise ValueError("Always fails")

        def fallback(value):
            return f"fallback: {value}"

        result = manager.execute_with_recovery(fails, "test", fallback=fallback)
        assert result == "fallback: test"

    def test_checkpoint_save_restore(self):
        """Test checkpoint save and restore."""
        manager = RecoveryManager()

        state = {"progress": 50, "data": [1, 2, 3]}
        manager.save_checkpoint("test_checkpoint", state)

        restored = manager.restore_checkpoint("test_checkpoint")
        assert restored == state

    def test_checkpoint_not_found(self):
        """Test restoring non-existent checkpoint."""
        manager = RecoveryManager()

        restored = manager.restore_checkpoint("nonexistent")
        assert restored is None

    def test_clear_checkpoints(self):
        """Test clearing all checkpoints."""
        manager = RecoveryManager()

        manager.save_checkpoint("checkpoint1", {"data": 1})
        manager.save_checkpoint("checkpoint2", {"data": 2})

        manager.clear_checkpoints()

        assert manager.restore_checkpoint("checkpoint1") is None
        assert manager.restore_checkpoint("checkpoint2") is None

    @patch("utils.error_recovery.requests.get")
    def test_get_system_health(self, mock_get):
        """Test getting system health."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        manager = RecoveryManager()
        health = manager.get_system_health()

        assert "ollama" in health
        assert "system_resources" in health

    def test_shutdown_gracefully(self):
        """Test graceful shutdown."""
        manager = RecoveryManager()

        # Add some state
        manager.degradation.cache_response("key", "value")
        manager.save_checkpoint("checkpoint", {"data": 1})
        manager.classifier.classify(ConnectionError("Test"))

        # Shutdown
        manager.shutdown_gracefully()

        # Verify cleanup
        assert len(manager.degradation.cache) == 0
        assert len(manager.checkpoints) == 0
        assert len(manager.classifier.error_history) == 0


# ============================================================================
# Decorator Tests
# ============================================================================


class TestDecorators:
    """Tests for decorator functions."""

    def test_with_retry_decorator_success(self):
        """Test with_retry decorator on successful function."""

        @with_retry(max_retries=3, backoff_factor=2.0)
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_with_retry_decorator_eventual_success(self):
        """Test with_retry decorator with eventual success."""
        attempt = {"count": 0}

        @with_retry(max_retries=3, backoff_factor=1.5)
        def flaky_func():
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert attempt["count"] == 3

    def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""
        attempt = {"count": 0}

        @with_circuit_breaker(failure_threshold=3, recovery_timeout=10.0)
        def sometimes_fails():
            attempt["count"] += 1
            if attempt["count"] <= 3:
                raise ConnectionError("Failure")
            return "success"

        # First 3 calls fail
        for _ in range(3):
            with pytest.raises(ConnectionError):
                sometimes_fails()

        # Circuit should be open
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            sometimes_fails()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_recovery_workflow(self):
        """Test complete recovery workflow."""
        manager = RecoveryManager(fallback_models=["model1", "model2"])

        attempt = {"count": 0}

        def operation(value):
            attempt["count"] += 1
            if attempt["count"] < 2:
                raise ConnectionError("Temporary failure")
            return f"success: {value}"

        result = manager.execute_with_recovery(
            operation, "test_value", checkpoint_key="test_op"
        )

        assert result == "success: test_value"
        assert attempt["count"] == 2

        # Verify checkpoint was saved
        checkpoint = manager.restore_checkpoint("test_op")
        assert checkpoint is not None

    def test_retry_with_degradation(self):
        """Test retry with model degradation."""
        retry = RetryStrategy(max_retries=2, base_delay=0.1)
        degradation = GracefulDegradation(fallback_models=["model1", "model2"])

        def try_with_model(model):
            if model == "model1":
                raise TimeoutError("Model1 timeout")
            return f"success with {model}"

        # Try primary model, fail, get fallback
        try:
            result = retry.execute(try_with_model, "model1")
        except TimeoutError:
            fallback_model = degradation.get_fallback_model("model1")
            result = try_with_model(fallback_model)

        assert result == "success with model2"

    @patch("utils.error_recovery.requests.get")
    def test_health_check_before_operation(self, mock_get):
        """Test health check before risky operation."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        mock_get.return_value = mock_response

        health = HealthCheck()
        manager = RecoveryManager()

        # Check health first
        ollama_status = health.check_ollama()
        model_status = health.check_model_available("qwen3:8b")

        if ollama_status.healthy and model_status.healthy:
            result = manager.execute_with_recovery(lambda: "operation_result")
            assert result == "operation_result"
