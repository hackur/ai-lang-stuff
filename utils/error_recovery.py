"""Error recovery patterns library for local-first AI toolkit.

This module provides comprehensive error recovery utilities including retry strategies,
error classification, graceful degradation, health checks, and recovery orchestration.

Example:
    >>> from utils.error_recovery import RetryStrategy, RecoveryManager
    >>> retry = RetryStrategy(max_retries=3, backoff_factor=2.0)
    >>> manager = RecoveryManager()
    >>> result = manager.execute_with_recovery(risky_operation, fallback=safe_operation)
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeAlias, TypeVar

import psutil
import requests
from requests.exceptions import ConnectionError, Timeout

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type aliases
ContextDict: TypeAlias = dict[str, Any]
ErrorPatterns: TypeAlias = dict[str, int]
HealthDetails: TypeAlias = dict[str, Any]


# ============================================================================
# Error Classification
# ============================================================================


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""

    TRANSIENT = "transient"  # Temporary, retry likely to succeed
    PERMANENT = "permanent"  # Permanent failure, retry won't help
    RESOURCE = "resource"  # Resource exhaustion (memory, disk, etc.)
    CONNECTION = "connection"  # Network/connection issues
    TIMEOUT = "timeout"  # Operation timeout
    UNKNOWN = "unknown"  # Unknown error type


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"  # Non-critical, can continue
    MEDIUM = "medium"  # Important but recoverable
    HIGH = "high"  # Critical, requires immediate action
    CRITICAL = "critical"  # System failure, requires shutdown


@dataclass
class ErrorInfo:
    """Structured error information for classification and recovery."""

    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    retryable: bool
    suggested_action: str
    context: ContextDict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ErrorClassifier:
    """Classify errors and determine appropriate recovery strategies.

    Analyzes errors to determine their category, severity, and whether
    retry is appropriate. Logs error patterns for analysis.

    Attributes:
        error_history: List of recent errors for pattern detection
        max_history: Maximum number of errors to track
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the ErrorClassifier.

        Args:
            max_history: Maximum number of errors to track in history
        """
        self.error_history: list[ErrorInfo] = []
        self.max_history = max_history
        logger.info("Initialized ErrorClassifier")

    def classify(self, error: Exception, context: ContextDict | None = None) -> ErrorInfo:
        """Classify an error and determine recovery strategy.

        Args:
            error: The exception to classify
            context: Optional context about where error occurred

        Returns:
            ErrorInfo with classification and suggested action

        Example:
            >>> classifier = ErrorClassifier()
            >>> error_info = classifier.classify(ConnectionError("timeout"))
            >>> print(error_info.category)
        """
        context = context or {}
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Classify based on error type and message
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        retryable = False
        suggested_action = "Log error and continue"

        # Connection errors
        if isinstance(error, (ConnectionError, ConnectionRefusedError)):
            category = ErrorCategory.CONNECTION
            severity = ErrorSeverity.HIGH
            retryable = True
            suggested_action = "Check service availability, retry with backoff"

        # Timeout errors
        elif isinstance(error, (Timeout, TimeoutError, asyncio.TimeoutError)):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
            retryable = True
            suggested_action = "Increase timeout, retry with exponential backoff"

        # Memory errors
        elif isinstance(error, MemoryError) or "out of memory" in error_msg:
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.CRITICAL
            retryable = False
            suggested_action = "Clear caches, use smaller model, restart process"

        # File/disk errors
        elif isinstance(error, (OSError, IOError)) and "disk" in error_msg:
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.HIGH
            retryable = False
            suggested_action = "Check disk space, clean up temporary files"

        # Model not found
        elif "model not found" in error_msg or "404" in error_msg:
            category = ErrorCategory.PERMANENT
            severity = ErrorSeverity.HIGH
            retryable = False
            suggested_action = "Pull model or use fallback model"

        # Rate limiting
        elif "rate limit" in error_msg or "429" in error_msg:
            category = ErrorCategory.TRANSIENT
            severity = ErrorSeverity.MEDIUM
            retryable = True
            suggested_action = "Wait and retry with exponential backoff"

        # Server errors (5xx)
        elif any(code in error_msg for code in ["500", "502", "503", "504"]):
            category = ErrorCategory.TRANSIENT
            severity = ErrorSeverity.MEDIUM
            retryable = True
            suggested_action = "Retry with exponential backoff"

        # Permission errors
        elif isinstance(error, PermissionError):
            category = ErrorCategory.PERMANENT
            severity = ErrorSeverity.HIGH
            retryable = False
            suggested_action = "Check permissions and configuration"

        # Value errors (usually permanent)
        elif isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.PERMANENT
            severity = ErrorSeverity.LOW
            retryable = False
            suggested_action = "Fix input validation"

        error_info = ErrorInfo(
            error=error,
            category=category,
            severity=severity,
            retryable=retryable,
            suggested_action=suggested_action,
            context=context,
        )

        # Track error in history
        self._add_to_history(error_info)

        logger.warning(
            f"Classified error: {error_type} -> {category.value} "
            f"(severity: {severity.value}, retryable: {retryable})"
        )
        logger.debug(f"Error details: {error_msg[:200]}")
        logger.info(f"Suggested action: {suggested_action}")

        return error_info

    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """Add error to history, maintaining max size."""
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

    def get_error_patterns(self) -> ErrorPatterns:
        """Analyze error history for patterns.

        Returns:
            Dictionary mapping error categories to occurrence counts

        Example:
            >>> classifier = ErrorClassifier()
            >>> patterns = classifier.get_error_patterns()
            >>> print(f"Connection errors: {patterns.get('connection', 0)}")
        """
        patterns = {}
        for error_info in self.error_history:
            category = error_info.category.value
            patterns[category] = patterns.get(category, 0) + 1

        logger.info(f"Error patterns: {patterns}")
        return patterns

    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        logger.info("Cleared error history")


# ============================================================================
# Retry Strategy
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.8, 1.2)


class CircuitState(Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures.

    Monitors failures and trips open to prevent repeated calls to failing
    services, allowing them time to recover.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        success_threshold: Successes needed to close circuit from half-open
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        """Initialize the CircuitBreaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes to close from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None

        logger.info(
            f"Initialized CircuitBreaker: threshold={failure_threshold}, "
            f"recovery={recovery_timeout}s"
        )

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            RuntimeError: If circuit is open
            Exception: Original exception if circuit closed

        Example:
            >>> breaker = CircuitBreaker()
            >>> result = breaker.call(risky_function, arg1, arg2)
        """
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time and (
                time.time() - self.last_failure_time >= self.recovery_timeout
            ):
                logger.info("Circuit half-open: attempting recovery")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise RuntimeError(
                    f"Circuit breaker OPEN. Recovery in "
                    f"{self.recovery_timeout - (time.time() - (self.last_failure_time or 0)):.1f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"Circuit half-open: {self.success_count}/{self.success_threshold} successes"
            )

            if self.success_count >= self.success_threshold:
                logger.info("Circuit closed: service recovered")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit opened: recovery failed")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit opened: {self.failure_count} failures exceeded threshold")
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


class RetryStrategy:
    """Implements retry logic with exponential backoff and jitter.

    Provides configurable retry behavior with exponential backoff, jitter
    for distributed systems, circuit breaker pattern, and fallback mechanisms.

    Attributes:
        config: RetryConfig with retry parameters
        circuit_breaker: Optional CircuitBreaker instance
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        use_circuit_breaker: bool = True,
    ) -> None:
        """Initialize the RetryStrategy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Exponential backoff multiplier
            jitter: Whether to add random jitter to delays
            use_circuit_breaker: Whether to use circuit breaker pattern
        """
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter,
        )

        self.circuit_breaker: CircuitBreaker | None = (
            CircuitBreaker() if use_circuit_breaker else None
        )

        logger.info(
            f"Initialized RetryStrategy: max_retries={max_retries}, "
            f"base_delay={base_delay}s, backoff={backoff_factor}, "
            f"circuit_breaker={use_circuit_breaker}"
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.config.base_delay * (self.config.backoff_factor**attempt),
            self.config.max_delay,
        )

        # Add jitter if enabled
        if self.config.jitter:
            jitter_min, jitter_max = self.config.jitter_range
            delay *= random.uniform(jitter_min, jitter_max)

        return delay

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute with retries
            *args: Positional arguments for func
            fallback: Optional fallback function if all retries fail
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Raises:
            Exception: Last exception if all retries fail and no fallback

        Example:
            >>> retry = RetryStrategy(max_retries=3)
            >>> result = retry.execute(api_call, param1, fallback=cache_lookup)
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Use circuit breaker if enabled
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if error is retryable
                classifier = ErrorClassifier()
                error_info = classifier.classify(e)

                if not error_info.retryable or attempt >= self.config.max_retries:
                    logger.error(
                        f"Failed after {attempt + 1} attempts: {e}",
                        exc_info=True,
                    )

                    # Try fallback if available
                    if fallback:
                        logger.info("Executing fallback function")
                        try:
                            return fallback(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed: {fallback_error}")
                            raise last_exception

                    raise last_exception

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic completed without result or exception")

    def execute_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        fallback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with retry logic.

        Args:
            func: Async function to execute with retries
            *args: Positional arguments for func
            fallback: Optional async fallback function
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Example:
            >>> retry = RetryStrategy(max_retries=3)
            >>> result = await retry.execute_async(async_api_call, param1)
        """

        async def _execute() -> Any:
            last_exception: Exception | None = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    classifier = ErrorClassifier()
                    error_info = classifier.classify(e)

                    if not error_info.retryable or attempt >= self.config.max_retries:
                        logger.error(f"Failed after {attempt + 1} attempts: {e}")

                        if fallback:
                            logger.info("Executing fallback function")
                            try:
                                return await fallback(*args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback failed: {fallback_error}")
                                raise last_exception

                        raise last_exception

                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Retry in {delay:.2f}s...")
                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic completed without result")

        return _execute()


# ============================================================================
# Graceful Degradation
# ============================================================================


@dataclass
class ModelFallback:
    """Configuration for model fallback strategy."""

    primary_model: str
    fallback_models: list[str]
    cache_enabled: bool = True
    simplified_workflow: bool = False


class GracefulDegradation:
    """Implement graceful degradation strategies.

    Provides fallback mechanisms including smaller models, cached responses,
    simplified workflows, and user notifications.

    Attributes:
        fallback_chain: List of fallback models to try
        cache: Simple in-memory cache for responses
    """

    def __init__(self, fallback_models: list[str] | None = None) -> None:
        """Initialize GracefulDegradation.

        Args:
            fallback_models: List of models to try in order of preference
        """
        self.fallback_chain = fallback_models or [
            "qwen3:8b",  # Primary balanced model
            "gemma3:4b",  # Smaller, faster fallback
        ]
        self.cache: dict[str, Any] = {}
        logger.info(f"Initialized GracefulDegradation with fallbacks: {self.fallback_chain}")

    def get_fallback_model(self, failed_model: str) -> str | None:
        """Get next fallback model after failure.

        Args:
            failed_model: Model that failed

        Returns:
            Next model to try, or None if no fallbacks

        Example:
            >>> degradation = GracefulDegradation()
            >>> fallback = degradation.get_fallback_model("qwen3:30b")
            >>> print(f"Trying fallback: {fallback}")
        """
        try:
            current_index = self.fallback_chain.index(failed_model)
            if current_index < len(self.fallback_chain) - 1:
                fallback = self.fallback_chain[current_index + 1]
                logger.info(f"Falling back from {failed_model} to {fallback}")
                return fallback
        except ValueError:
            # Model not in chain, return first fallback
            if self.fallback_chain:
                fallback = self.fallback_chain[0]
                logger.info(f"Using default fallback: {fallback}")
                return fallback

        logger.warning("No fallback models available")
        return None

    def cache_response(self, key: str, value: Any) -> None:
        """Cache a response for future use.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = {"value": value, "timestamp": time.time()}
        logger.debug(f"Cached response for key: {key}")

    def get_cached_response(self, key: str, max_age: float = 3600.0) -> Any | None:
        """Get cached response if available and not expired.

        Args:
            key: Cache key
            max_age: Maximum age in seconds (default: 1 hour)

        Returns:
            Cached value if available and fresh, None otherwise

        Example:
            >>> degradation = GracefulDegradation()
            >>> cached = degradation.get_cached_response("query_123")
            >>> if cached:
            ...     print("Using cached response")
        """
        if key in self.cache:
            cached = self.cache[key]
            age = time.time() - cached["timestamp"]

            if age <= max_age:
                logger.info(f"Cache hit for key: {key} (age: {age:.1f}s)")
                return cached["value"]
            else:
                logger.debug(f"Cache expired for key: {key} (age: {age:.1f}s)")
                del self.cache[key]

        return None

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cleared response cache")

    def simplify_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Simplify workflow for degraded mode.

        Args:
            workflow: Original workflow configuration

        Returns:
            Simplified workflow configuration

        Example:
            >>> degradation = GracefulDegradation()
            >>> simplified = degradation.simplify_workflow(complex_workflow)
        """
        # Create simplified version
        simplified = workflow.copy()

        # Reduce complexity
        if "max_iterations" in simplified:
            simplified["max_iterations"] = min(simplified["max_iterations"], 3)
        if "tools" in simplified:
            # Keep only essential tools
            simplified["tools"] = simplified["tools"][:3]
        if "temperature" in simplified:
            # Lower temperature for more deterministic results
            simplified["temperature"] = 0.3

        logger.info("Simplified workflow for degraded mode")
        logger.debug(f"Simplified config: {simplified}")

        return simplified


# ============================================================================
# Health Checks
# ============================================================================


@dataclass
class HealthStatus:
    """Health status for a component."""

    component: str
    healthy: bool
    message: str
    details: HealthDetails = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthCheck:
    """Perform health checks on system components.

    Monitors Ollama server, model availability, MCP servers, vector stores,
    and system resources (memory, disk).

    Attributes:
        ollama_base_url: Base URL for Ollama server
        timeout: Request timeout in seconds
    """

    def __init__(self, ollama_base_url: str = "http://localhost:11434", timeout: int = 5) -> None:
        """Initialize HealthCheck.

        Args:
            ollama_base_url: Base URL for Ollama server
            timeout: Request timeout in seconds
        """
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout = timeout
        logger.info("Initialized HealthCheck")

    def check_ollama(self) -> HealthStatus:
        """Check Ollama server health.

        Returns:
            HealthStatus for Ollama server

        Example:
            >>> health = HealthCheck()
            >>> status = health.check_ollama()
            >>> if status.healthy:
            ...     print("Ollama is running")
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()

            return HealthStatus(
                component="ollama",
                healthy=True,
                message="Ollama server is running",
                details={"url": self.ollama_base_url},
            )

        except ConnectionError:
            return HealthStatus(
                component="ollama",
                healthy=False,
                message="Cannot connect to Ollama server",
                details={"url": self.ollama_base_url, "error": "Connection refused"},
            )

        except Exception as e:
            return HealthStatus(
                component="ollama",
                healthy=False,
                message=f"Ollama health check failed: {e}",
                details={"url": self.ollama_base_url, "error": str(e)},
            )

    def check_model_available(self, model: str) -> HealthStatus:
        """Check if a specific model is available.

        Args:
            model: Model name to check

        Returns:
            HealthStatus for the model

        Example:
            >>> health = HealthCheck()
            >>> status = health.check_model_available("qwen3:8b")
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            models = [m.get("name", "") for m in data.get("models", [])]

            if model in models:
                return HealthStatus(
                    component=f"model:{model}",
                    healthy=True,
                    message=f"Model {model} is available",
                    details={"model": model},
                )
            else:
                return HealthStatus(
                    component=f"model:{model}",
                    healthy=False,
                    message=f"Model {model} not found",
                    details={"model": model, "available_models": models},
                )

        except Exception as e:
            return HealthStatus(
                component=f"model:{model}",
                healthy=False,
                message=f"Failed to check model: {e}",
                details={"model": model, "error": str(e)},
            )

    def check_system_resources(self) -> HealthStatus:
        """Check system resource availability.

        Returns:
            HealthStatus for system resources (memory, disk, CPU)

        Example:
            >>> health = HealthCheck()
            >>> status = health.check_system_resources()
            >>> print(f"Memory available: {status.details['memory_available_gb']:.1f}GB")
        """
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent

            # Disk check
            disk = psutil.disk_usage("/")
            disk_available_gb = disk.free / (1024**3)
            disk_percent = disk.percent

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)

            # Determine health
            healthy = True
            issues = []

            if memory_percent > 90:
                healthy = False
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            if disk_percent > 90:
                healthy = False
                issues.append(f"Low disk space: {disk_available_gb:.1f}GB available")

            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            message = "System resources healthy" if healthy else "; ".join(issues)

            return HealthStatus(
                component="system_resources",
                healthy=healthy,
                message=message,
                details={
                    "memory_available_gb": memory_available_gb,
                    "memory_percent": memory_percent,
                    "disk_available_gb": disk_available_gb,
                    "disk_percent": disk_percent,
                    "cpu_percent": cpu_percent,
                },
            )

        except Exception as e:
            return HealthStatus(
                component="system_resources",
                healthy=False,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
            )

    def check_all(self, models: list[str] | None = None) -> dict[str, HealthStatus]:
        """Perform all health checks.

        Args:
            models: Optional list of models to check

        Returns:
            Dictionary mapping component names to HealthStatus

        Example:
            >>> health = HealthCheck()
            >>> statuses = health.check_all(models=["qwen3:8b"])
            >>> for component, status in statuses.items():
            ...     print(f"{component}: {'OK' if status.healthy else 'FAIL'}")
        """
        results = {}

        # Check Ollama
        results["ollama"] = self.check_ollama()

        # Check models if specified
        if models:
            for model in models:
                results[f"model:{model}"] = self.check_model_available(model)

        # Check system resources
        results["system_resources"] = self.check_system_resources()

        # Log summary
        healthy_count = sum(1 for status in results.values() if status.healthy)
        total_count = len(results)

        logger.info(f"Health check complete: {healthy_count}/{total_count} components healthy")

        return results


# ============================================================================
# Recovery Manager
# ============================================================================


class RecoveryManager:
    """Orchestrate recovery strategies for comprehensive error handling.

    Coordinates retry strategies, error classification, graceful degradation,
    health checks, checkpoint restoration, and clean shutdown.

    Attributes:
        retry_strategy: RetryStrategy instance
        classifier: ErrorClassifier instance
        degradation: GracefulDegradation instance
        health_check: HealthCheck instance
    """

    def __init__(
        self,
        retry_strategy: RetryStrategy | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        """Initialize RecoveryManager.

        Args:
            retry_strategy: Optional custom RetryStrategy
            fallback_models: Optional list of fallback models
        """
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.classifier = ErrorClassifier()
        self.degradation = GracefulDegradation(fallback_models)
        self.health_check = HealthCheck()

        self.checkpoints: dict[str, Any] = {}

        logger.info("Initialized RecoveryManager")

    def execute_with_recovery(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        checkpoint_key: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with comprehensive recovery strategies.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function
            checkpoint_key: Optional key for checkpointing state
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Example:
            >>> manager = RecoveryManager()
            >>> result = manager.execute_with_recovery(
            ...     risky_operation,
            ...     fallback=safe_operation,
            ...     checkpoint_key="operation_state"
            ... )
        """
        # Save checkpoint if requested
        if checkpoint_key:
            self.save_checkpoint(checkpoint_key, {"args": args, "kwargs": kwargs})

        try:
            # Execute with retry strategy
            return self.retry_strategy.execute(func, *args, fallback=fallback, **kwargs)

        except Exception as e:
            # Classify error
            error_info = self.classifier.classify(e, context={"function": func.__name__})

            logger.error(
                f"Execution failed with {error_info.category.value} error: {e}",
                exc_info=True,
            )
            logger.info(f"Suggested action: {error_info.suggested_action}")

            # Attempt recovery based on error type
            if error_info.category == ErrorCategory.RESOURCE:
                logger.info("Resource error detected, checking system health")
                health = self.health_check.check_system_resources()
                if not health.healthy:
                    logger.warning(f"System health issues: {health.message}")

            # Try checkpoint restoration if available
            if checkpoint_key:
                logger.info(f"Attempting checkpoint restoration: {checkpoint_key}")
                checkpoint = self.restore_checkpoint(checkpoint_key)
                if checkpoint:
                    logger.info("Checkpoint restored, attempting recovery")
                    # Could implement custom recovery logic here

            raise e

    def save_checkpoint(self, key: str, state: Any) -> None:
        """Save a checkpoint for state recovery.

        Args:
            key: Checkpoint identifier
            state: State to save
        """
        self.checkpoints[key] = {"state": state, "timestamp": time.time()}
        logger.debug(f"Saved checkpoint: {key}")

    def restore_checkpoint(self, key: str) -> Any | None:
        """Restore a saved checkpoint.

        Args:
            key: Checkpoint identifier

        Returns:
            Saved state if available, None otherwise
        """
        if key in self.checkpoints:
            checkpoint = self.checkpoints[key]
            logger.info(f"Restored checkpoint: {key}")
            return checkpoint["state"]

        logger.warning(f"Checkpoint not found: {key}")
        return None

    def clear_checkpoints(self) -> None:
        """Clear all saved checkpoints."""
        self.checkpoints.clear()
        logger.info("Cleared all checkpoints")

    def get_system_health(self, models: list[str] | None = None) -> dict[str, HealthStatus]:
        """Get comprehensive system health status.

        Args:
            models: Optional list of models to check

        Returns:
            Dictionary of component health statuses

        Example:
            >>> manager = RecoveryManager()
            >>> health = manager.get_system_health(models=["qwen3:8b"])
        """
        return self.health_check.check_all(models)

    def shutdown_gracefully(self) -> None:
        """Perform graceful shutdown, cleaning up resources.

        Example:
            >>> manager = RecoveryManager()
            >>> manager.shutdown_gracefully()
        """
        logger.info("Initiating graceful shutdown")

        # Clear caches
        self.degradation.clear_cache()

        # Clear checkpoints
        self.clear_checkpoints()

        # Clear error history
        self.classifier.clear_history()

        # Reset circuit breakers
        if self.retry_strategy.circuit_breaker:
            self.retry_strategy.circuit_breaker.reset()

        logger.info("Graceful shutdown complete")


# ============================================================================
# Convenience Decorators
# ============================================================================


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    fallback: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        fallback: Optional fallback function

    Example:
        >>> @with_retry(max_retries=3)
        ... def fetch_data():
        ...     return api_call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retry = RetryStrategy(max_retries=max_retries, backoff_factor=backoff_factor)
            return retry.execute(func, *args, fallback=fallback, **kwargs)

        return wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: float = 60.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add circuit breaker to a function.

    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery

    Example:
        >>> @with_circuit_breaker(failure_threshold=5)
        ... def call_external_service():
        ...     return service.request()
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator
