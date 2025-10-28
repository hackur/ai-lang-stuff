#!/usr/bin/env python3
"""Error Recovery Patterns Demo.

This example demonstrates comprehensive error recovery utilities including:
- Retry strategies with exponential backoff
- Error classification and analysis
- Graceful degradation with model fallbacks
- Health checks for system components
- Circuit breaker pattern
- Recovery orchestration

Prerequisites:
    - Ollama running (ollama serve)
    - At least one model available (qwen3:8b or gemma3:4b)

Expected Output:
    - Demonstrations of various error recovery patterns
    - Health check results
    - Retry behavior with backoff
    - Fallback mechanisms
    - System resource monitoring
"""

import logging
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import error recovery utilities
from utils.error_recovery import (
    CircuitBreaker,
    ErrorClassifier,
    GracefulDegradation,
    HealthCheck,
    RecoveryManager,
    RetryStrategy,
    with_circuit_breaker,
    with_retry,
)


# ============================================================================
# Example 1: Basic Retry with Exponential Backoff
# ============================================================================


def example_1_basic_retry():
    """Demonstrate basic retry with exponential backoff."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Retry with Exponential Backoff")
    print("=" * 70)

    retry = RetryStrategy(
        max_retries=3,
        base_delay=1.0,
        backoff_factor=2.0,
        jitter=True,
    )

    # Simulate flaky function that fails first 2 times
    attempt_count = {"count": 0}

    def flaky_function():
        attempt_count["count"] += 1
        logger.info(f"Attempt {attempt_count['count']}")

        if attempt_count["count"] < 3:
            raise ConnectionError(f"Connection failed on attempt {attempt_count['count']}")

        return "Success!"

    try:
        result = retry.execute(flaky_function)
        print(f"Result: {result}")
        print(f"Total attempts: {attempt_count['count']}")
    except Exception as e:
        print(f"Failed after all retries: {e}")


# ============================================================================
# Example 2: Retry with Fallback
# ============================================================================


def example_2_retry_with_fallback():
    """Demonstrate retry with fallback function."""
    print("\n" + "=" * 70)
    print("Example 2: Retry with Fallback")
    print("=" * 70)

    retry = RetryStrategy(max_retries=2, base_delay=0.5)

    def primary_function():
        """Always fails."""
        raise TimeoutError("Primary service timeout")

    def fallback_function():
        """Fallback returns cached data."""
        logger.info("Executing fallback function")
        return "Cached result from fallback"

    try:
        result = retry.execute(primary_function, fallback=fallback_function)
        print(f"Result: {result}")
        print("Successfully used fallback after primary failed")
    except Exception as e:
        print(f"Failed: {e}")


# ============================================================================
# Example 3: Error Classification
# ============================================================================


def example_3_error_classification():
    """Demonstrate error classification and analysis."""
    print("\n" + "=" * 70)
    print("Example 3: Error Classification")
    print("=" * 70)

    classifier = ErrorClassifier(max_history=100)

    # Classify various error types
    errors = [
        ConnectionError("Connection refused"),
        TimeoutError("Request timeout"),
        MemoryError("Out of memory"),
        ValueError("Invalid model name"),
        Exception("Rate limit exceeded"),
        Exception("Server error 503"),
    ]

    for error in errors:
        error_info = classifier.classify(error)
        print(f"\nError: {type(error).__name__}")
        print(f"  Category: {error_info.category.value}")
        print(f"  Severity: {error_info.severity.value}")
        print(f"  Retryable: {error_info.retryable}")
        print(f"  Action: {error_info.suggested_action}")

    # Show error patterns
    print("\nError Patterns:")
    patterns = classifier.get_error_patterns()
    for category, count in patterns.items():
        print(f"  {category}: {count}")


# ============================================================================
# Example 4: Circuit Breaker Pattern
# ============================================================================


def example_4_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n" + "=" * 70)
    print("Example 4: Circuit Breaker Pattern")
    print("=" * 70)

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
        success_threshold=2,
    )

    def unreliable_service():
        """Simulates unreliable service."""
        if random.random() < 0.8:  # 80% failure rate
            raise ConnectionError("Service unavailable")
        return "Success"

    # Try multiple calls
    for i in range(10):
        try:
            result = breaker.call(unreliable_service)
            print(f"Call {i + 1}: {result} (state: {breaker.state.value})")
        except ConnectionError as e:
            print(f"Call {i + 1}: Failed - {e} (state: {breaker.state.value})")
        except RuntimeError as e:
            print(f"Call {i + 1}: Circuit open - {e}")

        time.sleep(0.5)


# ============================================================================
# Example 5: Graceful Degradation
# ============================================================================


def example_5_graceful_degradation():
    """Demonstrate graceful degradation with model fallbacks."""
    print("\n" + "=" * 70)
    print("Example 5: Graceful Degradation")
    print("=" * 70)

    degradation = GracefulDegradation(fallback_models=["qwen3:30b", "qwen3:8b", "gemma3:4b"])

    # Simulate model failures
    failed_models = ["qwen3:30b", "qwen3:8b"]

    for failed_model in failed_models:
        fallback = degradation.get_fallback_model(failed_model)
        print(f"Model {failed_model} failed -> Fallback to: {fallback}")

    # Demonstrate caching
    print("\nCaching Demonstration:")
    degradation.cache_response("query_123", "Cached result for query 123")
    degradation.cache_response("query_456", "Cached result for query 456")

    cached = degradation.get_cached_response("query_123", max_age=60)
    print(f"Cache hit: {cached}")

    # Demonstrate workflow simplification
    print("\nWorkflow Simplification:")
    complex_workflow = {
        "max_iterations": 10,
        "tools": ["tool1", "tool2", "tool3", "tool4", "tool5"],
        "temperature": 0.7,
        "top_k": 50,
    }

    print(f"Original: {complex_workflow}")

    simplified = degradation.simplify_workflow(complex_workflow)
    print(f"Simplified: {simplified}")


# ============================================================================
# Example 6: Health Checks
# ============================================================================


def example_6_health_checks():
    """Demonstrate system health checks."""
    print("\n" + "=" * 70)
    print("Example 6: Health Checks")
    print("=" * 70)

    health = HealthCheck()

    # Check Ollama server
    print("\nOllama Server Health:")
    status = health.check_ollama()
    print(f"  Status: {'OK' if status.healthy else 'FAIL'}")
    print(f"  Message: {status.message}")

    # Check specific models
    print("\nModel Availability:")
    for model in ["qwen3:8b", "gemma3:4b", "nonexistent:model"]:
        status = health.check_model_available(model)
        print(f"  {model}: {'Available' if status.healthy else 'Not found'}")

    # Check system resources
    print("\nSystem Resources:")
    status = health.check_system_resources()
    print(f"  Status: {'OK' if status.healthy else 'WARNING'}")
    print(f"  Message: {status.message}")
    print(f"  Memory Available: {status.details['memory_available_gb']:.1f} GB")
    print(f"  Memory Usage: {status.details['memory_percent']:.1f}%")
    print(f"  Disk Available: {status.details['disk_available_gb']:.1f} GB")
    print(f"  Disk Usage: {status.details['disk_percent']:.1f}%")
    print(f"  CPU Usage: {status.details['cpu_percent']:.1f}%")

    # Check all components
    print("\nComprehensive Health Check:")
    all_statuses = health.check_all(models=["qwen3:8b", "gemma3:4b"])
    for component, status in all_statuses.items():
        icon = "✓" if status.healthy else "✗"
        print(f"  {icon} {component}: {status.message}")


# ============================================================================
# Example 7: Recovery Manager
# ============================================================================


def example_7_recovery_manager():
    """Demonstrate recovery manager orchestration."""
    print("\n" + "=" * 70)
    print("Example 7: Recovery Manager")
    print("=" * 70)

    manager = RecoveryManager(fallback_models=["qwen3:8b", "gemma3:4b"])

    # Execute with recovery
    print("\nExecuting with Recovery:")

    attempt = {"count": 0}

    def risky_operation(value):
        attempt["count"] += 1
        logger.info(f"Risky operation attempt {attempt['count']}")

        if attempt["count"] < 2:
            raise ConnectionError("Temporary failure")

        return f"Success with value: {value}"

    try:
        result = manager.execute_with_recovery(
            risky_operation,
            "test_value",
            checkpoint_key="operation_1",
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Checkpoint demonstration
    print("\nCheckpoint Management:")
    manager.save_checkpoint("state_1", {"progress": 50, "data": [1, 2, 3]})
    manager.save_checkpoint("state_2", {"progress": 75, "data": [4, 5, 6]})

    restored = manager.restore_checkpoint("state_1")
    print(f"Restored checkpoint 'state_1': {restored}")

    # System health
    print("\nSystem Health Report:")
    health = manager.get_system_health(models=["qwen3:8b"])
    for component, status in health.items():
        print(f"  {component}: {'OK' if status.healthy else 'FAIL'}")


# ============================================================================
# Example 8: Decorator Patterns
# ============================================================================


def example_8_decorators():
    """Demonstrate decorator patterns."""
    print("\n" + "=" * 70)
    print("Example 8: Decorator Patterns")
    print("=" * 70)

    # Retry decorator
    @with_retry(max_retries=3, backoff_factor=1.5)
    def fetch_data():
        """Simulated API call."""
        logger.info("Fetching data...")
        if random.random() < 0.5:
            raise TimeoutError("Request timeout")
        return {"data": [1, 2, 3]}

    print("\nRetry Decorator:")
    try:
        result = fetch_data()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Circuit breaker decorator
    @with_circuit_breaker(failure_threshold=2, recovery_timeout=3.0)
    def call_api():
        """Simulated external API call."""
        if random.random() < 0.7:
            raise ConnectionError("API unavailable")
        return "API Response"

    print("\nCircuit Breaker Decorator:")
    for i in range(5):
        try:
            result = call_api()
            print(f"Call {i + 1}: {result}")
        except (ConnectionError, RuntimeError) as e:
            print(f"Call {i + 1}: {e}")
        time.sleep(0.5)


# ============================================================================
# Example 9: Real-World Pattern - Connection Failures
# ============================================================================


def example_9_connection_failure_pattern():
    """Demonstrate handling connection failures."""
    print("\n" + "=" * 70)
    print("Example 9: Connection Failure Pattern")
    print("=" * 70)

    retry = RetryStrategy(max_retries=3, base_delay=1.0)
    classifier = ErrorClassifier()
    health = HealthCheck()

    def connect_to_ollama():
        """Simulate connecting to Ollama."""
        # Check if Ollama is actually running
        status = health.check_ollama()
        if not status.healthy:
            raise ConnectionError("Ollama server not running")
        return "Connected"

    try:
        result = retry.execute(connect_to_ollama)
        print(f"Connection result: {result}")
    except ConnectionError as e:
        error_info = classifier.classify(e)
        print(f"Connection failed: {e}")
        print(f"Category: {error_info.category.value}")
        print(f"Suggested action: {error_info.suggested_action}")


# ============================================================================
# Example 10: Real-World Pattern - Model Timeout with Fallback
# ============================================================================


def example_10_model_timeout_pattern():
    """Demonstrate handling model timeouts with fallback."""
    print("\n" + "=" * 70)
    print("Example 10: Model Timeout with Fallback")
    print("=" * 70)

    RecoveryManager(fallback_models=["qwen3:8b", "gemma3:4b"])
    degradation = GracefulDegradation()

    def generate_text(prompt, model="qwen3:8b", timeout=10):
        """Simulate text generation with potential timeout."""
        cache_key = f"{model}:{hash(prompt)}"

        # Check cache first
        cached = degradation.get_cached_response(cache_key, max_age=300)
        if cached:
            print(f"Cache hit for model {model}")
            return cached

        # Simulate generation
        logger.info(f"Generating with model {model}")

        # Simulate random timeout
        if random.random() < 0.3:
            raise TimeoutError(f"Model {model} timeout after {timeout}s")

        result = f"Generated text with {model}"

        # Cache result
        degradation.cache_response(cache_key, result)

        return result

    # Try with fallback chain
    models = ["qwen3:30b", "qwen3:8b", "gemma3:4b"]
    prompt = "What is machine learning?"

    for model in models:
        try:
            result = generate_text(prompt, model=model)
            print(f"Success with {model}: {result}")
            break
        except TimeoutError:
            print(f"Timeout with {model}, trying fallback...")
            fallback = degradation.get_fallback_model(model)
            if not fallback:
                print("No more fallbacks available")
                break


# ============================================================================
# Example 11: Real-World Pattern - Resource Management
# ============================================================================


def example_11_resource_management_pattern():
    """Demonstrate resource monitoring and management."""
    print("\n" + "=" * 70)
    print("Example 11: Resource Management Pattern")
    print("=" * 70)

    health = HealthCheck()
    degradation = GracefulDegradation()

    def process_large_dataset(items, batch_size=100):
        """Process data with resource awareness."""
        # Check memory before processing
        status = health.check_system_resources()
        memory_percent = status.details["memory_percent"]

        print(f"Memory usage: {memory_percent:.1f}%")

        if memory_percent > 80:
            # Reduce batch size if memory is high
            batch_size = max(10, batch_size // 2)
            print(f"High memory usage detected, reducing batch size to {batch_size}")

        if memory_percent > 90:
            # Clear caches if critically low
            print("Critical memory usage, clearing caches")
            degradation.clear_cache()

        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} items)")
            results.extend([f"processed_{item}" for item in batch])

        return results

    # Simulate processing
    test_items = list(range(250))
    results = process_large_dataset(test_items, batch_size=100)
    print(f"Processed {len(results)} items total")


# ============================================================================
# Example 12: Complete Integration
# ============================================================================


def example_12_complete_integration():
    """Demonstrate complete integration of all recovery patterns."""
    print("\n" + "=" * 70)
    print("Example 12: Complete Integration")
    print("=" * 70)

    # Initialize all components
    manager = RecoveryManager(fallback_models=["qwen3:8b", "gemma3:4b"])

    # Pre-flight health check
    print("\nPre-flight Health Check:")
    health = manager.get_system_health(models=["qwen3:8b"])

    all_healthy = all(status.healthy for status in health.values())

    if all_healthy:
        print("All systems healthy ✓")
    else:
        print("System health issues detected:")
        for component, status in health.items():
            if not status.healthy:
                print(f"  - {component}: {status.message}")

    # Execute complex operation with full recovery
    print("\nExecuting Complex Operation:")

    def complex_operation(data):
        """Simulated complex operation."""
        logger.info(f"Processing data: {data}")

        # Simulate occasional failure
        if random.random() < 0.3:
            raise ConnectionError("Service temporarily unavailable")

        return {"result": "processed", "data": data}

    def fallback_operation(data):
        """Fallback returns simplified result."""
        logger.info("Using fallback operation")
        return {"result": "fallback", "data": data}

    try:
        result = manager.execute_with_recovery(
            complex_operation,
            {"input": "test_data"},
            fallback=fallback_operation,
            checkpoint_key="complex_op",
        )
        print(f"Operation result: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")

    # Cleanup
    print("\nPerforming Graceful Shutdown:")
    manager.shutdown_gracefully()
    print("Shutdown complete ✓")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ERROR RECOVERY PATTERNS DEMONSTRATION")
    print("=" * 70)

    examples = [
        ("Basic Retry", example_1_basic_retry),
        ("Retry with Fallback", example_2_retry_with_fallback),
        ("Error Classification", example_3_error_classification),
        ("Circuit Breaker", example_4_circuit_breaker),
        ("Graceful Degradation", example_5_graceful_degradation),
        ("Health Checks", example_6_health_checks),
        ("Recovery Manager", example_7_recovery_manager),
        ("Decorators", example_8_decorators),
        ("Connection Failures", example_9_connection_failure_pattern),
        ("Model Timeout", example_10_model_timeout_pattern),
        ("Resource Management", example_11_resource_management_pattern),
        ("Complete Integration", example_12_complete_integration),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n[ERROR in {name}] {e}")
            logger.exception(f"Example {name} failed")

        # Pause between examples
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(
        "\nFor more details, see:"
        "\n  - Source: /Volumes/JS-DEV/ai-lang-stuff/utils/error_recovery.py"
        "\n  - Docs: /Volumes/JS-DEV/ai-lang-stuff/docs/error-recovery-patterns.md"
        "\n  - Tests: /Volumes/JS-DEV/ai-lang-stuff/tests/test_error_recovery.py"
    )


if __name__ == "__main__":
    main()
