# Error Recovery Patterns

Comprehensive guide to error recovery utilities in the local-first AI toolkit.

## Overview

The `utils/error_recovery.py` module provides production-ready error recovery patterns including:

- **RetryStrategy**: Exponential backoff with jitter
- **ErrorClassifier**: Intelligent error categorization
- **GracefulDegradation**: Fallback mechanisms
- **HealthCheck**: System health monitoring
- **RecoveryManager**: Orchestrated recovery strategies
- **CircuitBreaker**: Prevent cascading failures

---

## Quick Start

```python
from utils.error_recovery import RecoveryManager, RetryStrategy, with_retry

# Simple retry decorator
@with_retry(max_retries=3, backoff_factor=2.0)
def fetch_data():
    return api_call()

# Comprehensive recovery
manager = RecoveryManager()
result = manager.execute_with_recovery(
    risky_operation,
    fallback=safe_operation,
    checkpoint_key="op_state"
)

# Check system health
health = manager.get_system_health(models=["qwen3:8b"])
for component, status in health.items():
    print(f"{component}: {'OK' if status.healthy else 'FAIL'}")
```

---

## RetryStrategy

Implements retry logic with exponential backoff, jitter, and circuit breaker pattern.

### Features

- **Exponential Backoff**: Delays increase exponentially between retries
- **Jitter**: Random variation prevents thundering herd in distributed systems
- **Circuit Breaker**: Prevents repeated calls to failing services
- **Fallback**: Execute alternative function if all retries fail

### Basic Usage

```python
from utils.error_recovery import RetryStrategy

retry = RetryStrategy(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True,
    use_circuit_breaker=True
)

# Execute with retries
result = retry.execute(api_call, param1, param2)

# Execute with fallback
result = retry.execute(
    api_call,
    param1,
    fallback=cache_lookup
)
```

### Async Support

```python
async def fetch_async():
    retry = RetryStrategy(max_retries=3)
    result = await retry.execute_async(
        async_api_call,
        param1,
        fallback=async_cache_lookup
    )
    return result
```

### Configuration

```python
from utils.error_recovery import RetryConfig

config = RetryConfig(
    max_retries=5,           # Maximum retry attempts
    base_delay=2.0,          # Initial delay (seconds)
    max_delay=120.0,         # Maximum delay cap
    backoff_factor=3.0,      # Exponential multiplier
    jitter=True,             # Add random jitter
    jitter_range=(0.8, 1.2)  # Jitter multiplier range
)
```

### Delay Calculation

Delay is calculated as:
```
delay = min(base_delay * (backoff_factor ** attempt), max_delay)

# With jitter enabled:
delay *= random.uniform(0.8, 1.2)
```

**Example delays with base_delay=1.0, backoff_factor=2.0:**

| Attempt | Delay (no jitter) | Delay (with jitter) |
|---------|-------------------|---------------------|
| 0       | 1.0s              | 0.8s - 1.2s         |
| 1       | 2.0s              | 1.6s - 2.4s         |
| 2       | 4.0s              | 3.2s - 4.8s         |
| 3       | 8.0s              | 6.4s - 9.6s         |

---

## CircuitBreaker

Prevents cascading failures by temporarily blocking calls to failing services.

### States

1. **CLOSED**: Normal operation, calls pass through
2. **OPEN**: Service failing, reject all calls
3. **HALF_OPEN**: Testing if service recovered

### Usage

```python
from utils.error_recovery import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,    # Failures before opening
    recovery_timeout=60.0,  # Seconds before retry
    success_threshold=2     # Successes to close from half-open
)

# Execute with circuit breaker
try:
    result = breaker.call(api_function, arg1, arg2)
except RuntimeError as e:
    # Circuit is open
    print(f"Service unavailable: {e}")
```

### Decorator Pattern

```python
from utils.error_recovery import with_circuit_breaker

@with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
def call_external_service():
    return service.request()
```

### State Transitions

```
CLOSED --(failures >= threshold)--> OPEN
OPEN --(timeout elapsed)--> HALF_OPEN
HALF_OPEN --(success >= threshold)--> CLOSED
HALF_OPEN --(any failure)--> OPEN
```

---

## ErrorClassifier

Classifies errors to determine appropriate recovery strategies.

### Error Categories

- **TRANSIENT**: Temporary errors, retry likely to succeed (5xx, timeouts)
- **PERMANENT**: Permanent failures, retry won't help (404, validation errors)
- **RESOURCE**: Resource exhaustion (memory, disk)
- **CONNECTION**: Network/connection issues
- **TIMEOUT**: Operation timeout
- **UNKNOWN**: Unclassified errors

### Error Severity

- **LOW**: Non-critical, can continue
- **MEDIUM**: Important but recoverable
- **HIGH**: Critical, requires immediate action
- **CRITICAL**: System failure, requires shutdown

### Usage

```python
from utils.error_recovery import ErrorClassifier

classifier = ErrorClassifier(max_history=100)

try:
    risky_operation()
except Exception as e:
    error_info = classifier.classify(e, context={"operation": "data_fetch"})

    print(f"Category: {error_info.category.value}")
    print(f"Severity: {error_info.severity.value}")
    print(f"Retryable: {error_info.retryable}")
    print(f"Action: {error_info.suggested_action}")
```

### Error Pattern Analysis

```python
# Track errors over time
patterns = classifier.get_error_patterns()
print(f"Connection errors: {patterns.get('connection', 0)}")
print(f"Timeout errors: {patterns.get('timeout', 0)}")

# Clear history
classifier.clear_history()
```

### Classification Rules

| Error Type              | Category   | Retryable | Suggested Action                        |
|-------------------------|------------|-----------|------------------------------------------|
| ConnectionError         | CONNECTION | Yes       | Check service, retry with backoff       |
| TimeoutError            | TIMEOUT    | Yes       | Increase timeout, retry                 |
| MemoryError             | RESOURCE   | No        | Clear caches, use smaller model         |
| "model not found"       | PERMANENT  | No        | Pull model or use fallback              |
| "rate limit" / 429      | TRANSIENT  | Yes       | Wait and retry                          |
| 500/502/503/504         | TRANSIENT  | Yes       | Retry with backoff                      |
| ValueError/TypeError    | PERMANENT  | No        | Fix input validation                    |

---

## GracefulDegradation

Provides fallback mechanisms when primary operations fail.

### Features

- Fallback to smaller/faster models
- Response caching
- Workflow simplification
- Automatic degradation chain

### Usage

```python
from utils.error_recovery import GracefulDegradation

degradation = GracefulDegradation(fallback_models=[
    "qwen3:8b",    # Primary
    "gemma3:4b"    # Fallback
])

# Get next fallback model
fallback = degradation.get_fallback_model("qwen3:30b")
print(f"Falling back to: {fallback}")

# Cache responses
degradation.cache_response("query_123", result)
cached = degradation.get_cached_response("query_123", max_age=3600)

# Simplify workflow
simplified = degradation.simplify_workflow(complex_workflow)
```

### Model Fallback Chain

```python
# Define fallback chain by use case
speed_chain = ["qwen3:30b-a3b", "qwen3:8b", "gemma3:4b"]
quality_chain = ["qwen3:30b", "qwen3:8b"]
edge_chain = ["gemma3:4b", "gemma3:3b"]

degradation = GracefulDegradation(fallback_models=speed_chain)
```

### Response Caching

```python
# Save expensive computation
key = f"embedding_{text_hash}"
degradation.cache_response(key, embeddings)

# Retrieve later (max age 1 hour)
cached = degradation.get_cached_response(key, max_age=3600)
if cached:
    return cached

# Clear cache
degradation.clear_cache()
```

### Workflow Simplification

```python
complex_workflow = {
    "max_iterations": 10,
    "tools": ["tool1", "tool2", "tool3", "tool4"],
    "temperature": 0.7,
    "top_k": 50
}

# Simplify for degraded mode
simplified = degradation.simplify_workflow(complex_workflow)
# Result: {
#     "max_iterations": 3,
#     "tools": ["tool1", "tool2", "tool3"],
#     "temperature": 0.3
# }
```

---

## HealthCheck

Monitor health of system components.

### Components Monitored

1. **Ollama Server**: Connection and responsiveness
2. **Models**: Availability of specific models
3. **System Resources**: Memory, disk, CPU

### Usage

```python
from utils.error_recovery import HealthCheck

health = HealthCheck(
    ollama_base_url="http://localhost:11434",
    timeout=5
)

# Check Ollama server
status = health.check_ollama()
if status.healthy:
    print("Ollama is running")
else:
    print(f"Ollama issue: {status.message}")

# Check specific model
status = health.check_model_available("qwen3:8b")
print(f"Model available: {status.healthy}")

# Check system resources
status = health.check_system_resources()
print(f"Memory: {status.details['memory_available_gb']:.1f}GB available")
print(f"Disk: {status.details['disk_available_gb']:.1f}GB free")
print(f"CPU: {status.details['cpu_percent']:.1f}% usage")

# Check everything
all_statuses = health.check_all(models=["qwen3:8b", "gemma3:4b"])
for component, status in all_statuses.items():
    print(f"{component}: {'OK' if status.healthy else 'FAIL'} - {status.message}")
```

### Health Status Structure

```python
@dataclass
class HealthStatus:
    component: str          # Component name
    healthy: bool           # Health status
    message: str            # Status message
    details: Dict[str, Any] # Additional details
    timestamp: float        # Check timestamp
```

### Resource Thresholds

- **Memory**: Warning if >90% used
- **Disk**: Warning if >90% used
- **CPU**: Warning if >90% used (informational)

---

## RecoveryManager

Orchestrates all recovery strategies for comprehensive error handling.

### Features

- Coordinates retry, classification, degradation, health checks
- Checkpoint/restore state
- Graceful shutdown
- Integrated error handling

### Usage

```python
from utils.error_recovery import RecoveryManager

manager = RecoveryManager(
    retry_strategy=None,  # Use default
    fallback_models=["qwen3:8b", "gemma3:4b"]
)

# Execute with full recovery
result = manager.execute_with_recovery(
    risky_function,
    arg1, arg2,
    fallback=safe_function,
    checkpoint_key="operation_1"
)

# Save/restore checkpoints
manager.save_checkpoint("state_1", {"data": [1, 2, 3]})
state = manager.restore_checkpoint("state_1")

# Get system health
health = manager.get_system_health(models=["qwen3:8b"])

# Graceful shutdown
manager.shutdown_gracefully()
```

### Custom Retry Strategy

```python
from utils.error_recovery import RetryStrategy, RecoveryManager

custom_retry = RetryStrategy(
    max_retries=5,
    base_delay=2.0,
    backoff_factor=3.0,
    jitter=True
)

manager = RecoveryManager(retry_strategy=custom_retry)
```

### Checkpoint Pattern

```python
# Save state before risky operation
manager.save_checkpoint("import_job", {
    "current_row": 1000,
    "file_path": "/path/to/file.csv",
    "errors": []
})

try:
    continue_import()
except Exception as e:
    # Restore and resume
    state = manager.restore_checkpoint("import_job")
    resume_import(from_row=state["current_row"])
```

---

## Common Patterns

### Pattern 1: Connection Failures

```python
from utils.error_recovery import RetryStrategy, ErrorClassifier

retry = RetryStrategy(max_retries=3, base_delay=2.0)
classifier = ErrorClassifier()

try:
    result = retry.execute(connect_to_service)
except ConnectionError as e:
    error_info = classifier.classify(e)
    print(f"Connection failed: {error_info.suggested_action}")
    # Fall back to cached data or alternative service
```

### Pattern 2: Model Timeouts

```python
from utils.error_recovery import RecoveryManager, GracefulDegradation

manager = RecoveryManager(fallback_models=["qwen3:8b", "gemma3:4b"])
degradation = GracefulDegradation()

def generate_with_fallback(prompt, model="qwen3:30b"):
    # Check cache first
    cache_key = f"{model}:{hash(prompt)}"
    cached = degradation.get_cached_response(cache_key)
    if cached:
        return cached

    try:
        result = manager.execute_with_recovery(
            llm_generate,
            prompt,
            model=model
        )
        degradation.cache_response(cache_key, result)
        return result
    except TimeoutError:
        # Try smaller model
        fallback_model = degradation.get_fallback_model(model)
        if fallback_model:
            return generate_with_fallback(prompt, model=fallback_model)
        raise
```

### Pattern 3: Out of Memory

```python
from utils.error_recovery import HealthCheck, GracefulDegradation

health = HealthCheck()
degradation = GracefulDegradation()

def process_batch(items, batch_size=100):
    # Check memory before processing
    status = health.check_system_resources()

    if status.details["memory_percent"] > 80:
        # Reduce batch size
        batch_size = max(10, batch_size // 2)
        print(f"High memory usage, reducing batch to {batch_size}")

    try:
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            results.extend(process(batch))
        return results
    except MemoryError:
        # Clear caches and retry with smaller batches
        degradation.clear_cache()
        return process_batch(items, batch_size=batch_size // 2)
```

### Pattern 4: Corrupted State

```python
from utils.error_recovery import RecoveryManager

manager = RecoveryManager()

def long_running_operation(data):
    # Checkpoint every 100 items
    for i, item in enumerate(data):
        if i % 100 == 0:
            manager.save_checkpoint("progress", {
                "index": i,
                "processed": i,
                "errors": []
            })

        try:
            process_item(item)
        except Exception as e:
            # Restore from checkpoint
            state = manager.restore_checkpoint("progress")
            print(f"Resuming from index {state['index']}")
            # Could retry or skip
```

### Pattern 5: Missing Dependencies

```python
from utils.error_recovery import HealthCheck

health = HealthCheck()

def startup_checks():
    """Validate all dependencies before running."""
    issues = []

    # Check Ollama
    if not health.check_ollama().healthy:
        issues.append("Ollama server not running. Start with: ollama serve")

    # Check required models
    for model in ["qwen3:8b", "gemma3:4b"]:
        status = health.check_model_available(model)
        if not status.healthy:
            issues.append(f"Model {model} not found. Pull with: ollama pull {model}")

    # Check resources
    status = health.check_system_resources()
    if status.details["memory_available_gb"] < 4.0:
        issues.append("Low memory: less than 4GB available")

    if issues:
        print("Startup issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("All startup checks passed")
    return True
```

---

## Decorators

### @with_retry

Add retry logic to any function.

```python
from utils.error_recovery import with_retry

@with_retry(max_retries=3, backoff_factor=2.0)
def fetch_data(url):
    return requests.get(url).json()

@with_retry(max_retries=5, fallback=get_cached_data)
def fetch_with_fallback(key):
    return expensive_api_call(key)
```

### @with_circuit_breaker

Add circuit breaker to prevent cascading failures.

```python
from utils.error_recovery import with_circuit_breaker

@with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
def call_external_api():
    return api.request()

# Circuit opens after 5 failures
# Stays open for 60 seconds
# Then tries half-open state
```

---

## Integration Examples

### With LangChain Agents

```python
from langchain_ollama import ChatOllama
from utils.error_recovery import RecoveryManager, GracefulDegradation

manager = RecoveryManager(fallback_models=["qwen3:8b", "gemma3:4b"])
degradation = GracefulDegradation()

def create_agent_with_recovery(model="qwen3:8b"):
    def invoke_with_recovery(prompt):
        cache_key = f"{model}:{hash(prompt)}"

        # Try cache first
        cached = degradation.get_cached_response(cache_key)
        if cached:
            return cached

        # Execute with recovery
        try:
            llm = ChatOllama(model=model)
            result = manager.execute_with_recovery(
                llm.invoke,
                prompt
            )
            degradation.cache_response(cache_key, result)
            return result
        except Exception as e:
            # Try fallback model
            fallback_model = degradation.get_fallback_model(model)
            if fallback_model and fallback_model != model:
                return create_agent_with_recovery(fallback_model)(prompt)
            raise

    return invoke_with_recovery
```

### With MCP Servers

```python
from utils.error_recovery import RetryStrategy, with_circuit_breaker

retry = RetryStrategy(max_retries=3)

@with_circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
def call_mcp_tool(tool_name, args):
    return mcp_client.call_tool(tool_name, args)

def safe_mcp_call(tool_name, args, fallback_result=None):
    try:
        return retry.execute(
            call_mcp_tool,
            tool_name,
            args,
            fallback=lambda *a, **k: fallback_result
        )
    except RuntimeError as e:
        if "Circuit breaker OPEN" in str(e):
            print("MCP server circuit open, using fallback")
            return fallback_result
        raise
```

### With Vector Stores

```python
from utils.error_recovery import HealthCheck, RetryStrategy

health = HealthCheck()
retry = RetryStrategy(max_retries=3)

def vector_search_with_recovery(query, collection):
    # Check resources first
    status = health.check_system_resources()
    if status.details["memory_percent"] > 90:
        print("High memory, clearing caches")
        collection.clear_cache()

    # Search with retry
    try:
        return retry.execute(
            collection.search,
            query,
            fallback=lambda *a, **k: []
        )
    except MemoryError:
        # Use smaller index or reduce results
        print("Memory error, reducing result limit")
        return collection.search(query, limit=10)
```

---

## Best Practices

### 1. Choose Appropriate Retry Counts

- **Network calls**: 3-5 retries
- **Local operations**: 1-2 retries
- **Database**: 3 retries
- **External APIs**: 5+ retries with longer delays

### 2. Set Reasonable Timeouts

```python
# Quick operations
retry = RetryStrategy(max_retries=3, base_delay=0.5, max_delay=5.0)

# Slow operations (model inference)
retry = RetryStrategy(max_retries=3, base_delay=2.0, max_delay=60.0)

# Very slow operations (model downloads)
retry = RetryStrategy(max_retries=5, base_delay=5.0, max_delay=300.0)
```

### 3. Always Use Jitter for Distributed Systems

```python
# Good: Prevents thundering herd
retry = RetryStrategy(jitter=True)

# Bad: All clients retry simultaneously
retry = RetryStrategy(jitter=False)
```

### 4. Combine Strategies

```python
# Use all recovery patterns together
manager = RecoveryManager()
degradation = GracefulDegradation()
health = HealthCheck()

# Pre-flight check
if not health.check_ollama().healthy:
    raise RuntimeError("Ollama not running")

# Execute with recovery
result = manager.execute_with_recovery(
    operation,
    fallback=degradation.get_cached_response("op_cache"),
    checkpoint_key="operation_state"
)
```

### 5. Monitor and Log

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Error recovery automatically logs
manager = RecoveryManager()

# Check logs for patterns
classifier = ErrorClassifier()
patterns = classifier.get_error_patterns()
logger.info(f"Error patterns: {patterns}")
```

---

## Performance Considerations

### Memory Usage

- ErrorClassifier history: ~100 errors (configurable)
- GracefulDegradation cache: Bounded by max_age expiration
- RecoveryManager checkpoints: Manual cleanup required

### CPU Usage

- Minimal overhead for retry logic
- Health checks: ~1% CPU for resource monitoring
- Circuit breaker: Negligible overhead

### Recommendations

```python
# For high-throughput systems
manager = RecoveryManager()
manager.classifier.max_history = 50  # Reduce history

# Clear caches periodically
manager.degradation.clear_cache()
manager.clear_checkpoints()

# Use circuit breakers for external calls
retry = RetryStrategy(use_circuit_breaker=True)
```

---

## Testing

See `tests/test_error_recovery.py` for comprehensive test suite.

```python
# Run tests
pytest tests/test_error_recovery.py -v

# Run with coverage
pytest tests/test_error_recovery.py --cov=utils/error_recovery

# Run specific test
pytest tests/test_error_recovery.py::test_retry_with_backoff -v
```

---

## Troubleshooting

### Issue: Retries Exhausted Too Quickly

**Solution**: Increase max_retries or base_delay

```python
# Too aggressive
retry = RetryStrategy(max_retries=3, base_delay=0.5)

# Better
retry = RetryStrategy(max_retries=5, base_delay=2.0)
```

### Issue: Circuit Breaker Always Open

**Solution**: Adjust failure threshold or recovery timeout

```python
# Too sensitive
breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

# More tolerant
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
```

### Issue: Memory Leaks from Caching

**Solution**: Clear caches periodically or reduce max_age

```python
# Clear caches regularly
degradation.clear_cache()

# Use shorter max_age
cached = degradation.get_cached_response(key, max_age=300)  # 5 minutes
```

### Issue: Checkpoints Growing Too Large

**Solution**: Clear old checkpoints

```python
# Clear after successful operation
manager.execute_with_recovery(operation, checkpoint_key="op")
manager.clear_checkpoints()

# Or clear specific checkpoint
manager.checkpoints.pop("old_checkpoint", None)
```

---

## API Reference

See source code docstrings for complete API documentation:

```python
help(RetryStrategy)
help(ErrorClassifier)
help(GracefulDegradation)
help(HealthCheck)
help(RecoveryManager)
help(CircuitBreaker)
```

---

## Related Documentation

- [Ollama Manager](/Volumes/JS-DEV/ai-lang-stuff/utils/ollama_manager.py)
- [MCP Client](/Volumes/JS-DEV/ai-lang-stuff/utils/mcp_client.py)
- [State Manager](/Volumes/JS-DEV/ai-lang-stuff/utils/state_manager.py)
- [Development Plan](/Volumes/JS-DEV/ai-lang-stuff/docs/DEVELOPMENT-PLAN-20-POINTS.md)
