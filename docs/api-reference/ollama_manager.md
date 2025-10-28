# OllamaManager API Reference

Comprehensive Ollama server management, model operations, and intelligent recommendations.

## Overview

The `OllamaManager` class provides a production-ready interface for managing Ollama server operations including health checks, model management, benchmarking, and intelligent model recommendations based on task requirements.

**Module:** `utils.ollama_manager`

**Dependencies:**
- `requests` - HTTP client for Ollama API
- Python 3.9+

---

## Class: OllamaManager

```python
class OllamaManager:
    """Manage Ollama server and model operations."""
```

### Constructor

```python
def __init__(
    self,
    base_url: str = "http://localhost:11434",
    timeout: int = 30
) -> None
```

Initialize the OllamaManager.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:11434"` | Base URL of Ollama server |
| `timeout` | `int` | `30` | Request timeout in seconds |

**Example:**

```python
from utils.ollama_manager import OllamaManager

# Default configuration
manager = OllamaManager()

# Custom configuration
manager = OllamaManager(
    base_url="http://192.168.1.100:11434",
    timeout=60
)
```

---

## Health Check Methods

### check_ollama_running()

```python
def check_ollama_running(self) -> bool
```

Test connection to Ollama server and verify it's responsive.

**Returns:**
- `bool` - `True` if server is running and responsive, `False` otherwise

**Raises:**
- No exceptions raised - returns `False` on any error

**Example:**

```python
manager = OllamaManager()

if manager.check_ollama_running():
    print("Ollama is ready")
    models = manager.list_models()
else:
    print("Ollama is not running. Start it with: ollama serve")
```

**Implementation Notes:**
- Uses short 5-second timeout for quick checks
- Handles `ConnectionError`, `Timeout`, and `RequestException`
- Logs appropriate warnings on failure
- Safe to call repeatedly

---

## Model Management Methods

### list_models()

```python
def list_models(self) -> List[str]
```

Get list of installed models from Ollama.

**Returns:**
- `List[str]` - List of model names (e.g., `["qwen3:8b", "gemma3:4b"]`)
- Returns empty list on error

**Example:**

```python
manager = OllamaManager()
models = manager.list_models()

print(f"Found {len(models)} installed models:")
for model in models:
    print(f"  - {model}")
```

**Error Handling:**
```python
models = manager.list_models()
if not models:
    print("No models installed or Ollama not running")
```

---

### ensure_model_available()

```python
def ensure_model_available(self, model: str) -> bool
```

Ensure a model is available, pulling if necessary.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name (e.g., `"qwen3:8b"`) |

**Returns:**
- `bool` - `True` if model is available, `False` if pull failed

**Example:**

```python
manager = OllamaManager()

# Ensure model is available
if manager.ensure_model_available("qwen3:8b"):
    print("Model ready to use")
    # Proceed with model operations
else:
    print("Failed to get model")
```

**Behavior:**
1. Checks if model is already installed
2. Returns `True` immediately if found
3. Attempts to pull model if not found
4. Shows progress during download

**Use Cases:**
- Pre-flight checks before agent execution
- Automatic dependency resolution
- Setup scripts and initialization

---

### pull_model()

```python
def pull_model(self, model: str) -> bool
```

Pull a model from Ollama registry with progress updates.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name to pull |

**Returns:**
- `bool` - `True` on success, `False` on failure

**Example:**

```python
manager = OllamaManager()

print("Downloading model...")
success = manager.pull_model("qwen3:8b")

if success:
    print("Model downloaded successfully")
else:
    print("Download failed")
```

**Progress Output:**
```
Pulling model 'qwen3:8b'...
pulling manifest: 100.0%
downloading digestname: 45.2%
```

**Error Conditions:**
- `ConnectionError` - Cannot connect to Ollama
- `Timeout` - Download took too long
- `RequestException` - HTTP error

---

### get_model_info()

```python
def get_model_info(self, model: str) -> Dict[str, Any]
```

Get detailed information about a model.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name |

**Returns:**
- `Dict[str, Any]` - Model metadata dictionary
- Returns empty dict if model not found

**Response Keys:**
- `modelfile` - Model configuration
- `parameters` - Model parameters
- `template` - Prompt template
- `details` - Size, family, parameter count

**Example:**

```python
manager = OllamaManager()
info = manager.get_model_info("qwen3:8b")

if info:
    details = info.get("details", {})
    print(f"Model size: {details.get('parameter_size')}")
    print(f"Model family: {details.get('family')}")
    print(f"Quantization: {details.get('quantization_level')}")
```

**Use Cases:**
- Verifying model capabilities
- Displaying model information to users
- Selecting models based on size/parameters

---

## Benchmarking Methods

### benchmark_model()

```python
def benchmark_model(
    self,
    model: str,
    prompt: str = "Hello, how are you?"
) -> Dict[str, Any]
```

Benchmark a model's performance with latency and throughput metrics.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | - | Model name to benchmark |
| `prompt` | `str` | No | `"Hello, how are you?"` | Test prompt |

**Returns:**

Dictionary with benchmarking results:

```python
{
    "model": str,           # Model name
    "latency": float,       # Total time in seconds
    "tokens_per_sec": float,# Throughput
    "prompt": str,          # Input prompt
    "response": str,        # Model response
    "error": str           # Error message (if failed)
}
```

**Example:**

```python
manager = OllamaManager()

# Basic benchmark
results = manager.benchmark_model("qwen3:8b")
print(f"Latency: {results['latency']:.2f}s")
print(f"Throughput: {results['tokens_per_sec']:.1f} tokens/sec")

# Custom prompt
results = manager.benchmark_model(
    "qwen3:8b",
    prompt="Write a Python function to calculate fibonacci"
)
```

**Comparing Models:**

```python
models = ["qwen3:8b", "gemma3:4b", "qwen3:30b-a3b"]
results = []

for model in models:
    if manager.ensure_model_available(model):
        benchmark = manager.benchmark_model(model)
        results.append(benchmark)

# Sort by speed
results.sort(key=lambda x: x["tokens_per_sec"], reverse=True)

print("Models ranked by speed:")
for r in results:
    print(f"{r['model']}: {r['tokens_per_sec']:.1f} tok/s")
```

---

## Recommendation Methods

### recommend_model()

```python
def recommend_model(self, task_type: str) -> str
```

Recommend optimal model based on task requirements.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_type` | `str` | Yes | Task category (see table below) |

**Supported Task Types:**

| Task Type | Recommended Model | Rationale |
|-----------|------------------|-----------|
| `"fast"` | `qwen3:30b-a3b` | MoE optimized for speed |
| `"balanced"` | `qwen3:8b` | Good quality/speed tradeoff |
| `"quality"` | `qwen3:30b` | Best reasoning quality |
| `"embeddings"` | `nomic-embed-text` | Optimized for embeddings |
| `"vision"` | `qwen3-vl:8b` | Best local vision model |
| `"edge"` | `gemma3:4b` | Minimal resource usage |
| `"multilingual"` | `gemma3:12b` | 140+ languages |
| `"coding"` | `qwen3:30b-a3b` | Fast code generation |

**Returns:**
- `str` - Recommended model name
- Returns `"qwen3:8b"` (balanced) if task_type not recognized

**Example:**

```python
manager = OllamaManager()

# Get recommendation
model = manager.recommend_model("vision")
print(f"Recommended: {model}")  # "qwen3-vl:8b"

# Ensure it's available
if manager.ensure_model_available(model):
    print(f"Ready to use {model}")
```

**Task-Based Selection:**

```python
def select_model_for_task(task: str) -> str:
    manager = OllamaManager()

    # Map task to task_type
    task_map = {
        "image_analysis": "vision",
        "quick_response": "fast",
        "translation": "multilingual",
        "edge_device": "edge",
        "code_gen": "coding",
    }

    task_type = task_map.get(task, "balanced")
    return manager.recommend_model(task_type)
```

---

## Runtime Information Methods

### get_running_models()

```python
def get_running_models(self) -> List[Dict[str, Any]]
```

Get list of currently running models in Ollama.

**Returns:**
- `List[Dict[str, Any]]` - List of running model information
- Returns empty list on error

**Response Format:**

```python
[
    {
        "name": "qwen3:8b",
        "size": 4678000000,
        "digest": "sha256:abc123...",
        "details": {...}
    }
]
```

**Example:**

```python
manager = OllamaManager()
running = manager.get_running_models()

if running:
    print(f"Currently running {len(running)} models:")
    for model in running:
        name = model.get("name")
        size_gb = model.get("size", 0) / 1e9
        print(f"  - {name} ({size_gb:.2f} GB)")
else:
    print("No models currently running")
```

**Use Cases:**
- Resource monitoring
- Cleanup operations
- System diagnostics

---

## Module-Level Convenience Functions

### check_ollama()

```python
def check_ollama() -> bool
```

Quick function to check if Ollama is running.

**Example:**

```python
from utils.ollama_manager import check_ollama

if check_ollama():
    print("Ready")
```

---

### get_available_models()

```python
def get_available_models() -> List[str]
```

Quick function to get list of installed models.

**Example:**

```python
from utils.ollama_manager import get_available_models

models = get_available_models()
print(f"Installed: {', '.join(models)}")
```

---

### ensure_model()

```python
def ensure_model(model: str) -> bool
```

Quick function to ensure a model is available.

**Example:**

```python
from utils.ollama_manager import ensure_model

if ensure_model("qwen3:8b"):
    # Use model
    pass
```

---

## Class Attributes

### MODEL_RECOMMENDATIONS

```python
MODEL_RECOMMENDATIONS: Dict[str, str] = {
    "fast": "qwen3:30b-a3b",
    "balanced": "qwen3:8b",
    "quality": "qwen3:30b",
    "embeddings": "nomic-embed-text",
    "vision": "qwen3-vl:8b",
    "edge": "gemma3:4b",
    "multilingual": "gemma3:12b",
    "coding": "qwen3:30b-a3b",
}
```

Static mapping of task types to recommended models.

---

## Error Handling

### Exception Types

The module handles these exceptions internally:

- `ConnectionError` - Cannot connect to Ollama server
- `Timeout` - Request took too long
- `RequestException` - HTTP/network error

### Logging Levels

- `INFO` - Successful operations
- `WARNING` - Recoverable issues (server not running)
- `ERROR` - Operation failures
- `DEBUG` - Detailed diagnostics

### Best Practices

```python
import logging
from utils.ollama_manager import OllamaManager

# Configure logging
logging.basicConfig(level=logging.INFO)

manager = OllamaManager()

# Always check server first
if not manager.check_ollama_running():
    print("Start Ollama with: ollama serve")
    exit(1)

# Then verify models
model = "qwen3:8b"
if not manager.ensure_model_available(model):
    print(f"Failed to get model: {model}")
    exit(1)

# Now safe to use
info = manager.get_model_info(model)
```

---

## Performance Considerations

### Timeouts

- Health checks: 5 seconds (hardcoded)
- Regular operations: 30 seconds (configurable)
- Model pulls: 30 seconds per chunk (no total timeout)

**Custom Timeout:**

```python
# For slow connections
manager = OllamaManager(timeout=120)
```

### Caching

The manager doesn't cache responses. Implement application-level caching for repeated queries:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_models() -> List[str]:
    manager = OllamaManager()
    return manager.list_models()
```

### Connection Pooling

Each manager instance creates new connections. Reuse instances:

```python
# Good - reuse manager
manager = OllamaManager()
for model in models:
    manager.get_model_info(model)

# Avoid - creates new connections
for model in models:
    manager = OllamaManager()
    manager.get_model_info(model)
```

---

## Integration Examples

### With LangChain

```python
from langchain_ollama import ChatOllama
from utils.ollama_manager import OllamaManager

manager = OllamaManager()

# Verify server and model
if not manager.check_ollama_running():
    raise RuntimeError("Ollama not running")

model_name = manager.recommend_model("balanced")
manager.ensure_model_available(model_name)

# Create LangChain LLM
llm = ChatOllama(model=model_name)
response = llm.invoke("Hello!")
```

### With Agents

```python
from utils.ollama_manager import OllamaManager
from langchain.agents import AgentExecutor, create_tool_calling_agent

manager = OllamaManager()

# Select fastest model for agent
model = manager.recommend_model("fast")
manager.ensure_model_available(model)

# Benchmark to verify performance
results = manager.benchmark_model(model)
print(f"Agent will use {model} ({results['tokens_per_sec']:.1f} tok/s)")

# Create agent with selected model
llm = ChatOllama(model=model)
agent = create_tool_calling_agent(llm, tools, prompt)
```

### Setup Scripts

```python
#!/usr/bin/env python3
"""Setup script to verify Ollama environment."""

from utils.ollama_manager import OllamaManager

def setup():
    manager = OllamaManager()

    print("Checking Ollama...")
    if not manager.check_ollama_running():
        print("ERROR: Ollama not running")
        print("Start with: ollama serve")
        return False

    print("Installing required models...")
    required = ["qwen3:8b", "nomic-embed-text"]

    for model in required:
        if manager.ensure_model_available(model):
            print(f"✓ {model}")
        else:
            print(f"✗ {model} - failed to install")
            return False

    print("Setup complete!")
    return True

if __name__ == "__main__":
    setup()
```

---

## See Also

- [Vector Store Manager](./vector_store.md) - Uses embeddings from Ollama
- [MCP Client](./mcp_client.md) - External integrations
- [Tool Registry](./tool_registry.md) - Register Ollama operations as tools
- [Examples](../../examples/01-foundation/) - Usage examples

---

**Module Location:** `/Volumes/JS-DEV/ai-lang-stuff/utils/ollama_manager.py`

**Tests:** `/Volumes/JS-DEV/ai-lang-stuff/tests/test_ollama_manager.py`
