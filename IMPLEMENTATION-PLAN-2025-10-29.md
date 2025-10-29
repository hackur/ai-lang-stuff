# Local-First AI Toolkit - Implementation Plan
## Strategic 24-Task Roadmap (2025-10-29)

**Generated**: October 29, 2025
**Status**: Building on Phase 2 completion, focusing on quality and production readiness
**Previous Plans**: DEVELOPMENT-TASKS-28.md (9 tasks completed), MASTER-PLAN-SEQUENTIAL.md (Phase 2 done)

---

## Executive Summary

This plan focuses on **quality, reliability, and production readiness** after successfully completing infrastructure setup. We now have:
- Clean GitHub workflows (all passing)
- Type hints on all utils (100% coverage)
- Comprehensive docstrings
- Batch processing implemented (3-5x speedup)
- 269 tests collected (267 valid, 2 with import errors)
- 35 examples ready for testing

**Strategic Focus**:
1. Fix failing tests and import errors (unblock testing)
2. Complete example validation (28 remaining)
3. Add missing functionality gaps
4. Improve developer experience
5. Production hardening
6. Community preparation

**Total Estimated Time**: 35-45 hours (1-2 weeks of focused work)

---

## Task Breakdown by Priority

### Priority Distribution
- **P0 (Critical)**: 6 tasks - Fix blockers, complete validation
- **P1 (High)**: 9 tasks - Core features, production readiness
- **P2 (Medium)**: 6 tasks - Enhanced features, optimization
- **P3 (Low)**: 3 tasks - Community, polish, future prep

### Category Distribution
- **Testing & Quality**: 7 tasks
- **Features & Capabilities**: 6 tasks
- **Documentation & DX**: 4 tasks
- **Production Readiness**: 4 tasks
- **Performance**: 2 tasks
- **Community**: 1 task

---

## PRIORITY P0: CRITICAL (Do First - 12-15 hours)

### Task 1: Fix Test Collection Errors
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 1-2 hours

**Problem**: 2 test files have import errors preventing 269 tests from running:
- `tests/integration/test_mcp_integration.py`
- `tests/integration/test_multi_agent_workflows.py`

**Action**:
```bash
# Identify exact import errors
python -c "import tests.integration.test_mcp_integration"
python -c "import tests.integration.test_multi_agent_workflows"

# Fix missing imports or circular dependencies
# Run tests again to verify
pytest --collect-only
```

**Acceptance Criteria**:
- [ ] All 269 tests collect without errors
- [ ] Import errors resolved
- [ ] Tests can be run with: `pytest -v`
- [ ] No circular import issues

**Dependencies**: None (blocks all test execution)

**Files Affected**:
- `tests/integration/test_mcp_integration.py`
- `tests/integration/test_multi_agent_workflows.py`

---

### Task 2: Fix Pre-commit Python Version Mismatch
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 30 minutes

**Problem**: Pre-commit hooks installed but looking for Python 3.11, project uses 3.12

**Action**:
```bash
# Update .pre-commit-config.yaml
# Change python version from 3.11 to 3.12
# Or use system python

# Test hooks
pre-commit run --all-files
```

**Acceptance Criteria**:
- [ ] Pre-commit runs without Python version errors
- [ ] All hooks execute successfully
- [ ] Can commit without `--no-verify` flag
- [ ] Documented in DEVELOPMENT.md

**Dependencies**: None

**Files Affected**:
- `.pre-commit-config.yaml`
- `DEVELOPMENT.md` (update instructions)

---

### Task 3: Complete Example Validation (28 Remaining)
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 6-8 hours

**Status**: 2/35 examples tested (error_handling_demo.py, tool_registry_demo.py)

**Action**: Test all examples systematically by category:

```bash
# 01-foundation (6 examples)
uv run python examples/01-foundation/basic_llm_interaction.py
uv run python examples/01-foundation/streaming_responses.py
uv run python examples/01-foundation/prompt_engineering.py
uv run python examples/01-foundation/chat_with_history.py
uv run python examples/01-foundation/model_comparison.py
uv run python examples/01-foundation/function_calling.py

# 02-mcp (6 examples)
uv run python examples/02-mcp/filesystem_mcp.py
uv run python examples/02-mcp/web_search_mcp.py
uv run python examples/02-mcp/combined_tools.py
uv run python examples/02-mcp/tool_chaining.py
uv run python examples/02-mcp/parallel_tools.py
uv run python examples/02-mcp/error_handling_mcp.py

# 03-multi-agent (6 examples)
uv run python examples/03-multi-agent/research_pipeline.py
uv run python examples/03-multi-agent/code_review_pipeline.py
uv run python examples/03-multi-agent/parallel_agents.py
uv run python examples/03-multi-agent/conditional_routing.py
uv run python examples/03-multi-agent/state_checkpointing.py
uv run python examples/03-multi-agent/human_in_loop.py

# 04-rag (8 examples)
uv run python examples/04-rag/document_qa.py
uv run python examples/04-rag/code_search.py
uv run python examples/04-rag/streaming_rag.py
uv run python examples/04-rag/reranking_rag.py
uv run python examples/04-rag/multimodal_rag.py
uv run python examples/04-rag/contextual_compression.py
uv run python examples/04-rag/parent_document.py
uv run python examples/04-rag/vision_rag.py

# 05-interpretability (6 examples)
uv run python examples/05-interpretability/activation_patching.py
uv run python examples/05-interpretability/circuit_discovery.py
uv run python examples/05-interpretability/attention_visualization.py
uv run python examples/05-interpretability/logit_lens.py
uv run python examples/05-interpretability/neuron_analysis.py
uv run python examples/05-interpretability/layer_analysis.py

# 06-production (7 examples)
uv run python examples/06-production/monitoring_setup.py
uv run python examples/06-production/config_management.py
uv run python examples/06-production/health_checks.py
uv run python examples/06-production/deployment_patterns.py
uv run python examples/06-production/batch_processing.py
uv run python examples/06-production/rate_limiting.py
uv run python examples/06-production/circuit_breaker.py

# 07-advanced (7 examples)
uv run python examples/07-advanced/vision_agent.py
uv run python examples/07-advanced/audio_transcription.py
uv run python examples/07-advanced/multimodal_rag.py
uv run python examples/07-advanced/document_understanding.py
uv run python examples/07-advanced/code_generation.py
uv run python examples/07-advanced/agent_memory.py
uv run python examples/07-advanced/tool_creation.py
```

**Acceptance Criteria**:
- [ ] All 35 examples run without errors OR
- [ ] Clear documentation of which examples require Ollama/specific models
- [ ] Add pytest markers for examples that need Ollama: `@pytest.mark.ollama`
- [ ] Update examples/REQUIREMENTS.md with any missing prerequisites
- [ ] Document expected runtime for each example
- [ ] Fix any broken imports or missing dependencies

**Dependencies**: Task 1 (test infrastructure working)

**Files Affected**:
- All 35 example files
- `examples/REQUIREMENTS.md` (update)
- `tests/conftest.py` (add markers)

---

### Task 4: Measure and Improve Test Coverage
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 2-3 hours

**Current State**: Coverage infrastructure installed, not yet measured

**Action**:
```bash
# Run coverage measurement
pytest --cov=utils --cov=workflows --cov-report=html --cov-report=term

# Identify gaps
# Add tests for uncovered code paths
# Focus on utils/ first (most critical)
```

**Target Coverage**:
- `utils/`: 85%+ (critical paths)
- `workflows/`: 75%+ (complex logic)
- Overall: 80%+

**Acceptance Criteria**:
- [ ] Coverage report generated
- [ ] Utils coverage at 85%+
- [ ] HTML coverage report available at `htmlcov/index.html`
- [ ] Coverage badge added to README
- [ ] Coverage CI check added to workflows

**Dependencies**: Task 1 (tests must run)

**Files Affected**:
- All utils files (add missing tests)
- `tests/test_*.py` (expand coverage)
- `.github/workflows/test.yml` (add coverage check)
- `README.md` (add badge)

---

### Task 5: Fix Formatter Conflicts (Ruff vs Black)
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 30 minutes

**Current Issue**: Both black and ruff-format are configured, causing conflicts

**Action**:
```bash
# Choose one formatter (recommend ruff-format for speed)
# Update pyproject.toml to disable conflicting options
# Update pre-commit hooks
# Run formatter on all code

ruff format .
```

**Acceptance Criteria**:
- [ ] Single formatter configured (ruff-format recommended)
- [ ] All code formatted consistently
- [ ] No formatter conflicts in pre-commit
- [ ] CI passes with formatter checks
- [ ] Documentation updated

**Dependencies**: Task 2 (pre-commit working)

**Files Affected**:
- `pyproject.toml` (formatter config)
- `.pre-commit-config.yaml` (remove black or ruff-format)
- `DEVELOPMENT.md` (update instructions)

---

### Task 6: Add pytest Markers for Test Categories
**Priority**: P0
**Category**: Testing & Quality
**Estimated Time**: 1-2 hours

**Problem**: No way to run tests selectively (e.g., skip tests that need Ollama)

**Action**:
```python
# In conftest.py, register markers
pytest_markers = [
    "ollama: tests that require Ollama server running",
    "integration: integration tests (slower)",
    "unit: fast unit tests",
    "benchmark: performance benchmarks",
    "slow: tests that take >5 seconds",
]

# In tests, add markers:
@pytest.mark.ollama
def test_ollama_connection():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_rag_pipeline():
    pass
```

**Run Examples**:
```bash
# Run only unit tests (fast)
pytest -m unit

# Skip Ollama tests
pytest -m "not ollama"

# Run integration tests only
pytest -m integration
```

**Acceptance Criteria**:
- [ ] Markers registered in conftest.py
- [ ] All tests properly marked
- [ ] Can run test subsets independently
- [ ] CI runs different test tiers
- [ ] Documentation in DEVELOPMENT.md

**Dependencies**: Task 1 (tests collecting properly)

**Files Affected**:
- `tests/conftest.py` (register markers)
- All test files (add markers)
- `.github/workflows/test.yml` (use markers)
- `DEVELOPMENT.md` (document usage)

---

## PRIORITY P1: HIGH (Next Phase - 15-18 hours)

### Task 7: Implement Async Vector Store Operations
**Priority**: P1
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Current State**: VectorStoreManager is synchronous only

**Action**: Add async methods for non-blocking RAG operations

```python
# In utils/vector_store.py

async def async_create_from_documents(
    self,
    documents: list[Document],
    collection_name: str,
    embedding_model: str = "qwen3-embedding"
) -> VectorStore:
    """Async version of create_from_documents"""
    pass

async def async_similarity_search(
    self,
    query: str,
    collection_name: str,
    k: int = 5
) -> list[Document]:
    """Async similarity search"""
    pass

async def async_add_documents(
    self,
    documents: list[Document],
    collection_name: str
) -> None:
    """Async document addition"""
    pass
```

**Acceptance Criteria**:
- [ ] Async methods added to VectorStoreManager
- [ ] Tests validate async behavior
- [ ] Example: `examples/04-rag/async_rag.py`
- [ ] Performance comparison vs sync (should be faster)
- [ ] Documentation updated

**Dependencies**: Task 4 (test coverage baseline)

**Files Affected**:
- `utils/vector_store.py` (add async methods)
- `tests/test_vector_store.py` (async tests)
- `examples/04-rag/async_rag.py` (new example)

---

### Task 8: Add Configuration Validation with Pydantic
**Priority**: P1
**Category**: Production Readiness
**Estimated Time**: 2-3 hours

**Problem**: Configuration loaded from dicts/env vars without validation

**Action**: Create Pydantic models for all config

```python
# utils/config.py (new file)

from pydantic import BaseModel, Field, HttpUrl
from typing import Literal

class OllamaConfig(BaseModel):
    """Ollama server configuration"""
    base_url: HttpUrl = Field(default="http://localhost:11434")
    timeout: int = Field(default=120, ge=10, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    store_type: Literal["chroma", "faiss"] = "chroma"
    persist_directory: str = Field(default="./vector_store")
    collection_name: str = Field(min_length=1)

class MCPConfig(BaseModel):
    """MCP server configuration"""
    server_type: Literal["filesystem", "web-search"]
    enabled: bool = True
    config: dict = Field(default_factory=dict)

# Load from environment
config = OllamaConfig(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    timeout=int(os.getenv("OLLAMA_TIMEOUT", "120"))
)
```

**Acceptance Criteria**:
- [ ] Pydantic models for all major configs
- [ ] Environment variable loading
- [ ] YAML/JSON config file loading
- [ ] Validation error messages
- [ ] Tests for config validation
- [ ] Example: `examples/06-production/config_management.py`

**Dependencies**: None

**Files Affected**:
- `utils/config.py` (new file)
- `utils/ollama_manager.py` (use config)
- `utils/vector_store.py` (use config)
- `tests/test_config.py` (new tests)

---

### Task 9: Implement Streaming RAG with Progress
**Priority**: P1
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Goal**: Better UX for RAG with streaming responses and progress indicators

**Action**:
```python
# utils/streaming_utils.py (new file)

async def stream_rag_response(
    query: str,
    retriever: VectorStoreRetriever,
    llm: ChatOllama,
    show_progress: bool = True
) -> AsyncIterator[str]:
    """Stream RAG response with progress indicators"""

    # Phase 1: Retrieve documents
    if show_progress:
        print("Retrieving relevant documents...")
    docs = await retriever.aget_relevant_documents(query)

    # Phase 2: Generate response
    if show_progress:
        print(f"Found {len(docs)} documents. Generating response...")

    async for chunk in llm.astream(prompt):
        yield chunk
```

**Acceptance Criteria**:
- [ ] StreamingRAG utility class
- [ ] Progress indicators for retrieval
- [ ] Token-by-token response streaming
- [ ] Example: `examples/04-rag/streaming_rag.py` (enhance existing)
- [ ] Tests validate streaming behavior
- [ ] Performance comparison

**Dependencies**: Task 7 (async vector store)

**Files Affected**:
- `utils/streaming_utils.py` (new file)
- `examples/04-rag/streaming_rag.py` (enhance)
- `tests/test_streaming_utils.py` (new tests)

---

### Task 10: Add Embedding Cache Layer
**Priority**: P1
**Category**: Performance
**Estimated Time**: 3-4 hours

**Goal**: 10x+ speedup for repeated queries by caching embeddings

**Action**:
```python
# utils/embedding_cache.py (new file)

from functools import lru_cache
import sqlite3
import hashlib

class EmbeddingCache:
    """LRU + persistent cache for embeddings"""

    def __init__(self, db_path: str = "embeddings.db", max_size: int = 10000):
        self.db_path = db_path
        self.max_size = max_size
        self._init_db()

    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding"""
        key = self._make_key(text, model)
        return self._query_cache(key)

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache embedding"""
        key = self._make_key(text, model)
        self._store_cache(key, embedding)

    def stats(self) -> dict:
        """Return cache hit/miss statistics"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses)
        }
```

**Acceptance Criteria**:
- [ ] LRU cache for fast in-memory access
- [ ] SQLite persistent cache
- [ ] Hit/miss statistics tracking
- [ ] Integration with VectorStoreManager
- [ ] Example showing 10x+ speedup on cache hits
- [ ] Tests validate caching behavior

**Dependencies**: None

**Files Affected**:
- `utils/embedding_cache.py` (new file)
- `utils/vector_store.py` (integrate cache)
- `examples/04-rag/cached_rag.py` (new example)
- `tests/test_embedding_cache.py` (new tests)

---

### Task 11: Add Observability and Telemetry
**Priority**: P1
**Category**: Production Readiness
**Estimated Time**: 4-5 hours

**Goal**: Production-grade observability for debugging and monitoring

**Action**:
```python
# utils/telemetry.py (new file)

import logging
import time
from contextlib import contextmanager
from typing import Any

class StructuredLogger:
    """Structured logging with context"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}

    def log(self, level: str, message: str, **kwargs):
        """Log with context"""
        data = {**self.context, **kwargs}
        self.logger.log(level, message, extra=data)

class MetricsCollector:
    """Collect operation metrics"""

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def measure(self, operation: str):
        """Measure operation duration"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._record(operation, duration)

class Tracer:
    """Distributed tracing for multi-agent workflows"""

    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or self._generate_trace_id()
        self.spans = []

    @contextmanager
    def span(self, name: str):
        """Create a trace span"""
        span_id = self._generate_span_id()
        start = time.time()
        try:
            yield span_id
        finally:
            duration = time.time() - start
            self._record_span(span_id, name, duration)
```

**Acceptance Criteria**:
- [ ] Structured logging with context
- [ ] Metrics collection (counts, durations)
- [ ] Distributed tracing for workflows
- [ ] OpenTelemetry export support (optional)
- [ ] LangSmith integration hooks
- [ ] Example: `examples/06-production/observability.py`
- [ ] Zero performance impact when disabled

**Dependencies**: None

**Files Affected**:
- `utils/telemetry.py` (new file)
- All utils files (add telemetry calls)
- `examples/06-production/observability.py` (new example)
- `tests/test_telemetry.py` (new tests)

---

### Task 12: Create Model Performance Benchmarking Suite
**Priority**: P1
**Category**: Performance
**Estimated Time**: 3-4 hours

**Goal**: Comprehensive benchmarks for all recommended models

**Action**:
```bash
# tests/benchmarks/comprehensive_benchmarks.py (new file)

# Benchmark all models
pytest -m benchmark --benchmark-json=results.json

# Generate comparison report
python scripts/generate_benchmark_report.py
```

**Benchmarks**:
- Latency (first token, total)
- Throughput (tokens/second)
- Memory usage
- CPU usage
- Context window utilization
- Streaming vs batch

**Models to Benchmark**:
- qwen3:8b (baseline)
- qwen3:30b-a3b (MoE)
- gemma3:4b (lightweight)
- gemma3:12b (balanced)
- qwen3:70b (quality)

**Acceptance Criteria**:
- [ ] Benchmark suite for all models
- [ ] Tests with varying prompt lengths
- [ ] Comparison tables generated
- [ ] Results documented in `docs/BENCHMARKS.md`
- [ ] Hardware specs included
- [ ] Charts/graphs for visualization
- [ ] Run with: `pytest -m benchmark`

**Dependencies**: Task 6 (pytest markers)

**Files Affected**:
- `tests/benchmarks/comprehensive_benchmarks.py` (new)
- `utils/benchmark_runner.py` (new helper)
- `docs/BENCHMARKS.md` (new documentation)
- `scripts/generate_benchmark_report.py` (new script)

---

### Task 13: Add Context Window Management
**Priority**: P1
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Problem**: Long conversations exceed context windows, causing silent truncation

**Action**:
```python
# utils/context_manager.py (new file)

class ContextWindowManager:
    """Manage context window usage"""

    def __init__(self, model_name: str, max_tokens: int = None):
        self.model_name = model_name
        self.max_tokens = max_tokens or self._get_model_limit(model_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        # Use tiktoken or similar
        pass

    def truncate_messages(
        self,
        messages: list[dict],
        strategy: Literal["oldest", "summarize", "sliding"]
    ) -> list[dict]:
        """Truncate messages to fit context window"""
        pass

    def get_usage_stats(self) -> dict:
        """Get context window usage statistics"""
        return {
            "used_tokens": self.used_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (self.used_tokens / self.max_tokens) * 100,
            "remaining": self.max_tokens - self.used_tokens
        }
```

**Truncation Strategies**:
1. **Oldest First**: Remove oldest messages
2. **Summarize**: Compress old messages into summary
3. **Sliding Window**: Keep most recent N messages

**Acceptance Criteria**:
- [ ] Token counting for different models
- [ ] Multiple truncation strategies
- [ ] Usage tracking and warnings
- [ ] Example: `examples/03-multi-agent/context_aware_agent.py`
- [ ] Tests validate accuracy
- [ ] Documentation with best practices

**Dependencies**: None

**Files Affected**:
- `utils/context_manager.py` (new file)
- `examples/03-multi-agent/context_aware_agent.py` (new)
- `tests/test_context_manager.py` (new tests)

---

### Task 14: Implement Graceful Shutdown for Agents
**Priority**: P1
**Category**: Production Readiness
**Estimated Time**: 2-3 hours

**Problem**: Long-running agents don't handle interrupts gracefully

**Action**:
```python
# utils/lifecycle.py (new file)

import signal
import atexit

class GracefulShutdown:
    """Handle graceful shutdown for long-running agents"""

    def __init__(self, cleanup_timeout: int = 30):
        self.cleanup_timeout = cleanup_timeout
        self.shutdown_requested = False
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup signal handlers"""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        atexit.register(self._cleanup)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal"""
        print(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown_requested = True

    def _cleanup(self):
        """Cleanup resources"""
        # Save checkpoints
        # Close connections
        # Delete temp files
        pass

    def should_continue(self) -> bool:
        """Check if agent should continue running"""
        return not self.shutdown_requested
```

**Acceptance Criteria**:
- [ ] Signal handling (SIGTERM, SIGINT)
- [ ] Checkpoint saving before exit
- [ ] Resource cleanup
- [ ] Timeout for shutdown operations
- [ ] Example: `examples/06-production/graceful_shutdown.py`
- [ ] Tests validate cleanup
- [ ] Documentation for deployment

**Dependencies**: None

**Files Affected**:
- `utils/lifecycle.py` (new file)
- `examples/06-production/graceful_shutdown.py` (new)
- All long-running examples (add shutdown support)
- `tests/test_lifecycle.py` (new tests)

---

### Task 15: Create RAG Evaluation Framework
**Priority**: P1
**Category**: Features & Capabilities
**Estimated Time**: 4-5 hours

**Goal**: Systematic evaluation of RAG system quality

**Action**:
```python
# utils/rag_evaluator.py (new file)

class RAGEvaluator:
    """Evaluate RAG system performance"""

    def evaluate(
        self,
        test_dataset: list[dict],
        rag_chain: RunnableSequence
    ) -> dict[str, float]:
        """Run evaluation on test dataset"""

        results = {
            "relevance": self._measure_relevance(test_dataset, rag_chain),
            "faithfulness": self._measure_faithfulness(test_dataset, rag_chain),
            "answer_correctness": self._measure_correctness(test_dataset, rag_chain),
            "latency": self._measure_latency(test_dataset, rag_chain),
        }

        return results

    def compare_configs(
        self,
        configs: list[dict],
        test_dataset: list[dict]
    ) -> pd.DataFrame:
        """Compare different RAG configurations"""
        pass
```

**Metrics**:
- **Relevance**: Retrieved docs match query
- **Faithfulness**: Answer supported by docs
- **Correctness**: Answer matches ground truth
- **Latency**: Response time

**Acceptance Criteria**:
- [ ] RAGEvaluator class with multiple metrics
- [ ] Test dataset management
- [ ] A/B testing support
- [ ] Results export (CSV, JSON)
- [ ] Example: `examples/04-rag/evaluate_rag.py`
- [ ] Documentation: `docs/RAG-EVALUATION.md`
- [ ] Tests validate metrics

**Dependencies**: Task 7 (async vector store)

**Files Affected**:
- `utils/rag_evaluator.py` (new file)
- `examples/04-rag/evaluate_rag.py` (new)
- `tests/integration/test_rag_evaluation.py` (new)
- `docs/RAG-EVALUATION.md` (new guide)

---

## PRIORITY P2: MEDIUM (This Week - 10-12 hours)

### Task 16: Add Vector Store Migration Utilities
**Priority**: P2
**Category**: Features & Capabilities
**Estimated Time**: 2-3 hours

**Goal**: Migrate between ChromaDB and FAISS without data loss

**Action**:
```python
# In utils/vector_store.py, add migration methods

def migrate_to_faiss(
    self,
    chroma_collection: str,
    faiss_index_path: str
) -> None:
    """Migrate from ChromaDB to FAISS"""
    pass

def migrate_to_chroma(
    self,
    faiss_index_path: str,
    chroma_collection: str
) -> None:
    """Migrate from FAISS to ChromaDB"""
    pass

def export_collection(
    self,
    collection_name: str,
    output_path: str
) -> None:
    """Export collection to portable format (JSON + embeddings)"""
    pass
```

**CLI Tool**:
```bash
# Create migration CLI
python scripts/migrate_vector_store.py \
  --from chroma \
  --to faiss \
  --collection my_docs \
  --output ./faiss_index
```

**Acceptance Criteria**:
- [ ] Migrate between Chroma and FAISS
- [ ] Export to portable format
- [ ] Import from exported format
- [ ] Progress bars for large collections
- [ ] Data integrity validation
- [ ] CLI tool with documentation
- [ ] Tests validate all migration paths

**Dependencies**: Task 4 (vector store tests)

**Files Affected**:
- `utils/vector_store.py` (add migration methods)
- `scripts/migrate_vector_store.py` (new CLI)
- `tests/test_vector_store.py` (migration tests)

---

### Task 17: Implement Agent Pool Management
**Priority**: P2
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Goal**: Manage concurrent agent execution with resource limits

**Action**:
```python
# utils/agent_pool.py (new file)

import asyncio
from typing import Callable

class AgentPool:
    """Pool of agents with concurrency limits"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.task_queue = asyncio.Queue()
        self.active_agents = {}

    async def submit(self, agent_fn: Callable, *args, **kwargs):
        """Submit agent task to pool"""
        async with self.semaphore:
            agent_id = self._generate_id()
            self.active_agents[agent_id] = agent_fn
            try:
                result = await agent_fn(*args, **kwargs)
                return result
            finally:
                del self.active_agents[agent_id]

    async def map(self, agent_fn: Callable, items: list):
        """Map agent function over items with concurrency limit"""
        tasks = [self.submit(agent_fn, item) for item in items]
        return await asyncio.gather(*tasks)

    def health_check(self) -> dict:
        """Check pool health"""
        return {
            "active": len(self.active_agents),
            "max_workers": self.max_workers,
            "queue_size": self.task_queue.qsize()
        }
```

**Acceptance Criteria**:
- [ ] Agent pool with max concurrency
- [ ] Work queue for pending tasks
- [ ] Load balancing
- [ ] Health monitoring
- [ ] Automatic restart on failure
- [ ] Example: `examples/03-multi-agent/pooled_agents.py`
- [ ] Tests validate concurrency limits

**Dependencies**: None

**Files Affected**:
- `utils/agent_pool.py` (new file)
- `examples/03-multi-agent/pooled_agents.py` (new)
- `tests/test_agent_pool.py` (new tests)

---

### Task 18: Add Prompt Template Library
**Priority**: P2
**Category**: Features & Capabilities
**Estimated Time**: 2-3 hours

**Goal**: Reusable prompt templates for common tasks

**Action**:
```python
# utils/prompt_templates.py (new file)

TEMPLATES = {
    "research": {
        "system": "You are a research assistant...",
        "user": "Research the following topic: {topic}\n\nFocus on: {focus_areas}",
        "examples": [...]
    },
    "code_review": {
        "system": "You are a senior software engineer...",
        "user": "Review this code:\n\n{code}\n\nFocus on: {criteria}",
        "examples": [...]
    },
    "summarization": {
        "system": "You are a summarization expert...",
        "user": "Summarize this text:\n\n{text}\n\nLength: {length}",
        "examples": [...]
    },
    # ... 15+ templates total
}

class PromptTemplate:
    """Prompt template with validation"""

    def __init__(self, name: str, version: str = "v1"):
        self.name = name
        self.version = version
        self.template = TEMPLATES[name]

    def format(self, **kwargs) -> str:
        """Format template with variables"""
        self._validate_args(kwargs)
        return self.template["user"].format(**kwargs)
```

**Categories**:
- Research
- Code review
- Summarization
- Q&A
- Data analysis
- Creative writing

**Acceptance Criteria**:
- [ ] 15+ templates covering common patterns
- [ ] Variable substitution with validation
- [ ] Few-shot example support
- [ ] Template versioning
- [ ] Example: `examples/01-foundation/template_usage.py`
- [ ] Tests validate all templates
- [ ] Documentation with usage guide

**Dependencies**: None

**Files Affected**:
- `utils/prompt_templates.py` (new file)
- `examples/01-foundation/template_usage.py` (new)
- `tests/test_prompt_templates.py` (new tests)

---

### Task 19: Add Model Quantization Guidance
**Priority**: P2
**Category**: Documentation & DX
**Estimated Time**: 2-3 hours

**Goal**: Help users choose right quantization for their hardware

**Action**:
```python
# In utils/ollama_manager.py, add quantization methods

def recommend_quantization(
    self,
    model: str,
    available_ram_gb: float,
    target: Literal["speed", "quality", "balanced"] = "balanced"
) -> str:
    """Recommend quantization level based on hardware"""

    model_sizes = {
        "qwen3:8b-q4": 4.4,
        "qwen3:8b-q5": 5.5,
        "qwen3:8b-q8": 8.5,
        "qwen3:8b-fp16": 16.0,
    }

    # Logic to recommend based on RAM and target
    pass

def compare_quantizations(
    self,
    model: str,
    prompt: str,
    quantizations: list[str]
) -> pd.DataFrame:
    """Compare different quantizations"""
    pass
```

**Documentation**:
```markdown
# docs/MODEL-OPTIMIZATION.md

## Quantization Guide

### When to Use Each Level
- **Q4**: Fast inference, 60-75% size reduction, minimal quality loss
- **Q5**: Balanced, 50-60% size reduction, <5% quality loss
- **Q8**: High quality, 40-50% size reduction, <2% quality loss
- **FP16**: Best quality, no compression, slowest

### Hardware Requirements
| Quantization | qwen3:8b | qwen3:30b | qwen3:70b |
|--------------|----------|-----------|-----------|
| Q4           | 4.4 GB   | 13 GB     | 30 GB     |
| Q5           | 5.5 GB   | 16 GB     | 38 GB     |
| Q8           | 8.5 GB   | 24 GB     | 56 GB     |
```

**Acceptance Criteria**:
- [ ] Quantization recommendation logic
- [ ] Memory usage tracking
- [ ] Performance/quality tradeoff data
- [ ] Example: `examples/01-foundation/quantization_comparison.py`
- [ ] Documentation: `docs/MODEL-OPTIMIZATION.md`
- [ ] Guide explains when to use each level

**Dependencies**: None

**Files Affected**:
- `utils/ollama_manager.py` (add methods)
- `examples/01-foundation/quantization_comparison.py` (new)
- `docs/MODEL-OPTIMIZATION.md` (new guide)

---

### Task 20: Create Interactive CLI Improvements
**Priority**: P2
**Category**: Documentation & DX
**Estimated Time**: 3-4 hours

**Goal**: Better CLI experience with interactive features

**Action**:
```bash
# Enhance cli/main.py with:
# - Interactive model selection (fzf-like interface)
# - Progress bars for long operations
# - Colored output with rich library
# - Tab completion
# - Better error messages
```

**Features to Add**:
```python
# cli/commands/init.py (new)
def interactive_setup():
    """Interactive project setup wizard"""
    # Ask about models to download
    # Configure vector store
    # Setup MCP servers
    # Generate config file
    pass

# cli/commands/run.py (enhanced)
def run_example(name: str, follow: bool = False):
    """Run example with live output"""
    # Show progress bar
    # Stream output in real-time
    # Show summary at end
    pass
```

**Acceptance Criteria**:
- [ ] Interactive configuration wizard
- [ ] Progress bars for long operations
- [ ] Colored output with rich
- [ ] Tab completion support
- [ ] Better help text
- [ ] Tests for CLI commands
- [ ] Documentation: `docs/CLI-GUIDE.md`

**Dependencies**: Task 3 (all examples working)

**Files Affected**:
- `cli/main.py` (enhance)
- `cli/commands/` (new command modules)
- `docs/CLI-GUIDE.md` (new documentation)
- `tests/test_cli.py` (new tests)

---

### Task 21: Add Health Check Endpoints
**Priority**: P2
**Category**: Production Readiness
**Estimated Time**: 2-3 hours

**Goal**: Production health monitoring

**Action**:
```python
# api/health_server.py (new file)

from fastapi import FastAPI
from prometheus_client import Counter, Histogram

app = FastAPI()

@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness():
    """Readiness probe - checks dependencies"""
    checks = {
        "ollama": check_ollama_running(),
        "vector_store": check_vector_store(),
        "models": check_models_loaded()
    }

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, 503

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_prometheus_metrics()
```

**Acceptance Criteria**:
- [ ] FastAPI health check server
- [ ] Readiness probe (models loaded)
- [ ] Liveness probe (system responsive)
- [ ] Metrics endpoint (Prometheus format)
- [ ] Example: `examples/06-production/health_monitoring.py`
- [ ] Tests validate all endpoints
- [ ] Kubernetes deployment examples

**Dependencies**: None

**Files Affected**:
- `api/health_server.py` (new file)
- `examples/06-production/health_monitoring.py` (new)
- `tests/test_health_server.py` (new tests)
- `docs/DEPLOYMENT.md` (add K8s examples)

---

## PRIORITY P3: LOW (Nice to Have - 6-8 hours)

### Task 22: Add Vision Model Support Enhancements
**Priority**: P3
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Goal**: Better vision model integration

**Action**:
```python
# utils/vision_utils.py (new file)

class VisionProcessor:
    """Vision model utilities"""

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for vision model"""
        pass

    def run_ocr(self, image_path: str) -> str:
        """Extract text from image"""
        pass

    def analyze_image(
        self,
        image_path: str,
        model: str = "qwen3-vl:8b",
        prompt: str = "Describe this image"
    ) -> str:
        """Analyze image with vision model"""
        pass
```

**Acceptance Criteria**:
- [ ] Image preprocessing utilities
- [ ] Vision model wrapper (qwen3-vl, llava)
- [ ] OCR integration
- [ ] Image + text prompt composition
- [ ] Vision RAG support
- [ ] Examples: enhance `examples/07-advanced/vision_agent.py`
- [ ] Tests validate image handling

**Dependencies**: None

**Files Affected**:
- `utils/vision_utils.py` (new file)
- `examples/07-advanced/vision_agent.py` (enhance)
- `examples/07-advanced/multimodal_rag.py` (enhance)
- `tests/test_vision_utils.py` (new tests)

---

### Task 23: Create Contribution Guidelines
**Priority**: P3
**Category**: Community
**Estimated Time**: 2-3 hours

**Goal**: Prepare for community contributions

**Action**: Create comprehensive contribution documentation

**Files to Create**:
```markdown
# CONTRIBUTING.md
- How to contribute
- Code style guidelines
- Testing requirements
- PR checklist
- Development setup

# .github/PULL_REQUEST_TEMPLATE.md
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Follows code style
- [ ] No breaking changes

# .github/ISSUE_TEMPLATE/bug_report.md
# .github/ISSUE_TEMPLATE/feature_request.md
# .github/ISSUE_TEMPLATE/question.md
```

**Acceptance Criteria**:
- [ ] CONTRIBUTING.md with clear process
- [ ] PR template with checklist
- [ ] Issue templates (bug, feature, question)
- [ ] Code of conduct
- [ ] Developer setup guide
- [ ] First-time contributor guide
- [ ] Links from README

**Dependencies**: None

**Files Affected**:
- `CONTRIBUTING.md` (new)
- `.github/PULL_REQUEST_TEMPLATE.md` (new)
- `.github/ISSUE_TEMPLATE/` (new templates)
- `docs/DEVELOPMENT.md` (enhance)

---

### Task 24: Add Conversation Memory Patterns
**Priority**: P3
**Category**: Features & Capabilities
**Estimated Time**: 3-4 hours

**Goal**: Flexible memory patterns for agents

**Action**:
```python
# utils/memory.py (new file)

class BufferMemory:
    """Keep recent N messages"""
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []

class SummaryMemory:
    """Compress old messages into summary"""
    def __init__(self, summarizer: ChatOllama):
        self.summarizer = summarizer
        self.summary = ""

class EntityMemory:
    """Track entities across conversation"""
    def __init__(self):
        self.entities = {}

class VectorMemory:
    """Semantic similarity-based retrieval"""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
```

**Acceptance Criteria**:
- [ ] Buffer memory (recent N messages)
- [ ] Summary memory (compressed history)
- [ ] Entity memory (track entities)
- [ ] Vector memory (semantic search)
- [ ] Persistence to disk
- [ ] Example: `examples/03-multi-agent/memory_agent.py`
- [ ] Tests validate memory operations

**Dependencies**: Task 7 (async operations)

**Files Affected**:
- `utils/memory.py` (new file)
- `examples/03-multi-agent/memory_agent.py` (new)
- `tests/test_memory.py` (new tests)

---

## Implementation Strategy

### Week 1: Fix Blockers & Core Quality (P0)
**Days 1-2**: Testing infrastructure
- Task 1: Fix test collection errors
- Task 2: Fix pre-commit Python version
- Task 6: Add pytest markers

**Days 3-5**: Validation & coverage
- Task 3: Complete example validation (28 remaining)
- Task 4: Measure and improve test coverage
- Task 5: Fix formatter conflicts

### Week 2: High-Value Features (P1)
**Days 1-2**: Async & performance
- Task 7: Async vector store operations
- Task 10: Embedding cache layer
- Task 12: Model benchmarking suite

**Days 3-4**: Production readiness
- Task 8: Configuration validation
- Task 11: Observability and telemetry
- Task 14: Graceful shutdown

**Day 5**: Advanced features
- Task 9: Streaming RAG
- Task 13: Context window management
- Task 15: RAG evaluation

### Week 3+: Polish & Community (P2, P3)
- Complete P2 tasks based on priority
- Select P3 tasks based on user feedback
- Community preparation

---

## Success Metrics

### Code Quality Targets
- [ ] Test coverage: 85%+ (utils), 75%+ (workflows)
- [ ] Type hint coverage: 95%+ (already done for utils)
- [ ] Linting: 0 errors, 0 warnings
- [ ] All 35 examples working or documented
- [ ] All 269+ tests passing

### Performance Targets
- [ ] Batch operations: 3-5x faster (already achieved)
- [ ] Cache hit rate: 70%+ for embeddings
- [ ] Streaming latency: <100ms first token
- [ ] Benchmark data for all recommended models

### Developer Experience Targets
- [ ] Setup time: <10 minutes
- [ ] All examples work out-of-box
- [ ] Clear error messages with solutions
- [ ] Comprehensive documentation
- [ ] Interactive CLI working

### Production Readiness Targets
- [ ] Health check endpoints
- [ ] Graceful shutdown handling
- [ ] Observability instrumentation
- [ ] Configuration validation
- [ ] Deployment documentation

---

## Task Selection Guide

**For Maximum Impact**: Tasks 1, 3, 4, 8, 11
- Unblock testing (1)
- Validate everything works (3)
- Measure quality (4)
- Production config (8)
- Production observability (11)

**For Performance**: Tasks 7, 10, 12
- Async operations (7)
- Caching (10)
- Benchmarking (12)

**For Production**: Tasks 8, 11, 14, 21
- Config validation (8)
- Telemetry (11)
- Graceful shutdown (14)
- Health checks (21)

**For Developer Experience**: Tasks 2, 5, 6, 20, 23
- Pre-commit fixed (2)
- Formatter conflicts resolved (5)
- Test markers (6)
- CLI improvements (20)
- Contribution guidelines (23)

**For Advanced Features**: Tasks 9, 13, 15, 17, 24
- Streaming RAG (9)
- Context management (13)
- RAG evaluation (15)
- Agent pooling (17)
- Memory patterns (24)

---

## Risk Mitigation

### High Risk Tasks
1. **Task 3**: Example validation might reveal widespread issues
   - **Mitigation**: Test in small batches, fix common issues first

2. **Task 4**: Coverage might be lower than expected
   - **Mitigation**: Focus on critical paths first (utils), expand gradually

3. **Task 7**: Async changes might break existing code
   - **Mitigation**: Add async methods alongside sync, don't remove existing

### Medium Risk Tasks
1. **Task 11**: Telemetry might impact performance
   - **Mitigation**: Make it optional, measure overhead, optimize

2. **Task 15**: RAG evaluation might be complex
   - **Mitigation**: Start with simple metrics, add advanced later

---

## Dependencies Graph

```
Task 1 (Fix tests) → Task 4 (Coverage) → Task 7 (Async)
                    → Task 6 (Markers) → Task 12 (Benchmarks)
                    → Task 3 (Examples) → Task 20 (CLI)

Task 2 (Pre-commit) → Task 5 (Formatter)

Task 7 (Async) → Task 9 (Streaming RAG)
              → Task 15 (RAG eval)
              → Task 24 (Memory)

Task 8 (Config) → (Production tasks)
Task 11 (Telemetry) → (Production tasks)
```

**Critical Path**: 1 → 4 → 7 → 9 → 15
**Quality Path**: 1 → 6 → 12
**Production Path**: 8 → 11 → 14 → 21

---

## Notes

- All tasks assume working knowledge of Python 3.10+, LangChain, and existing codebase
- Tasks designed to be completed independently where possible
- Each task includes clear acceptance criteria for verification
- Estimated times are for focused, uninterrupted work
- All code must follow CLAUDE.md guidelines (no emojis, proper testing)
- Tasks prioritized by: blockers first, then value, then polish

---

## Summary

**Total Tasks**: 24 (6 P0, 9 P1, 6 P2, 3 P3)
**Total Estimated Time**: 35-45 hours (1-2 weeks)

**Top 5 Recommended Starting Tasks**:
1. **Task 1**: Fix test collection errors (unblocks everything)
2. **Task 3**: Complete example validation (validates all work)
3. **Task 4**: Measure test coverage (quality baseline)
4. **Task 8**: Configuration validation (production foundation)
5. **Task 11**: Observability (production debugging)

This plan builds logically on completed work, focuses on maximum value delivery, and maintains the project's local-first, production-ready principles.
