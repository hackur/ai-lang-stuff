# Example Prerequisites & Requirements

This document lists all prerequisites, dependencies, and expected outputs for each example in this repository.

## Quick Reference

| Example Category | Requires Ollama | Requires Models | Requires Internet | Can Run Offline |
|-----------------|----------------|-----------------|-------------------|-----------------|
| 01-foundation   | ✅ Yes | qwen3:8b, gemma3:4b | ❌ No | ✅ Yes |
| 02-mcp          | ✅ Yes | qwen3:8b | ⚠️ web_search only | ⚠️ Partial |
| 03-multi-agent  | ✅ Yes | qwen3:8b | ❌ No | ✅ Yes |
| 04-rag          | ✅ Yes | qwen3:8b, qwen3-embedding | ❌ No | ✅ Yes |
| 05-interpretability | ❌ No | N/A (uses transformers) | ❌ No | ✅ Yes |
| 06-production   | ✅ Yes | qwen3:8b | ❌ No | ✅ Yes |
| 07-advanced     | ✅ Yes | qwen3-vl:8b, whisper | ❌ No | ✅ Yes |
| error_handling_demo | ❌ No | N/A | ❌ No | ✅ Yes |
| tool_registry_demo | ❌ No | N/A | ❌ No | ✅ Yes |

---

## 01-foundation: Basic LLM Interactions

### simple_chat.py
**Purpose**: Basic chat interaction with local LLM

**Prerequisites**:
- Ollama running (`ollama serve`)
- Model: qwen3:8b (`ollama pull qwen3:8b`)

**Expected Output**:
```
Human: What is the capital of France?
AI: The capital of France is Paris. [continues...]
```

**Run**: `uv run python examples/01-foundation/simple_chat.py`

---

### streaming_chat.py
**Purpose**: Streaming responses from LLM

**Prerequisites**:
- Ollama running
- Model: qwen3:8b

**Expected Output**:
```
Streaming response:
The... capital... of... France... is... Paris...
```

**Run**: `uv run python examples/01-foundation/streaming_chat.py`

---

### compare_models.py
**Purpose**: Compare outputs from different models

**Prerequisites**:
- Ollama running
- Models: qwen3:8b, gemma3:4b
  ```bash
  ollama pull qwen3:8b
  ollama pull gemma3:4b
  ```

**Expected Output**:
```
Testing prompt: "Explain quantum computing in simple terms"
--- qwen3:8b Response ---
[Response from qwen3:8b]
--- gemma3:4b Response ---
[Response from gemma3:4b]
```

**Run**: `uv run python examples/01-foundation/compare_models.py`

---

## 02-mcp: MCP Tool Integration

### filesystem_agent.py
**Purpose**: Agent with filesystem access via MCP

**Prerequisites**:
- Ollama running
- Model: qwen3:8b
- MCP filesystem server running (automatic in example)

**Expected Output**:
```
Agent can:
- List files in directory
- Read file contents
- Search for files
- Create/write files (with permission)
```

**Run**: `uv run python examples/02-mcp/filesystem_agent.py`

**Notes**: Uses custom MCP server in `mcp-servers/custom/filesystem/`

---

### web_search_agent.py
**Purpose**: Agent with web search capabilities

**Prerequisites**:
- Ollama running
- Model: qwen3:8b
- Internet connection (for search)
- MCP web search server running

**Expected Output**:
```
Searching for: "local LLMs 2024"
Found 5 results:
1. [Title] - [URL]
2. [Title] - [URL]
...
```

**Run**: `uv run python examples/02-mcp/web_search_agent.py`

**Notes**: Requires internet for web search functionality

---

### combined_tools_agent.py
**Purpose**: Agent using both filesystem and web search

**Prerequisites**:
- All prerequisites from filesystem_agent.py
- All prerequisites from web_search_agent.py
- Internet connection

**Expected Output**:
```
Agent can:
- Search web for information
- Save results to files
- Read and analyze saved data
```

**Run**: `uv run python examples/02-mcp/combined_tools_agent.py`

---

## 03-multi-agent: LangGraph Workflows

### research_pipeline.py
**Purpose**: Multi-agent research workflow

**Prerequisites**:
- Ollama running
- Model: qwen3:8b
- ChromaDB (installed automatically)

**Expected Output**:
```
=== Research Pipeline ===
1. Planner agent: Breaking down research topic
2. Researcher agent: Gathering information
3. Writer agent: Synthesizing findings
Final report: [markdown report]
```

**Run**: `uv run python examples/03-multi-agent/research_pipeline.py`

**Duration**: ~2-5 minutes depending on topic complexity

---

### code_review_pipeline.py
**Purpose**: Automated code review with multiple agents

**Prerequisites**:
- Ollama running
- Model: qwen3:8b
- Sample code file to review

**Expected Output**:
```
=== Code Review ===
Reviewer 1 (Style): [feedback]
Reviewer 2 (Security): [feedback]
Reviewer 3 (Performance): [feedback]
Summary: [consolidated report]
```

**Run**: `uv run python examples/03-multi-agent/code_review_pipeline.py`

---

### parallel_comparison.py
**Purpose**: Run multiple agents in parallel

**Prerequisites**:
- Ollama running
- Models: qwen3:8b, gemma3:4b

**Expected Output**:
```
Running 3 agents in parallel...
Agent 1 completed in 2.3s
Agent 2 completed in 2.1s
Agent 3 completed in 2.5s
Results: [comparison table]
```

**Run**: `uv run python examples/03-multi-agent/parallel_comparison.py`

---

## 04-rag: Retrieval-Augmented Generation

### document_qa.py
**Purpose**: Question answering over documents

**Prerequisites**:
- Ollama running
- Models: qwen3:8b, qwen3-embedding
  ```bash
  ollama pull qwen3:8b
  ollama pull qwen3-embedding
  ```
- Sample documents (provided in example)
- ChromaDB (auto-installed)

**Expected Output**:
```
Indexing 10 documents...
Index created with 10 documents
Query: "What is the main topic?"
Answer: [AI-generated answer with sources]
Sources: doc1.txt, doc3.txt
```

**Run**: `uv run python examples/04-rag/document_qa.py`

---

### codebase_search.py
**Purpose**: Semantic search over codebase

**Prerequisites**:
- Ollama running
- Model: qwen3-embedding
- Target codebase to index

**Expected Output**:
```
Indexing codebase at ./utils/
Found 50 Python files
Indexed 200 code chunks
Query: "error handling patterns"
Results:
1. utils/error_recovery.py:123 (score: 0.92)
2. examples/error_handling_demo.py:45 (score: 0.88)
```

**Run**: `uv run python examples/04-rag/codebase_search.py`

---

### advanced_rag_reranking.py
**Purpose**: RAG with reranking for better results

**Prerequisites**:
- All prerequisites from document_qa.py
- Additional: cross-encoder model for reranking

**Expected Output**:
```
Initial retrieval: 20 documents
After reranking: Top 5 most relevant
1. [doc] (relevance: 0.95)
2. [doc] (relevance: 0.91)
...
```

**Run**: `uv run python examples/04-rag/advanced_rag_reranking.py`

---

### streaming_rag.py
**Purpose**: Streaming RAG responses

**Prerequisites**:
- All prerequisites from document_qa.py

**Expected Output**:
```
Query: "Explain the concept"
Streaming answer:
Based on the documents... [streams word by word]
```

**Run**: `uv run python examples/04-rag/streaming_rag.py`

---

### vision_rag.py
**Purpose**: RAG over images and documents

**Prerequisites**:
- Ollama running
- Models: qwen3-vl:8b, qwen3-embedding
  ```bash
  ollama pull qwen3-vl:8b
  ollama pull qwen3-embedding
  ```
- Sample images (provided)

**Expected Output**:
```
Indexing images and captions...
Query: "Show me images of nature"
Results:
1. forest.jpg - "A dense forest with tall trees"
2. mountain.jpg - "Snow-capped mountain peak"
```

**Run**: `uv run python examples/04-rag/vision_rag.py`

---

## 05-interpretability: Mechanistic Interpretability

### activation_patching.py
**Purpose**: Activation patching experiments

**Prerequisites**:
- Python 3.10-3.12 (NOT 3.13)
- transformer-lens (`uv sync`)
- Small transformer model (auto-downloaded)

**Expected Output**:
```
Loading model: gpt2-small
Running activation patching...
Layer 0 impact: 0.12
Layer 1 impact: 0.34
...
Visualization saved to activations.png
```

**Run**: `uv run python examples/05-interpretability/activation_patching.py`

**Notes**: No Ollama required, uses HuggingFace models

---

### attention_visualization.ipynb
**Purpose**: Interactive attention pattern visualization

**Prerequisites**:
- Jupyter Lab (`uv run jupyter lab`)
- transformer-lens
- matplotlib, plotly

**Expected Output**:
- Interactive attention heatmaps
- Layer-by-layer analysis
- Token attribution scores

**Run**: `uv run jupyter lab examples/05-interpretability/attention_visualization.ipynb`

---

### circuit_discovery.py
**Purpose**: Discover computational circuits in models

**Prerequisites**:
- Same as activation_patching.py
- Longer runtime (5-10 minutes)

**Expected Output**:
```
Analyzing circuits for: "The cat sat on the"
Found 3 significant circuits:
1. Subject-verb agreement (layers 2-4)
2. Preposition handling (layers 5-6)
3. Context integration (layers 8-10)
```

**Run**: `uv run python examples/05-interpretability/circuit_discovery.py`

---

## 06-production: Production Patterns

### config_management.py
**Purpose**: Configuration management best practices

**Prerequisites**:
- config/models.yaml exists
- .env.example copied to .env (optional)

**Expected Output**:
```
Loading config from: config/models.yaml
Model: qwen3:8b
  Temperature: 0.7
  Max tokens: 2048
  System prompt: [loaded]
Config validation: ✓
```

**Run**: `uv run python examples/06-production/config_management.py`

---

### monitoring_logging.py
**Purpose**: Production logging and monitoring

**Prerequisites**:
- Ollama running
- Model: qwen3:8b

**Expected Output**:
```
[2025-10-28 13:00:00] INFO: Agent started
[2025-10-28 13:00:01] INFO: Processing query
[2025-10-28 13:00:03] INFO: Response generated (2.1s)
Metrics:
  - Requests: 10
  - Avg latency: 2.3s
  - Errors: 0
```

**Run**: `uv run python examples/06-production/monitoring_logging.py`

**Notes**: Creates logs/ directory

---

### deployment_ready.py
**Purpose**: Production-ready agent template

**Prerequisites**:
- Ollama running
- Model: qwen3:8b
- Redis (optional, for caching)

**Expected Output**:
```
Starting production agent...
Health check: ✓
Cache: enabled
Rate limiting: 10 req/min
Ready to serve requests
```

**Run**: `uv run python examples/06-production/deployment_ready.py`

---

### production_agent.py
**Purpose**: Complete production agent with all features

**Prerequisites**:
- All prerequisites from deployment_ready.py
- Monitoring stack (optional)

**Run**: `uv run python examples/06-production/production_agent.py`

---

## 07-advanced: Advanced Features

### vision_agent.py
**Purpose**: Vision understanding agent

**Prerequisites**:
- Ollama running
- Model: qwen3-vl:8b
  ```bash
  ollama pull qwen3-vl:8b
  ```
- Sample images (provided)

**Expected Output**:
```
Loading vision model: qwen3-vl:8b
Analyzing image: sample.jpg
Description: A person sitting at a desk with a laptop
Objects detected: person, desk, laptop, chair
Scene: indoor office environment
```

**Run**: `uv run python examples/07-advanced/vision_agent.py`

---

### audio_transcription_agent.py
**Purpose**: Audio transcription and analysis

**Prerequisites**:
- Ollama running
- Models: whisper (via Ollama)
  ```bash
  ollama pull whisper
  ```
- Sample audio file

**Expected Output**:
```
Transcribing: audio.mp3
Transcription: "Hello, this is a test recording..."
Summary: [AI-generated summary]
Key points: [bullet list]
```

**Run**: `uv run python examples/07-advanced/audio_transcription_agent.py`

---

### multimodal_rag.py
**Purpose**: RAG over text, images, and audio

**Prerequisites**:
- Ollama running
- Models: qwen3-vl:8b, qwen3:8b, qwen3-embedding, whisper
- Sample multimedia documents

**Expected Output**:
```
Indexing multimedia collection...
  - 10 text documents
  - 5 images
  - 3 audio files
Query: "Show me content about machine learning"
Results (ranked):
1. ml_paper.pdf (text, score: 0.94)
2. neural_net.jpg (image, score: 0.88)
3. lecture.mp3 (audio, score: 0.85)
```

**Run**: `uv run python examples/07-advanced/multimodal_rag.py`

---

### document_understanding.py
**Purpose**: Deep document understanding and analysis

**Prerequisites**:
- Ollama running
- Models: qwen3-vl:8b (for visual doc analysis)
- PDF processing: pypdf

**Expected Output**:
```
Analyzing document: report.pdf
Pages: 10
Structure detected:
  - Title
  - Abstract
  - 3 sections
  - 5 figures
  - References
Summary: [comprehensive summary]
Key figures: [analysis of charts/diagrams]
```

**Run**: `uv run python examples/07-advanced/document_understanding.py`

---

## Standalone Examples

### error_handling_demo.py
**Purpose**: Comprehensive error recovery patterns

**Prerequisites**:
- None (runs without Ollama)

**Expected Output**:
```
=== ERROR RECOVERY PATTERNS DEMONSTRATION ===
Example 1: Basic Retry with Exponential Backoff
Example 2: Retry with Fallback
Example 3: Error Classification
...
DEMONSTRATION COMPLETE
```

**Run**: `uv run python examples/error_handling_demo.py`

**Duration**: ~30 seconds

---

### tool_registry_demo.py
**Purpose**: Tool registry and management

**Prerequisites**:
- None (runs without Ollama)

**Expected Output**:
```
=== TOOL REGISTRY DEMONSTRATION ===
Registered tools:
  - calculator (math)
  - web_search (web)
  - file_read (filesystem)
Total: 3 tools in 3 categories
```

**Run**: `uv run python examples/tool_registry_demo.py`

---

## Troubleshooting

### Ollama Not Running
```bash
# Check status
ps aux | grep ollama

# Start Ollama
ollama serve

# Or as service (macOS)
brew services start ollama
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull missing model
ollama pull qwen3:8b
```

### Import Errors
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync

# Verify installation
python -c "import langchain; print('OK')"
```

### Slow Performance
- Use smaller models (gemma3:4b vs qwen3:70b)
- Use quantized models (Q4, Q5)
- Reduce context window
- Enable GPU acceleration (if available)

### ChromaDB Errors
```bash
# Clear ChromaDB cache
rm -rf chroma_db/

# Reinstall
uv sync --force-reinstall
```

---

## Performance Guidelines

### Model Selection by Task

| Task | Recommended Model | Speed | Quality |
|------|------------------|-------|---------|
| Quick answers | gemma3:4b | 🚀🚀🚀 | ⭐⭐⭐ |
| General chat | qwen3:8b | 🚀🚀 | ⭐⭐⭐⭐ |
| Complex reasoning | qwen3:70b | 🚀 | ⭐⭐⭐⭐⭐ |
| Vision tasks | qwen3-vl:8b | 🚀🚀 | ⭐⭐⭐⭐ |
| Embeddings | qwen3-embedding | 🚀🚀🚀 | ⭐⭐⭐⭐ |

### Hardware Performance

| Hardware | qwen3:8b Speed | Recommendations |
|----------|---------------|-----------------|
| M3 Max | ~45 tok/s | Use any model |
| M2 Pro | ~35 tok/s | Stick to <30B models |
| M1 Max | ~40 tok/s | Excellent |
| Intel i7 | ~15 tok/s | Use <8B models |

---

## Testing All Examples

Use the integration test suite:

```bash
# Test all examples
uv run pytest tests/integration/test_examples_run.py -v

# Test specific category
uv run pytest tests/integration/test_examples_run.py -v -k "foundation"

# Skip slow tests
uv run pytest tests/integration/test_examples_run.py -v -m "not slow"
```

---

**Last Updated**: 2025-10-28
**Maintained By**: See CONTRIBUTING.md for how to update this document
