# ADR 004: Vector Store Selection for RAG Systems

## Status
Accepted

## Context
We need vector databases for Retrieval-Augmented Generation (RAG) systems, semantic search, and memory management. The choice of vector store impacts performance, ease of use, resource requirements, and local-first compatibility.

### Problem Statement
- RAG systems require efficient similarity search over embeddings
- Need to support both persistent and in-memory use cases
- Local-first architecture requires databases that run without cloud services
- Must handle datasets from hundreds to millions of vectors
- Performance critical for real-time applications

### Requirements
- **Local-First**: Must run entirely on-device, no cloud dependencies
- **Performance**: Sub-second search on 100k+ vectors
- **Persistence**: Optional durability for long-term storage
- **Ease of Use**: Simple setup, minimal configuration
- **Integration**: LangChain compatibility
- **Resource Efficiency**: Reasonable RAM and disk usage
- **Flexibility**: Support different distance metrics and filtering

## Decision
We will use a **dual vector store strategy**:
1. **Chroma** as the primary vector store for most use cases
2. **FAISS** as a lightweight alternative for performance-critical scenarios
3. Optional **Qdrant** support for advanced features (when installed separately)
4. All-MiniLM-L6-v2 as default embedding model (384 dimensions)

## Rationale

### Why Chroma (Primary)

**Ease of Use:**
- Zero configuration required
- Automatic persistence to disk
- Simple Python API
- Built-in HTTP server option
- Excellent documentation

**Local-First Benefits:**
- Pure Python implementation
- SQLite backend (no separate database)
- Works completely offline
- Single-file storage
- No external processes required

**Developer Experience:**
```python
import chromadb
from chromadb.config import Settings

# Persistent
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("docs")

# Ephemeral
client = chromadb.EphemeralClient()
collection = client.create_collection("docs")

# That's it!
```

**LangChain Integration:**
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

**Features:**
- Metadata filtering (where clauses)
- Multiple distance metrics (L2, cosine, IP)
- Collections for multi-tenant scenarios
- Automatic schema migration
- Update and delete operations

**Performance:**
- 100k vectors: ~50ms search (M3 Max)
- 1M vectors: ~200ms search
- Acceptable for most RAG applications
- Scales to millions of vectors

**Trade-offs:**
- Slower than FAISS for large-scale search
- More disk space than pure FAISS (SQLite overhead)
- Memory usage higher (Python overhead)

### Why FAISS (Alternative)

**Performance Benefits:**
- CPU-optimized (SIMD, AVX2)
- Sub-10ms search on 100k vectors
- Handles millions of vectors efficiently
- Quantization support (PQ, OPQ)
- GPU acceleration available

**Use Cases:**
- Very large datasets (>1M vectors)
- Latency-critical applications
- Batch processing scenarios
- Research and experimentation

**Local-First Compatibility:**
- Runs entirely in-process
- No external dependencies beyond numpy
- Serializable to disk (pickle-able)
- Memory-mappable indexes

**LangChain Integration:**
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# Save/load
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

**Trade-offs:**
- More complex API than Chroma
- Manual persistence required
- No built-in metadata filtering
- Steeper learning curve
- Index type selection affects performance

### Why Not Others

**Pinecone:**
- ❌ Cloud-only, monthly costs
- ❌ Violates local-first principle
- ✅ Excellent performance and features
- **Verdict**: Consider for optional cloud integration only

**Weaviate:**
- ❌ Requires Docker/separate server
- ❌ Higher resource overhead
- ✅ Advanced features (hybrid search, ML models)
- **Verdict**: Too heavyweight for local-first toolkit

**Milvus:**
- ❌ Requires Docker/separate server
- ❌ Complex setup and configuration
- ✅ Excellent performance at scale
- **Verdict**: Overkill for single-machine use

**Qdrant:**
- ⚠️ Requires Rust installation for Python client
- ⚠️ More complex setup than Chroma
- ✅ Excellent performance and features
- ✅ Can run embedded or as server
- **Verdict**: Optional for users wanting advanced features

**Lance DB:**
- ⚠️ Relatively new, less mature
- ✅ Columnar format, efficient for ML
- ⚠️ Smaller ecosystem
- **Verdict**: Watch for future adoption

## Consequences

### Positive
- Chroma provides best developer experience for most use cases
- FAISS available when maximum performance needed
- Both work completely offline
- LangChain integration mature and well-documented
- Can switch between stores with minimal code changes
- Covers spectrum from simple prototypes to production use
- Zero infrastructure overhead
- No recurring costs

### Negative
- Two vector stores to maintain and document
- FAISS requires more manual management (persistence, filtering)
- Chroma slower than specialized cloud solutions
- Users must choose between options (decision fatigue)
- Advanced features require understanding trade-offs
- Index optimization needs experimentation

### Mitigation Strategies
1. **Complexity**: Provide clear decision tree for choosing store
2. **Performance**: Document optimization techniques for both stores
3. **Learning Curve**: Templates and examples for common patterns
4. **Decision Fatigue**: Default to Chroma, document when to use FAISS
5. **Advanced Features**: Optional Qdrant guide for power users

## Alternatives Considered

### Alternative 1: Chroma Only
**Pros:**
- Simpler (single choice)
- Easier to document and support
- Sufficient for 80% of use cases

**Cons:**
- Leaves performance on table
- Not suitable for large-scale RAG
- May frustrate power users

**Why Rejected:** FAISS overhead minimal, performance benefits significant for certain use cases. Worth supporting both.

### Alternative 2: FAISS Only
**Pros:**
- Maximum performance
- Industry-standard
- Proven at scale

**Cons:**
- Worse developer experience
- Manual persistence
- No built-in filtering
- Steeper learning curve

**Why Rejected:** Too much friction for beginners. Chroma's ease of use more valuable for experimentation toolkit.

### Alternative 3: NumPy/Scipy Only
**Pros:**
- No dependencies beyond NumPy
- Simple implementation
- Full control

**Cons:**
- Poor performance (>1s for 10k vectors)
- No optimization
- Reinventing wheel
- No persistence helpers

**Why Rejected:** Performance unacceptable even for small datasets. Not worth the implementation effort.

### Alternative 4: SQLite with Vector Extension
**Pros:**
- Familiar SQL interface
- Excellent persistence
- Mature ecosystem

**Cons:**
- Extension not widely available
- Poor vector search performance
- Limited to small datasets
- No advanced index types

**Why Rejected:** Vector search not SQLite's strength. Better to use purpose-built vector stores.

## Implementation

### Project Structure
```
examples/04-rag/
├── 01-simple-rag-chroma.py           # Basic RAG with Chroma
├── 02-simple-rag-faiss.py            # Basic RAG with FAISS
├── 03-metadata-filtering.py          # Advanced filtering (Chroma)
├── 04-hybrid-search.py               # Keyword + semantic
├── 05-faiss-optimization.py          # Index tuning
├── 06-conversation-memory.py         # Chat history with vectors
└── 07-multimodal-rag.py             # Images + text (CLIP)
```

### Standard Chroma Pattern
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# 1. Load documents
loader = DirectoryLoader("./docs", glob="**/*.md")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # or "mps" for M3
)

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="docs"
)

# 5. Search
results = vectorstore.similarity_search(
    query="How do I use LangGraph?",
    k=5,
    filter={"source": {"$contains": "langgraph"}}
)
```

### Standard FAISS Pattern
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Only from trusted sources
)

# Search
results = vectorstore.similarity_search("query", k=5)

# Advanced: Custom FAISS index
import faiss
index = faiss.IndexFlatL2(384)  # dimension
vectorstore = FAISS(embeddings, index, {}, {})
```

### Decision Tree
```python
def choose_vector_store(use_case: str) -> str:
    """Help users choose appropriate vector store"""

    if use_case == "getting_started":
        return "Chroma"  # Easiest to learn

    if use_case == "prototype":
        return "Chroma"  # Fast iteration

    if use_case == "large_dataset" and vectors > 1_000_000:
        return "FAISS"  # Performance critical

    if use_case == "metadata_filtering":
        return "Chroma"  # Built-in WHERE clauses

    if use_case == "batch_processing":
        return "FAISS"  # Optimized for throughput

    if use_case == "low_latency" and p99_latency_ms < 50:
        return "FAISS"  # Sub-10ms possible

    # Default
    return "Chroma"
```

### Embedding Model Selection
```python
# config/embeddings.py
from enum import Enum

class EmbeddingModel(Enum):
    # Balanced (default)
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"  # 384d, 14MB

    # Quality
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"  # 768d, 438MB

    # Multilingual
    PARAPHRASE_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384d

    # Speed
    ALL_MINILM_L12_V2 = "all-MiniLM-L12-v2"  # 384d, faster than L6

    # Code
    CODE_SEARCH_NET = "code-search-net"  # 768d, code-specific

# Recommendations
EMBEDDING_RECOMMENDATIONS = {
    "general": EmbeddingModel.ALL_MINILM_L6_V2,
    "quality": EmbeddingModel.ALL_MPNET_BASE_V2,
    "multilingual": EmbeddingModel.PARAPHRASE_MULTILINGUAL,
    "code": EmbeddingModel.CODE_SEARCH_NET,
}
```

## Verification

### Success Criteria
- [ ] Chroma examples work out of box
- [ ] FAISS examples demonstrate performance benefits
- [ ] Decision tree documented
- [ ] Benchmarks for both stores on M3 Max
- [ ] Migration guide between stores
- [ ] Performance <100ms for 100k vectors (both stores)

### Testing Strategy
```python
# tests/test_vector_stores.py
def test_chroma_crud():
    """Test create, read, update, delete with Chroma"""

def test_faiss_persistence():
    """Test FAISS save and load"""

def test_embedding_consistency():
    """Verify same embeddings in both stores"""

def test_metadata_filtering():
    """Test Chroma WHERE clauses"""

# tests/benchmarks/vector_store_benchmark.py
def benchmark_search_speed(num_vectors: int, query_count: int):
    """Compare Chroma vs FAISS search speed"""

def benchmark_indexing_speed(num_vectors: int):
    """Compare indexing performance"""

def benchmark_memory_usage(num_vectors: int):
    """Measure RAM consumption"""
```

### Performance Targets

| Operation | Chroma (100k) | FAISS (100k) | Chroma (1M) | FAISS (1M) |
|-----------|---------------|--------------|-------------|------------|
| Index | 30s | 15s | 300s | 150s |
| Search (k=5) | 50ms | 5ms | 200ms | 20ms |
| Batch (100q) | 2s | 500ms | 10s | 2s |
| Memory | 500MB | 200MB | 5GB | 2GB |

## Migration Path

### Phase 1: Foundation (Current)
- Chroma integration and examples
- Basic FAISS examples
- Embedding model selection guide
- Decision tree documentation

### Phase 2: Optimization
- FAISS index optimization guide
- Chroma performance tuning
- Benchmarking tools
- Memory profiling

### Phase 3: Advanced Features
- Optional Qdrant integration
- Hybrid search (keyword + semantic)
- Multimodal embeddings (CLIP)
- Distributed search patterns

### Between Vector Stores
```python
# Chroma to FAISS
chroma_store = Chroma(persist_directory="./chroma_db", ...)
docs_and_embeddings = chroma_store._collection.get(include=["documents", "embeddings"])

faiss_store = FAISS.from_embeddings(
    text_embeddings=list(zip(
        docs_and_embeddings["documents"],
        docs_and_embeddings["embeddings"]
    )),
    embedding=embeddings
)
faiss_store.save_local("faiss_index")

# FAISS to Chroma
faiss_store = FAISS.load_local("faiss_index", embeddings)
docs = faiss_store.docstore._dict.values()

chroma_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

## References
- [Chroma Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [LangChain Vector Stores](https://python.langchain.com/docs/integrations/vectorstores/)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)
- [Embedding Models Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

## Related ADRs
- ADR-001: Local-First Architecture (storage requirements)
- ADR-006: Apple Silicon Optimization (embedding performance)
- Future: ADR on hybrid search strategies

## Changelog
- 2025-10-26: Initial version - dual vector store strategy with Chroma and FAISS
