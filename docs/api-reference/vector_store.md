# VectorStoreManager API Reference

Unified interface for Chroma and FAISS vector stores with Ollama embeddings for local-first RAG systems.

## Overview

The `VectorStoreManager` class provides comprehensive vector store management with support for both Chroma (optimized for persistence) and FAISS (optimized for performance). Integrates seamlessly with Ollama for local embeddings.

**Module:** `utils.vector_store`

**Dependencies:**
- `langchain-chroma` - Chroma vector store
- `langchain-community` - FAISS vector store
- `langchain-ollama` - Ollama embeddings
- `langchain-core` - Base classes

---

## Type Aliases

```python
VectorStoreType = Literal["chroma", "faiss"]
```

Valid vector store backend types.

---

## Class: VectorStoreManager

```python
class VectorStoreManager:
    """Manages vector store operations with Chroma and FAISS backends."""
```

### Constructor

```python
def __init__(
    self,
    embedding_model: str = "qwen3-embedding",
    base_url: str = "http://localhost:11434"
) -> None
```

Initialize the vector store manager.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"qwen3-embedding"` | Ollama embedding model name |
| `base_url` | `str` | `"http://localhost:11434"` | Ollama API endpoint |

**Example:**

```python
from utils.vector_store import VectorStoreManager

# Default configuration (qwen3-embedding)
manager = VectorStoreManager()

# Custom embedding model
manager = VectorStoreManager(
    embedding_model="nomic-embed-text",
    base_url="http://localhost:11434"
)
```

**Recommended Embedding Models:**

| Model | Size | Use Case |
|-------|------|----------|
| `nomic-embed-text` | 137M | General purpose, best quality |
| `qwen3-embedding` | Small | Fast, balanced |
| `mxbai-embed-large` | 335M | High dimension (1024) |

---

## Vector Store Creation

### create_from_documents()

```python
def create_from_documents(
    self,
    documents: List[Document],
    collection_name: str,
    persist_dir: str,
    store_type: VectorStoreType = "chroma",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> VectorStore
```

Create a new vector store from documents.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `documents` | `List[Document]` | Yes | - | Documents to index |
| `collection_name` | `str` | Yes | - | Collection/index name |
| `persist_dir` | `str` | Yes | - | Persistence directory |
| `store_type` | `VectorStoreType` | No | `"chroma"` | Backend type ("chroma" or "faiss") |
| `chunk_size` | `Optional[int]` | No | `None` | Custom chunk size for splitting |
| `chunk_overlap` | `Optional[int]` | No | `None` | Overlap between chunks |

**Returns:**
- `VectorStore` - Chroma or FAISS vector store instance

**Raises:**
- `ValueError` - If documents is empty or store_type is invalid
- `ConnectionError` - If Ollama is not available

**Example:**

```python
from langchain_community.document_loaders import TextLoader
from utils.vector_store import VectorStoreManager

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Create vector store
manager = VectorStoreManager()
vectorstore = manager.create_from_documents(
    documents=documents,
    collection_name="my_docs",
    persist_dir="./data/vectors"
)

# Search
results = vectorstore.similarity_search("query", k=3)
```

**With Chunking:**

```python
vectorstore = manager.create_from_documents(
    documents=documents,
    collection_name="chunked_docs",
    persist_dir="./data/vectors",
    chunk_size=512,
    chunk_overlap=50
)
```

**FAISS Backend:**

```python
vectorstore = manager.create_from_documents(
    documents=documents,
    collection_name="fast_docs",
    persist_dir="./data/vectors",
    store_type="faiss"
)
```

---

## Vector Store Loading

### load_existing()

```python
def load_existing(
    self,
    collection_name: str,
    persist_dir: str,
    store_type: VectorStoreType = "chroma"
) -> VectorStore
```

Load an existing vector store from disk.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection_name` | `str` | Yes | - | Collection to load |
| `persist_dir` | `str` | Yes | - | Persistence directory |
| `store_type` | `VectorStoreType` | No | `"chroma"` | Backend type |

**Returns:**
- `VectorStore` - Loaded vector store instance

**Raises:**
- `FileNotFoundError` - If collection doesn't exist
- `ValueError` - If store_type is invalid

**Example:**

```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager()

# Load existing Chroma collection
vectorstore = manager.load_existing(
    collection_name="my_docs",
    persist_dir="./data/vectors"
)

# Perform searches
results = vectorstore.similarity_search("query")
```

**Error Handling:**

```python
try:
    vectorstore = manager.load_existing("my_docs", "./data/vectors")
except FileNotFoundError as e:
    print(f"Collection not found: {e}")
    # List available collections
    collections = manager.list_collections("./data/vectors")
    print(f"Available: {collections}")
```

---

## Document Management

### add_documents()

```python
def add_documents(
    self,
    vectorstore: VectorStore,
    documents: List[Document],
    persist_dir: Optional[str] = None
) -> None
```

Add new documents to an existing vector store.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `vectorstore` | `VectorStore` | Yes | Existing vector store |
| `documents` | `List[Document]` | Yes | Documents to add |
| `persist_dir` | `Optional[str]` | No | Required for FAISS |

**Raises:**
- `ValueError` - If documents is empty or persist_dir missing for FAISS

**Example:**

```python
manager = VectorStoreManager()

# Load existing store
vectorstore = manager.load_existing("my_docs", "./data/vectors")

# Add new documents
new_docs = [
    Document(page_content="New content", metadata={"source": "new.txt"})
]
manager.add_documents(vectorstore, new_docs)
```

**FAISS Persistence:**

```python
# FAISS requires persist_dir for saving
vectorstore = manager.load_existing("docs", "./data/vectors", store_type="faiss")
manager.add_documents(
    vectorstore,
    new_docs,
    persist_dir="./data/vectors/faiss/docs"
)
```

---

### delete_collection()

```python
def delete_collection(
    self,
    collection_name: str,
    persist_dir: str,
    store_type: VectorStoreType = "chroma"
) -> None
```

Delete a vector store collection from disk.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection_name` | `str` | Yes | - | Collection to delete |
| `persist_dir` | `str` | Yes | - | Persistence directory |
| `store_type` | `VectorStoreType` | No | `"chroma"` | Backend type |

**Raises:**
- `FileNotFoundError` - If collection doesn't exist

**Example:**

```python
manager = VectorStoreManager()

# Delete old collection
manager.delete_collection("old_docs", "./data/vectors")
```

**Warning:** This permanently deletes the collection. Ensure you have backups if needed.

---

### list_collections()

```python
def list_collections(
    self,
    persist_dir: str,
    store_type: Optional[VectorStoreType] = None
) -> dict[str, List[str]]
```

List all available vector store collections.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `persist_dir` | `str` | Yes | Persistence directory |
| `store_type` | `Optional[VectorStoreType]` | No | Filter by type |

**Returns:**

Dictionary mapping store types to collection lists:

```python
{
    "chroma": ["collection1", "collection2"],
    "faiss": ["collection3"]
}
```

**Example:**

```python
manager = VectorStoreManager()

# List all collections
collections = manager.list_collections("./data/vectors")
print(f"Chroma: {collections['chroma']}")
print(f"FAISS: {collections['faiss']}")

# Filter by type
chroma_only = manager.list_collections("./data/vectors", store_type="chroma")
```

---

## Search Operations

### similarity_search()

```python
def similarity_search(
    self,
    vectorstore: VectorStore,
    query: str,
    k: int = 4,
    filter: Optional[dict] = None
) -> List[Document]
```

Perform similarity search with optional metadata filtering.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vectorstore` | `VectorStore` | Yes | - | Vector store to search |
| `query` | `str` | Yes | - | Query string |
| `k` | `int` | No | `4` | Number of results |
| `filter` | `Optional[dict]` | No | `None` | Metadata filter |

**Returns:**
- `List[Document]` - Documents ranked by similarity

**Example:**

```python
manager = VectorStoreManager()
vectorstore = manager.load_existing("my_docs", "./data/vectors")

# Basic search
results = manager.similarity_search(
    vectorstore,
    "What are AI agents?",
    k=3
)

for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}\n")
```

**Metadata Filtering:**

```python
# Search only in specific source
results = manager.similarity_search(
    vectorstore,
    "AI agents",
    k=5,
    filter={"source": "research_papers"}
)

# Multiple filters
results = manager.similarity_search(
    vectorstore,
    "AI agents",
    k=5,
    filter={"source": "research_papers", "year": 2024}
)
```

---

### similarity_search_with_score()

```python
def similarity_search_with_score(
    self,
    vectorstore: VectorStore,
    query: str,
    k: int = 4,
    filter: Optional[dict] = None
) -> List[tuple[Document, float]]
```

Perform similarity search with relevance scores.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vectorstore` | `VectorStore` | Yes | - | Vector store to search |
| `query` | `str` | Yes | - | Query string |
| `k` | `int` | No | `4` | Number of results |
| `filter` | `Optional[dict]` | No | `None` | Metadata filter |

**Returns:**
- `List[tuple[Document, float]]` - (Document, score) pairs
- **Lower scores indicate higher similarity**

**Example:**

```python
manager = VectorStoreManager()
vectorstore = manager.load_existing("my_docs", "./data/vectors")

results = manager.similarity_search_with_score(
    vectorstore,
    "AI agents",
    k=3
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...\n")
```

**Score Thresholding:**

```python
results = manager.similarity_search_with_score(vectorstore, query, k=10)

# Filter by score threshold
relevant = [(doc, score) for doc, score in results if score < 0.5]
print(f"Found {len(relevant)} highly relevant results")
```

---

## Convenience Functions

### create_chroma_store()

```python
def create_chroma_store(
    documents: List[Document],
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    embedding_model: str = "qwen3-embedding"
) -> Chroma
```

Quick function to create a Chroma vector store.

**Example:**

```python
from utils.vector_store import create_chroma_store

vectorstore = create_chroma_store(
    documents=docs,
    collection_name="my_docs",
    persist_dir="./data/vectors",
    embedding_model="nomic-embed-text"
)
```

---

### create_faiss_store()

```python
def create_faiss_store(
    documents: List[Document],
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    embedding_model: str = "qwen3-embedding"
) -> FAISS
```

Quick function to create a FAISS vector store.

**Example:**

```python
from utils.vector_store import create_faiss_store

vectorstore = create_faiss_store(
    documents=docs,
    collection_name="fast_docs",
    persist_dir="./data/vectors"
)
```

---

### load_vector_store()

```python
def load_vector_store(
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    store_type: VectorStoreType = "chroma",
    embedding_model: str = "qwen3-embedding"
) -> VectorStore
```

Quick function to load an existing vector store.

**Example:**

```python
from utils.vector_store import load_vector_store

vectorstore = load_vector_store(
    collection_name="my_docs",
    persist_dir="./data/vectors"
)
```

---

## Backend Comparison

### Chroma vs FAISS

| Feature | Chroma | FAISS |
|---------|--------|-------|
| **Persistence** | Automatic | Manual save required |
| **Performance** | Good | Excellent |
| **Metadata Filtering** | Native | Limited |
| **Memory Usage** | Moderate | Low |
| **Setup Complexity** | Simple | Simple |
| **Best For** | Production RAG | High-performance search |

### When to Use Chroma

- Need automatic persistence
- Require complex metadata filtering
- Building production RAG systems
- Want simple setup and management

**Example:**

```python
# Chroma - automatically persisted
manager = VectorStoreManager()
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="production_docs",
    persist_dir="./data/vectors",
    store_type="chroma"
)
# Automatically saved to disk
```

---

### When to Use FAISS

- Need maximum search performance
- Working with large datasets (millions of vectors)
- Memory-constrained environments
- Batch processing workflows

**Example:**

```python
# FAISS - high performance
manager = VectorStoreManager()
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="fast_search",
    persist_dir="./data/vectors",
    store_type="faiss"
)
# Manual save
vectorstore.save_local("./data/vectors/faiss/fast_search")
```

---

## Performance Tips

### Chunking Strategy

```python
# Small chunks (more granular, slower)
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="granular",
    persist_dir="./data/vectors",
    chunk_size=256,
    chunk_overlap=25
)

# Medium chunks (balanced)
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="balanced",
    persist_dir="./data/vectors",
    chunk_size=512,
    chunk_overlap=50
)

# Large chunks (faster, less granular)
vectorstore = manager.create_from_documents(
    documents=docs,
    collection_name="fast",
    persist_dir="./data/vectors",
    chunk_size=1024,
    chunk_overlap=100
)
```

**Recommended:** Start with 512 tokens, adjust based on results.

---

### Embedding Model Selection

```python
# Fast, good quality
manager = VectorStoreManager(embedding_model="nomic-embed-text")

# Custom model
manager = VectorStoreManager(embedding_model="mxbai-embed-large")
```

**Benchmark Your Models:**

```python
import time

models = ["nomic-embed-text", "qwen3-embedding", "mxbai-embed-large"]

for model_name in models:
    manager = VectorStoreManager(embedding_model=model_name)

    start = time.time()
    vectorstore = manager.create_from_documents(docs, model_name, "./bench")
    elapsed = time.time() - start

    print(f"{model_name}: {elapsed:.2f}s for {len(docs)} docs")
```

---

### Search Optimization

```python
# Use metadata filters to reduce search space
results = manager.similarity_search(
    vectorstore,
    query="query",
    k=5,
    filter={"category": "technical"}  # Narrows search
)

# Adjust k based on use case
results = manager.similarity_search(
    vectorstore,
    query="query",
    k=10  # Retrieve more, then rerank
)
```

---

## Integration Examples

### RAG System

```python
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from utils.vector_store import VectorStoreManager

# Create vector store
manager = VectorStoreManager()
vectorstore = manager.load_existing("knowledge_base", "./data/vectors")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG chain
llm = ChatOllama(model="qwen3:8b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Query
response = qa_chain.invoke("What are AI agents?")
print(response["result"])
```

---

### Document Ingestion Pipeline

```python
from langchain_community.document_loaders import DirectoryLoader
from utils.vector_store import VectorStoreManager

def ingest_documents(directory: str, collection_name: str):
    # Load documents
    loader = DirectoryLoader(directory, glob="**/*.txt")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    # Create vector store
    manager = VectorStoreManager(embedding_model="nomic-embed-text")
    vectorstore = manager.create_from_documents(
        documents=documents,
        collection_name=collection_name,
        persist_dir="./data/vectors",
        chunk_size=512,
        chunk_overlap=50
    )

    print(f"Created collection: {collection_name}")
    return vectorstore

# Usage
vectorstore = ingest_documents("./docs", "documentation")
```

---

### Incremental Updates

```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager()

# Load existing store
vectorstore = manager.load_existing("docs", "./data/vectors")

# Add new documents incrementally
new_docs = load_new_documents()
manager.add_documents(vectorstore, new_docs)

print(f"Added {len(new_docs)} new documents")
```

---

## Error Handling

```python
from utils.vector_store import VectorStoreManager
import logging

logging.basicConfig(level=logging.INFO)
manager = VectorStoreManager()

try:
    vectorstore = manager.create_from_documents(
        documents=docs,
        collection_name="my_docs",
        persist_dir="./data/vectors"
    )
except ValueError as e:
    print(f"Invalid input: {e}")
except ConnectionError as e:
    print(f"Ollama not available: {e}")
    print("Ensure Ollama is running and model is pulled:")
    print("  ollama pull nomic-embed-text")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## See Also

- [Ollama Manager](./ollama_manager.md) - Embedding model management
- [State Manager](./state_manager.md) - Workflow persistence
- [MCP Client](./mcp_client.md) - Document loading tools
- [Examples](../../examples/04-rag/) - RAG system examples

---

**Module Location:** `/Volumes/JS-DEV/ai-lang-stuff/utils/vector_store.py`

**Tests:** `/Volumes/JS-DEV/ai-lang-stuff/tests/benchmarks/vector_store_performance.py`
