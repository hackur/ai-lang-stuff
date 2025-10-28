# RAG Systems - Retrieval-Augmented Generation Examples

This directory contains examples demonstrating RAG (Retrieval-Augmented Generation) systems using local models, embeddings, and vector stores. All examples run entirely locally without cloud dependencies.

## Table of Contents
- [What is RAG?](#what-is-rag)
- [RAG Concepts](#rag-concepts)
- [Vector Store Options](#vector-store-options)
- [Prerequisites](#prerequisites)
- [Examples](#examples)
- [Chunking Strategies](#chunking-strategies)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## What is RAG?

RAG (Retrieval-Augmented Generation) combines information retrieval with language generation to answer questions based on your own documents. Instead of relying solely on a model's training data, RAG:

1. **Retrieves** relevant information from your document collection
2. **Augments** the LLM prompt with this retrieved context
3. **Generates** accurate answers grounded in your documents

This approach is ideal for:
- Question answering over private documents
- Knowledge base systems
- Document analysis and summarization
- Research assistants
- Chatbots with domain-specific knowledge

---

## RAG Concepts

### Embeddings
Embeddings are numerical representations (vectors) of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search.

**Local Embedding Models:**
```bash
# Small, fast model for embeddings
ollama pull nomic-embed-text

# Alternative embedding models
ollama pull mxbai-embed-large  # Higher quality, slower
ollama pull all-minilm         # Lightweight option
```

**How Embeddings Work:**
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Convert text to vector
text = "The cat sat on the mat"
vector = embeddings.embed_query(text)  # Returns list of floats (768 dimensions)

# Similar texts have similar vectors
text1 = "A feline rested on a rug"
text2 = "The weather is sunny today"
# cosine_similarity(text, text1) > cosine_similarity(text, text2)
```

### Retrieval Process
1. **Indexing**: Convert documents to embeddings and store in vector database
2. **Query**: Convert user question to embedding
3. **Search**: Find most similar document chunks using vector similarity
4. **Ranking**: Order results by relevance score
5. **Context Building**: Combine top results into context

### Generation with Context
```python
# Retrieved context is inserted into prompt
prompt = f"""
Answer the question based on the following context:

Context:
{retrieved_documents}

Question: {user_question}

Answer:
"""
```

---

## Vector Store Options

### Chroma (Recommended)
**Best for:** General purpose, easy setup, persistent storage

**Pros:**
- Simple API and setup
- Built-in persistence
- Metadata filtering
- Good performance for medium datasets (up to 1M documents)
- Active development and community

**Cons:**
- Higher memory usage than FAISS
- Slower for very large datasets (10M+ documents)

**Use Case:** Most local RAG applications

### FAISS
**Best for:** Maximum speed, very large datasets

**Pros:**
- Extremely fast similarity search
- Memory efficient with quantization
- Scales to billions of vectors
- Battle-tested (Facebook/Meta)

**Cons:**
- More complex setup
- No built-in persistence (requires manual save/load)
- Limited metadata filtering
- Less intuitive API

**Use Case:** High-performance applications with large document collections

### Comparison Table

| Feature | Chroma | FAISS |
|---------|--------|-------|
| Setup Difficulty | Easy | Medium |
| Speed (small datasets) | Fast | Very Fast |
| Speed (large datasets) | Medium | Excellent |
| Memory Usage | Higher | Lower |
| Persistence | Built-in | Manual |
| Metadata Filtering | Yes | Limited |
| Best Dataset Size | <1M docs | Any size |

---

## Prerequisites

### 1. Install Dependencies
```bash
# Core RAG dependencies
uv add langchain-ollama langchain-chroma chromadb

# Text processing
uv add langchain-text-splitters

# Document loaders
uv add langchain-community pypdf python-docx

# For FAISS (optional)
uv add faiss-cpu  # or faiss-gpu if you have CUDA

# For advanced features
uv add sentence-transformers  # Better embeddings
uv add unstructured  # More document formats
```

### 2. Pull Required Models
```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Generation models (choose one or more)
ollama pull qwen3:8b         # Good balance
ollama pull qwen3:30b-a3b    # Faster, similar quality
ollama pull gemma3:12b       # Alternative option

# Vision model (for image analysis)
ollama pull qwen3-vl:8b
```

### 3. Verify Setup
```bash
# Check Ollama is running
curl http://localhost:11434

# Verify models are available
ollama list

# Test embedding model
ollama run nomic-embed-text "test"
```

### 4. Prepare Sample Data
```bash
# Create data directory
mkdir -p ./data/documents

# Add your documents (PDF, TXT, MD, DOCX, etc.)
# Example: copy your documentation to ./data/documents/
```

---

## Examples

### Example 1: Basic Document QA
**File:** `document_qa.py`

Simple RAG system for answering questions from markdown documentation.

**Purpose:** Get started with RAG concepts using local embeddings and Chroma.

**Features:**
- Load markdown files from directory
- Chunk documents intelligently
- Store in Chroma vector database
- Answer questions with source attribution

**Usage:**
```bash
# Run the example
python examples/04-rag/document_qa.py

# Example queries it handles:
# - "What are the key features of LangGraph?"
# - "How do I install dependencies?"
# - "Explain the RAG architecture"
```

**Expected Output:**
```
Answer: LangGraph's key features include...

Sources:
- ./docs/langgraph-intro.md
- ./docs/architecture.md
```

### Example 2: Advanced RAG with Multi-Stage Re-ranking
**File:** `advanced_rag_reranking.py`

Multi-stage retrieval pipeline with hybrid search, quality scoring, and context compression.

**Purpose:** Maximize retrieval precision through advanced ranking techniques.

**Features:**
- Hybrid retrieval (BM25 + semantic search)
- Quality scoring with multiple metrics
- Multi-stage re-ranking pipeline
- Context compression for optimal token usage
- Deduplication and filtering

**Usage:**
```bash
# Basic usage
python examples/04-rag/advanced_rag_reranking.py ./data/documents

# With custom collection
python examples/04-rag/advanced_rag_reranking.py ./data/documents --collection tech_docs

# Rebuild index
python examples/04-rag/advanced_rag_reranking.py ./data/documents --rebuild
```

**Example Query:**
```
Question: What are the best practices for API design?

Stage 1: Hybrid retrieval -> 10 documents
Stage 2: Deduplication -> 8 unique documents
Stage 3: Quality scoring and re-ranking -> Top 4 selected
Stage 4: Context compression -> Optimized for LLM

Answer: Based on the retrieved documents, API design best practices include...
[Source 1] [Source 2]
```

### Example 3: Vision RAG - Multi-Modal Question Answering
**File:** `vision_rag.py`

Multi-modal RAG system that works with both images and text using vision-language models.

**Purpose:** Answer questions about image collections with AI-generated descriptions.

**Features:**
- Automatic image description generation
- Visual question answering with qwen3-vl:8b
- Combined text + image retrieval
- Multi-modal context integration
- Support for JPG, PNG, GIF, WebP formats

**Usage:**
```bash
# Index image directory
python examples/04-rag/vision_rag.py ./data/images

# With custom models
python examples/04-rag/vision_rag.py ./data/images \
  --vision-model qwen3-vl:8b \
  --text-model qwen3:8b

# Rebuild index
python examples/04-rag/vision_rag.py ./data/images --rebuild
```

**Example Queries:**
```
Question: Show me images with blue cars
Question: What colors appear most frequently in these images?
Question: Describe the architecture in the building photos
Question: Are there any images with text? What do they say?
```

**How It Works:**
1. Scans directory for images
2. Generates detailed descriptions using vision model
3. Stores descriptions in vector database
4. For queries requesting visual details, uses vision model directly
5. For general queries, uses text model with descriptions

### Example 4: Streaming RAG - Real-Time Response Generation
**File:** `streaming_rag.py`

RAG system with progressive answer generation and real-time source citations.

**Purpose:** Improve user experience with immediate feedback during answer generation.

**Features:**
- Token-by-token streaming output
- Progressive source citation display
- Real-time retrieval feedback
- Relevance scores with results
- Context-aware streaming

**Usage:**
```bash
# Basic usage
python examples/04-rag/streaming_rag.py ./data/documents

# With custom configuration
python examples/04-rag/streaming_rag.py ./data/documents \
  --llm-model qwen3:8b \
  --embedding-model qwen3-embedding
```

**Example Output:**
```
Question: How do I implement authentication?

Retrieving relevant documents...
Retrieved 4 relevant documents.

Generating answer...

Answer:
--------------------------------------------------------------------------------
To implement authentication, you need to [Source 1] configure a secure session
management system. Start by [Source 2] setting up password hashing using bcrypt
or similar algorithms...
--------------------------------------------------------------------------------

Sources (2 cited, 4 retrieved):

[Source 1] - Relevance: 92%
  File: auth_guide.txt
  Preview: Authentication is a critical security component...

[Source 2] - Relevance: 88%
  File: security_best_practices.txt
  Preview: Password hashing should use industry-standard...
```

### Example 5: Codebase Search (Advanced RAG Application)
**File:** `codebase_search.py`

Search and analyze code repositories with RAG-powered semantic search.

**Purpose:** Navigate and understand large codebases using natural language.

**Features:**
- Code-aware text splitting
- Multi-language support
- Function and class extraction
- Semantic code search
- Cross-file references

**Usage:**
```bash
# Index codebase
python examples/04-rag/codebase_search.py ./src --build-index

# Search with natural language
python examples/04-rag/codebase_search.py ./src --query "authentication middleware"
python examples/04-rag/codebase_search.py ./src --query "how to handle errors"
```

---

## Chunking Strategies

Chunking splits documents into smaller pieces for better retrieval. The right strategy depends on your content type.

### 1. RecursiveCharacterTextSplitter (Recommended)
**Best for:** General text, code, documentation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Target size in characters
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Split hierarchy
)
```

**How it works:**
1. Try to split on paragraphs (`\n\n`)
2. If too large, split on sentences (`\n`)
3. If still too large, split on words (` `)
4. Last resort: split on characters

**Settings Guide:**
- **Short Q&A**: chunk_size=500, overlap=50
- **General docs**: chunk_size=1000, overlap=200 (default)
- **Technical docs**: chunk_size=1500, overlap=300
- **Long articles**: chunk_size=2000, overlap=400

### 2. MarkdownHeaderTextSplitter
**Best for:** Markdown documentation with headers

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
```

**Advantages:**
- Preserves document structure
- Maintains heading context
- Better for hierarchical docs

### 3. CodeTextSplitter
**Best for:** Source code files

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # or JAVASCRIPT, JAVA, etc.
    chunk_size=1000,
    chunk_overlap=200
)
```

**Features:**
- Respects code structure (functions, classes)
- Preserves syntax validity
- Language-aware separators

### 4. SemanticChunker
**Best for:** Maximum coherence, slower processing

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)
```

**How it works:**
- Uses embeddings to find natural breakpoints
- Splits where topic/meaning changes
- More accurate but computationally expensive

### Choosing Chunk Size

**Small chunks (200-500 chars):**
- Pros: Precise retrieval, lower noise
- Cons: May lose context, more chunks to process
- Use for: Q&A, fact extraction

**Medium chunks (1000-1500 chars):**
- Pros: Good balance of context and precision
- Cons: May include irrelevant information
- Use for: General documentation, articles

**Large chunks (2000+ chars):**
- Pros: Maximum context, fewer chunks
- Cons: Lower retrieval precision, more noise
- Use for: Long-form content, books

**Overlap Guidelines:**
- 10-20% of chunk_size (typical)
- Larger overlap: Better context continuity, more storage
- No overlap: Risk losing information at boundaries

---

## Performance Tips

### 1. Optimize Embedding Generation
```python
# Batch embed for faster indexing
texts = [chunk.page_content for chunk in chunks]
embeddings_list = embeddings.embed_documents(texts)  # Batch operation

# vs. slower loop
for chunk in chunks:
    embedding = embeddings.embed_query(chunk.page_content)  # One at a time
```

### 2. Choose Right Retrieval Parameters
```python
# Start with these defaults
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={
        "k": 5,  # Number of chunks to retrieve
        # Increase for more context, decrease for precision
    }
)

# For diverse results (avoid redundancy)
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Candidates to consider
        "lambda_mult": 0.5,  # Diversity vs relevance (0=diverse, 1=similar)
    }
)
```

### 3. Use Metadata Filtering
```python
# Add metadata during indexing
texts = text_splitter.split_documents(documents)
for chunk in texts:
    chunk.metadata["source"] = doc.metadata["source"]
    chunk.metadata["doc_type"] = "tutorial"
    chunk.metadata["section"] = extract_section(chunk)

# Filter during retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"doc_type": "tutorial"}  # Only search tutorials
    }
)
```

### 4. Cache Embeddings
```python
import pickle
from pathlib import Path

cache_file = Path("./data/embeddings_cache.pkl")

if cache_file.exists():
    # Load cached vectorstore
    with open(cache_file, "rb") as f:
        vectorstore = pickle.load(f)
else:
    # Create and cache vectorstore
    vectorstore = Chroma.from_documents(...)
    with open(cache_file, "wb") as f:
        pickle.dump(vectorstore, f)
```

### 5. Use Lighter Models for Speed
```python
# Fast embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 384 dims, fast

# Fast generation model
llm = ChatOllama(model="qwen3:30b-a3b")  # MoE, ~2x faster than qwen3:30b
```

### 6. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def process_document(doc_path):
    loader = TextLoader(doc_path)
    return loader.load()

# Process multiple documents in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    all_docs = list(executor.map(process_document, doc_paths))
```

### 7. Incremental Indexing
```python
# Add documents incrementally instead of rebuilding
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

# Add new documents
new_texts = text_splitter.split_documents(new_documents)
vectorstore.add_documents(new_texts)
```

### Performance Benchmarks
| Configuration | Indexing Speed | Query Speed | Quality |
|---------------|----------------|-------------|---------|
| Small chunks + nomic-embed | ~100 docs/min | ~200ms | Good |
| Medium chunks + nomic-embed | ~150 docs/min | ~150ms | Better |
| Large chunks + mxbai-embed | ~50 docs/min | ~100ms | Best |
| Semantic chunks + mxbai-embed | ~20 docs/min | ~100ms | Best |

---

## Troubleshooting

### Issue: "Ollama connection refused"
**Symptoms:** `ConnectionError: [Errno 61] Connection refused`

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Verify connection
curl http://localhost:11434
```

### Issue: "Model not found: nomic-embed-text"
**Symptoms:** Model doesn't exist error

**Solution:**
```bash
# List available models
ollama list

# Pull embedding model
ollama pull nomic-embed-text

# Verify
ollama list | grep nomic
```

### Issue: "ChromaDB not found or empty results"
**Symptoms:** No results returned, or vectorstore is empty

**Solution:**
```python
# Check if collection exists
import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")
collections = client.list_collections()
print(f"Collections: {[c.name for c in collections]}")

# Check collection size
if collections:
    collection = collections[0]
    print(f"Documents in collection: {collection.count()}")

# If empty, re-index
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./data/chroma_db"
)
```

### Issue: "Poor retrieval quality / irrelevant results"
**Symptoms:** Retrieved chunks don't match query

**Solutions:**
1. **Adjust chunk size:**
   ```python
   # Try smaller chunks for precision
   splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
   ```

2. **Increase k (retrieve more chunks):**
   ```python
   retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
   ```

3. **Use MMR for diversity:**
   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",
       search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
   )
   ```

4. **Try different embedding model:**
   ```bash
   # Higher quality embeddings
   ollama pull mxbai-embed-large
   ```

5. **Add re-ranking:**
   ```python
   from langchain.retrievers import ContextualCompressionRetriever
   from langchain.retrievers.document_compressors import LLMChainExtractor

   compressor = LLMChainExtractor.from_llm(llm)
   retriever = ContextualCompressionRetriever(
       base_compressor=compressor,
       base_retriever=vectorstore.as_retriever()
   )
   ```

### Issue: "Slow indexing / out of memory"
**Symptoms:** Process hangs or crashes during indexing

**Solutions:**
1. **Process in batches:**
   ```python
   batch_size = 100
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       texts = text_splitter.split_documents(batch)
       vectorstore.add_documents(texts)
   ```

2. **Use lighter embedding model:**
   ```bash
   ollama pull all-minilm  # Smaller, faster
   ```

3. **Reduce chunk overlap:**
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=50  # Reduced from 200
   )
   ```

### Issue: "FAISS index not persisting"
**Symptoms:** Index lost after restart

**Solution:**
```python
# Save FAISS index
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("./data/faiss_index")

# Load FAISS index
vectorstore = FAISS.load_local(
    "./data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Only if you trust the source
)
```

### Issue: "Embedding dimension mismatch"
**Symptoms:** `ValueError: dimension mismatch`

**Cause:** Trying to use different embedding models on same vectorstore

**Solution:**
```bash
# Delete existing vectorstore
rm -rf ./data/chroma_db

# Or create new collection
vectorstore = Chroma(
    collection_name="my_collection_v2",  # New collection
    persist_directory="./data/chroma_db",
    embedding_function=new_embeddings
)
```

### Issue: "LLM generates answers not in documents"
**Symptoms:** Hallucinations despite providing context

**Solution:**
```python
# Use stricter prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
""")

# Use lower temperature
llm = ChatOllama(model="qwen3:8b", temperature=0.0)
```

### Debug Checklist
```python
# Comprehensive RAG debugging script
def debug_rag_system():
    # 1. Check Ollama connection
    try:
        llm = ChatOllama(model="qwen3:8b")
        llm.invoke("test")
        print("✓ Ollama connection OK")
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return

    # 2. Check embedding model
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embed = embeddings.embed_query("test")
        print(f"✓ Embeddings OK (dimension: {len(test_embed)})")
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")
        return

    # 3. Check vectorstore
    try:
        vectorstore = Chroma(
            persist_directory="./data/chroma_db",
            embedding_function=embeddings
        )
        count = vectorstore._collection.count()
        print(f"✓ Vectorstore OK ({count} documents)")
    except Exception as e:
        print(f"✗ Vectorstore failed: {e}")
        return

    # 4. Test retrieval
    try:
        results = vectorstore.similarity_search("test query", k=3)
        print(f"✓ Retrieval OK ({len(results)} results)")
        for i, doc in enumerate(results):
            print(f"  Result {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        return

    # 5. Test end-to-end
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        result = qa_chain({"query": "test question"})
        print(f"✓ End-to-end OK")
        print(f"  Answer: {result['result'][:100]}...")
    except Exception as e:
        print(f"✗ End-to-end failed: {e}")

# Run diagnostics
debug_rag_system()
```

---

## Additional Resources

### Documentation
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Ollama Embeddings](https://ollama.com/blog/embedding-models)

### Related Examples
- `examples/01-foundation/` - Basic LLM interaction
- `examples/02-mcp/` - Tool integration with MCP
- `examples/03-multi-agent/` - Orchestrating multiple agents
- `examples/05-interpretability/` - Model internals analysis

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [Ollama Discord](https://discord.gg/ollama)
- Project issues: See main README

---

## Next Steps

1. **Start with Example 1** (`document_qa.py`) to understand basic RAG
2. **Experiment with chunking** strategies for your document type
3. **Try Example 2** (`advanced_rag.py`) for better retrieval quality
4. **Add vision support** with Example 3 if you have image-heavy docs
5. **Scale up** with proper performance optimization and caching

All examples are designed to run locally with minimal setup. See [Prerequisites](#prerequisites) for installation steps.
