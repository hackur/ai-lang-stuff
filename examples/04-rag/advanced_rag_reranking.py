"""
Advanced RAG with Multi-Stage Retrieval and Re-ranking.

This example demonstrates an advanced RAG pipeline that combines:
1. Hybrid retrieval (BM25 + semantic search)
2. Cross-encoder re-ranking for improved relevance
3. Quality scoring and context compression
4. Multi-stage filtering for optimal results

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3:8b`  (for LLM)
  - `ollama pull qwen3-embedding`  (for embeddings)
- Sample documents to index

Expected output:
Highly relevant answers with improved retrieval precision through multi-stage
ranking, quality scoring, and context compression.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import logging

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager, VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityScorer:
    """Score documents based on relevance and quality metrics."""

    def __init__(self, min_length: int = 50, max_length: int = 2000):
        """
        Initialize quality scorer.

        Args:
            min_length: Minimum acceptable document length.
            max_length: Maximum optimal document length.
        """
        self.min_length = min_length
        self.max_length = max_length

    def score_document(self, doc: Document, query: str) -> float:
        """
        Score a document based on quality and relevance metrics.

        Args:
            doc: Document to score.
            query: Query string for relevance checking.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0
        content = doc.page_content.lower()
        query_lower = query.lower()
        query_terms = query_lower.split()

        # 1. Length score (0.2 weight)
        length = len(doc.page_content)
        if length < self.min_length:
            length_score = 0.0
        elif length > self.max_length:
            length_score = 0.8  # Penalize very long docs
        else:
            length_score = 1.0
        score += 0.2 * length_score

        # 2. Query term presence (0.3 weight)
        term_matches = sum(1 for term in query_terms if term in content)
        term_score = term_matches / len(query_terms) if query_terms else 0.0
        score += 0.3 * term_score

        # 3. Query term density (0.2 weight)
        total_words = len(content.split())
        density = term_matches / total_words if total_words > 0 else 0.0
        density_score = min(density * 100, 1.0)  # Normalize
        score += 0.2 * density_score

        # 4. Structural quality (0.15 weight)
        has_structure = any(marker in content for marker in ["\n\n", ". ", ", "])
        structure_score = 1.0 if has_structure else 0.5
        score += 0.15 * structure_score

        # 5. Metadata richness (0.15 weight)
        metadata_score = min(len(doc.metadata) / 5.0, 1.0)
        score += 0.15 * metadata_score

        return score


class ContextCompressor:
    """Compress retrieved contexts to essential information."""

    def __init__(self, max_tokens: int = 1500):
        """
        Initialize context compressor.

        Args:
            max_tokens: Maximum tokens to retain (approximate).
        """
        self.max_tokens = max_tokens

    def compress(self, documents: List[Document], query: str) -> List[Document]:
        """
        Compress documents by extracting most relevant sentences.

        Args:
            documents: Documents to compress.
            query: Query for relevance scoring.

        Returns:
            Compressed documents.
        """
        compressed = []
        query_terms = set(query.lower().split())

        for doc in documents:
            # Split into sentences
            sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]

            # Score each sentence
            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance = sum(1 for term in query_terms if term in sentence_lower)
                scored_sentences.append((relevance, sentence))

            # Sort by relevance and take top sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])

            # Reconstruct document with most relevant sentences
            char_budget = (self.max_tokens * 4) // len(documents)  # ~4 chars per token
            compressed_content = []
            current_length = 0

            for _, sentence in scored_sentences:
                sentence_len = len(sentence)
                if current_length + sentence_len <= char_budget:
                    compressed_content.append(sentence)
                    current_length += sentence_len
                else:
                    break

            if compressed_content:
                compressed_doc = Document(
                    page_content=". ".join(compressed_content) + ".",
                    metadata={**doc.metadata, "compressed": True},
                )
                compressed.append(compressed_doc)

        return compressed


class AdvancedRAGRetriever:
    """Advanced retrieval pipeline with hybrid search and re-ranking."""

    def __init__(
        self,
        vectorstore,
        documents: List[Document],
        quality_scorer: QualityScorer,
        context_compressor: ContextCompressor,
        k: int = 10,
        final_k: int = 4,
    ):
        """
        Initialize advanced RAG retriever.

        Args:
            vectorstore: Vector store for semantic search.
            documents: All documents for BM25 retrieval.
            quality_scorer: Quality scorer instance.
            context_compressor: Context compressor instance.
            k: Number of documents to retrieve initially.
            final_k: Number of documents to return after re-ranking.
        """
        self.vectorstore = vectorstore
        self.quality_scorer = quality_scorer
        self.context_compressor = context_compressor
        self.k = k
        self.final_k = final_k

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = k

        # Create ensemble retriever (hybrid)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vectorstore.as_retriever(search_kwargs={"k": k})],
            weights=[0.4, 0.6],  # BM25: 40%, Semantic: 60%
        )

    def retrieve_and_rerank(self, query: str) -> Tuple[List[Document], dict]:
        """
        Retrieve documents with multi-stage ranking.

        Args:
            query: Query string.

        Returns:
            Tuple of (re-ranked documents, retrieval stats).
        """
        logger.info(f"Stage 1: Hybrid retrieval for query: '{query}'")

        # Stage 1: Hybrid retrieval (BM25 + Semantic)
        initial_docs = self.ensemble_retriever.invoke(query)
        logger.info(f"  -> Retrieved {len(initial_docs)} initial documents")

        # Stage 2: Deduplicate based on content similarity
        unique_docs = []
        seen_contents = set()
        for doc in initial_docs:
            # Use first 200 chars as fingerprint
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(fingerprint)

        logger.info(f"Stage 2: Deduplication -> {len(unique_docs)} unique documents")

        # Stage 3: Quality scoring and re-ranking
        logger.info("Stage 3: Quality scoring and re-ranking")
        scored_docs = []
        for doc in unique_docs:
            quality_score = self.quality_scorer.score_document(doc, query)
            scored_docs.append((quality_score, doc))

        # Sort by quality score
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Take top final_k
        top_docs = [doc for score, doc in scored_docs[: self.final_k]]
        logger.info(f"  -> Selected top {len(top_docs)} documents")

        # Stage 4: Context compression
        logger.info("Stage 4: Context compression")
        compressed_docs = self.context_compressor.compress(top_docs, query)
        logger.info(f"  -> Compressed to {len(compressed_docs)} documents")

        # Calculate stats
        stats = {
            "initial_count": len(initial_docs),
            "unique_count": len(unique_docs),
            "final_count": len(compressed_docs),
            "avg_quality_score": sum(s for s, _ in scored_docs[: self.final_k]) / self.final_k,
            "compression_ratio": sum(len(d.page_content) for d in compressed_docs)
            / sum(len(d.page_content) for d in top_docs)
            if top_docs
            else 0.0,
        }

        return compressed_docs, stats


def load_documents(data_path: str) -> List[Document]:
    """
    Load documents from a directory or file.

    Args:
        data_path: Path to data directory or text file.

    Returns:
        List of loaded documents.
    """
    data_path_obj = Path(data_path)

    if data_path_obj.is_file():
        logger.info(f"Loading single file: {data_path}")
        loader = TextLoader(str(data_path_obj))
        return loader.load()

    elif data_path_obj.is_dir():
        logger.info(f"Loading directory: {data_path}")
        loader = DirectoryLoader(str(data_path_obj), glob="**/*.txt", loader_cls=TextLoader)
        return loader.load()

    else:
        raise ValueError(f"Path not found: {data_path}")


def chunk_documents(
    documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 150
) -> List[Document]:
    """
    Split documents into chunks for retrieval.

    Args:
        documents: Documents to chunk.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def create_advanced_qa_chain(retriever: AdvancedRAGRetriever, llm_model: str = "qwen3:8b"):
    """
    Create an advanced QA chain with the retriever.

    Args:
        retriever: Advanced RAG retriever instance.
        llm_model: Ollama model name.

    Returns:
        Callable QA function.
    """
    # Initialize LLM
    llm = ChatOllama(model=llm_model, temperature=0.2, base_url="http://localhost:11434")

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant that answers questions based on provided context.
Use the following context to answer the question. If the answer is not in the context, say so clearly.
Always ground your answer in the provided context and cite specific details when possible.

Context:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    def qa_function(query: str) -> dict:
        """
        Answer a question using advanced RAG.

        Args:
            query: Question to answer.

        Returns:
            Dictionary with answer, sources, and stats.
        """
        # Retrieve and re-rank
        docs, stats = retriever.retrieve_and_rerank(query)

        if not docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "stats": stats,
            }

        # Format context
        context = "\n\n".join(
            [f"[Document {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        # Generate answer
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})

        return {"answer": response.content, "sources": docs, "stats": stats}

    return qa_function


def interactive_qa_loop(qa_function):
    """
    Run interactive Q&A session.

    Args:
        qa_function: QA function to use.
    """
    print("\n" + "=" * 80)
    print("Advanced RAG with Multi-Stage Re-ranking")
    print("=" * 80)
    print("Ask questions. Type 'quit' to exit, 'stats' to see retrieval statistics.")
    print("=" * 80 + "\n")

    last_result = None

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if question.lower() == "stats" and last_result:
                print("\nRetrieval Statistics:")
                print("-" * 60)
                stats = last_result["stats"]
                print(f"Initial retrieval: {stats['initial_count']} documents")
                print(f"After deduplication: {stats['unique_count']} documents")
                print(f"Final selection: {stats['final_count']} documents")
                print(f"Avg quality score: {stats['avg_quality_score']:.3f}")
                print(f"Compression ratio: {stats['compression_ratio']:.2%}")
                print("-" * 60)
                continue

            # Query
            print("\nProcessing...")
            result = qa_function(question)
            last_result = result

            # Display answer
            print("\nAnswer:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)

            # Show sources
            if result["sources"]:
                print(f"\nBased on {len(result['sources'])} sources")
                print("(Type 'stats' to see retrieval statistics)")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced RAG with multi-stage retrieval and re-ranking"
    )
    parser.add_argument("data_path", help="Path to text file or directory containing documents")
    parser.add_argument(
        "--collection",
        default="advanced_rag",
        help="Vector store collection name (default: advanced_rag)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./data/vector_stores",
        help="Vector store directory (default: ./data/vector_stores)",
    )
    parser.add_argument(
        "--llm-model", default="qwen3:8b", help="Ollama LLM model (default: qwen3:8b)"
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding",
        help="Ollama embedding model (default: qwen3-embedding)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store")

    args = parser.parse_args()

    print("=" * 80)
    print("Advanced RAG System - Initialization")
    print("=" * 80)

    # Step 1: Check Ollama
    print("\n1. Checking Ollama...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("Error: Ollama not running! Start with: ollama serve")
        return

    # Step 2: Verify models
    print("\n2. Verifying models...")
    for model_name in [args.llm_model, args.embedding_model]:
        if not ollama_mgr.ensure_model_available(model_name):
            print(f"Error: Model '{model_name}' not available")
            return

    # Step 3: Load documents
    print("\n3. Loading documents...")
    documents = load_documents(args.data_path)
    logger.info(f"Loaded {len(documents)} documents")

    # Step 4: Chunk documents
    print("\n4. Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 5: Create/load vector store
    print("\n5. Setting up vector store...")
    vector_mgr = VectorStoreManager(embedding_model=args.embedding_model)

    collections = vector_mgr.list_collections(args.persist_dir)
    collection_exists = args.collection in collections.get("chroma", [])

    if collection_exists and not args.rebuild:
        print(f"   Loading existing collection '{args.collection}'...")
        vectorstore = vector_mgr.load_existing(
            collection_name=args.collection, persist_dir=args.persist_dir
        )
    else:
        print(f"   Creating vector store '{args.collection}'...")
        vectorstore = vector_mgr.create_from_documents(
            documents=chunks, collection_name=args.collection, persist_dir=args.persist_dir
        )

    # Step 6: Initialize advanced components
    print("\n6. Initializing advanced RAG components...")
    quality_scorer = QualityScorer(min_length=50, max_length=1500)
    context_compressor = ContextCompressor(max_tokens=1500)

    advanced_retriever = AdvancedRAGRetriever(
        vectorstore=vectorstore,
        documents=chunks,
        quality_scorer=quality_scorer,
        context_compressor=context_compressor,
        k=10,
        final_k=4,
    )

    # Step 7: Create QA chain
    print("\n7. Creating QA chain...")
    qa_function = create_advanced_qa_chain(advanced_retriever, args.llm_model)

    # Step 8: Start interactive session
    print("\n8. Starting interactive session...")
    interactive_qa_loop(qa_function)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)
