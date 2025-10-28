"""
Streaming RAG with Progressive Answer Generation.

This example demonstrates a RAG system with streaming capabilities:
1. Stream tokens as they are generated
2. Progressive answer construction
3. Real-time source citations
4. Token-by-token output for better UX
5. Context-aware streaming

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3:8b`  (for LLM)
  - `ollama pull qwen3-embedding`  (for embeddings)
- Documents to index

Expected output:
Answers that stream in real-time, showing progressive generation with
source citations appearing as they become relevant.
"""

import sys
from pathlib import Path
from typing import List, Iterator
import logging
import re

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


class StreamingRAG:
    """RAG system with streaming response generation."""

    def __init__(
        self, vectorstore, llm_model: str = "qwen3:8b", base_url: str = "http://localhost:11434"
    ):
        """
        Initialize streaming RAG system.

        Args:
            vectorstore: Vector store for retrieval.
            llm_model: Ollama model name.
            base_url: Ollama API endpoint.
        """
        self.vectorstore = vectorstore
        self.llm = ChatOllama(
            model=llm_model,
            base_url=base_url,
            temperature=0.3,
            streaming=True,  # Enable streaming
        )

        # Prompt template with citation instructions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant that answers questions using provided context.

Important guidelines:
1. Base your answer on the provided context
2. When referencing information, cite sources using [Source N] format
3. If information is not in the context, say so clearly
4. Be concise but thorough

Context:
{context}

Sources:
{sources}""",
                ),
                ("human", "{question}"),
            ]
        )

        logger.info(f"Initialized StreamingRAG with model: {llm_model}")

    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.

        Args:
            query: Query string.
            k: Number of documents to retrieve.

        Returns:
            List of (document, score) tuples.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results

    def format_context(self, docs_with_scores: List[tuple[Document, float]]) -> tuple[str, str]:
        """
        Format retrieved documents into context and source list.

        Args:
            docs_with_scores: Documents with relevance scores.

        Returns:
            Tuple of (formatted_context, formatted_sources).
        """
        context_parts = []
        source_parts = []

        for i, (doc, score) in enumerate(docs_with_scores, 1):
            # Format context
            context_parts.append(f"[Source {i}]\n{doc.page_content}")

            # Format source citation
            source_info = f"[Source {i}]"
            if "source" in doc.metadata:
                source_path = Path(doc.metadata["source"])
                source_info += f" {source_path.name}"
            if "page" in doc.metadata:
                source_info += f" (page {doc.metadata['page']})"

            source_parts.append(source_info)

        context = "\n\n".join(context_parts)
        sources = "\n".join(source_parts)

        return context, sources

    def stream_answer(self, query: str, k: int = 4) -> Iterator[dict]:
        """
        Stream answer generation with progressive citations.

        Args:
            query: Question to answer.
            k: Number of documents to retrieve.

        Yields:
            Dictionaries with type ('retrieval', 'token', 'source', 'complete').
        """
        # Step 1: Retrieval
        yield {
            "type": "retrieval",
            "status": "started",
            "message": "Retrieving relevant documents...",
        }

        docs_with_scores = self.retrieve_with_scores(query, k=k)

        if not docs_with_scores:
            yield {"type": "retrieval", "status": "complete", "count": 0}
            yield {
                "type": "token",
                "content": "I couldn't find relevant information to answer your question.",
            }
            yield {"type": "complete"}
            return

        yield {
            "type": "retrieval",
            "status": "complete",
            "count": len(docs_with_scores),
            "documents": [doc for doc, _ in docs_with_scores],
        }

        # Step 2: Format context
        context, sources = self.format_context(docs_with_scores)

        # Step 3: Stream generation
        yield {"type": "generation", "status": "started", "message": "Generating answer..."}

        chain = self.prompt | self.llm

        # Track which sources have been mentioned
        mentioned_sources = set()
        buffer = ""

        # Stream tokens
        for chunk in chain.stream({"context": context, "sources": sources, "question": query}):
            if hasattr(chunk, "content"):
                content = chunk.content
            else:
                content = str(chunk)

            buffer += content

            # Check for source citations in buffer
            source_pattern = r"\[Source (\d+)\]"
            matches = re.finditer(source_pattern, buffer)

            for match in matches:
                source_num = int(match.group(1))
                if source_num not in mentioned_sources and source_num <= len(docs_with_scores):
                    mentioned_sources.add(source_num)
                    doc, score = docs_with_scores[source_num - 1]

                    # Emit source reference
                    yield {"type": "source", "number": source_num, "document": doc, "score": score}

            # Emit token
            yield {"type": "token", "content": content}

        # Step 4: Complete
        yield {
            "type": "complete",
            "sources": docs_with_scores,
            "mentioned_count": len(mentioned_sources),
        }


def load_documents(data_path: str) -> List[Document]:
    """
    Load documents from file or directory.

    Args:
        data_path: Path to data.

    Returns:
        List of documents.
    """
    data_path_obj = Path(data_path)

    if data_path_obj.is_file():
        logger.info(f"Loading file: {data_path}")
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
    Chunk documents for retrieval.

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


def display_streaming_response(stream: Iterator[dict]):
    """
    Display streaming response with formatting.

    Args:
        stream: Iterator of response chunks.
    """
    answer_buffer = []

    for event in stream:
        event_type = event["type"]

        if event_type == "retrieval":
            if event["status"] == "started":
                print(f"\n{event['message']}", flush=True)
            elif event["status"] == "complete":
                count = event["count"]
                print(f"Retrieved {count} relevant documents.\n", flush=True)
                event.get("documents", [])

        elif event_type == "generation":
            if event["status"] == "started":
                print(f"{event['message']}\n", flush=True)
                print("Answer:", flush=True)
                print("-" * 60, flush=True)

        elif event_type == "token":
            content = event["content"]
            answer_buffer.append(content)
            print(content, end="", flush=True)

        elif event_type == "source":
            # Note: Source is mentioned in the answer
            pass

        elif event_type == "complete":
            print("\n" + "-" * 60, flush=True)

            # Display sources
            sources = event.get("sources", [])
            mentioned_count = event.get("mentioned_count", 0)

            if sources:
                print(f"\nSources ({mentioned_count} cited, {len(sources)} retrieved):")
                print("-" * 60)
                for i, (doc, score) in enumerate(sources, 1):
                    source_path = doc.metadata.get("source", "unknown")
                    if source_path != "unknown":
                        source_path = Path(source_path).name

                    print(f"\n[Source {i}] - Relevance: {(1 - score):.2%}")
                    print(f"  File: {source_path}")
                    print(f"  Preview: {doc.page_content[:150]}...")

                print("-" * 60)


def interactive_streaming_qa_loop(streaming_rag: StreamingRAG):
    """
    Run interactive Q&A with streaming responses.

    Args:
        streaming_rag: StreamingRAG instance.
    """
    print("\n" + "=" * 80)
    print("Streaming RAG System")
    print("=" * 80)
    print("Ask questions to see streaming answers with real-time citations.")
    print("Type 'quit' to exit.")
    print("=" * 80 + "\n")

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            # Stream response
            stream = streaming_rag.stream_answer(question)
            display_streaming_response(stream)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Streaming RAG system")
    parser.add_argument("data_path", help="Path to text file or directory")
    parser.add_argument(
        "--collection", default="streaming_rag", help="Collection name (default: streaming_rag)"
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
    print("Streaming RAG System - Initialization")
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

    # Step 6: Create streaming RAG
    print("\n6. Initializing streaming RAG...")
    streaming_rag = StreamingRAG(vectorstore=vectorstore, llm_model=args.llm_model)

    # Step 7: Start interactive session
    print("\n7. Starting interactive session...")
    interactive_streaming_qa_loop(streaming_rag)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)
