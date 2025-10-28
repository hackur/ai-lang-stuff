"""
Semantic Code Search with RAG using Local LLMs.

This example demonstrates how to build a RAG system for semantic search
and understanding of code repositories using local models.

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3:8b`  (for LLM)
  - `ollama pull qwen3-embedding`  (for embeddings)
- Python codebase to analyze

Expected output:
Interactive code search that can find code by semantic meaning and explain
what code does in natural language.
"""

import sys
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager, VectorStoreManager


def load_codebase(directory: str, extensions: List[str] = None) -> List[Document]:
    """Load Python files from a directory with metadata.

    Args:
        directory: Path to code directory.
        extensions: List of file extensions to load (default: ['.py']).

    Returns:
        List of Document objects with code content and metadata.
    """
    if extensions is None:
        extensions = [".py"]

    directory_path = Path(directory)

    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    print(f"Loading codebase from: {directory}")
    print(f"Looking for files with extensions: {extensions}")

    documents = []

    for ext in extensions:
        files = list(directory_path.rglob(f"*{ext}"))
        print(f"Found {len(files)} {ext} files")

        for file_path in files:
            # Skip common directories to ignore
            skip_dirs = [
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "node_modules",
                ".pytest_cache",
                "build",
                "dist",
                ".egg-info"
            ]

            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

                # Add enhanced metadata
                for doc in docs:
                    doc.metadata.update({
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "relative_path": str(file_path.relative_to(directory_path)),
                        "extension": file_path.suffix,
                        "directory": str(file_path.parent.relative_to(directory_path))
                    })

                documents.extend(docs)

            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
                continue

    print(f"Successfully loaded {len(documents)} files")
    return documents


def chunk_code_documents(
    documents: List[Document],
    language: Language = Language.PYTHON,
    chunk_size: int = 1500,
    chunk_overlap: int = 300
) -> List[Document]:
    """Split code documents using language-aware chunking.

    This preserves code structure like classes, functions, and blocks.

    Args:
        documents: List of documents to chunk.
        language: Programming language for smart splitting.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between chunks for context.

    Returns:
        List of chunked documents.
    """
    # Use language-aware text splitter
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} code chunks")

    return chunks


def create_code_search_chain(vectorstore, llm_model: str = "qwen3:8b"):
    """Create a RetrievalQA chain optimized for code understanding.

    Args:
        vectorstore: Vector store for code retrieval.
        llm_model: Name of the Ollama model to use.

    Returns:
        RetrievalQA chain instance.
    """
    # Initialize LLM
    llm = ChatOllama(
        model=llm_model,
        temperature=0.1,  # Very low temperature for code explanation
        base_url="http://localhost:11434"
    )

    # Custom prompt template for code understanding
    prompt_template = """You are a helpful code assistant. Use the following code snippets to answer the question.
If the code snippets don't contain enough information, say so - don't make up functionality.
When explaining code, be specific about what it does, how it works, and mention file names.

Code Context:
{context}

Question: {question}

Helpful Answer (include file paths when relevant):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create RetrievalQA chain with higher k for code search
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 6}  # Higher k for code context
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def format_code_snippet(doc: Document, max_lines: int = 20) -> str:
    """Format a code document for display.

    Args:
        doc: Document containing code.
        max_lines: Maximum number of lines to show.

    Returns:
        Formatted code snippet string.
    """
    lines = doc.page_content.split("\n")

    # Truncate if too long
    if len(lines) > max_lines:
        snippet = "\n".join(lines[:max_lines])
        snippet += f"\n... ({len(lines) - max_lines} more lines)"
    else:
        snippet = doc.page_content

    # Format with file info
    file_path = doc.metadata.get("relative_path", "unknown")
    output = f"File: {file_path}\n"
    output += "-" * 60 + "\n"
    output += snippet
    output += "\n" + "-" * 60

    return output


def interactive_code_search(qa_chain):
    """Run interactive code search session.

    Args:
        qa_chain: Initialized RetrievalQA chain.
    """
    print("\n" + "=" * 80)
    print("Semantic Code Search System")
    print("=" * 80)
    print("Search your codebase using natural language.")
    print("\nExample queries:")
    print("  - 'How does authentication work?'")
    print("  - 'Find functions that handle file uploads'")
    print("  - 'Show me error handling code'")
    print("  - 'What does the VectorStoreManager class do?'")
    print("\nCommands:")
    print("  - Type 'sources' to see code snippets for the last answer")
    print("  - Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")

    last_result = None

    while True:
        try:
            query = input("\nSearch: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if query.lower() == "sources" and last_result:
                print("\nRelevant Code Snippets:")
                print("=" * 80)
                for i, doc in enumerate(last_result["source_documents"], 1):
                    print(f"\nSnippet {i}:")
                    print(format_code_snippet(doc))
                    print()
                continue

            # Search the codebase
            print("\nSearching codebase...")
            result = qa_chain.invoke({"query": query})
            last_result = result

            # Display answer
            print("\nAnswer:")
            print("=" * 80)
            print(result["result"])
            print("=" * 80)

            # Show stats
            num_sources = len(result.get("source_documents", []))
            unique_files = set(
                doc.metadata.get("file_name", "unknown")
                for doc in result.get("source_documents", [])
            )
            print(f"\n(Found relevant code in {len(unique_files)} files, "
                  f"{num_sources} snippets total)")
            print("Type 'sources' to view the code snippets")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different query.")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Code Search with RAG")
    parser.add_argument(
        "directory",
        help="Path to code directory to index"
    )
    parser.add_argument(
        "--collection",
        default="codebase_search",
        help="Name for the vector store collection (default: codebase_search)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./data/vector_stores",
        help="Directory to store vector database (default: ./data/vector_stores)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".py"],
        help="File extensions to index (default: .py)"
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3:8b",
        help="Ollama LLM model to use (default: qwen3:8b)"
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding",
        help="Ollama embedding model to use (default: qwen3-embedding)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of vector store even if it exists"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Semantic Code Search System - Setup")
    print("=" * 80)

    # Step 1: Check Ollama
    print("\n1. Checking Ollama status...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("\nError: Ollama is not running!")
        print("Start it with: ollama serve")
        return

    # Step 2: Ensure models are available
    print("\n2. Checking required models...")

    print(f"   - LLM model: {args.llm_model}")
    if not ollama_mgr.ensure_model_available(args.llm_model):
        print(f"\nError: Could not load LLM model '{args.llm_model}'")
        return

    print(f"   - Embedding model: {args.embedding_model}")
    if not ollama_mgr.ensure_model_available(args.embedding_model):
        print(f"\nError: Could not load embedding model '{args.embedding_model}'")
        return

    # Step 3: Initialize vector store manager
    print("\n3. Initializing vector store manager...")
    vector_mgr = VectorStoreManager(embedding_model=args.embedding_model)

    # Step 4: Check if collection exists
    persist_path = Path(args.persist_dir)
    collections = vector_mgr.list_collections(args.persist_dir)
    collection_exists = args.collection in collections.get("chroma", [])

    if collection_exists and not args.rebuild:
        print(f"\n4. Loading existing collection '{args.collection}'...")
        vectorstore = vector_mgr.load_existing(
            collection_name=args.collection,
            persist_dir=args.persist_dir
        )
        print(f"   -> Loaded from {persist_path / 'chroma' / args.collection}")

    else:
        if args.rebuild:
            print(f"\n4. Rebuilding collection '{args.collection}'...")
        else:
            print(f"\n4. Creating new collection '{args.collection}'...")

        # Load codebase
        print("\n   Loading codebase...")
        documents = load_codebase(args.directory, args.extensions)

        if not documents:
            print("\nError: No code files loaded!")
            return

        # Chunk code
        print("\n   Chunking code (preserving structure)...")
        chunks = chunk_code_documents(documents)

        # Create vector store
        print("\n   Creating vector store (this may take a while)...")
        vectorstore = vector_mgr.create_from_documents(
            documents=chunks,
            collection_name=args.collection,
            persist_dir=args.persist_dir
        )
        print(f"   -> Saved to {persist_path / 'chroma' / args.collection}")

    # Step 5: Create search chain
    print("\n5. Creating code search chain...")
    search_chain = create_code_search_chain(vectorstore, args.llm_model)

    # Step 6: Start interactive search
    print("\n6. Starting interactive code search...")
    interactive_code_search(search_chain)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Are models installed?")
        print("   - ollama pull qwen3:8b")
        print("   - ollama pull qwen3-embedding")
        print("3. Is the directory path correct?")
        print("4. Do you have read permissions for the code directory?")
        print("5. Do you have write permissions for the persist directory?")
        sys.exit(1)
