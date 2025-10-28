"""
Document Question Answering with RAG using Local LLMs.

This example demonstrates how to build a RAG (Retrieval Augmented Generation)
system for question answering over PDF documents using local models.

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3:8b`  (for LLM)
  - `ollama pull qwen3-embedding`  (for embeddings)
- PDF documents to query

Expected output:
Interactive Q&A loop that answers questions based on PDF content with source citations.
"""

import sys
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager, VectorStoreManager


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load and parse PDF documents.

    Args:
        pdf_path: Path to PDF file or directory containing PDFs.

    Returns:
        List of Document objects with content and metadata.
    """
    pdf_path_obj = Path(pdf_path)
    documents = []

    if pdf_path_obj.is_file():
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path_obj))
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

    elif pdf_path_obj.is_dir():
        pdf_files = list(pdf_path_obj.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in directory")

        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            print(f"  -> {len(docs)} pages")

        print(f"Total: {len(documents)} pages loaded")
    else:
        raise ValueError(f"Path not found: {pdf_path}")

    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks for better retrieval.

    Args:
        documents: List of documents to chunk.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between chunks for context preservation.

    Returns:
        List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def create_qa_chain(vectorstore, llm_model: str = "qwen3:8b"):
    """Create a RetrievalQA chain with custom prompt.

    Args:
        vectorstore: Vector store for document retrieval.
        llm_model: Name of the Ollama model to use.

    Returns:
        RetrievalQA chain instance.
    """
    # Initialize LLM
    llm = ChatOllama(
        model=llm_model,
        temperature=0.3,  # Lower temperature for factual answers
        base_url="http://localhost:11434"
    )

    # Custom prompt template for Q&A
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know - don't try to make up an answer.
Always cite the source page when possible.

Context:
{context}

Question: {question}

Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def interactive_qa_loop(qa_chain):
    """Run interactive Q&A session.

    Args:
        qa_chain: Initialized RetrievalQA chain.
    """
    print("\n" + "=" * 80)
    print("PDF Question Answering System")
    print("=" * 80)
    print("Ask questions about your documents. Type 'quit' or 'exit' to stop.")
    print("Type 'sources' to see source documents for the last answer.")
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

            if question.lower() == "sources" and last_result:
                print("\nSource Documents:")
                print("-" * 60)
                for i, doc in enumerate(last_result["source_documents"], 1):
                    print(f"\nSource {i}:")
                    print(f"Page: {doc.metadata.get('page', 'N/A')}")
                    print(f"File: {doc.metadata.get('source', 'N/A')}")
                    print(f"Content: {doc.page_content[:300]}...")
                print("-" * 60)
                continue

            # Query the chain
            print("\nThinking...")
            result = qa_chain.invoke({"query": question})
            last_result = result

            # Display answer
            print("\nAnswer:")
            print("-" * 60)
            print(result["result"])
            print("-" * 60)

            # Show number of sources
            num_sources = len(result.get("source_documents", []))
            print(f"\n(Based on {num_sources} source documents. Type 'sources' to view them)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Question Answering with RAG")
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--collection",
        default="pdf_qa",
        help="Name for the vector store collection (default: pdf_qa)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./data/vector_stores",
        help="Directory to store vector database (default: ./data/vector_stores)"
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
    print("PDF Question Answering System - Setup")
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

        # Load PDFs
        print("\n   Loading PDFs...")
        documents = load_pdf_documents(args.pdf_path)

        if not documents:
            print("\nError: No documents loaded!")
            return

        # Chunk documents
        print("\n   Chunking documents...")
        chunks = chunk_documents(documents)

        # Create vector store
        print("\n   Creating vector store (this may take a while)...")
        vectorstore = vector_mgr.create_from_documents(
            documents=chunks,
            collection_name=args.collection,
            persist_dir=args.persist_dir
        )
        print(f"   -> Saved to {persist_path / 'chroma' / args.collection}")

    # Step 5: Create QA chain
    print("\n5. Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore, args.llm_model)

    # Step 6: Start interactive Q&A
    print("\n6. Starting interactive Q&A session...")
    interactive_qa_loop(qa_chain)


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
        print("3. Is the PDF path correct?")
        print("4. Do you have write permissions for the persist directory?")
        sys.exit(1)
