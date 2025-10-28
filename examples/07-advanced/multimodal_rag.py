"""
Multi-Modal RAG: Cross-Modal Retrieval System.

This example demonstrates a unified RAG system that:
1. Indexes both images and text documents together
2. Performs cross-modal retrieval (query text, retrieve images and vice versa)
3. Provides unified answers combining visual and textual information
4. Supports mixed-media knowledge bases

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3-vl:8b`  (vision-language model)
  - `ollama pull qwen3:8b`  (text model)
  - `ollama pull qwen3-embedding`  (embeddings)

Expected output:
Unified search across text and images with comprehensive answers.

Usage:
    python multimodal_rag.py /path/to/data_dir --index
    python multimodal_rag.py /path/to/data_dir --query "What does the diagram show?"
    python multimodal_rag.py /path/to/data_dir --interactive
"""

import base64
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager, VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Types of media in the knowledge base."""

    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass
class MultiModalDocument:
    """Document with multi-modal content."""

    content: str
    media_type: MediaType
    source_path: Optional[Path] = None
    metadata: Optional[Dict] = None
    image_path: Optional[Path] = None  # For text docs with associated images


@dataclass
class SearchResult:
    """Result from multi-modal search."""

    answer: str
    sources: List[Document]
    media_types: List[str]
    confidence: str


class MultiModalRAG:
    """
    Unified RAG system for text and images.

    Features:
    - Index text documents and images together
    - Cross-modal retrieval
    - Unified answer generation
    - Support for documents with embedded images
    """

    def __init__(
        self,
        vision_model: str = "qwen3-vl:8b",
        text_model: str = "qwen3:8b",
        embedding_model: str = "qwen3-embedding",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize multi-modal RAG system.

        Args:
            vision_model: Vision-language model.
            text_model: Text-only model.
            embedding_model: Embedding model.
            base_url: Ollama API endpoint.
        """
        self.vision_model = vision_model
        self.text_model = text_model
        self.embedding_model = embedding_model
        self.base_url = base_url

        # Initialize models
        self.vision_llm = ChatOllama(model=vision_model, base_url=base_url, temperature=0.3)

        self.text_llm = ChatOllama(model=text_model, base_url=base_url, temperature=0.2)

        # Vector store manager
        self.vector_mgr = VectorStoreManager(embedding_model=embedding_model)

        logger.info(f"Initialized MultiModalRAG with models: {vision_model}, {text_model}")

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def describe_image(self, image_path: Path) -> str:
        """
        Generate searchable description of image.

        Args:
            image_path: Path to image.

        Returns:
            Detailed description for indexing.
        """
        prompt = """Describe this image in detail for a searchable knowledge base. Include:

1. Main subject and content
2. Key visual elements and features
3. Any text, labels, or annotations
4. Context and purpose (e.g., diagram, photo, chart)
5. Notable details that someone might search for

Make it comprehensive and searchable."""

        try:
            image_b64 = self.encode_image(image_path)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ]
            )

            response = self.vision_llm.invoke([message])
            logger.debug(f"Generated description for {image_path.name}")
            return response.content

        except Exception as e:
            logger.error(f"Error describing image {image_path}: {e}")
            raise

    def index_directory(
        self, data_dir: Path, collection_name: str, persist_dir: str, rebuild: bool = False
    ) -> tuple:
        """
        Index all text and image files in directory.

        Args:
            data_dir: Directory containing mixed media files.
            collection_name: Collection name for vector store.
            persist_dir: Directory to persist vector store.
            rebuild: Whether to rebuild existing index.

        Returns:
            Tuple of (vectorstore, indexed_documents).
        """
        # Check if collection exists
        collections = self.vector_mgr.list_collections(persist_dir)
        collection_exists = collection_name in collections.get("chroma", [])

        if collection_exists and not rebuild:
            logger.info(f"Loading existing collection: {collection_name}")
            vectorstore = self.vector_mgr.load_existing(
                collection_name=collection_name, persist_dir=persist_dir
            )
            return vectorstore, []

        logger.info("Building multi-modal index...")

        # Find all supported files
        text_extensions = {".txt", ".md", ".pdf", ".csv"}
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        text_files = []
        image_files = []

        for ext in text_extensions:
            text_files.extend(data_dir.glob(f"**/*{ext}"))

        for ext in image_extensions:
            image_files.extend(data_dir.glob(f"**/*{ext}"))

        logger.info(f"Found {len(text_files)} text files, {len(image_files)} images")

        all_documents = []

        # Process text files
        if text_files:
            logger.info("Processing text files...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

            for i, file_path in enumerate(text_files, 1):
                logger.info(f"  Text {i}/{len(text_files)}: {file_path.name}")

                try:
                    # Read file
                    if file_path.suffix == ".pdf":
                        # For PDF, you'd use PyPDF loader
                        logger.warning(
                            f"PDF support requires pypdf loader - skipping {file_path.name}"
                        )
                        continue
                    else:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                    # Split into chunks
                    chunks = text_splitter.split_text(content)

                    # Create documents
                    for j, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "type": "text",
                                "format": file_path.suffix[1:],
                                "chunk": j,
                            },
                        )
                        all_documents.append(doc)

                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    continue

        # Process images
        if image_files:
            logger.info("Processing images...")

            for i, image_path in enumerate(image_files, 1):
                logger.info(f"  Image {i}/{len(image_files)}: {image_path.name}")

                try:
                    # Generate description
                    description = self.describe_image(image_path)

                    # Create document
                    doc = Document(
                        page_content=description,
                        metadata={
                            "source": str(image_path),
                            "filename": image_path.name,
                            "type": "image",
                            "format": image_path.suffix[1:].lower(),
                            "image_path": str(image_path),
                        },
                    )
                    all_documents.append(doc)

                except Exception as e:
                    logger.error(f"Failed to process {image_path.name}: {e}")
                    continue

        if not all_documents:
            raise ValueError("No documents were successfully indexed")

        logger.info(f"Successfully processed {len(all_documents)} total documents")

        # Create vector store
        logger.info("Creating vector store...")
        vectorstore = self.vector_mgr.create_from_documents(
            documents=all_documents, collection_name=collection_name, persist_dir=persist_dir
        )

        return vectorstore, all_documents

    def search(
        self, vectorstore, query: str, k: int = 5, media_filter: Optional[MediaType] = None
    ) -> List[Document]:
        """
        Search across multi-modal knowledge base.

        Args:
            vectorstore: Vector store to search.
            query: Search query.
            k: Number of results to return.
            media_filter: Optional filter by media type.

        Returns:
            List of relevant documents.
        """
        logger.info(f"Searching: '{query}'")

        # Perform similarity search
        if media_filter:
            # Filter by media type
            filter_dict = {"type": media_filter.value}
            results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = vectorstore.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} results")
        return results

    def answer_with_sources(
        self, vectorstore, query: str, k: int = 5, use_vision_for_images: bool = True
    ) -> SearchResult:
        """
        Answer query using multi-modal sources.

        Args:
            vectorstore: Vector store to search.
            query: Question to answer.
            k: Number of sources to retrieve.
            use_vision_for_images: Whether to re-analyze images with vision model.

        Returns:
            SearchResult with answer and sources.
        """
        # Search for relevant documents
        docs = self.search(vectorstore, query, k=k)

        if not docs:
            return SearchResult(
                answer="No relevant information found.",
                sources=[],
                media_types=[],
                confidence="none",
            )

        # Separate text and image sources
        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        image_docs = [d for d in docs if d.metadata.get("type") == "image"]

        logger.info(f"Retrieved {len(text_docs)} text docs, {len(image_docs)} image docs")

        # Build context
        context_parts = []

        # Add text content
        if text_docs:
            context_parts.append("TEXT SOURCES:")
            for i, doc in enumerate(text_docs, 1):
                filename = doc.metadata.get("filename", "unknown")
                context_parts.append(f"\n[Text {i} - {filename}]")
                context_parts.append(doc.page_content)

        # Add image descriptions
        if image_docs:
            context_parts.append("\n\nIMAGE SOURCES:")

            # Optionally re-analyze images with vision model
            if use_vision_for_images and image_docs:
                logger.info("Re-analyzing images with vision model...")
                # Use vision model on the most relevant image
                top_image = image_docs[0]
                image_path = Path(top_image.metadata.get("image_path", ""))

                if image_path.exists():
                    try:
                        # Ask vision model the specific question
                        image_b64 = self.encode_image(image_path)
                        message = HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": f"Question: {query}\n\nAnswer based on this image.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                                },
                            ]
                        )
                        vision_response = self.vision_llm.invoke([message])
                        context_parts.append(f"\n[Visual Analysis of {image_path.name}]")
                        context_parts.append(vision_response.content)
                    except Exception as e:
                        logger.error(f"Error re-analyzing image: {e}")

            # Add stored descriptions
            for i, doc in enumerate(image_docs, 1):
                filename = doc.metadata.get("filename", "unknown")
                context_parts.append(f"\n[Image {i} - {filename}]")
                context_parts.append(doc.page_content)

        context = "\n".join(context_parts)

        # Generate answer
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable assistant with access to both text documents and images.
Answer questions accurately based on the provided sources.

Guidelines:
- Cite specific sources (e.g., "According to [Text 1]..." or "The image shows...")
- If information comes from an image, mention it explicitly
- If sources don't contain the answer, say so
- Combine information from multiple sources when relevant""",
                ),
                (
                    "human",
                    """Sources:
{context}

Question: {query}

Answer:""",
                ),
            ]
        )

        logger.info("Generating answer...")
        chain = prompt | self.text_llm
        response = chain.invoke({"context": context, "query": query})

        # Determine media types in results
        media_types = list(set(d.metadata.get("type", "unknown") for d in docs))

        return SearchResult(
            answer=response.content,
            sources=docs,
            media_types=media_types,
            confidence="high" if len(docs) >= 3 else "medium" if docs else "low",
        )

    def interactive_session(self, vectorstore):
        """
        Run interactive multi-modal Q&A session.

        Args:
            vectorstore: Vector store to query.
        """
        print("\n" + "=" * 80)
        print("Multi-Modal RAG - Interactive Session")
        print("=" * 80)
        print("Ask questions about your text and image knowledge base.")
        print("Type 'quit' to exit.")
        print("=" * 80 + "\n")

        while True:
            try:
                query = input("\nQuestion: ").strip()

                if not query:
                    continue

                if query.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                # Process query
                print("\nSearching knowledge base...")
                result = self.answer_with_sources(
                    vectorstore=vectorstore, query=query, k=5, use_vision_for_images=True
                )

                # Display answer
                print("\nAnswer:")
                print("-" * 80)
                print(result.answer)
                print("-" * 80)

                # Show sources
                if result.sources:
                    print(f"\nSources ({len(result.sources)}):")
                    for i, doc in enumerate(result.sources, 1):
                        doc_type = doc.metadata.get("type", "unknown")
                        filename = doc.metadata.get("filename", "unknown")
                        print(f"  {i}. [{doc_type.upper()}] {filename}")

                print(
                    f"\n[Confidence: {result.confidence} | Media types: {', '.join(result.media_types)}]"
                )

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
        description="Multi-Modal RAG with unified text and image search"
    )
    parser.add_argument("data_dir", help="Directory containing text and image files")
    parser.add_argument("--index", action="store_true", help="Index the directory")
    parser.add_argument("--query", "-q", help="Query the knowledge base")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive session"
    )
    parser.add_argument(
        "--collection", default="multimodal_rag", help="Collection name (default: multimodal_rag)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./data/vector_stores",
        help="Vector store directory (default: ./data/vector_stores)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index")
    parser.add_argument(
        "--vision-model", default="qwen3-vl:8b", help="Vision model (default: qwen3-vl:8b)"
    )
    parser.add_argument("--text-model", default="qwen3:8b", help="Text model (default: qwen3:8b)")
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding",
        help="Embedding model (default: qwen3-embedding)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Modal RAG System - Initialization")
    print("=" * 80)

    # Verify data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return

    # Check Ollama
    print("\n1. Checking Ollama...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("Error: Ollama not running! Start with: ollama serve")
        return

    # Verify models
    print("\n2. Verifying models...")
    required_models = [args.vision_model, args.text_model, args.embedding_model]
    for model_name in required_models:
        print(f"   - {model_name}")
        if not ollama_mgr.ensure_model_available(model_name):
            print(f"Error: Model '{model_name}' not available")
            print(f"Install with: ollama pull {model_name}")
            return

    # Initialize RAG system
    print("\n3. Initializing multi-modal RAG...")
    rag = MultiModalRAG(
        vision_model=args.vision_model,
        text_model=args.text_model,
        embedding_model=args.embedding_model,
    )

    # Index or load
    print("\n4. Preparing knowledge base...")
    try:
        vectorstore, docs = rag.index_directory(
            data_dir=data_dir,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            rebuild=args.rebuild or args.index,
        )
    except Exception as e:
        print(f"Error indexing directory: {e}")
        return

    # Execute based on mode
    try:
        if args.query:
            # Single query mode
            print("\n5. Processing query...")
            result = rag.answer_with_sources(vectorstore=vectorstore, query=args.query, k=5)

            print("\nAnswer:")
            print("=" * 80)
            print(result.answer)
            print("=" * 80)

            if result.sources:
                print(f"\nSources ({len(result.sources)}):")
                for i, doc in enumerate(result.sources, 1):
                    doc_type = doc.metadata.get("type", "unknown")
                    filename = doc.metadata.get("filename", "unknown")
                    print(f"  {i}. [{doc_type.upper()}] {filename}")

        elif args.interactive:
            # Interactive mode
            print("\n5. Starting interactive session...")
            rag.interactive_session(vectorstore)

        else:
            print("\nSuccess! Knowledge base ready.")
            print("Use --query 'your question' or --interactive to search.")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return

    print("\n" + "=" * 80)
    print("Execution completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)
