"""Vector store management utilities for local-first RAG systems.

This module provides a comprehensive VectorStoreManager class for creating,
loading, and managing vector stores using Chroma and FAISS backends with
Ollama embeddings.

Example:
    ```python
    from utils.vector_store import VectorStoreManager
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    # Initialize manager
    manager = VectorStoreManager()

    # Load and chunk documents
    loader = TextLoader("document.txt")
    documents = loader.load()

    # Create vector store
    vectorstore = manager.create_from_documents(
        documents=documents,
        collection_name="my_docs",
        persist_dir="./data/vector_stores"
    )

    # Search
    results = vectorstore.similarity_search("query", k=3)
    ```
"""

import logging
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Union

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# Type aliases
VectorStoreType = Literal["chroma", "faiss"]


class VectorStoreManager:
    """Manages vector store operations with Chroma and FAISS backends.

    This class provides a unified interface for creating, loading, and managing
    vector stores with support for both Chroma (recommended for persistence) and
    FAISS (recommended for performance).

    Attributes:
        embedding_model: Name of the Ollama embedding model to use.
        base_url: Ollama API endpoint URL.
    """

    def __init__(
        self,
        embedding_model: str = "qwen3-embedding",
        base_url: str = "http://localhost:11434"
    ):
        """Initialize the vector store manager.

        Args:
            embedding_model: Ollama embedding model name. Default: qwen3-embedding
            base_url: Ollama API endpoint. Default: http://localhost:11434
        """
        self.embedding_model = embedding_model
        self.base_url = base_url
        logger.info(
            f"Initialized VectorStoreManager with model={embedding_model}, "
            f"base_url={base_url}"
        )

    def _get_embeddings(self) -> Embeddings:
        """Create an embeddings instance with error handling.

        Returns:
            OllamaEmbeddings instance configured with the specified model.

        Raises:
            ConnectionError: If Ollama is not running or unreachable.
            ValueError: If the embedding model is not available.
        """
        try:
            embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url
            )
            logger.debug(f"Created embeddings instance for model: {self.embedding_model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running and model '{self.embedding_model}' is available. "
                f"Pull it with: ollama pull {self.embedding_model}"
            ) from e

    def create_from_documents(
        self,
        documents: List[Document],
        collection_name: str,
        persist_dir: str,
        store_type: VectorStoreType = "chroma",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> VectorStore:
        """Create a new vector store from documents.

        Args:
            documents: List of LangChain Document objects to index.
            collection_name: Name for the collection/index.
            persist_dir: Directory where the vector store will be persisted.
            store_type: Vector store backend ("chroma" or "faiss"). Default: "chroma"
            chunk_size: Optional custom chunk size for text splitting.
            chunk_overlap: Optional overlap between chunks.

        Returns:
            VectorStore instance (Chroma or FAISS).

        Raises:
            ValueError: If documents list is empty or store_type is invalid.
            ConnectionError: If Ollama is not available.

        Example:
            ```python
            manager = VectorStoreManager()
            docs = [Document(page_content="text", metadata={"source": "file.txt"})]
            vs = manager.create_from_documents(
                documents=docs,
                collection_name="my_collection",
                persist_dir="./data/vectors"
            )
            ```
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")

        if store_type not in ["chroma", "faiss"]:
            raise ValueError(f"Invalid store_type: {store_type}. Must be 'chroma' or 'faiss'")

        logger.info(
            f"Creating {store_type} vector store '{collection_name}' "
            f"with {len(documents)} documents"
        )

        # Get embeddings
        embeddings = self._get_embeddings()

        # Optional: Apply text splitting if chunk parameters provided
        if chunk_size is not None:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap or chunk_size // 10
            )
            documents = splitter.split_documents(documents)
            logger.info(f"Split documents into {len(documents)} chunks")

        # Create vector store
        persist_path = Path(persist_dir) / store_type / collection_name
        persist_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if store_type == "chroma":
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=str(persist_path)
                )
                logger.info(f"Created Chroma collection at {persist_path}")

            else:  # FAISS
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=embeddings
                )
                vectorstore.save_local(str(persist_path))
                logger.info(f"Created FAISS index at {persist_path}")

            logger.info(
                f"Successfully created {store_type} vector store with "
                f"{len(documents)} documents"
            )
            return vectorstore

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def load_existing(
        self,
        collection_name: str,
        persist_dir: str,
        store_type: VectorStoreType = "chroma"
    ) -> VectorStore:
        """Load an existing vector store from disk.

        Args:
            collection_name: Name of the collection to load.
            persist_dir: Directory where vector stores are persisted.
            store_type: Vector store backend ("chroma" or "faiss"). Default: "chroma"

        Returns:
            VectorStore instance loaded from disk.

        Raises:
            FileNotFoundError: If the collection doesn't exist.
            ValueError: If store_type is invalid.

        Example:
            ```python
            manager = VectorStoreManager()
            vs = manager.load_existing(
                collection_name="my_collection",
                persist_dir="./data/vectors"
            )
            results = vs.similarity_search("query")
            ```
        """
        if store_type not in ["chroma", "faiss"]:
            raise ValueError(f"Invalid store_type: {store_type}. Must be 'chroma' or 'faiss'")

        persist_path = Path(persist_dir) / store_type / collection_name

        if not persist_path.exists():
            raise FileNotFoundError(
                f"Collection '{collection_name}' not found at {persist_path}. "
                f"Available collections: {self.list_collections(persist_dir, store_type)}"
            )

        logger.info(f"Loading {store_type} collection '{collection_name}' from {persist_path}")

        embeddings = self._get_embeddings()

        try:
            if store_type == "chroma":
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(persist_path)
                )
                logger.info(f"Loaded Chroma collection: {collection_name}")

            else:  # FAISS
                vectorstore = FAISS.load_local(
                    str(persist_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded FAISS index: {collection_name}")

            return vectorstore

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def add_documents(
        self,
        vectorstore: VectorStore,
        documents: List[Document],
        persist_dir: Optional[str] = None
    ) -> None:
        """Add new documents to an existing vector store.

        Args:
            vectorstore: Existing VectorStore instance to update.
            documents: List of Document objects to add.
            persist_dir: Optional directory to save FAISS index (required for FAISS).

        Raises:
            ValueError: If documents list is empty or persist_dir missing for FAISS.

        Example:
            ```python
            manager = VectorStoreManager()
            vs = manager.load_existing("my_collection", "./data/vectors")

            new_docs = [Document(page_content="new content")]
            manager.add_documents(vs, new_docs)
            ```
        """
        if not documents:
            raise ValueError("Cannot add empty document list")

        logger.info(f"Adding {len(documents)} documents to vector store")

        try:
            vectorstore.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents")

            # Persist FAISS if needed
            if isinstance(vectorstore, FAISS):
                if not persist_dir:
                    raise ValueError(
                        "persist_dir is required when adding documents to FAISS store"
                    )
                vectorstore.save_local(persist_dir)
                logger.info(f"Saved updated FAISS index to {persist_dir}")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete_collection(
        self,
        collection_name: str,
        persist_dir: str,
        store_type: VectorStoreType = "chroma"
    ) -> None:
        """Delete a vector store collection from disk.

        Args:
            collection_name: Name of the collection to delete.
            persist_dir: Directory where vector stores are persisted.
            store_type: Vector store backend ("chroma" or "faiss"). Default: "chroma"

        Raises:
            FileNotFoundError: If the collection doesn't exist.

        Example:
            ```python
            manager = VectorStoreManager()
            manager.delete_collection("old_collection", "./data/vectors")
            ```
        """
        if store_type not in ["chroma", "faiss"]:
            raise ValueError(f"Invalid store_type: {store_type}. Must be 'chroma' or 'faiss'")

        persist_path = Path(persist_dir) / store_type / collection_name

        if not persist_path.exists():
            raise FileNotFoundError(
                f"Collection '{collection_name}' not found at {persist_path}"
            )

        logger.info(f"Deleting {store_type} collection '{collection_name}' at {persist_path}")

        try:
            shutil.rmtree(persist_path)
            logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def list_collections(
        self,
        persist_dir: str,
        store_type: Optional[VectorStoreType] = None
    ) -> dict[str, List[str]]:
        """List all available vector store collections.

        Args:
            persist_dir: Directory where vector stores are persisted.
            store_type: Optional filter for specific store type.

        Returns:
            Dictionary mapping store types to lists of collection names.
            Example: {"chroma": ["collection1", "collection2"], "faiss": ["collection3"]}

        Example:
            ```python
            manager = VectorStoreManager()
            collections = manager.list_collections("./data/vectors")
            print(f"Chroma collections: {collections['chroma']}")
            print(f"FAISS collections: {collections['faiss']}")
            ```
        """
        base_path = Path(persist_dir)
        collections = {"chroma": [], "faiss": []}

        if not base_path.exists():
            logger.warning(f"Directory {persist_dir} does not exist")
            return collections

        # Determine which store types to check
        types_to_check = [store_type] if store_type else ["chroma", "faiss"]

        for stype in types_to_check:
            store_path = base_path / stype
            if store_path.exists() and store_path.is_dir():
                collections[stype] = [
                    d.name for d in store_path.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
                logger.debug(f"Found {len(collections[stype])} {stype} collections")

        return collections

    def similarity_search(
        self,
        vectorstore: VectorStore,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform similarity search with optional metadata filtering.

        Args:
            vectorstore: VectorStore instance to search.
            query: Query string to search for.
            k: Number of results to return. Default: 4
            filter: Optional metadata filter dictionary.

        Returns:
            List of Document objects ranked by similarity.

        Example:
            ```python
            manager = VectorStoreManager()
            vs = manager.load_existing("my_collection", "./data/vectors")

            # Basic search
            results = manager.similarity_search(vs, "AI agents", k=3)

            # Filtered search
            results = manager.similarity_search(
                vs,
                "AI agents",
                k=3,
                filter={"source": "research_papers"}
            )
            ```
        """
        logger.info(f"Performing similarity search: query='{query}', k={k}, filter={filter}")

        try:
            if filter:
                results = vectorstore.similarity_search(query, k=k, filter=filter)
            else:
                results = vectorstore.similarity_search(query, k=k)

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def similarity_search_with_score(
        self,
        vectorstore: VectorStore,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores.

        Args:
            vectorstore: VectorStore instance to search.
            query: Query string to search for.
            k: Number of results to return. Default: 4
            filter: Optional metadata filter dictionary.

        Returns:
            List of (Document, score) tuples ranked by similarity.
            Lower scores indicate higher similarity.

        Example:
            ```python
            manager = VectorStoreManager()
            vs = manager.load_existing("my_collection", "./data/vectors")

            results = manager.similarity_search_with_score(vs, "AI agents", k=3)
            for doc, score in results:
                print(f"Score: {score:.4f} - {doc.page_content[:100]}")
            ```
        """
        logger.info(
            f"Performing similarity search with scores: query='{query}', k={k}, filter={filter}"
        )

        try:
            if filter:
                results = vectorstore.similarity_search_with_score(query, k=k, filter=filter)
            else:
                results = vectorstore.similarity_search_with_score(query, k=k)

            logger.info(f"Found {len(results)} results with scores")
            return results

        except Exception as e:
            logger.error(f"Similarity search with scores failed: {e}")
            raise


# Convenience functions for common operations

def create_chroma_store(
    documents: List[Document],
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    embedding_model: str = "qwen3-embedding"
) -> Chroma:
    """Convenience function to create a Chroma vector store.

    Args:
        documents: Documents to index.
        collection_name: Collection name.
        persist_dir: Persistence directory.
        embedding_model: Ollama embedding model.

    Returns:
        Chroma vector store instance.
    """
    manager = VectorStoreManager(embedding_model=embedding_model)
    return manager.create_from_documents(
        documents=documents,
        collection_name=collection_name,
        persist_dir=persist_dir,
        store_type="chroma"
    )


def create_faiss_store(
    documents: List[Document],
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    embedding_model: str = "qwen3-embedding"
) -> FAISS:
    """Convenience function to create a FAISS vector store.

    Args:
        documents: Documents to index.
        collection_name: Collection name.
        persist_dir: Persistence directory.
        embedding_model: Ollama embedding model.

    Returns:
        FAISS vector store instance.
    """
    manager = VectorStoreManager(embedding_model=embedding_model)
    return manager.create_from_documents(
        documents=documents,
        collection_name=collection_name,
        persist_dir=persist_dir,
        store_type="faiss"
    )


def load_vector_store(
    collection_name: str,
    persist_dir: str = "./data/vector_stores",
    store_type: VectorStoreType = "chroma",
    embedding_model: str = "qwen3-embedding"
) -> VectorStore:
    """Convenience function to load an existing vector store.

    Args:
        collection_name: Collection name.
        persist_dir: Persistence directory.
        store_type: Store backend ("chroma" or "faiss").
        embedding_model: Ollama embedding model.

    Returns:
        VectorStore instance.
    """
    manager = VectorStoreManager(embedding_model=embedding_model)
    return manager.load_existing(
        collection_name=collection_name,
        persist_dir=persist_dir,
        store_type=store_type
    )
