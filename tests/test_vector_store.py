"""
Comprehensive test suite for utils/vector_store.py.

Tests VectorStoreManager class methods, convenience functions, error handling,
and integration with Chroma and FAISS backends using mocked Ollama embeddings.
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.vector_store import (
    VectorStoreManager,
    create_chroma_store,
    create_faiss_store,
    load_vector_store,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_persist_dir(tmp_path):
    """Create a temporary directory for vector store persistence.

    Args:
        tmp_path: pytest's built-in temporary directory fixture.

    Yields:
        Path: Temporary directory path.

    Cleanup:
        Automatically cleaned up by pytest after test completion.
    """
    persist_dir = tmp_path / "vector_stores"
    persist_dir.mkdir(exist_ok=True)
    yield str(persist_dir)
    # Cleanup handled automatically by tmp_path


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing.

    Returns:
        List of Document objects with varied content and metadata.
    """
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain.txt", "category": "framework"},
        ),
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            metadata={"source": "langgraph.txt", "category": "library"},
        ),
        Document(
            page_content="Ollama provides a simple API for running large language models locally.",
            metadata={"source": "ollama.txt", "category": "tools"},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings instance.

    Returns:
        Mock Embeddings object that returns deterministic embeddings.
    """
    mock = MagicMock(spec=Embeddings)
    # Return deterministic embeddings (3-dimensional for testing)
    mock.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    mock.embed_query.return_value = [0.5, 0.5, 0.5]
    return mock


@pytest.fixture
def mock_ollama_embeddings(mock_embeddings):
    """Mock OllamaEmbeddings to avoid actual model calls.

    Args:
        mock_embeddings: Mock embeddings fixture.

    Yields:
        Mock patch context for OllamaEmbeddings.
    """
    with patch("utils.vector_store.OllamaEmbeddings") as mock:
        mock.return_value = mock_embeddings
        yield mock


@pytest.fixture
def vector_store_manager(mock_ollama_embeddings):
    """Create a VectorStoreManager instance with mocked embeddings.

    Args:
        mock_ollama_embeddings: Mock for Ollama embeddings.

    Returns:
        VectorStoreManager instance ready for testing.
    """
    return VectorStoreManager(embedding_model="test-model", base_url="http://localhost:11434")


# ============================================================================
# VectorStoreManager Initialization Tests
# ============================================================================


def test_vector_store_manager_init():
    """Test VectorStoreManager initialization with default parameters."""
    manager = VectorStoreManager()
    assert manager.embedding_model == "qwen3-embedding"
    assert manager.base_url == "http://localhost:11434"


def test_vector_store_manager_init_custom():
    """Test VectorStoreManager initialization with custom parameters."""
    manager = VectorStoreManager(embedding_model="custom-model", base_url="http://localhost:9999")
    assert manager.embedding_model == "custom-model"
    assert manager.base_url == "http://localhost:9999"


# ============================================================================
# _get_embeddings() Tests
# ============================================================================


def test_get_embeddings_success(mock_ollama_embeddings):
    """Test successful embeddings creation."""
    manager = VectorStoreManager(embedding_model="test-model")
    embeddings = manager._get_embeddings()

    assert embeddings is not None
    mock_ollama_embeddings.assert_called_once_with(
        model="test-model", base_url="http://localhost:11434"
    )


def test_get_embeddings_connection_error():
    """Test embeddings creation failure when Ollama is unavailable."""
    with patch("utils.vector_store.OllamaEmbeddings") as mock:
        mock.side_effect = Exception("Connection refused")

        manager = VectorStoreManager(embedding_model="test-model")

        with pytest.raises(ConnectionError) as exc_info:
            manager._get_embeddings()

        assert "Could not connect to Ollama" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)


# ============================================================================
# create_from_documents() Tests - Chroma
# ============================================================================


def test_create_chroma_from_documents(vector_store_manager, sample_documents, temp_persist_dir):
    """Test creating a Chroma vector store from documents."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="test_collection",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Verify directory structure
    expected_path = Path(temp_persist_dir) / "chroma" / "test_collection"
    assert expected_path.exists()


def test_create_chroma_with_chunking(vector_store_manager, sample_documents, temp_persist_dir):
    """Test creating Chroma store with document chunking."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="chunked_collection",
        persist_dir=temp_persist_dir,
        store_type="chroma",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)


# ============================================================================
# create_from_documents() Tests - FAISS
# ============================================================================


def test_create_faiss_from_documents(vector_store_manager, sample_documents, temp_persist_dir):
    """Test creating a FAISS vector store from documents."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="test_faiss",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, FAISS)

    # Verify directory structure
    expected_path = Path(temp_persist_dir) / "faiss" / "test_faiss"
    assert expected_path.exists()


def test_create_faiss_with_chunking(vector_store_manager, sample_documents, temp_persist_dir):
    """Test creating FAISS store with document chunking."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="chunked_faiss",
        persist_dir=temp_persist_dir,
        store_type="faiss",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, FAISS)


# ============================================================================
# create_from_documents() Error Tests
# ============================================================================


def test_create_from_empty_documents(vector_store_manager, temp_persist_dir):
    """Test error handling when creating store from empty document list."""
    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.create_from_documents(
            documents=[], collection_name="test", persist_dir=temp_persist_dir
        )

    assert "Cannot create vector store from empty document list" in str(exc_info.value)


def test_create_from_invalid_store_type(vector_store_manager, sample_documents, temp_persist_dir):
    """Test error handling with invalid store type."""
    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.create_from_documents(
            documents=sample_documents,
            collection_name="test",
            persist_dir=temp_persist_dir,
            store_type="invalid",
        )

    assert "Invalid store_type" in str(exc_info.value)
    assert "Must be 'chroma' or 'faiss'" in str(exc_info.value)


# ============================================================================
# load_existing() Tests - Chroma
# ============================================================================


def test_load_existing_chroma(vector_store_manager, sample_documents, temp_persist_dir):
    """Test loading an existing Chroma collection."""
    # Create a collection first
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="load_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    # Load it
    loaded = vector_store_manager.load_existing(
        collection_name="load_test", persist_dir=temp_persist_dir, store_type="chroma"
    )

    assert loaded is not None
    assert isinstance(loaded, Chroma)


# ============================================================================
# load_existing() Tests - FAISS
# ============================================================================


def test_load_existing_faiss(vector_store_manager, sample_documents, temp_persist_dir):
    """Test loading an existing FAISS index."""
    # Create a FAISS store first
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="faiss_load_test",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    # Load it
    loaded = vector_store_manager.load_existing(
        collection_name="faiss_load_test", persist_dir=temp_persist_dir, store_type="faiss"
    )

    assert loaded is not None
    assert isinstance(loaded, FAISS)


# ============================================================================
# load_existing() Error Tests
# ============================================================================


def test_load_nonexistent_collection(vector_store_manager, temp_persist_dir):
    """Test error handling when loading non-existent collection."""
    with pytest.raises(FileNotFoundError) as exc_info:
        vector_store_manager.load_existing(
            collection_name="nonexistent", persist_dir=temp_persist_dir, store_type="chroma"
        )

    assert "not found" in str(exc_info.value)
    assert "nonexistent" in str(exc_info.value)


def test_load_invalid_store_type(vector_store_manager, temp_persist_dir):
    """Test error handling with invalid store type when loading."""
    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.load_existing(
            collection_name="test", persist_dir=temp_persist_dir, store_type="invalid"
        )

    assert "Invalid store_type" in str(exc_info.value)


# ============================================================================
# add_documents() Tests
# ============================================================================


def test_add_documents_to_chroma(vector_store_manager, sample_documents, temp_persist_dir):
    """Test adding documents to existing Chroma store."""
    # Create initial store
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents[:2],
        collection_name="add_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    # Add more documents
    new_docs = [Document(page_content="New content to add", metadata={"source": "new.txt"})]

    vector_store_manager.add_documents(vectorstore, new_docs)

    # Verify by loading and checking (basic validation)
    assert vectorstore is not None


def test_add_documents_to_faiss(vector_store_manager, sample_documents, temp_persist_dir):
    """Test adding documents to existing FAISS store."""
    # Create initial store
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents[:2],
        collection_name="faiss_add_test",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    # Add more documents
    new_docs = [Document(page_content="New FAISS content", metadata={"source": "new_faiss.txt"})]

    persist_path = Path(temp_persist_dir) / "faiss" / "faiss_add_test"
    vector_store_manager.add_documents(vectorstore, new_docs, persist_dir=str(persist_path))

    assert vectorstore is not None


def test_add_empty_documents_error(vector_store_manager, sample_documents, temp_persist_dir):
    """Test error handling when adding empty document list."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="empty_add_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.add_documents(vectorstore, [])

    assert "Cannot add empty document list" in str(exc_info.value)


def test_add_documents_faiss_without_persist_dir(
    vector_store_manager, sample_documents, temp_persist_dir
):
    """Test error when adding to FAISS without persist_dir."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="faiss_no_persist",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    new_docs = [Document(page_content="New content")]

    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.add_documents(vectorstore, new_docs)

    assert "persist_dir is required" in str(exc_info.value)


# ============================================================================
# delete_collection() Tests
# ============================================================================


def test_delete_chroma_collection(vector_store_manager, sample_documents, temp_persist_dir):
    """Test deleting a Chroma collection."""
    # Create collection
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="delete_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    collection_path = Path(temp_persist_dir) / "chroma" / "delete_test"
    assert collection_path.exists()

    # Delete it
    vector_store_manager.delete_collection(
        collection_name="delete_test", persist_dir=temp_persist_dir, store_type="chroma"
    )

    assert not collection_path.exists()


def test_delete_faiss_collection(vector_store_manager, sample_documents, temp_persist_dir):
    """Test deleting a FAISS collection."""
    # Create collection
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="faiss_delete_test",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    collection_path = Path(temp_persist_dir) / "faiss" / "faiss_delete_test"
    assert collection_path.exists()

    # Delete it
    vector_store_manager.delete_collection(
        collection_name="faiss_delete_test", persist_dir=temp_persist_dir, store_type="faiss"
    )

    assert not collection_path.exists()


def test_delete_nonexistent_collection(vector_store_manager, temp_persist_dir):
    """Test error handling when deleting non-existent collection."""
    with pytest.raises(FileNotFoundError) as exc_info:
        vector_store_manager.delete_collection(
            collection_name="nonexistent", persist_dir=temp_persist_dir, store_type="chroma"
        )

    assert "not found" in str(exc_info.value)


def test_delete_invalid_store_type(vector_store_manager, temp_persist_dir):
    """Test error handling with invalid store type when deleting."""
    with pytest.raises(ValueError) as exc_info:
        vector_store_manager.delete_collection(
            collection_name="test", persist_dir=temp_persist_dir, store_type="invalid"
        )

    assert "Invalid store_type" in str(exc_info.value)


# ============================================================================
# list_collections() Tests
# ============================================================================


def test_list_empty_collections(vector_store_manager, temp_persist_dir):
    """Test listing collections when none exist."""
    collections = vector_store_manager.list_collections(temp_persist_dir)

    assert isinstance(collections, dict)
    assert "chroma" in collections
    assert "faiss" in collections
    assert len(collections["chroma"]) == 0
    assert len(collections["faiss"]) == 0


def test_list_collections_mixed(vector_store_manager, sample_documents, temp_persist_dir):
    """Test listing collections with both Chroma and FAISS stores."""
    # Create multiple collections
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="chroma1",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="chroma2",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="faiss1",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    collections = vector_store_manager.list_collections(temp_persist_dir)

    assert len(collections["chroma"]) == 2
    assert "chroma1" in collections["chroma"]
    assert "chroma2" in collections["chroma"]
    assert len(collections["faiss"]) == 1
    assert "faiss1" in collections["faiss"]


def test_list_collections_filtered(vector_store_manager, sample_documents, temp_persist_dir):
    """Test listing collections filtered by store type."""
    # Create collections
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="chroma_only",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )
    vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="faiss_only",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    # Filter for chroma only
    collections = vector_store_manager.list_collections(temp_persist_dir, store_type="chroma")

    assert len(collections["chroma"]) == 1
    assert "chroma_only" in collections["chroma"]


def test_list_collections_nonexistent_dir(vector_store_manager):
    """Test listing collections in non-existent directory."""
    collections = vector_store_manager.list_collections("/nonexistent/path")

    assert collections == {"chroma": [], "faiss": []}


# ============================================================================
# similarity_search() Tests
# ============================================================================


def test_similarity_search_basic(vector_store_manager, sample_documents, temp_persist_dir):
    """Test basic similarity search."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="search_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    results = vector_store_manager.similarity_search(
        vectorstore=vectorstore, query="LangChain framework", k=2
    )

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


def test_similarity_search_with_filter(vector_store_manager, sample_documents, temp_persist_dir):
    """Test similarity search with metadata filtering."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="filter_search_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    results = vector_store_manager.similarity_search(
        vectorstore=vectorstore, query="framework", k=5, filter={"category": "framework"}
    )

    assert isinstance(results, list)


def test_similarity_search_with_score(vector_store_manager, sample_documents, temp_persist_dir):
    """Test similarity search with relevance scores."""
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents,
        collection_name="score_search_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    results = vector_store_manager.similarity_search_with_score(
        vectorstore=vectorstore, query="LangChain", k=2
    )

    assert isinstance(results, list)
    assert len(results) <= 2
    # Each result should be a tuple of (Document, score)
    if results:
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, (int, float))


# ============================================================================
# Convenience Functions Tests
# ============================================================================


def test_create_chroma_store_convenience(
    mock_ollama_embeddings, sample_documents, temp_persist_dir
):
    """Test create_chroma_store convenience function."""
    vectorstore = create_chroma_store(
        documents=sample_documents,
        collection_name="convenience_chroma",
        persist_dir=temp_persist_dir,
        embedding_model="test-model",
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)


def test_create_faiss_store_convenience(mock_ollama_embeddings, sample_documents, temp_persist_dir):
    """Test create_faiss_store convenience function."""
    vectorstore = create_faiss_store(
        documents=sample_documents,
        collection_name="convenience_faiss",
        persist_dir=temp_persist_dir,
        embedding_model="test-model",
    )

    assert vectorstore is not None
    assert isinstance(vectorstore, FAISS)


def test_load_vector_store_convenience_chroma(
    mock_ollama_embeddings, sample_documents, temp_persist_dir
):
    """Test load_vector_store convenience function for Chroma."""
    # Create store first
    create_chroma_store(
        documents=sample_documents,
        collection_name="load_convenience",
        persist_dir=temp_persist_dir,
        embedding_model="test-model",
    )

    # Load it
    loaded = load_vector_store(
        collection_name="load_convenience",
        persist_dir=temp_persist_dir,
        store_type="chroma",
        embedding_model="test-model",
    )

    assert loaded is not None
    assert isinstance(loaded, Chroma)


def test_load_vector_store_convenience_faiss(
    mock_ollama_embeddings, sample_documents, temp_persist_dir
):
    """Test load_vector_store convenience function for FAISS."""
    # Create store first
    create_faiss_store(
        documents=sample_documents,
        collection_name="load_faiss_convenience",
        persist_dir=temp_persist_dir,
        embedding_model="test-model",
    )

    # Load it
    loaded = load_vector_store(
        collection_name="load_faiss_convenience",
        persist_dir=temp_persist_dir,
        store_type="faiss",
        embedding_model="test-model",
    )

    assert loaded is not None
    assert isinstance(loaded, FAISS)


# ============================================================================
# Integration-like Tests (with mocking)
# ============================================================================


def test_full_workflow_chroma(vector_store_manager, sample_documents, temp_persist_dir):
    """Test complete workflow: create -> add -> search -> delete."""
    # Create
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents[:2],
        collection_name="workflow_test",
        persist_dir=temp_persist_dir,
        store_type="chroma",
    )

    # Add documents
    new_docs = [Document(page_content="Additional content", metadata={"source": "new.txt"})]
    vector_store_manager.add_documents(vectorstore, new_docs)

    # Search
    results = vector_store_manager.similarity_search(vectorstore, "content", k=2)
    assert isinstance(results, list)

    # List
    collections = vector_store_manager.list_collections(temp_persist_dir)
    assert "workflow_test" in collections["chroma"]

    # Delete
    vector_store_manager.delete_collection("workflow_test", temp_persist_dir, store_type="chroma")

    # Verify deletion
    collections = vector_store_manager.list_collections(temp_persist_dir)
    assert "workflow_test" not in collections["chroma"]


def test_full_workflow_faiss(vector_store_manager, sample_documents, temp_persist_dir):
    """Test complete workflow with FAISS: create -> add -> search -> delete."""
    # Create
    vectorstore = vector_store_manager.create_from_documents(
        documents=sample_documents[:2],
        collection_name="faiss_workflow",
        persist_dir=temp_persist_dir,
        store_type="faiss",
    )

    # Add documents
    new_docs = [Document(page_content="FAISS additional content")]
    persist_path = Path(temp_persist_dir) / "faiss" / "faiss_workflow"
    vector_store_manager.add_documents(vectorstore, new_docs, persist_dir=str(persist_path))

    # Search
    results = vector_store_manager.similarity_search(vectorstore, "content", k=2)
    assert isinstance(results, list)

    # List
    collections = vector_store_manager.list_collections(temp_persist_dir)
    assert "faiss_workflow" in collections["faiss"]

    # Delete
    vector_store_manager.delete_collection("faiss_workflow", temp_persist_dir, store_type="faiss")

    # Verify deletion
    collections = vector_store_manager.list_collections(temp_persist_dir)
    assert "faiss_workflow" not in collections["faiss"]
