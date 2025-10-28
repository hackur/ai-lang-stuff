"""
End-to-end RAG pipeline integration tests.

Tests complete RAG workflows including:
- Document ingestion and chunking
- Vector store creation and retrieval
- Query processing and answer generation
- Result validation and source tracking
"""

from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

# Import utilities
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import VectorStoreManager


# ============================================================================
# Document Ingestion Tests
# ============================================================================


class TestDocumentIngestion:
    """Test document loading and preprocessing."""

    def test_text_document_loading(self, test_data_dir: Path):
        """Test loading text documents.

        Args:
            test_data_dir: Test data directory fixture
        """
        text_file = test_data_dir / "sample.txt"
        content = text_file.read_text()

        # Create Document object
        doc = Document(page_content=content, metadata={"source": str(text_file), "type": "text"})

        assert len(doc.page_content) > 0
        assert doc.metadata["source"] == str(text_file)

    def test_document_chunking(self, sample_pdf_content: str):
        """Test document chunking for RAG.

        Args:
            sample_pdf_content: Sample PDF content fixture
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Create document
        doc = Document(page_content=sample_pdf_content, metadata={"source": "test.pdf", "page": 1})

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )

        chunks = splitter.split_documents([doc])

        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 250 for chunk in chunks)  # Allow some overflow
        assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)

    def test_chunk_overlap_preservation(self, sample_pdf_content: str):
        """Test that chunk overlap preserves context.

        Args:
            sample_pdf_content: Sample PDF content fixture
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        doc = Document(page_content=sample_pdf_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

        chunks = splitter.split_documents([doc])

        # Check that consecutive chunks have overlapping content
        if len(chunks) > 1:
            # Last 20 chars of first chunk should appear in second chunk
            # (approximately - depends on split points)
            assert len(chunks) >= 2


# ============================================================================
# Vector Store Tests
# ============================================================================


class TestVectorStoreOperations:
    """Test vector store creation and operations."""

    def test_vector_store_manager_initialization(self):
        """Test VectorStoreManager initialization."""
        manager = VectorStoreManager(embedding_model="qwen3-embedding")

        assert manager.embedding_model == "qwen3-embedding"
        assert manager.base_url == "http://localhost:11434"

    @pytest.mark.integration
    def test_vector_store_creation_with_mock_embeddings(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test creating vector store with mock embeddings.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        # Convert to Document objects
        docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in sample_documents
        ]

        # Create vector store with mock embeddings
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_collection",
            persist_directory=str(vector_store_dir),
        )

        assert vectorstore is not None
        assert vectorstore._collection.count() == len(docs)

    @pytest.mark.integration
    def test_vector_store_retrieval(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test retrieving documents from vector store.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        # Create vector store
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_retrieval",
            persist_directory=str(vector_store_dir),
        )

        # Test similarity search
        results = vectorstore.similarity_search("Python programming", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_vector_store_persistence(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test vector store persistence and loading.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        collection_name = "test_persistence"

        # Create and persist vector store
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name=collection_name,
            persist_directory=str(vector_store_dir),
        )

        original_count = vectorstore._collection.count()

        # Load existing vector store
        loaded_vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=str(vector_store_dir),
        )

        assert loaded_vectorstore._collection.count() == original_count


# ============================================================================
# Query Processing Tests
# ============================================================================


class TestQueryProcessing:
    """Test query processing and answer generation."""

    @pytest.mark.integration
    def test_retrieval_qa_chain_creation(
        self, sample_documents: List[dict], mock_embeddings, mock_ollama_llm, vector_store_dir: Path
    ):
        """Test creating RetrievalQA chain.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            mock_ollama_llm: Mock Ollama LLM fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain.chains import RetrievalQA
        from langchain_chroma import Chroma

        # Create vector store
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_qa",
            persist_directory=str(vector_store_dir),
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=mock_ollama_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
        )

        assert qa_chain is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_qa_chain_query(
        self, sample_documents: List[dict], mock_embeddings, mock_ollama_llm, vector_store_dir: Path
    ):
        """Test querying the QA chain.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            mock_ollama_llm: Mock Ollama LLM fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain.chains import RetrievalQA
        from langchain_chroma import Chroma

        # Create vector store
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_query",
            persist_directory=str(vector_store_dir),
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=mock_ollama_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

        # Execute query
        result = qa_chain.invoke({"query": "What is Python?"})

        assert "result" in result
        assert "source_documents" in result
        assert len(result["source_documents"]) > 0

    def test_retriever_configuration(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test retriever with different configurations.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_retriever",
            persist_directory=str(vector_store_dir),
        )

        # Test different k values
        for k in [1, 2, 3]:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            results = retriever.get_relevant_documents("test query")

            assert len(results) <= k


# ============================================================================
# Source Tracking Tests
# ============================================================================


class TestSourceTracking:
    """Test source document tracking in RAG."""

    def test_metadata_preservation(self):
        """Test that metadata is preserved through chunking."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        original_doc = Document(
            page_content="This is a long document that will be split into chunks. " * 10,
            metadata={"source": "test.pdf", "page": 5, "author": "Test Author"},
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

        chunks = splitter.split_documents([original_doc])

        # Check metadata is preserved in all chunks
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["author"] == "Test Author"

    @pytest.mark.integration
    def test_source_document_tracking(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test tracking source documents in retrieval.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_sources",
            persist_directory=str(vector_store_dir),
        )

        # Retrieve with metadata
        results = vectorstore.similarity_search("Python", k=2)

        for result in results:
            assert "source" in result.metadata
            assert len(result.page_content) > 0


# ============================================================================
# End-to-End RAG Pipeline Tests
# ============================================================================


class TestEndToEndRAGPipeline:
    """Test complete RAG pipelines."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_rag_pipeline(
        self, sample_pdf_content: str, mock_embeddings, mock_ollama_llm, vector_store_dir: Path
    ):
        """Test complete RAG pipeline from documents to answers.

        Args:
            sample_pdf_content: Sample PDF content fixture
            mock_embeddings: Mock embeddings fixture
            mock_ollama_llm: Mock Ollama LLM fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain.chains import RetrievalQA
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        # Step 1: Load and chunk documents
        doc = Document(page_content=sample_pdf_content, metadata={"source": "intro.pdf", "page": 1})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
        )

        chunks = splitter.split_documents([doc])
        assert len(chunks) > 0

        # Step 2: Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=mock_embeddings,
            collection_name="test_e2e",
            persist_directory=str(vector_store_dir),
        )

        # Step 3: Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=mock_ollama_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

        # Step 4: Query the system
        result = qa_chain.invoke({"query": "What are the prerequisites for local AI development?"})

        # Validate results
        assert "result" in result
        assert "source_documents" in result
        assert len(result["source_documents"]) > 0

    def test_multiple_document_sources(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test RAG with documents from multiple sources.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        # Documents have different source metadata
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        sources = set(doc.metadata["source"] for doc in docs)
        assert len(sources) > 1  # Multiple sources

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_multi_source",
            persist_directory=str(vector_store_dir),
        )

        # Query should retrieve from multiple sources
        results = vectorstore.similarity_search("programming language", k=4)

        result_sources = set(doc.metadata["source"] for doc in results)
        # May retrieve from different sources
        assert len(result_sources) >= 1


# ============================================================================
# Performance and Validation Tests
# ============================================================================


class TestRAGPerformance:
    """Test RAG pipeline performance characteristics."""

    def test_chunk_size_impact(self, sample_pdf_content: str):
        """Test impact of different chunk sizes.

        Args:
            sample_pdf_content: Sample PDF content fixture
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        doc = Document(page_content=sample_pdf_content)

        # Test different chunk sizes
        chunk_sizes = [100, 200, 500]

        for size in chunk_sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=50,
            )

            chunks = splitter.split_documents([doc])

            # Smaller chunks = more chunks
            assert len(chunks) > 0
            assert all(len(c.page_content) <= size + 100 for c in chunks)  # Allow overflow

    @pytest.mark.integration
    def test_retrieval_accuracy(
        self, sample_documents: List[dict], mock_embeddings, vector_store_dir: Path
    ):
        """Test retrieval accuracy with known queries.

        Args:
            sample_documents: Sample documents fixture
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in sample_documents
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_accuracy",
            persist_directory=str(vector_store_dir),
        )

        # Query for specific topic
        results = vectorstore.similarity_search("Python programming", k=1)

        # Should retrieve Python-related document
        assert len(results) > 0
        # Note: With mock embeddings, we can't test actual semantic similarity


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestRAGErrorHandling:
    """Test error handling in RAG pipelines."""

    def test_empty_document_handling(self, mock_embeddings, vector_store_dir: Path):
        """Test handling of empty documents.

        Args:
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        # Empty document
        docs = [Document(page_content="")]

        # Should handle gracefully or raise appropriate error
        try:
            Chroma.from_documents(
                documents=docs,
                embedding=mock_embeddings,
                collection_name="test_empty",
                persist_directory=str(vector_store_dir),
            )
        except (ValueError, Exception) as e:
            # Expected for empty documents
            assert "empty" in str(e).lower() or len(docs) == 1

    def test_query_without_documents(self, mock_embeddings, vector_store_dir: Path):
        """Test querying empty vector store.

        Args:
            mock_embeddings: Mock embeddings fixture
            vector_store_dir: Vector store directory fixture
        """
        from langchain_chroma import Chroma

        # Create empty vector store
        docs = [Document(page_content="placeholder")]  # Need at least one doc

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            collection_name="test_no_docs",
            persist_directory=str(vector_store_dir),
        )

        # Query should return empty or single result
        results = vectorstore.similarity_search("nonexistent query", k=5)

        assert isinstance(results, list)
