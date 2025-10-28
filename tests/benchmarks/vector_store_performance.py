"""
Vector store performance benchmarks.

Compares performance of different vector store implementations:
- Chroma (default, persistent)
- FAISS (in-memory, fast)

Metrics:
- Indexing speed (documents/second)
- Query latency (ms per query)
- Memory usage
- Scaling characteristics (100, 1K, 10K documents)
"""

import time
import psutil
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import csv
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


@dataclass
class VectorStoreBenchmarkResult:
    """Results from a vector store benchmark run."""

    store_type: str
    operation: str  # 'index' or 'query'
    num_documents: int
    latency_ms: float
    throughput: float  # docs/sec for indexing, queries/sec for searching
    memory_mb: float
    error: Optional[str] = None


class VectorStoreBenchmark:
    """Benchmark suite for vector store performance testing."""

    # Document sizes to test
    DOC_COUNTS = [100, 500, 1000]

    # Store implementations
    STORES = ["chroma", "faiss"]

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[VectorStoreBenchmarkResult] = []
        self.temp_dirs: List[Path] = []

        # Initialize embeddings (reuse across tests)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _generate_documents(self, count: int) -> List[Document]:
        """
        Generate test documents.

        Args:
            count: Number of documents to generate

        Returns:
            List of Document objects
        """
        documents = []
        for i in range(count):
            content = f"""
            Document {i}: This is a test document for benchmarking vector store performance.
            It contains information about topic {i % 10}, category {i % 5}, and priority {i % 3}.
            The document includes various keywords: performance, benchmark, testing, vector, embeddings.
            Additional content to make document size realistic: Lorem ipsum dolor sit amet,
            consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            """
            documents.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        "doc_id": i,
                        "topic": i % 10,
                        "category": i % 5,
                        "priority": i % 3,
                    }
                )
            )
        return documents

    def _create_temp_dir(self) -> Path:
        """Create temporary directory for vector store."""
        temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def benchmark_indexing(
        self,
        store_type: str,
        num_documents: int,
        num_runs: int = 3
    ) -> List[VectorStoreBenchmarkResult]:
        """
        Benchmark document indexing performance.

        Args:
            store_type: Type of vector store ('chroma' or 'faiss')
            num_documents: Number of documents to index
            num_runs: Number of runs to average

        Returns:
            List of benchmark results
        """
        results = []
        documents = self._generate_documents(num_documents)

        for run in range(num_runs):
            try:
                mem_before = self._get_memory_usage()
                start_time = time.perf_counter()

                if store_type == "chroma":
                    persist_dir = self._create_temp_dir()
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=str(persist_dir)
                    )
                elif store_type == "faiss":
                    vectorstore = FAISS.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                else:
                    raise ValueError(f"Unknown store type: {store_type}")

                end_time = time.perf_counter()
                mem_after = self._get_memory_usage()

                latency_ms = (end_time - start_time) * 1000
                throughput = num_documents / (latency_ms / 1000) if latency_ms > 0 else 0
                memory_mb = mem_after - mem_before

                result = VectorStoreBenchmarkResult(
                    store_type=store_type,
                    operation="index",
                    num_documents=num_documents,
                    latency_ms=latency_ms,
                    throughput=throughput,
                    memory_mb=memory_mb,
                )

                results.append(result)
                self.results.append(result)

                # Cleanup
                if store_type == "chroma":
                    vectorstore.delete_collection()

                time.sleep(1)

            except Exception as e:
                result = VectorStoreBenchmarkResult(
                    store_type=store_type,
                    operation="index",
                    num_documents=num_documents,
                    latency_ms=0,
                    throughput=0,
                    memory_mb=0,
                    error=str(e),
                )
                results.append(result)
                self.results.append(result)

        return results

    def benchmark_querying(
        self,
        store_type: str,
        num_documents: int,
        num_queries: int = 10
    ) -> List[VectorStoreBenchmarkResult]:
        """
        Benchmark query performance.

        Args:
            store_type: Type of vector store
            num_documents: Number of documents in the index
            num_queries: Number of queries to run

        Returns:
            List of benchmark results
        """
        results = []
        documents = self._generate_documents(num_documents)

        # Create and populate vector store
        try:
            if store_type == "chroma":
                persist_dir = self._create_temp_dir()
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(persist_dir)
                )
            elif store_type == "faiss":
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                raise ValueError(f"Unknown store type: {store_type}")

            # Test queries
            test_queries = [
                "performance testing",
                "benchmark results",
                "document category",
                "vector embeddings",
                "priority information",
            ]

            mem_before = self._get_memory_usage()
            start_time = time.perf_counter()

            for i in range(num_queries):
                query = test_queries[i % len(test_queries)]
                _ = vectorstore.similarity_search(query, k=5)

            end_time = time.perf_counter()
            mem_after = self._get_memory_usage()

            latency_ms = (end_time - start_time) * 1000
            throughput = num_queries / (latency_ms / 1000) if latency_ms > 0 else 0
            memory_mb = mem_after - mem_before

            result = VectorStoreBenchmarkResult(
                store_type=store_type,
                operation="query",
                num_documents=num_documents,
                latency_ms=latency_ms,
                throughput=throughput,
                memory_mb=memory_mb,
            )

            results.append(result)
            self.results.append(result)

            # Cleanup
            if store_type == "chroma":
                vectorstore.delete_collection()

        except Exception as e:
            result = VectorStoreBenchmarkResult(
                store_type=store_type,
                operation="query",
                num_documents=num_documents,
                latency_ms=0,
                throughput=0,
                memory_mb=0,
                error=str(e),
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_all_stores(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks on all vector stores.

        Returns:
            Summary statistics dictionary
        """
        print("Vector Store Performance Benchmark")
        print("=" * 70)

        for store_type in self.STORES:
            print(f"\nBenchmarking {store_type.upper()}:")

            for doc_count in self.DOC_COUNTS:
                # Indexing benchmark
                print(f"  - Indexing {doc_count} documents...", end=" ")
                index_results = self.benchmark_indexing(store_type, doc_count, num_runs=3)
                valid_results = [r for r in index_results if r.error is None]
                if valid_results:
                    avg_throughput = statistics.mean(r.throughput for r in valid_results)
                    print(f"{avg_throughput:.1f} docs/sec")
                else:
                    print("FAILED")

                # Query benchmark
                print(f"  - Querying {doc_count} documents...", end=" ")
                query_results = self.benchmark_querying(store_type, doc_count, num_queries=10)
                if query_results and query_results[0].error is None:
                    print(f"{query_results[0].latency_ms:.1f}ms total")
                else:
                    print("FAILED")

        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            "total_runs": len(self.results),
            "by_store": {},
            "by_operation": {},
        }

        # Group by store type
        for store in self.STORES:
            store_results = [r for r in self.results if r.store_type == store and r.error is None]
            if store_results:
                summary["by_store"][store] = {
                    "avg_latency_ms": statistics.mean(r.latency_ms for r in store_results),
                    "avg_throughput": statistics.mean(r.throughput for r in store_results),
                    "avg_memory_mb": statistics.mean(r.memory_mb for r in store_results),
                    "runs": len(store_results),
                }

        # Group by operation
        for operation in ["index", "query"]:
            op_results = [r for r in self.results if r.operation == operation and r.error is None]
            if op_results:
                summary["by_operation"][operation] = {
                    "avg_latency_ms": statistics.mean(r.latency_ms for r in op_results),
                    "avg_throughput": statistics.mean(r.throughput for r in op_results),
                    "runs": len(op_results),
                }

        return summary

    def save_results(self, format: str = "json"):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = self.output_dir / f"vector_store_benchmark_{timestamp}.json"
            data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_runs": len(self.results),
                },
                "results": [asdict(r) for r in self.results],
                "summary": self.generate_summary(),
            }
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to {output_file}")

        elif format == "csv":
            output_file = self.output_dir / f"vector_store_benchmark_{timestamp}.csv"
            with open(output_file, "w", newline="") as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(asdict(result))
            print(f"\nResults saved to {output_file}")

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


# Pytest integration
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("store_type", VectorStoreBenchmark.STORES)
def test_vector_store_indexing(store_type: str):
    """Test vector store indexing performance."""
    benchmark = VectorStoreBenchmark()
    try:
        results = benchmark.benchmark_indexing(store_type, num_documents=100, num_runs=1)

        assert len(results) == 1
        result = results[0]

        assert result.error is None, f"Indexing failed: {result.error}"
        assert result.throughput > 0, "No throughput measured"
    finally:
        benchmark.cleanup()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("store_type", VectorStoreBenchmark.STORES)
def test_vector_store_querying(store_type: str):
    """Test vector store query performance."""
    benchmark = VectorStoreBenchmark()
    try:
        results = benchmark.benchmark_querying(store_type, num_documents=100, num_queries=5)

        assert len(results) == 1
        result = results[0]

        assert result.error is None, f"Querying failed: {result.error}"
        assert result.latency_ms > 0, "No latency measured"
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    """Run benchmarks standalone."""
    benchmark = VectorStoreBenchmark()

    try:
        # Run full benchmark suite
        summary = benchmark.benchmark_all_stores()

        # Save results
        benchmark.save_results(format="json")
        benchmark.save_results(format="csv")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(json.dumps(summary, indent=2))

    finally:
        # Cleanup temp directories
        benchmark.cleanup()
