"""
Benchmark tests for LLM performance.

These tests measure performance of core LLM operations to track
regressions over time.
"""

import pytest
from langchain_ollama import ChatOllama


@pytest.fixture
def llm():
    """Create a test LLM instance."""
    return ChatOllama(model="qwen2.5:0.5b", temperature=0)


def test_simple_completion(benchmark, llm):
    """Benchmark simple completion performance."""
    def run_completion():
        return llm.invoke("What is 2+2?")

    result = benchmark(run_completion)
    assert result is not None


def test_batch_completion(benchmark, llm):
    """Benchmark batch completion performance."""
    prompts = ["What is 2+2?"] * 5

    def run_batch():
        return llm.batch(prompts)

    results = benchmark(run_batch)
    assert len(results) == 5


def test_streaming_completion(benchmark, llm):
    """Benchmark streaming completion performance."""
    def run_streaming():
        chunks = []
        for chunk in llm.stream("Count from 1 to 10"):
            chunks.append(chunk)
        return chunks

    results = benchmark(run_streaming)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_async_completion(benchmark, llm):
    """Benchmark async completion performance."""
    async def run_async():
        return await llm.ainvoke("What is 2+2?")

    result = await benchmark(run_async)
    assert result is not None
