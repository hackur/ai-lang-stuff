"""
RAG Pipeline Workflow for LangGraph Studio.

Pipeline: Document Ingestion → Retrieval → Response Generation
Demonstrates vector store integration and retrieval-augmented generation.
"""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# State Definition
# ============================================================================


class RAGState(TypedDict):
    """State for RAG pipeline.

    Attributes:
        query: User's question
        documents: Documents to ingest
        retrieved_docs: Retrieved relevant documents
        context: Formatted context from retrieval
        response: Generated answer
        sources: Source references
        messages: Message history
        iteration: Current step
    """

    query: str
    documents: List[str]
    retrieved_docs: List[Document]
    context: str
    response: str
    sources: List[str]
    messages: Annotated[List, operator.add]
    iteration: int


# ============================================================================
# RAG Nodes
# ============================================================================

# Global vector store (initialized lazily)
_vector_store = None


def get_vector_store():
    """Get or create vector store."""
    global _vector_store
    if _vector_store is None:
        embeddings = OllamaEmbeddings(model="qwen3:8b")
        _vector_store = Chroma(
            collection_name="rag_demo",
            embedding_function=embeddings,
            persist_directory="./data/chroma",
        )
    return _vector_store


def ingestion_node(state: RAGState) -> RAGState:
    """Ingest documents into vector store."""
    if not state.get("documents"):
        return {
            "messages": [AIMessage(content="No documents to ingest", name="Ingestion")],
            "iteration": state.get("iteration", 0) + 1,
        }

    vector_store = get_vector_store()

    # Create documents
    docs = [
        Document(page_content=doc, metadata={"source": f"doc_{i}"})
        for i, doc in enumerate(state["documents"])
    ]

    # Add to vector store
    vector_store.add_documents(docs)

    return {
        "messages": [
            AIMessage(content=f"Ingested {len(docs)} documents", name="Ingestion")
        ],
        "iteration": state.get("iteration", 0) + 1,
    }


def retrieval_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents."""
    vector_store = get_vector_store()

    # Retrieve top-k documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved = retriever.invoke(state["query"])

    # Format context
    context = "\n\n".join([doc.page_content for doc in retrieved])
    sources = [doc.metadata.get("source", "unknown") for doc in retrieved]

    return {
        "retrieved_docs": retrieved,
        "context": context,
        "sources": sources,
        "messages": [
            AIMessage(content=f"Retrieved {len(retrieved)} documents", name="Retrieval")
        ],
        "iteration": state.get("iteration", 0) + 1,
    }


def generation_node(state: RAGState) -> RAGState:
    """Generate response using retrieved context."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.7)

    system = SystemMessage(
        content="""You are a helpful assistant. Answer questions using ONLY the
        provided context. If the context doesn't contain enough information,
        say so clearly. Cite sources when possible."""
    )

    user = HumanMessage(
        content=f"""Question: {state['query']}

Context:
{state.get('context', 'No context available')}

Provide a clear, accurate answer based on the context."""
    )

    response = llm.invoke([system, user])

    return {
        "response": response.content,
        "messages": [AIMessage(content="Response generated", name="Generation")],
        "iteration": state.get("iteration", 0) + 1,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def create_graph() -> StateGraph:
    """Create RAG pipeline graph."""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("ingestion", ingestion_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)

    # Define flow
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "retrieval")
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", END)

    return workflow


# ============================================================================
# LangGraph Studio Entry Point
# ============================================================================

rag_pipeline = create_graph().compile(
    checkpointer=SqliteSaver.from_conn_string("./checkpoints/rag.db")
)

# For standalone testing
if __name__ == "__main__":
    # Sample documents about local LLMs
    sample_docs = [
        """Local LLMs run entirely on your device, providing privacy and offline
        capabilities. They don't send data to external servers, making them ideal
        for sensitive work. Popular frameworks include Ollama and LM Studio.""",
        """Ollama supports running models like Qwen3, Gemma3, and Llama on macOS,
        Linux, and Windows. Models can be quantized to run on consumer hardware.
        The smallest useful models are around 3-4B parameters.""",
        """Benefits of local LLMs: complete privacy, no API costs, offline operation,
        full control over the model. Challenges include: hardware requirements,
        slower inference, need to manage models locally, limited to smaller models.""",
    ]

    initial_state = {
        "query": "What are the benefits of using local LLMs?",
        "documents": sample_docs,
        "retrieved_docs": [],
        "context": "",
        "response": "",
        "sources": [],
        "messages": [],
        "iteration": 0,
    }

    config = {"configurable": {"thread_id": "test-001"}}

    print("RAG Pipeline Workflow")
    print("=" * 70)

    for step_output in rag_pipeline.stream(initial_state, config):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]
        print(f"\n[Step {node_state.get('iteration')}] {node_name.upper()}")

    final_state = rag_pipeline.get_state(config).values
    print("\n" + "=" * 70)
    print("RESPONSE:")
    print("=" * 70)
    print(final_state.get("response", "No response"))
    print("\nSources:", final_state.get("sources", []))
