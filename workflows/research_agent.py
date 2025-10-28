"""
Research Agent Workflow for LangGraph Studio.

Sequential pipeline: Researcher → Analyzer → Summarizer

This workflow is optimized for visualization in LangGraph Studio.
It demonstrates state management, sequential agent execution, and checkpointing.
"""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# State Definition
# ============================================================================


class ResearchState(TypedDict):
    """State for research pipeline.

    Attributes:
        question: Original research question
        research_findings: Raw research data
        analysis: Structured analysis
        summary: Final summary
        messages: Conversation history
        iteration: Current step
    """

    question: str
    research_findings: str
    analysis: str
    summary: str
    messages: Annotated[List, operator.add]
    iteration: int


# ============================================================================
# Agent Nodes
# ============================================================================


def researcher_node(state: ResearchState) -> ResearchState:
    """Gather information on the topic."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.7)

    system = SystemMessage(
        content="You are a researcher. Gather comprehensive information with multiple perspectives."
    )
    user = HumanMessage(
        content=f"Research: {state['question']}\n\nProvide 3-4 key findings."
    )

    response = llm.invoke([system, user])

    return {
        "research_findings": response.content,
        "messages": [AIMessage(content="Research complete", name="Researcher")],
        "iteration": state.get("iteration", 0) + 1,
    }


def analyzer_node(state: ResearchState) -> ResearchState:
    """Analyze research findings."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.5)

    system = SystemMessage(
        content="You are an analyst. Extract insights, patterns, and themes from research."
    )
    user = HumanMessage(
        content=f"""Analyze findings for: {state['question']}

Findings: {state['research_findings']}

Provide: themes, patterns, insights, implications."""
    )

    response = llm.invoke([system, user])

    return {
        "analysis": response.content,
        "messages": [AIMessage(content="Analysis complete", name="Analyzer")],
        "iteration": state.get("iteration", 0) + 1,
    }


def summarizer_node(state: ResearchState) -> ResearchState:
    """Create final summary."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    system = SystemMessage(
        content="You are a communicator. Create clear, actionable summaries."
    )
    user = HumanMessage(
        content=f"""Summarize for: {state['question']}

Findings: {state['research_findings']}
Analysis: {state['analysis']}

Provide concise summary (3-4 paragraphs) with actionable insights."""
    )

    response = llm.invoke([system, user])

    return {
        "summary": response.content,
        "messages": [AIMessage(content="Summary complete", name="Summarizer")],
        "iteration": state.get("iteration", 0) + 1,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def create_graph() -> StateGraph:
    """Create research pipeline graph."""
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("summarizer", summarizer_node)

    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow


# ============================================================================
# LangGraph Studio Entry Point
# ============================================================================

# This is what LangGraph Studio looks for
research_agent = create_graph().compile(
    checkpointer=SqliteSaver.from_conn_string("./checkpoints/research.db")
)

# For standalone testing
if __name__ == "__main__":
    import os

    # Test the graph
    initial_state = {
        "question": "What are the benefits of local LLMs versus cloud AI?",
        "research_findings": "",
        "analysis": "",
        "summary": "",
        "messages": [],
        "iteration": 0,
    }

    config = {"configurable": {"thread_id": "test-001"}}

    print("Research Agent Workflow")
    print("=" * 70)

    for step_output in research_agent.stream(initial_state, config):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]
        print(f"\n[Step {node_state.get('iteration')}] {node_name.upper()}")

    final_state = research_agent.get_state(config).values
    print("\n" + "=" * 70)
    print("FINAL SUMMARY:")
    print("=" * 70)
    print(final_state.get("summary", "No summary"))
