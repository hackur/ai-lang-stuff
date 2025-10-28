"""
Sequential Research Pipeline using LangGraph.

This example demonstrates a sequential multi-agent workflow where agents pass
information in a linear chain: Researcher → Analyzer → Summarizer.

Architecture:
- State: Tracks question, research findings, analysis, and final summary
- Nodes: Each agent performs a specific task and updates state
- Edges: Sequential flow with error handling
- Persistence: SQLite checkpointer saves state at each step

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded: `ollama pull qwen3:8b`

Expected output:
Complete research pipeline execution showing each agent's contribution.
"""

import operator

# Import utilities - using absolute import path
import sys
from pathlib import Path
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.logging import get_logger, setup_logging
from utils.state_manager import StateManager, create_thread_id

# ============================================================================
# State Definition
# ============================================================================


class ResearchPipelineState(TypedDict):
    """State for sequential research pipeline.

    Attributes:
        question: Original research question
        research_findings: Raw research data collected
        analysis: Structured analysis of findings
        summary: Final concise summary
        messages: Message history for conversation tracking
        iteration: Current step number
        errors: List of any errors encountered
    """

    question: str
    research_findings: str
    analysis: str
    summary: str
    messages: Annotated[List, operator.add]
    iteration: int
    errors: Annotated[List[str], operator.add]


# ============================================================================
# Agent Node Definitions
# ============================================================================


def researcher_node(state: ResearchPipelineState) -> ResearchPipelineState:
    """Research agent: Gathers information on the topic.

    This node simulates a research agent that collects information about
    the given question. In production, this would query databases, APIs,
    or search engines.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with research_findings populated

    Raises:
        Exception: If LLM invocation fails
    """
    logger = get_logger(__name__)
    logger.info("RESEARCHER: Starting research phase")

    try:
        # Initialize LLM
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.7,
        )

        # Create research prompt
        system_prompt = SystemMessage(
            content="""You are a thorough researcher. Your job is to gather
            comprehensive information about the given topic. Provide factual,
            detailed findings with multiple perspectives. Focus on key facts,
            statistics, and important details."""
        )

        user_prompt = HumanMessage(
            content=f"""Research this question and provide detailed findings:

Question: {state["question"]}

Provide 3-4 key findings with supporting details."""
        )

        # Get research findings
        logger.info(f"RESEARCHER: Investigating: {state['question']}")
        response = llm.invoke([system_prompt, user_prompt])

        research_findings = response.content
        logger.info(f"RESEARCHER: Collected {len(research_findings)} chars of data")

        # Update state
        return {
            "research_findings": research_findings,
            "messages": [
                AIMessage(
                    content=f"Research complete. Found {len(research_findings)} chars",
                    name="Researcher",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"RESEARCHER: Error - {str(e)}")
        return {
            "errors": [f"Researcher error: {str(e)}"],
            "messages": [AIMessage(content=f"Research failed: {str(e)}", name="Researcher")],
        }


def analyzer_node(state: ResearchPipelineState) -> ResearchPipelineState:
    """Analyzer agent: Analyzes research findings.

    This node takes raw research findings and performs structured analysis,
    identifying patterns, themes, and key insights.

    Args:
        state: Current pipeline state with research_findings

    Returns:
        Updated state with analysis populated

    Raises:
        Exception: If LLM invocation fails
    """
    logger = get_logger(__name__)
    logger.info("ANALYZER: Starting analysis phase")

    try:
        # Initialize LLM
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.5,  # Lower temp for more analytical output
        )

        # Create analysis prompt
        system_prompt = SystemMessage(
            content="""You are an analytical expert. Your job is to analyze
            research findings and extract key insights, patterns, and themes.
            Organize information logically and highlight important connections."""
        )

        user_prompt = HumanMessage(
            content=f"""Analyze these research findings and provide structured insights:

Original Question: {state["question"]}

Research Findings:
{state["research_findings"]}

Provide your analysis with:
1. Key themes identified
2. Important patterns or trends
3. Critical insights
4. Potential implications"""
        )

        # Get analysis
        logger.info("ANALYZER: Processing research findings")
        response = llm.invoke([system_prompt, user_prompt])

        analysis = response.content
        logger.info(f"ANALYZER: Generated {len(analysis)} chars of analysis")

        # Update state
        return {
            "analysis": analysis,
            "messages": [
                AIMessage(
                    content=f"Analysis complete. Generated {len(analysis)} chars",
                    name="Analyzer",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"ANALYZER: Error - {str(e)}")
        return {
            "errors": [f"Analyzer error: {str(e)}"],
            "messages": [AIMessage(content=f"Analysis failed: {str(e)}", name="Analyzer")],
        }


def summarizer_node(state: ResearchPipelineState) -> ResearchPipelineState:
    """Summarizer agent: Creates concise final summary.

    This node synthesizes research and analysis into a clear, actionable
    summary that directly answers the original question.

    Args:
        state: Current pipeline state with analysis

    Returns:
        Updated state with summary populated

    Raises:
        Exception: If LLM invocation fails
    """
    logger = get_logger(__name__)
    logger.info("SUMMARIZER: Starting summary phase")

    try:
        # Initialize LLM
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.3,  # Very low temp for concise, focused output
        )

        # Create summary prompt
        system_prompt = SystemMessage(
            content="""You are a skilled communicator who creates clear,
            concise summaries. Your job is to synthesize research and analysis
            into an actionable summary that directly answers the question."""
        )

        user_prompt = HumanMessage(
            content=f"""Create a concise summary based on this research pipeline:

Original Question: {state["question"]}

Research Findings:
{state["research_findings"]}

Analysis:
{state["analysis"]}

Provide a clear, actionable summary (3-4 paragraphs) that:
1. Directly answers the original question
2. Highlights the most important findings
3. Provides actionable insights or recommendations"""
        )

        # Get summary
        logger.info("SUMMARIZER: Synthesizing final summary")
        response = llm.invoke([system_prompt, user_prompt])

        summary = response.content
        logger.info(f"SUMMARIZER: Generated {len(summary)} chars summary")

        # Update state
        return {
            "summary": summary,
            "messages": [
                AIMessage(
                    content=f"Summary complete. Final output: {len(summary)} chars",
                    name="Summarizer",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"SUMMARIZER: Error - {str(e)}")
        return {
            "errors": [f"Summarizer error: {str(e)}"],
            "messages": [AIMessage(content=f"Summary failed: {str(e)}", name="Summarizer")],
        }


# ============================================================================
# Graph Construction
# ============================================================================


def create_research_pipeline() -> StateGraph:
    """Create the sequential research pipeline graph.

    Graph structure:
        START → Researcher → Analyzer → Summarizer → END

    Each node updates state sequentially, with checkpoints saved at each step.

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    logger = get_logger(__name__)
    logger.info("Creating research pipeline graph")

    # Create graph
    workflow = StateGraph(ResearchPipelineState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("summarizer", summarizer_node)

    # Define sequential edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "summarizer")
    workflow.add_edge("summarizer", END)

    logger.info("Research pipeline graph created successfully")
    return workflow


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run the research pipeline example."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    print("=" * 70)
    print("Sequential Research Pipeline Example")
    print("=" * 70)
    print()

    try:
        # Create pipeline
        workflow = create_research_pipeline()

        # Setup persistence
        checkpointer = StateManager.get_checkpointer("./checkpoints_research.db")
        app = workflow.compile(checkpointer=checkpointer)

        # Create unique thread ID
        thread_id = create_thread_id("research-pipeline")
        config = {"configurable": {"thread_id": thread_id}}

        print(f"Thread ID: {thread_id}")
        print()

        # Define research question
        question = "What are the key benefits and challenges of using local LLMs versus cloud-based AI services?"

        print(f"Research Question: {question}")
        print()
        print("-" * 70)

        # Initialize state
        initial_state = {
            "question": question,
            "research_findings": "",
            "analysis": "",
            "summary": "",
            "messages": [],
            "iteration": 0,
            "errors": [],
        }

        # Execute pipeline
        print("\nExecuting Research Pipeline...")
        print("-" * 70)

        for step_output in app.stream(initial_state, config):
            # Extract node name and state
            node_name = list(step_output.keys())[0]
            node_state = step_output[node_name]

            print(f"\n[Step {node_state.get('iteration', '?')}] {node_name.upper()}")
            print("-" * 70)

            # Show progress
            if node_name == "researcher" and node_state.get("research_findings"):
                print("Research findings collected:")
                print(node_state["research_findings"][:200] + "...")
            elif node_name == "analyzer" and node_state.get("analysis"):
                print("Analysis generated:")
                print(node_state["analysis"][:200] + "...")
            elif node_name == "summarizer" and node_state.get("summary"):
                print("Summary created:")
                print(node_state["summary"][:200] + "...")

        # Get final state
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        final_state = app.get_state(config).values

        print(f"\nTotal iterations: {final_state.get('iteration', 0)}")
        print(f"Errors encountered: {len(final_state.get('errors', []))}")

        if final_state.get("errors"):
            print("\nErrors:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        print("\n" + "-" * 70)
        print("FINAL SUMMARY:")
        print("-" * 70)
        print(final_state.get("summary", "No summary generated"))
        print("-" * 70)

        # Show checkpoint stats
        stats = StateManager.get_checkpoint_size("./checkpoints_research.db")
        print("\nCheckpoint stats:")
        print(f"  - Database size: {stats['file_size_mb']:.2f} MB")
        print(f"  - Total checkpoints: {stats['checkpoint_count']}")
        print(f"  - Thread count: {stats['thread_count']}")

        print("\n" + "=" * 70)
        print("Pipeline execution complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Is the model installed? Try: ollama pull qwen3:8b")
        print("3. Check logs for detailed error information")


if __name__ == "__main__":
    main()
