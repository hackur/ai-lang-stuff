"""
Parallel Model Comparison using LangGraph Send API.

This example demonstrates parallel agent execution where multiple LLM models
process the same prompt simultaneously and their outputs are compared.

Architecture:
- State: Tracks prompt, model responses, and comparison results
- Nodes: Multiple model nodes run in parallel, then comparison node
- Edges: Parallel execution using Send API, converging to comparison
- Persistence: SQLite checkpointer tracks all parallel executions

Key LangGraph Concepts:
- Send API: Dispatch parallel tasks to multiple nodes
- Dynamic routing: Route based on available models
- Reducer functions: Aggregate parallel results

Prerequisites:
- Ollama server running: `ollama serve`
- Multiple models downloaded:
  - `ollama pull qwen3:8b`
  - `ollama pull gemma3:4b`
  - `ollama pull deepseek-r1:8b`

Expected output:
Side-by-side comparison of responses from different models.
"""

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Import utilities
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.state_manager import StateManager, create_thread_id
from utils.logging import setup_logging, get_logger
from utils.retry import retry_with_backoff


# ============================================================================
# State Definition
# ============================================================================


class ModelResponse(TypedDict):
    """Individual model response container."""

    model_name: str
    response: str
    tokens: int
    duration_ms: float


class ParallelComparisonState(TypedDict):
    """State for parallel model comparison.

    Attributes:
        prompt: The prompt to send to all models
        models: List of model names to compare
        responses: Aggregated responses from all models
        comparison: Final comparison analysis
        winner: Best performing model (optional)
        messages: Message history
        errors: List of any errors encountered
    """

    prompt: str
    models: List[str]
    responses: Annotated[List[ModelResponse], operator.add]
    comparison: str
    winner: str
    messages: Annotated[List, operator.add]
    errors: Annotated[List[str], operator.add]


# ============================================================================
# Model Execution Nodes
# ============================================================================


@retry_with_backoff(max_retries=3)
def invoke_model_with_retry(llm: ChatOllama, prompt: str) -> str:
    """Invoke model with retry logic.

    Args:
        llm: ChatOllama instance
        prompt: User prompt

    Returns:
        Model response content
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def model_executor_node(
    state: dict,
) -> dict:
    """Execute a single model and return its response.

    This node is called in parallel for each model. It receives the model name
    and prompt, executes the model, and returns the response.

    Args:
        state: State containing model_name and prompt

    Returns:
        Dictionary with model response data
    """
    logger = get_logger(__name__)
    model_name = state["model_name"]
    prompt = state["prompt"]

    logger.info(f"MODEL[{model_name}]: Starting execution")

    try:
        import time

        start_time = time.time()

        # Initialize model
        llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.7,
        )

        # Get response with retry
        response_content = invoke_model_with_retry(llm, prompt)

        duration_ms = (time.time() - start_time) * 1000

        # Estimate tokens (rough approximation)
        tokens = len(response_content.split())

        logger.info(f"MODEL[{model_name}]: Complete. {tokens} tokens in {duration_ms:.0f}ms")

        return {
            "responses": [
                {
                    "model_name": model_name,
                    "response": response_content,
                    "tokens": tokens,
                    "duration_ms": duration_ms,
                }
            ],
            "messages": [
                AIMessage(
                    content=f"Completed: {tokens} tokens in {duration_ms:.0f}ms",
                    name=model_name,
                )
            ],
        }

    except Exception as e:
        logger.error(f"MODEL[{model_name}]: Error - {str(e)}")
        return {
            "errors": [f"Model {model_name} error: {str(e)}"],
            "messages": [
                AIMessage(
                    content=f"Failed: {str(e)}",
                    name=model_name,
                )
            ],
        }


def comparison_node(state: ParallelComparisonState) -> ParallelComparisonState:
    """Compare all model responses and determine winner.

    This node receives all parallel responses and performs comparative analysis.

    Args:
        state: State with all model responses

    Returns:
        Updated state with comparison and winner
    """
    logger = get_logger(__name__)
    logger.info("COMPARATOR: Starting comparison analysis")

    try:
        responses = state.get("responses", [])

        if not responses:
            return {
                "comparison": "No responses to compare",
                "errors": ["No model responses received"],
            }

        # Initialize comparison LLM (use reliable model)
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.5,
        )

        # Build comparison prompt
        comparison_prompt = f"""Compare these responses from different LLM models to the prompt: "{state["prompt"]}"

"""

        for i, resp in enumerate(responses, 1):
            comparison_prompt += f"""
Model {i}: {resp["model_name"]}
Response: {resp["response"]}
Stats: {resp["tokens"]} tokens, {resp["duration_ms"]:.0f}ms
---
"""

        comparison_prompt += """
Provide a detailed comparison including:
1. Quality of responses (accuracy, completeness, clarity)
2. Response style and tone differences
3. Performance metrics
4. Which model performed best and why
5. Strengths and weaknesses of each

Declare a winner based on overall quality."""

        # Get comparison
        logger.info(f"COMPARATOR: Analyzing {len(responses)} responses")
        comparison_response = llm.invoke([HumanMessage(content=comparison_prompt)])
        comparison = comparison_response.content

        # Extract winner (simple heuristic - look for model names in comparison)
        winner = "tie"
        for resp in responses:
            model_name = resp["model_name"]
            if "winner" in comparison.lower() and model_name in comparison.lower():
                # Found winner mention
                winner = model_name
                break

        logger.info(f"COMPARATOR: Analysis complete. Winner: {winner}")

        return {
            "comparison": comparison,
            "winner": winner,
            "messages": [
                AIMessage(
                    content=f"Comparison complete. Winner: {winner}",
                    name="Comparator",
                )
            ],
        }

    except Exception as e:
        logger.error(f"COMPARATOR: Error - {str(e)}")
        return {
            "errors": [f"Comparison error: {str(e)}"],
            "comparison": f"Comparison failed: {str(e)}",
            "messages": [
                AIMessage(
                    content=f"Comparison failed: {str(e)}",
                    name="Comparator",
                )
            ],
        }


# ============================================================================
# Routing Logic
# ============================================================================


def route_to_models(
    state: ParallelComparisonState,
) -> List[Send]:
    """Route prompt to all models in parallel using Send API.

    This function creates parallel execution tasks for each model.

    Args:
        state: Current state with models list and prompt

    Returns:
        List of Send objects for parallel execution
    """
    logger = get_logger(__name__)
    models = state.get("models", [])

    logger.info(f"ROUTER: Dispatching to {len(models)} models in parallel")

    # Create Send task for each model
    sends = []
    for model_name in models:
        sends.append(
            Send(
                "model_executor",
                {"model_name": model_name, "prompt": state["prompt"]},
            )
        )

    return sends


def should_compare(state: ParallelComparisonState) -> Literal["compare", "end"]:
    """Determine if we have enough responses to compare.

    Args:
        state: Current state

    Returns:
        "compare" if we have responses, "end" otherwise
    """
    responses = state.get("responses", [])
    if len(responses) >= 2:
        return "compare"
    return "end"


# ============================================================================
# Graph Construction
# ============================================================================


def create_parallel_comparison_graph() -> StateGraph:
    """Create the parallel model comparison graph.

    Graph structure:
        START → [Model1, Model2, Model3] (parallel) → Comparator → END

    Uses Send API for parallel dispatch to multiple model nodes.

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    logger = get_logger(__name__)
    logger.info("Creating parallel comparison graph")

    # Create graph
    workflow = StateGraph(ParallelComparisonState)

    # Add nodes
    workflow.add_node("model_executor", model_executor_node)
    workflow.add_node("comparator", comparison_node)

    # Conditional entry: dispatch to all models in parallel
    workflow.add_conditional_edges(
        START,
        route_to_models,
    )

    # After all models complete, route to comparator or end
    workflow.add_conditional_edges(
        "model_executor",
        should_compare,
        {
            "compare": "comparator",
            "end": END,
        },
    )

    # Comparator always goes to end
    workflow.add_edge("comparator", END)

    logger.info("Parallel comparison graph created successfully")
    return workflow


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run the parallel comparison example."""
    setup_logging()
    logger = get_logger(__name__)

    print("=" * 70)
    print("Parallel Model Comparison Example")
    print("=" * 70)
    print()

    try:
        # Create graph
        workflow = create_parallel_comparison_graph()

        # Setup persistence
        checkpointer = StateManager.get_checkpointer("./checkpoints_parallel_comparison.db")
        app = workflow.compile(checkpointer=checkpointer)

        # Create thread ID
        thread_id = create_thread_id("parallel-comparison")
        config = {"configurable": {"thread_id": thread_id}}

        print(f"Thread ID: {thread_id}")
        print()

        # Define test prompt and models
        prompt = "Explain the concept of recursion in programming with a simple example."
        models = ["qwen3:8b", "gemma3:4b", "deepseek-r1:8b"]

        print(f"Prompt: {prompt}")
        print(f"Models to compare: {', '.join(models)}")
        print()
        print("-" * 70)

        # Initialize state
        initial_state = {
            "prompt": prompt,
            "models": models,
            "responses": [],
            "comparison": "",
            "winner": "",
            "messages": [],
            "errors": [],
        }

        # Execute pipeline
        print("\nExecuting Parallel Comparison...")
        print("-" * 70)

        step_count = 0
        for step_output in app.stream(initial_state, config):
            step_count += 1
            node_name = list(step_output.keys())[0]
            node_state = step_output[node_name]

            print(f"\n[Step {step_count}] {node_name.upper()}")

            if node_name == "model_executor":
                # Show model execution progress
                if node_state.get("responses"):
                    resp = node_state["responses"][0]
                    print(
                        f"  Model: {resp['model_name']} - "
                        f"{resp['tokens']} tokens in {resp['duration_ms']:.0f}ms"
                    )
            elif node_name == "comparator":
                print("  Performing comparison analysis...")

        # Get final state
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        final_state = app.get_state(config).values

        # Show all responses
        print("\nModel Responses:")
        print("-" * 70)
        for i, resp in enumerate(final_state.get("responses", []), 1):
            print(f"\n{i}. {resp['model_name']}")
            print(f"   Performance: {resp['tokens']} tokens, {resp['duration_ms']:.0f}ms")
            print(f"   Response: {resp['response'][:150]}...")

        # Show comparison
        print("\n" + "-" * 70)
        print("COMPARISON ANALYSIS:")
        print("-" * 70)
        print(final_state.get("comparison", "No comparison available"))

        print("\n" + "-" * 70)
        print(f"Winner: {final_state.get('winner', 'Not determined')}")
        print("-" * 70)

        # Show errors if any
        if final_state.get("errors"):
            print("\nErrors encountered:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        # Show checkpoint stats
        stats = StateManager.get_checkpoint_size("./checkpoints_parallel_comparison.db")
        print("\nCheckpoint stats:")
        print(f"  - Database size: {stats['file_size_mb']:.2f} MB")
        print(f"  - Total checkpoints: {stats['checkpoint_count']}")

        print("\n" + "=" * 70)
        print("Parallel comparison complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Parallel comparison failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Are models installed?")
        print("   - ollama pull qwen3:8b")
        print("   - ollama pull gemma3:4b")
        print("   - ollama pull deepseek-r1:8b")
        print("3. Check logs for detailed error information")


if __name__ == "__main__":
    main()
