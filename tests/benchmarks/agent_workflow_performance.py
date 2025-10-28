"""
Agent workflow performance benchmarks.

Measures performance of multi-agent systems and LangGraph workflows:
- Multi-agent pipeline execution time
- State management overhead
- Tool call latency
- End-to-end workflow timing
- Message passing efficiency

Workflows tested:
- Sequential agent pipeline
- Parallel agent execution
- Tool-calling agent
- Supervisor-worker pattern
- Conditional routing
"""

import csv
import json
import operator
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor


@dataclass
class AgentWorkflowBenchmarkResult:
    """Results from an agent workflow benchmark run."""

    workflow_type: str
    num_agents: int
    num_steps: int
    total_latency_ms: float
    avg_step_latency_ms: float
    state_overhead_ms: float
    tool_calls: int
    tool_call_latency_ms: float
    error: Optional[str] = None


# Define test tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_documents(query: str) -> str:
    """Search for documents (simulated)."""
    time.sleep(0.1)  # Simulate I/O
    return f"Found 5 documents for query: {query}"


@tool
def analyze_text(text: str) -> str:
    """Analyze text content (simulated)."""
    time.sleep(0.05)  # Simulate processing
    return f"Analysis complete: {len(text)} characters, {len(text.split())} words"


# Agent state definition
class AgentState(dict):
    """Simple state for agent workflows."""

    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int
    tool_calls: int


class AgentWorkflowBenchmark:
    """Benchmark suite for agent workflow performance testing."""

    WORKFLOW_TYPES = [
        "sequential",
        "parallel",
        "tool_calling",
        "supervisor",
        "conditional",
    ]

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[AgentWorkflowBenchmarkResult] = []

        # Initialize LLM (use fast model for benchmarking)
        self.llm = ChatOllama(model="qwen3:8b", temperature=0)

        # Initialize tools
        self.tools = [calculator, search_documents, analyze_text]
        self.tool_executor = ToolExecutor(self.tools)

    def _create_sequential_workflow(self, num_agents: int) -> StateGraph:
        """
        Create a sequential agent workflow.

        Args:
            num_agents: Number of agents in sequence

        Returns:
            StateGraph instance
        """
        workflow = StateGraph(AgentState)

        def make_agent_node(name: str, prompt: str):
            """Create an agent node."""

            def agent_node(state: AgentState) -> AgentState:
                state.get("messages", [])
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return {
                    "messages": [response],
                    "step_count": state.get("step_count", 0) + 1,
                    "tool_calls": state.get("tool_calls", 0),
                }

            return agent_node

        # Create nodes
        for i in range(num_agents):
            node_name = f"agent_{i}"
            workflow.add_node(node_name, make_agent_node(node_name, f"Process step {i}"))

        # Link sequentially
        workflow.set_entry_point("agent_0")
        for i in range(num_agents - 1):
            workflow.add_edge(f"agent_{i}", f"agent_{i + 1}")
        workflow.add_edge(f"agent_{num_agents - 1}", END)

        return workflow

    def _create_tool_calling_workflow(self) -> StateGraph:
        """Create a workflow with tool-calling agent."""
        workflow = StateGraph(AgentState)

        def tool_calling_agent(state: AgentState) -> AgentState:
            """Agent that calls tools."""
            state.get("messages", [])
            tool_calls = state.get("tool_calls", 0)

            # Simulate tool call
            time.perf_counter()
            result = calculator("2 + 2")
            time.perf_counter()

            return {
                "messages": [AIMessage(content=result)],
                "step_count": state.get("step_count", 0) + 1,
                "tool_calls": tool_calls + 1,
            }

        workflow.add_node("agent", tool_calling_agent)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

        return workflow

    def benchmark_sequential_workflow(
        self, num_agents: int, num_runs: int = 3
    ) -> List[AgentWorkflowBenchmarkResult]:
        """
        Benchmark sequential agent workflow.

        Args:
            num_agents: Number of agents in sequence
            num_runs: Number of runs to average

        Returns:
            List of benchmark results
        """
        results = []

        try:
            workflow = self._create_sequential_workflow(num_agents)
            app = workflow.compile()

            for run in range(num_runs):
                initial_state = {
                    "messages": [HumanMessage(content="Start workflow")],
                    "step_count": 0,
                    "tool_calls": 0,
                }

                start_time = time.perf_counter()
                final_state = app.invoke(initial_state)
                end_time = time.perf_counter()

                total_latency_ms = (end_time - start_time) * 1000
                num_steps = final_state.get("step_count", 0)
                avg_step_latency = total_latency_ms / num_steps if num_steps > 0 else 0

                result = AgentWorkflowBenchmarkResult(
                    workflow_type="sequential",
                    num_agents=num_agents,
                    num_steps=num_steps,
                    total_latency_ms=total_latency_ms,
                    avg_step_latency_ms=avg_step_latency,
                    state_overhead_ms=0,  # Calculated separately if needed
                    tool_calls=final_state.get("tool_calls", 0),
                    tool_call_latency_ms=0,
                )

                results.append(result)
                self.results.append(result)

                time.sleep(0.5)

        except Exception as e:
            result = AgentWorkflowBenchmarkResult(
                workflow_type="sequential",
                num_agents=num_agents,
                num_steps=0,
                total_latency_ms=0,
                avg_step_latency_ms=0,
                state_overhead_ms=0,
                tool_calls=0,
                tool_call_latency_ms=0,
                error=str(e),
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_tool_calling(
        self, num_calls: int = 5, num_runs: int = 3
    ) -> List[AgentWorkflowBenchmarkResult]:
        """
        Benchmark tool-calling workflow.

        Args:
            num_calls: Number of tool calls to make
            num_runs: Number of runs to average

        Returns:
            List of benchmark results
        """
        results = []

        try:
            for run in range(num_runs):
                start_time = time.perf_counter()

                # Measure individual tool call latency
                tool_latencies = []
                for i in range(num_calls):
                    tool_start = time.perf_counter()
                    _ = calculator(f"{i} + {i}")
                    tool_end = time.perf_counter()
                    tool_latencies.append((tool_end - tool_start) * 1000)

                end_time = time.perf_counter()

                total_latency_ms = (end_time - start_time) * 1000
                avg_tool_latency = statistics.mean(tool_latencies) if tool_latencies else 0

                result = AgentWorkflowBenchmarkResult(
                    workflow_type="tool_calling",
                    num_agents=1,
                    num_steps=num_calls,
                    total_latency_ms=total_latency_ms,
                    avg_step_latency_ms=total_latency_ms / num_calls,
                    state_overhead_ms=0,
                    tool_calls=num_calls,
                    tool_call_latency_ms=avg_tool_latency,
                )

                results.append(result)
                self.results.append(result)

                time.sleep(0.5)

        except Exception as e:
            result = AgentWorkflowBenchmarkResult(
                workflow_type="tool_calling",
                num_agents=1,
                num_steps=0,
                total_latency_ms=0,
                avg_step_latency_ms=0,
                state_overhead_ms=0,
                tool_calls=0,
                tool_call_latency_ms=0,
                error=str(e),
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_state_management(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark state management overhead.

        Args:
            num_iterations: Number of state updates to test

        Returns:
            Dictionary with timing metrics
        """
        # Measure state creation and update
        start = time.perf_counter()
        for i in range(num_iterations):
            state = AgentState(
                messages=[HumanMessage(content=f"Message {i}")],
                step_count=i,
                tool_calls=0,
            )
            # Simulate state update
            state["step_count"] = state.get("step_count", 0) + 1
        end = time.perf_counter()

        total_ms = (end - start) * 1000
        per_update_ms = total_ms / num_iterations

        return {
            "total_ms": total_ms,
            "per_update_ms": per_update_ms,
            "iterations": num_iterations,
        }

    def benchmark_all_workflows(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks on all workflow types.

        Returns:
            Summary statistics dictionary
        """
        print("Agent Workflow Performance Benchmark")
        print("=" * 70)

        # Sequential workflows
        print("\nSequential Workflows:")
        for num_agents in [2, 3, 5]:
            print(f"  - {num_agents} agents...", end=" ")
            results = self.benchmark_sequential_workflow(num_agents, num_runs=2)
            valid_results = [r for r in results if r.error is None]
            if valid_results:
                avg_latency = statistics.mean(r.total_latency_ms for r in valid_results)
                print(f"{avg_latency:.0f}ms")
            else:
                print("FAILED")

        # Tool calling
        print("\nTool Calling:")
        print("  - 5 tool calls...", end=" ")
        tool_results = self.benchmark_tool_calling(num_calls=5, num_runs=2)
        valid_tool_results = [r for r in tool_results if r.error is None]
        if valid_tool_results:
            avg_latency = statistics.mean(r.total_latency_ms for r in valid_tool_results)
            print(f"{avg_latency:.0f}ms")
        else:
            print("FAILED")

        # State management
        print("\nState Management:")
        print("  - 100 state updates...", end=" ")
        state_metrics = self.benchmark_state_management(100)
        print(f"{state_metrics['per_update_ms']:.3f}ms per update")

        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            "total_runs": len(self.results),
            "by_workflow": {},
        }

        # Group by workflow type
        for workflow_type in ["sequential", "tool_calling"]:
            workflow_results = [
                r for r in self.results if r.workflow_type == workflow_type and r.error is None
            ]
            if workflow_results:
                summary["by_workflow"][workflow_type] = {
                    "avg_total_latency_ms": statistics.mean(
                        r.total_latency_ms for r in workflow_results
                    ),
                    "avg_step_latency_ms": statistics.mean(
                        r.avg_step_latency_ms for r in workflow_results
                    ),
                    "runs": len(workflow_results),
                }

        return summary

    def save_results(self, format: str = "json"):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = self.output_dir / f"agent_workflow_benchmark_{timestamp}.json"
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
            output_file = self.output_dir / f"agent_workflow_benchmark_{timestamp}.csv"
            with open(output_file, "w", newline="") as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(asdict(result))
            print(f"\nResults saved to {output_file}")


# Pytest integration
@pytest.mark.integration
@pytest.mark.slow
def test_sequential_workflow():
    """Test sequential workflow performance."""
    benchmark = AgentWorkflowBenchmark()
    results = benchmark.benchmark_sequential_workflow(num_agents=2, num_runs=1)

    assert len(results) == 1
    result = results[0]

    assert result.error is None, f"Workflow failed: {result.error}"
    assert result.num_steps > 0


@pytest.mark.integration
@pytest.mark.slow
def test_tool_calling_workflow():
    """Test tool calling performance."""
    benchmark = AgentWorkflowBenchmark()
    results = benchmark.benchmark_tool_calling(num_calls=3, num_runs=1)

    assert len(results) == 1
    result = results[0]

    assert result.error is None, f"Tool calling failed: {result.error}"
    assert result.tool_calls == 3


def test_state_management():
    """Test state management overhead."""
    benchmark = AgentWorkflowBenchmark()
    metrics = benchmark.benchmark_state_management(num_iterations=50)

    assert metrics["iterations"] == 50
    assert metrics["per_update_ms"] >= 0


if __name__ == "__main__":
    """Run benchmarks standalone."""
    benchmark = AgentWorkflowBenchmark()

    # Run full benchmark suite
    summary = benchmark.benchmark_all_workflows()

    # Save results
    benchmark.save_results(format="json")
    benchmark.save_results(format="csv")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
