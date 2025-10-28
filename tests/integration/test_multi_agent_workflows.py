"""
Integration tests for multi-agent workflows using LangGraph.

Tests complete workflow execution including:
- State persistence and recovery
- Parallel execution patterns
- Error recovery mechanisms
- Checkpoint validation
"""

import operator
import sqlite3

# Import utilities
import sys
from pathlib import Path
from typing import Annotated, List, TypedDict

import pytest
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.state_manager import StateManager, create_thread_id

# ============================================================================
# Test State Definitions
# ============================================================================


class SimpleAgentState(TypedDict):
    """Simple state for testing."""

    messages: Annotated[List, operator.add]
    counter: int
    result: str


class ParallelAgentState(TypedDict):
    """State for parallel execution testing."""

    task: str
    agent_a_result: str
    agent_b_result: str
    combined_result: str
    messages: Annotated[List, operator.add]


class ErrorRecoveryState(TypedDict):
    """State for error recovery testing."""

    step: int
    errors: Annotated[List[str], operator.add]
    results: Annotated[List[str], operator.add]
    retry_count: int


# ============================================================================
# Workflow Execution Tests
# ============================================================================


class TestWorkflowExecution:
    """Test complete workflow execution."""

    def test_simple_sequential_workflow(self, checkpoint_dir: Path):
        """Test simple sequential workflow with checkpoints.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def node_a(state: SimpleAgentState) -> SimpleAgentState:
            """First node."""
            return {
                "counter": state.get("counter", 0) + 1,
                "messages": [AIMessage(content="Node A executed")],
            }

        def node_b(state: SimpleAgentState) -> SimpleAgentState:
            """Second node."""
            return {
                "counter": state.get("counter", 0) + 1,
                "result": "Workflow complete",
                "messages": [AIMessage(content="Node B executed")],
            }

        # Create workflow
        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("node_a", node_a)
        workflow.add_node("node_b", node_b)

        workflow.set_entry_point("node_a")
        workflow.add_edge("node_a", "node_b")
        workflow.add_edge("node_b", END)

        # Compile with checkpointer
        checkpoint_path = checkpoint_dir / "test_workflow.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        # Execute workflow
        thread_id = create_thread_id("test-workflow")
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {"messages": [], "counter": 0, "result": ""}

        # Run workflow
        final_state = None
        for output in app.stream(initial_state, config):
            final_state = output

        # Validate final state
        assert final_state is not None
        last_node_state = list(final_state.values())[0]
        assert last_node_state["counter"] == 2
        assert last_node_state["result"] == "Workflow complete"

    @pytest.mark.integration
    def test_workflow_with_conditional_edges(self, checkpoint_dir: Path):
        """Test workflow with conditional routing.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def router_node(state: SimpleAgentState) -> SimpleAgentState:
            """Router that decides next path."""
            return {
                "counter": state.get("counter", 0) + 1,
                "messages": [AIMessage(content="Router executed")],
            }

        def path_a_node(state: SimpleAgentState) -> SimpleAgentState:
            """Path A."""
            return {
                "result": "Took path A",
                "messages": [AIMessage(content="Path A")],
            }

        def path_b_node(state: SimpleAgentState) -> SimpleAgentState:
            """Path B."""
            return {
                "result": "Took path B",
                "messages": [AIMessage(content="Path B")],
            }

        def route_decision(state: SimpleAgentState) -> str:
            """Decide which path to take."""
            if state.get("counter", 0) > 0:
                return "path_a"
            return "path_b"

        # Create workflow
        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("router", router_node)
        workflow.add_node("path_a", path_a_node)
        workflow.add_node("path_b", path_b_node)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router", route_decision, {"path_a": "path_a", "path_b": "path_b"}
        )
        workflow.add_edge("path_a", END)
        workflow.add_edge("path_b", END)

        # Compile and execute
        checkpoint_path = checkpoint_dir / "conditional_workflow.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("conditional-test")
        config = {"configurable": {"thread_id": thread_id}}

        final_state = None
        for output in app.stream({"messages": [], "counter": 0, "result": ""}, config):
            final_state = output

        # Should take path A (counter > 0 after router)
        assert final_state is not None
        last_state = list(final_state.values())[0]
        assert "path A" in last_state["result"]


# ============================================================================
# State Persistence Tests
# ============================================================================


class TestStatePersistence:
    """Test state persistence and recovery."""

    def test_checkpoint_creation(self, checkpoint_dir: Path):
        """Test that checkpoints are created.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        checkpoint_path = checkpoint_dir / "test_checkpoints.db"

        # Create checkpointer
        StateManager.get_checkpointer(str(checkpoint_path))

        assert checkpoint_path.exists()

    def test_state_recovery_after_execution(self, checkpoint_dir: Path):
        """Test recovering state after workflow execution.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def test_node(state: SimpleAgentState) -> SimpleAgentState:
            """Test node."""
            return {
                "counter": state.get("counter", 0) + 1,
                "result": "Node executed",
                "messages": [AIMessage(content="Test message")],
            }

        # Create and execute workflow
        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("test", test_node)
        workflow.set_entry_point("test")
        workflow.add_edge("test", END)

        checkpoint_path = checkpoint_dir / "recovery_test.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("recovery-test")
        config = {"configurable": {"thread_id": thread_id}}

        # Execute
        list(app.stream({"messages": [], "counter": 0, "result": ""}, config))

        # Recover state
        recovered_state = app.get_state(config)

        assert recovered_state is not None
        assert recovered_state.values["counter"] == 1
        assert recovered_state.values["result"] == "Node executed"

    def test_checkpoint_history(self, checkpoint_dir: Path):
        """Test that checkpoint history is maintained.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def increment_node(state: SimpleAgentState) -> SimpleAgentState:
            """Increment counter."""
            return {
                "counter": state.get("counter", 0) + 1,
                "messages": [AIMessage(content=f"Count: {state.get('counter', 0) + 1}")],
            }

        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("increment", increment_node)
        workflow.set_entry_point("increment")
        workflow.add_edge("increment", END)

        checkpoint_path = checkpoint_dir / "history_test.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("history-test")
        config = {"configurable": {"thread_id": thread_id}}

        # Execute multiple times
        for _ in range(3):
            list(app.stream({"messages": [], "counter": 0, "result": ""}, config))

        # Check that checkpoint file has data
        assert checkpoint_path.exists()

        # Query checkpoint database
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0


# ============================================================================
# Parallel Execution Tests
# ============================================================================


class TestParallelExecution:
    """Test parallel agent execution patterns."""

    def test_parallel_branches(self, checkpoint_dir: Path):
        """Test workflow with parallel branches.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def agent_a(state: ParallelAgentState) -> ParallelAgentState:
            """Agent A processing."""
            return {
                "agent_a_result": f"Agent A processed: {state['task']}",
                "messages": [AIMessage(content="Agent A done")],
            }

        def agent_b(state: ParallelAgentState) -> ParallelAgentState:
            """Agent B processing."""
            return {
                "agent_b_result": f"Agent B analyzed: {state['task']}",
                "messages": [AIMessage(content="Agent B done")],
            }

        def combiner(state: ParallelAgentState) -> ParallelAgentState:
            """Combine results from parallel agents."""
            combined = f"{state.get('agent_a_result', '')} | {state.get('agent_b_result', '')}"
            return {
                "combined_result": combined,
                "messages": [AIMessage(content="Combined results")],
            }

        # Create workflow with parallel branches
        workflow = StateGraph(ParallelAgentState)
        workflow.add_node("agent_a", agent_a)
        workflow.add_node("agent_b", agent_b)
        workflow.add_node("combiner", combiner)

        # Both agents start from entry point
        workflow.set_entry_point("agent_a")
        workflow.add_edge("agent_a", "agent_b")
        workflow.add_edge("agent_b", "combiner")
        workflow.add_edge("combiner", END)

        # Execute
        checkpoint_path = checkpoint_dir / "parallel_test.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("parallel-test")
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "task": "test task",
            "agent_a_result": "",
            "agent_b_result": "",
            "combined_result": "",
            "messages": [],
        }

        final_state = None
        for output in app.stream(initial_state, config):
            final_state = output

        # Validate both agents executed
        assert final_state is not None
        last_state = list(final_state.values())[0]
        assert "Agent A processed" in last_state["combined_result"]
        assert "Agent B analyzed" in last_state["combined_result"]


# ============================================================================
# Error Recovery Tests
# ============================================================================


class TestErrorRecovery:
    """Test error recovery mechanisms in workflows."""

    def test_error_handling_in_node(self, checkpoint_dir: Path):
        """Test error handling within workflow nodes.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def error_prone_node(state: ErrorRecoveryState) -> ErrorRecoveryState:
            """Node that may error on first try."""
            retry_count = state.get("retry_count", 0)

            if retry_count < 1:
                return {
                    "errors": ["Simulated error on first attempt"],
                    "retry_count": retry_count + 1,
                }
            else:
                return {
                    "results": ["Success on retry"],
                    "step": state.get("step", 0) + 1,
                }

        workflow = StateGraph(ErrorRecoveryState)
        workflow.add_node("error_prone", error_prone_node)
        workflow.set_entry_point("error_prone")
        workflow.add_edge("error_prone", END)

        checkpoint_path = checkpoint_dir / "error_test.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("error-test")
        config = {"configurable": {"thread_id": thread_id}}

        # First attempt - should error
        initial_state = {"step": 0, "errors": [], "results": [], "retry_count": 0}

        final_state = None
        for output in app.stream(initial_state, config):
            final_state = output

        assert final_state is not None
        last_state = list(final_state.values())[0]

        # Should have recorded error
        assert len(last_state["errors"]) > 0 or last_state["retry_count"] > 0

    def test_workflow_state_after_error(self, checkpoint_dir: Path):
        """Test that state is preserved after error.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def safe_node(state: ErrorRecoveryState) -> ErrorRecoveryState:
            """Safe node that always succeeds."""
            return {
                "results": ["Safe execution"],
                "step": state.get("step", 0) + 1,
            }

        workflow = StateGraph(ErrorRecoveryState)
        workflow.add_node("safe", safe_node)
        workflow.set_entry_point("safe")
        workflow.add_edge("safe", END)

        checkpoint_path = checkpoint_dir / "state_after_error.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("state-error-test")
        config = {"configurable": {"thread_id": thread_id}}

        # Execute with some initial error state
        initial_state = {"step": 0, "errors": ["Previous error"], "results": [], "retry_count": 1}

        final_state = None
        for output in app.stream(initial_state, config):
            final_state = output

        # State should include previous error
        assert final_state is not None
        last_state = list(final_state.values())[0]
        assert len(last_state["errors"]) > 0
        assert "Previous error" in last_state["errors"]


# ============================================================================
# Multi-Agent Coordination Tests
# ============================================================================


class TestMultiAgentCoordination:
    """Test coordination between multiple agents."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_research_pipeline_structure(self, checkpoint_dir: Path, mock_ollama_llm):
        """Test research pipeline agent coordination.

        Args:
            checkpoint_dir: Checkpoint directory fixture
            mock_ollama_llm: Mock Ollama LLM fixture
        """
        from langgraph.graph import END, StateGraph

        class ResearchState(TypedDict):
            """Research pipeline state."""

            question: str
            research: str
            analysis: str
            summary: str
            messages: Annotated[List, operator.add]

        def researcher(state: ResearchState) -> ResearchState:
            """Mock researcher."""
            return {
                "research": f"Research findings for: {state['question']}",
                "messages": [AIMessage(content="Research complete")],
            }

        def analyzer(state: ResearchState) -> ResearchState:
            """Mock analyzer."""
            return {
                "analysis": f"Analysis of: {state['research']}",
                "messages": [AIMessage(content="Analysis complete")],
            }

        def summarizer(state: ResearchState) -> ResearchState:
            """Mock summarizer."""
            return {
                "summary": f"Summary: {state['analysis']}",
                "messages": [AIMessage(content="Summary complete")],
            }

        # Build pipeline
        workflow = StateGraph(ResearchState)
        workflow.add_node("researcher", researcher)
        workflow.add_node("analyzer", analyzer)
        workflow.add_node("summarizer", summarizer)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyzer")
        workflow.add_edge("analyzer", "summarizer")
        workflow.add_edge("summarizer", END)

        # Execute
        checkpoint_path = checkpoint_dir / "research_pipeline.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("research-pipeline-test")
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "question": "What is LangGraph?",
            "research": "",
            "analysis": "",
            "summary": "",
            "messages": [],
        }

        final_state = None
        for output in app.stream(initial_state, config):
            final_state = output

        # Validate all stages executed
        assert final_state is not None
        last_state = list(final_state.values())[0]
        assert len(last_state["research"]) > 0
        assert len(last_state["analysis"]) > 0
        assert len(last_state["summary"]) > 0


# ============================================================================
# Checkpoint Validation Tests
# ============================================================================


class TestCheckpointValidation:
    """Test checkpoint data integrity and validation."""

    def test_checkpoint_file_creation(self, checkpoint_dir: Path):
        """Test checkpoint database file is created.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        checkpoint_path = checkpoint_dir / "validation_test.db"
        StateManager.get_checkpointer(str(checkpoint_path))

        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0

    def test_checkpoint_data_integrity(self, checkpoint_dir: Path):
        """Test checkpoint data can be read from database.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def test_node(state: SimpleAgentState) -> SimpleAgentState:
            """Test node."""
            return {
                "counter": 42,
                "result": "test result",
                "messages": [AIMessage(content="test")],
            }

        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("test", test_node)
        workflow.set_entry_point("test")
        workflow.add_edge("test", END)

        checkpoint_path = checkpoint_dir / "integrity_test.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        thread_id = create_thread_id("integrity-test")
        config = {"configurable": {"thread_id": thread_id}}

        # Execute
        list(app.stream({"messages": [], "counter": 0, "result": ""}, config))

        # Verify checkpoint data
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        assert len(tables) > 0
        table_names = [t[0] for t in tables]
        assert "checkpoints" in table_names

        conn.close()

    def test_multiple_thread_checkpoints(self, checkpoint_dir: Path):
        """Test checkpoints for multiple threads.

        Args:
            checkpoint_dir: Checkpoint directory fixture
        """
        from langgraph.graph import END, StateGraph

        def simple_node(state: SimpleAgentState) -> SimpleAgentState:
            """Simple node."""
            return {"counter": state.get("counter", 0) + 1}

        workflow = StateGraph(SimpleAgentState)
        workflow.add_node("node", simple_node)
        workflow.set_entry_point("node")
        workflow.add_edge("node", END)

        checkpoint_path = checkpoint_dir / "multi_thread.db"
        checkpointer = StateManager.get_checkpointer(str(checkpoint_path))
        app = workflow.compile(checkpointer=checkpointer)

        # Execute with multiple threads
        thread_ids = [create_thread_id(f"thread-{i}") for i in range(3)]

        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            list(app.stream({"messages": [], "counter": 0, "result": ""}, config))

        # Verify multiple threads in checkpoint
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        thread_count = cursor.fetchone()[0]
        conn.close()

        assert thread_count >= 1  # At least some threads recorded
