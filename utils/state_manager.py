"""LangGraph state management and persistence utilities.

This module provides helper functions for managing LangGraph state persistence
using SQLite checkpointers, creating state schemas, and managing checkpoint lifecycles.
"""

import logging
import operator
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Type, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver


logger = logging.getLogger(__name__)


class StateManager:
    """Manages LangGraph state persistence and checkpoint operations."""

    @staticmethod
    def get_checkpointer(db_path: str = "./checkpoints.db") -> SqliteSaver:
        """Create or connect to SQLite checkpointer for LangGraph persistence.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.

        Returns:
            SqliteSaver: Configured SQLite checkpointer for LangGraph.

        Raises:
            sqlite3.Error: If database connection fails.

        Example:
            >>> checkpointer = StateManager.get_checkpointer("./my_checkpoints.db")
            >>> graph = graph_builder.compile(checkpointer=checkpointer)
        """
        try:
            # Ensure directory exists
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Create checkpointer connection
            logger.info(f"Creating SQLite checkpointer at {db_path}")
            connection = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(connection)

            # Initialize checkpoint tables if needed
            checkpointer.setup()

            logger.info(f"Successfully initialized checkpointer at {db_path}")
            return checkpointer

        except sqlite3.Error as e:
            logger.error(f"Failed to create checkpointer at {db_path}: {e}")
            raise

    @staticmethod
    def create_state_schema(
        fields: Dict[str, type], class_name: str = "AgentState"
    ) -> Type[TypedDict]:
        """Create a TypedDict state schema for LangGraph workflows.

        Args:
            fields: Dictionary mapping field names to types.
                Use Annotated[Type, operator.add] for list reduction.
            class_name: Name for the generated TypedDict class.

        Returns:
            Type[TypedDict]: Generated state schema class.

        Raises:
            ValueError: If fields are invalid or empty.
            TypeError: If field types are not valid type annotations.

        Example:
            >>> from typing import Annotated
            >>> import operator
            >>> schema = StateManager.create_state_schema({
            ...     "messages": Annotated[List[BaseMessage], operator.add],
            ...     "context": str,
            ...     "iteration": int
            ... })
        """
        if not fields:
            raise ValueError("Fields dictionary cannot be empty")

        if not all(isinstance(name, str) for name in fields.keys()):
            raise ValueError("All field names must be strings")

        try:
            # Validate field types
            for field_name, field_type in fields.items():
                if not isinstance(field_type, type) and not hasattr(field_type, "__origin__"):
                    raise TypeError(f"Invalid type for field '{field_name}': {field_type}")

            # Create TypedDict dynamically
            state_schema = TypedDict(class_name, fields)

            logger.info(f"Created state schema '{class_name}' with fields: {list(fields.keys())}")
            return state_schema

        except Exception as e:
            logger.error(f"Failed to create state schema: {e}")
            raise

    @staticmethod
    def load_checkpoint(
        thread_id: str, db_path: str = "./checkpoints.db"
    ) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint by thread ID.

        Args:
            thread_id: Unique identifier for the checkpoint thread.
            db_path: Path to SQLite checkpoint database.

        Returns:
            Optional[Dict[str, Any]]: Checkpoint state dictionary or None if not found.

        Raises:
            sqlite3.Error: If database query fails.

        Example:
            >>> state = StateManager.load_checkpoint("thread-123")
            >>> if state:
            ...     print(f"Found checkpoint with {len(state['messages'])} messages")
        """
        try:
            connection = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(connection)

            # Query checkpoint for thread
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = checkpointer.get(config)

            if checkpoint is None:
                logger.warning(f"No checkpoint found for thread_id: {thread_id}")
                return None

            logger.info(f"Loaded checkpoint for thread_id: {thread_id}")
            return checkpoint.get("channel_values", {})

        except sqlite3.Error as e:
            logger.error(f"Failed to load checkpoint for thread_id {thread_id}: {e}")
            raise
        finally:
            if "connection" in locals():
                connection.close()

    @staticmethod
    def list_checkpoints(db_path: str = "./checkpoints.db") -> List[Dict[str, Any]]:
        """List all checkpoint thread IDs with metadata.

        Args:
            db_path: Path to SQLite checkpoint database.

        Returns:
            List[Dict[str, Any]]: List of checkpoint metadata dictionaries with:
                - thread_id: Unique thread identifier
                - checkpoint_id: Checkpoint version identifier
                - timestamp: When checkpoint was created

        Raises:
            sqlite3.Error: If database query fails.

        Example:
            >>> checkpoints = StateManager.list_checkpoints()
            >>> for cp in checkpoints:
            ...     print(f"Thread: {cp['thread_id']}, Time: {cp['timestamp']}")
        """
        try:
            if not Path(db_path).exists():
                logger.warning(f"Checkpoint database not found: {db_path}")
                return []

            connection = sqlite3.connect(db_path, check_same_thread=False)
            cursor = connection.cursor()

            # Query checkpoint metadata from LangGraph tables
            # SqliteSaver uses 'checkpoints' table
            cursor.execute(
                """
                SELECT thread_id, checkpoint_id, created_at
                FROM checkpoints
                ORDER BY created_at DESC
            """
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "thread_id": row[0],
                        "checkpoint_id": row[1],
                        "timestamp": row[2],
                    }
                )

            logger.info(f"Found {len(results)} checkpoints in {db_path}")
            return results

        except sqlite3.Error as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise
        finally:
            if "connection" in locals():
                connection.close()

    @staticmethod
    def clear_checkpoints(
        db_path: str = "./checkpoints.db",
        thread_id: Optional[str] = None,
        confirm: bool = False,
    ) -> int:
        """Clear checkpoint data from database.

        Args:
            db_path: Path to SQLite checkpoint database.
            thread_id: Specific thread to clear. If None, clears all checkpoints.
            confirm: Must be True to actually delete data (safety check).

        Returns:
            int: Number of checkpoints deleted.

        Raises:
            sqlite3.Error: If database operation fails.
            ValueError: If confirm is False (prevents accidental deletion).

        Example:
            >>> # Clear specific thread
            >>> deleted = StateManager.clear_checkpoints(
            ...     thread_id="thread-123",
            ...     confirm=True
            ... )
            >>> print(f"Deleted {deleted} checkpoints")
            >>>
            >>> # Clear all checkpoints
            >>> deleted = StateManager.clear_checkpoints(confirm=True)
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to delete checkpoints. "
                "This prevents accidental data loss."
            )

        try:
            if not Path(db_path).exists():
                logger.warning(f"Checkpoint database not found: {db_path}")
                return 0

            connection = sqlite3.connect(db_path, check_same_thread=False)
            cursor = connection.cursor()

            if thread_id:
                # Delete specific thread
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                deleted = cursor.rowcount
                logger.info(f"Deleted {deleted} checkpoints for thread_id: {thread_id}")
            else:
                # Delete all checkpoints
                cursor.execute("SELECT COUNT(*) FROM checkpoints")
                deleted = cursor.fetchone()[0]
                cursor.execute("DELETE FROM checkpoints")
                logger.warning(f"Deleted ALL {deleted} checkpoints from {db_path}")

            connection.commit()
            return deleted

        except sqlite3.Error as e:
            logger.error(f"Failed to clear checkpoints: {e}")
            raise
        finally:
            if "connection" in locals():
                connection.close()


# ============================================================================
# Common State Pattern Helpers
# ============================================================================


def basic_agent_state() -> Type[TypedDict]:
    """Create basic agent state with message history.

    Returns:
        Type[TypedDict]: State schema with messages list using operator.add.

    Example:
        >>> State = basic_agent_state()
        >>> def agent_node(state: State) -> State:
        ...     messages = state["messages"]
        ...     # ... process messages
        ...     return {"messages": [response]}
    """
    return StateManager.create_state_schema(
        {"messages": Annotated[List[BaseMessage], operator.add]},
        class_name="BasicAgentState",
    )


def research_state() -> Type[TypedDict]:
    """Create research agent state with question, context, and sources.

    Returns:
        Type[TypedDict]: State schema for research workflows.

    Example:
        >>> State = research_state()
        >>> def research_node(state: State) -> State:
        ...     question = state["question"]
        ...     context = state.get("context", [])
        ...     # ... perform research
        ...     return {
        ...         "context": context + [new_info],
        ...         "sources": [source_url]
        ...     }
    """
    return StateManager.create_state_schema(
        {
            "question": str,
            "context": Annotated[List[str], operator.add],
            "answer": str,
            "sources": Annotated[List[str], operator.add],
        },
        class_name="ResearchState",
    )


def code_review_state() -> Type[TypedDict]:
    """Create code review state with code, issues, and approval status.

    Returns:
        Type[TypedDict]: State schema for code review workflows.

    Example:
        >>> State = code_review_state()
        >>> def review_node(state: State) -> State:
        ...     code = state["code"]
        ...     issues = analyze_code(code)
        ...     return {
        ...         "issues": issues,
        ...         "suggestions": generate_suggestions(issues),
        ...         "approved": len(issues) == 0
        ...     }
    """
    return StateManager.create_state_schema(
        {
            "code": str,
            "issues": Annotated[List[str], operator.add],
            "suggestions": Annotated[List[str], operator.add],
            "approved": bool,
        },
        class_name="CodeReviewState",
    )


# ============================================================================
# Utility Functions
# ============================================================================


def create_thread_id(prefix: str = "thread") -> str:
    """Generate unique thread ID with timestamp.

    Args:
        prefix: Prefix for thread ID.

    Returns:
        str: Unique thread ID in format "{prefix}-{timestamp}".

    Example:
        >>> thread_id = create_thread_id("research")
        >>> print(thread_id)  # "research-1698765432"
    """
    timestamp = int(datetime.now().timestamp())
    thread_id = f"{prefix}-{timestamp}"
    logger.debug(f"Generated thread_id: {thread_id}")
    return thread_id


def get_checkpoint_size(db_path: str = "./checkpoints.db") -> Dict[str, Any]:
    """Get checkpoint database statistics.

    Args:
        db_path: Path to SQLite checkpoint database.

    Returns:
        Dict[str, Any]: Statistics including:
            - file_size_mb: Database file size in megabytes
            - checkpoint_count: Total number of checkpoints
            - thread_count: Number of unique threads

    Example:
        >>> stats = get_checkpoint_size()
        >>> print(f"Database size: {stats['file_size_mb']:.2f} MB")
        >>> print(f"Total checkpoints: {stats['checkpoint_count']}")
    """
    try:
        db_file = Path(db_path)

        if not db_file.exists():
            return {"file_size_mb": 0.0, "checkpoint_count": 0, "thread_count": 0}

        # Get file size
        file_size_mb = db_file.stat().st_size / (1024 * 1024)

        # Get checkpoint counts
        connection = sqlite3.connect(db_path, check_same_thread=False)
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        thread_count = cursor.fetchone()[0]

        connection.close()

        return {
            "file_size_mb": file_size_mb,
            "checkpoint_count": checkpoint_count,
            "thread_count": thread_count,
        }

    except Exception as e:
        logger.error(f"Failed to get checkpoint statistics: {e}")
        return {"file_size_mb": 0.0, "checkpoint_count": 0, "thread_count": 0, "error": str(e)}


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "StateManager",
    "basic_agent_state",
    "research_state",
    "code_review_state",
    "create_thread_id",
    "get_checkpoint_size",
]
