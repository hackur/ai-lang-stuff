"""Utility modules for local-first AI experimentation toolkit."""

# Core utilities - always available
from .tool_registry import ToolRegistry, get_registry

__all__ = ["ToolRegistry", "get_registry"]

# Ollama management
try:
    from .ollama_manager import OllamaManager

    __all__.extend(["OllamaManager"])
except ImportError:
    pass

# MCP client wrappers
try:
    from .mcp_client import MCPClient, FilesystemMCP, WebSearchMCP

    __all__.extend(["MCPClient", "FilesystemMCP", "WebSearchMCP"])
except ImportError:
    pass

# Vector store management
try:
    from .vector_store import (
        VectorStoreManager,
        create_chroma_store,
        create_faiss_store,
        load_vector_store,
    )

    __all__.extend(
        [
            "VectorStoreManager",
            "create_chroma_store",
            "create_faiss_store",
            "load_vector_store",
        ]
    )
except ImportError:
    pass

# State management for LangGraph
try:
    from .state_manager import (
        StateManager,
        create_thread_id,
        basic_agent_state,
        research_state,
        code_review_state,
    )

    __all__.extend(
        [
            "StateManager",
            "create_thread_id",
            "basic_agent_state",
            "research_state",
            "code_review_state",
        ]
    )
except ImportError:
    pass
