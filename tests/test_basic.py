"""
Basic tests for local AI setup.

Run with: uv run pytest tests/test_basic.py
"""

import pytest
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def test_ollama_import():
    """Test that langchain_ollama can be imported."""
    from langchain_ollama import ChatOllama

    assert ChatOllama is not None


def test_chat_ollama_initialization():
    """Test that ChatOllama can be initialized."""
    llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")
    assert llm is not None
    assert llm.model == "qwen3:8b"


@pytest.mark.integration
def test_model_response():
    """Test that model can generate a response (requires Ollama running)."""
    llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")

    response = llm.invoke([HumanMessage(content="Say 'test successful'")])

    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.integration
def test_streaming_response():
    """Test that model can stream responses."""
    llm = ChatOllama(model="qwen3:8b")

    chunks = list(llm.stream([HumanMessage(content="Count to 3")]))

    assert len(chunks) > 0
    assert all(hasattr(chunk, "content") for chunk in chunks)


def test_config_loading():
    """Test that configuration can be loaded."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "models.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "default" in config
        assert "model_name" in config["default"]
    else:
        pytest.skip("Config file not found")


def test_env_file_exists():
    """Test that .env file exists or example exists."""
    from pathlib import Path

    env_path = Path(__file__).parent.parent / ".env"
    env_example_path = Path(__file__).parent.parent / "config" / ".env.example"

    assert env_path.exists() or env_example_path.exists(), ".env or .env.example should exist"


def test_directory_structure():
    """Test that required directories exist."""
    from pathlib import Path

    base_path = Path(__file__).parent.parent

    required_dirs = [
        "examples",
        "config",
        "plans",
        "scripts",
        "mcp-servers",
        "tests",
    ]

    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"


@pytest.mark.parametrize(
    "model_name",
    ["qwen3:8b", "qwen3:30b-a3b", "gemma3:4b"],
)
def test_model_configuration(model_name):
    """Test that models can be configured."""
    llm = ChatOllama(model=model_name)
    assert llm.model == model_name
