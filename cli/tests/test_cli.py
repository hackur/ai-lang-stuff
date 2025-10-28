"""Basic CLI tests."""

from click.testing import CliRunner
from ailang.main import cli


def test_cli_help():
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ailang - Local-first AI development toolkit" in result.output


def test_cli_version():
    """Test that version flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_models_command():
    """Test models command group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "--help"])
    assert result.exit_code == 0
    assert "Manage local LLM models" in result.output


def test_examples_command():
    """Test examples command group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "--help"])
    assert result.exit_code == 0
    assert "Run and manage example projects" in result.output


def test_mcp_command():
    """Test mcp command group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "Model Context Protocol" in result.output


def test_rag_command():
    """Test rag command group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "--help"])
    assert result.exit_code == 0
    assert "Retrieval-Augmented Generation" in result.output


def test_models_recommend():
    """Test model recommendation command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "recommend", "coding"])
    assert result.exit_code == 0
    # Should show recommendations even without Ollama installed


def test_examples_list():
    """Test examples list command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "list"])
    assert result.exit_code == 0


def test_rag_list():
    """Test rag list command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "list"])
    assert result.exit_code == 0
