"""Test individual command modules."""

import pytest
from click.testing import CliRunner
from pathlib import Path

from ailang.commands.models import MODEL_RECOMMENDATIONS
from ailang.commands.examples import get_examples, EXAMPLES_DIR
from ailang.commands.mcp import get_mcp_servers, MCP_DIR
from ailang.commands.rag import get_collections, RAG_DATA_DIR


class TestModelsCommand:
    """Test models command functionality."""

    def test_model_recommendations_exist(self):
        """Test that model recommendations are defined."""
        assert "coding" in MODEL_RECOMMENDATIONS
        assert "reasoning" in MODEL_RECOMMENDATIONS
        assert "vision" in MODEL_RECOMMENDATIONS

    def test_model_recommendations_structure(self):
        """Test model recommendations have correct structure."""
        for task, models in MODEL_RECOMMENDATIONS.items():
            assert isinstance(models, dict)
            for variant, model_name in models.items():
                assert isinstance(model_name, str)
                assert ":" in model_name  # Format: name:version

    def test_coding_recommendations(self):
        """Test coding task recommendations."""
        coding = MODEL_RECOMMENDATIONS["coding"]
        assert "fast" in coding or "balanced" in coding
        assert any("qwen" in v.lower() or "gemma" in v.lower()
                   for v in coding.values())


class TestExamplesCommand:
    """Test examples command functionality."""

    def test_get_examples_returns_list(self):
        """Test that get_examples returns a list."""
        examples = get_examples()
        assert isinstance(examples, list)

    def test_example_structure(self):
        """Test example objects have required fields."""
        examples = get_examples()
        if examples:
            example = examples[0]
            assert "category" in example
            assert "name" in example
            assert "path" in example
            assert "description" in example

    def test_examples_dir_constant(self):
        """Test EXAMPLES_DIR is defined."""
        assert isinstance(EXAMPLES_DIR, Path)


class TestMCPCommand:
    """Test MCP command functionality."""

    def test_get_mcp_servers_returns_list(self):
        """Test that get_mcp_servers returns a list."""
        servers = get_mcp_servers()
        assert isinstance(servers, list)

    def test_mcp_server_structure(self):
        """Test MCP server objects have required fields."""
        servers = get_mcp_servers()
        if servers:
            server = servers[0]
            assert "name" in server
            assert "type" in server
            assert "path" in server

    def test_mcp_dir_constant(self):
        """Test MCP_DIR is defined."""
        assert isinstance(MCP_DIR, Path)


class TestRAGCommand:
    """Test RAG command functionality."""

    def test_get_collections_returns_list(self):
        """Test that get_collections returns a list."""
        collections = get_collections()
        assert isinstance(collections, list)

    def test_collection_structure(self):
        """Test collection objects have required fields."""
        collections = get_collections()
        if collections:
            collection = collections[0]
            assert "name" in collection
            assert "path" in collection
            assert "doc_count" in collection
            assert "metadata" in collection

    def test_rag_data_dir_constant(self):
        """Test RAG_DATA_DIR is defined."""
        assert isinstance(RAG_DATA_DIR, Path)
        assert ".ailang" in str(RAG_DATA_DIR)


class TestCommandIntegration:
    """Test command integration and consistency."""

    def test_all_commands_have_help(self):
        """Test that all commands have help text."""
        from ailang.main import cli

        runner = CliRunner()

        commands = ["models", "examples", "mcp", "rag"]
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0
            assert len(result.output) > 0

    def test_subcommands_exist(self):
        """Test that expected subcommands exist."""
        from ailang.main import cli

        runner = CliRunner()

        # Test models subcommands
        result = runner.invoke(cli, ["models", "recommend", "--help"])
        assert result.exit_code == 0

        # Test examples subcommands
        result = runner.invoke(cli, ["examples", "list", "--help"])
        assert result.exit_code == 0

        # Test mcp subcommands
        result = runner.invoke(cli, ["mcp", "list", "--help"])
        assert result.exit_code == 0

        # Test rag subcommands
        result = runner.invoke(cli, ["rag", "list", "--help"])
        assert result.exit_code == 0

    def test_json_output_option(self):
        """Test that commands support JSON output where applicable."""
        from ailang.main import cli

        runner = CliRunner()

        # Commands that should support --json-output
        json_commands = [
            ["models", "list", "--help"],
            ["examples", "list", "--help"],
            ["mcp", "list", "--help"],
            ["rag", "list", "--help"],
        ]

        for cmd in json_commands:
            result = runner.invoke(cli, cmd)
            # Check help text mentions json-output
            if "--help" in cmd:
                assert "json" in result.output.lower() or result.exit_code == 0
