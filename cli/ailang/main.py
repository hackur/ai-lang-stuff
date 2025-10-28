"""Main entry point for ailang CLI."""

import click
from ailang.commands import models, examples, mcp, rag


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    ailang - Local-first AI development toolkit.

    A comprehensive CLI for managing local LLMs, examples, MCP servers,
    and RAG systems. Built for privacy-preserving, on-device AI development.

    \b
    Quick Start:
      ailang models list          # List available models
      ailang examples list        # Show example projects
      ailang mcp list            # List MCP servers
      ailang rag list            # Show RAG collections

    \b
    Documentation:
      https://github.com/yourusername/ai-lang-stuff
    """
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["QUIET"] = quiet


# Register command groups
cli.add_command(models.models)
cli.add_command(examples.examples)
cli.add_command(mcp.mcp)
cli.add_command(rag.rag)


if __name__ == "__main__":
    cli()
