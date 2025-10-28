"""RAG (Retrieval-Augmented Generation) utilities."""

import json
import shutil
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.tree import Tree

console = Console()

RAG_DATA_DIR = Path.home() / ".ailang" / "rag"


def get_collections() -> List[dict]:
    """Get list of RAG collections."""
    collections = []

    if not RAG_DATA_DIR.exists():
        return collections

    for collection_dir in RAG_DATA_DIR.iterdir():
        if not collection_dir.is_dir():
            continue

        metadata_file = collection_dir / "metadata.json"
        metadata = {}

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        # Count documents
        doc_count = len(list(collection_dir.glob("*.json"))) - 1  # Exclude metadata

        collections.append({
            "name": collection_dir.name,
            "path": str(collection_dir),
            "doc_count": doc_count,
            "metadata": metadata,
        })

    return collections


@click.group()
def rag():
    """Manage RAG (Retrieval-Augmented Generation) systems."""
    pass


@rag.command()
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list(ctx, json_output):
    """List all RAG collections."""
    collections = get_collections()

    if not collections:
        console.print("[yellow]No RAG collections found.[/yellow]")
        console.print("[dim]Create a collection with: ailang rag index <dir>[/dim]")
        return

    if json_output:
        click.echo(json.dumps(collections, indent=2))
    else:
        table = Table(title="RAG Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Documents", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Description", style="dim")

        for collection in collections:
            created = collection["metadata"].get("created", "Unknown")
            description = collection["metadata"].get("description", "-")

            table.add_row(
                collection["name"],
                str(collection["doc_count"]),
                created,
                description[:50] + "..." if len(description) > 50 else description,
            )

        console.print(table)
        console.print(f"\n[cyan]Total collections: {len(collections)}[/cyan]")


@rag.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--name", help="Collection name (defaults to directory name)")
@click.option("--description", help="Collection description")
@click.option("--extensions", default=".txt,.md,.py,.js,.ts", help="File extensions to index")
@click.option("--exclude", help="Patterns to exclude (comma-separated)")
@click.pass_context
def index(ctx, directory, name, description, extensions, exclude):
    """
    Index documents from a directory into a RAG collection.

    \b
    Examples:
      ailang rag index ./docs --name my-docs
      ailang rag index ./code --extensions .py,.js --exclude test,__pycache__
    """
    source_dir = Path(directory).resolve()

    if not source_dir.is_dir():
        console.print(f"[red]Error: {directory} is not a directory[/red]")
        raise click.Abort()

    collection_name = name or source_dir.name
    collection_path = RAG_DATA_DIR / collection_name

    # Check if collection exists
    if collection_path.exists():
        console.print(f"[yellow]Collection '{collection_name}' already exists.[/yellow]")
        if not click.confirm("Overwrite?"):
            return
        shutil.rmtree(collection_path)

    collection_path.mkdir(parents=True, exist_ok=True)

    # Parse extensions and exclusions
    ext_list = [e.strip() for e in extensions.split(",")]
    exclude_list = [e.strip() for e in exclude.split(",")] if exclude else []

    console.print(f"[cyan]Indexing directory: {source_dir}[/cyan]")
    console.print(f"[cyan]Extensions: {', '.join(ext_list)}[/cyan]")
    if exclude_list:
        console.print(f"[cyan]Excluding: {', '.join(exclude_list)}[/cyan]")
    console.print()

    # Scan for files
    files_to_index = []
    for ext in ext_list:
        for file in source_dir.rglob(f"*{ext}"):
            # Check exclusions
            if any(excl in str(file) for excl in exclude_list):
                continue
            files_to_index.append(file)

    if not files_to_index:
        console.print("[yellow]No files found to index.[/yellow]")
        return

    console.print(f"[green]Found {len(files_to_index)} files to index[/green]\n")

    # Index files
    indexed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(files_to_index))

        for file in files_to_index:
            try:
                # Read file content
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Create document
                doc = {
                    "path": str(file.relative_to(source_dir)),
                    "full_path": str(file),
                    "content": content,
                    "size": len(content),
                    "extension": file.suffix,
                }

                # Save document
                doc_file = collection_path / f"{file.stem}_{hash(str(file))}.json"
                with open(doc_file, "w") as f:
                    json.dump(doc, f, indent=2)

                indexed_count += 1

            except Exception as e:
                if ctx.obj.get("VERBOSE"):
                    console.print(f"[yellow]Warning: Failed to index {file}: {e}[/yellow]")

            progress.update(task, advance=1)

    # Save metadata
    metadata = {
        "name": collection_name,
        "description": description or f"Indexed from {source_dir}",
        "source_directory": str(source_dir),
        "created": str(Path.ctime(collection_path)),
        "document_count": indexed_count,
        "extensions": ext_list,
    }

    with open(collection_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[green]✓ Successfully indexed {indexed_count} documents[/green]")
    console.print(f"[cyan]Collection: {collection_name}[/cyan]")
    console.print(f"[dim]Location: {collection_path}[/dim]")


@rag.command()
@click.argument("collection")
@click.argument("query")
@click.option("--top-k", type=int, default=5, help="Number of results to return")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def query(ctx, collection, query, top_k, json_output):
    """
    Query a RAG collection.

    \b
    Examples:
      ailang rag query my-docs "how to install"
      ailang rag query my-docs "authentication" --top-k 10
    """
    collections = get_collections()
    matching = next((c for c in collections if c["name"] == collection), None)

    if not matching:
        console.print(f"[red]Collection '{collection}' not found.[/red]")
        console.print("\n[yellow]Available collections:[/yellow]")
        for c in collections:
            console.print(f"  {c['name']}")
        return

    collection_path = Path(matching["path"])

    console.print(f"[cyan]Querying collection '{collection}'...[/cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    # Simple keyword-based search (for demo purposes)
    # In production, use vector embeddings
    results = []

    for doc_file in collection_path.glob("*.json"):
        if doc_file.name == "metadata.json":
            continue

        try:
            with open(doc_file) as f:
                doc = json.load(f)

            # Simple scoring based on keyword matches
            content_lower = doc["content"].lower()
            query_lower = query.lower()

            score = content_lower.count(query_lower)

            if score > 0:
                results.append({
                    "path": doc["path"],
                    "score": score,
                    "snippet": doc["content"][:200],
                })
        except Exception:
            continue

    # Sort by score and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if json_output:
        click.echo(json.dumps(results, indent=2))
    else:
        console.print(f"[green]Found {len(results)} results:[/green]\n")

        for i, result in enumerate(results, 1):
            console.print(f"[cyan]{i}. {result['path']}[/cyan]")
            console.print(f"   [yellow]Score: {result['score']}[/yellow]")
            console.print(f"   [dim]{result['snippet']}...[/dim]\n")


@rag.command()
@click.argument("collection")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, collection, force):
    """Delete a RAG collection."""
    collections = get_collections()
    matching = next((c for c in collections if c["name"] == collection), None)

    if not matching:
        console.print(f"[red]Collection '{collection}' not found.[/red]")
        return

    if not force:
        console.print(f"[yellow]This will delete collection '{collection}' with {matching['doc_count']} documents.[/yellow]")
        if not click.confirm("Are you sure?"):
            return

    try:
        shutil.rmtree(matching["path"])
        console.print(f"[green]✓ Collection '{collection}' deleted successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting collection: {e}[/red]")
        raise click.Abort()
