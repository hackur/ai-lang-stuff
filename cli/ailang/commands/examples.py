"""Example management commands."""

import json
import subprocess
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

console = Console()

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples"


def get_examples() -> List[dict]:
    """Scan examples directory and return list of examples."""
    examples = []

    if not EXAMPLES_DIR.exists():
        return examples

    for category_dir in sorted(EXAMPLES_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue

        category = category_dir.name

        for example_file in sorted(category_dir.glob("*.py")):
            # Read docstring for description
            description = "No description available"
            try:
                with open(example_file) as f:
                    content = f.read()
                    if '"""' in content:
                        start = content.find('"""') + 3
                        end = content.find('"""', start)
                        if end > start:
                            description = content[start:end].strip().split("\n")[0]
            except Exception:
                pass

            examples.append({
                "category": category,
                "name": example_file.stem,
                "path": str(example_file),
                "description": description,
            })

    return examples


@click.group()
def examples():
    """Run and manage example projects."""
    pass


@examples.command()
@click.option("--category", help="Filter by category (e.g., '01-foundation')")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list(ctx, category, json_output):
    """Show all available examples."""
    all_examples = get_examples()

    if category:
        all_examples = [e for e in all_examples if e["category"] == category]

    if not all_examples:
        console.print("[yellow]No examples found.[/yellow]")
        return

    if json_output:
        click.echo(json.dumps(all_examples, indent=2))
    else:
        # Group by category
        tree = Tree("[bold cyan]Available Examples[/bold cyan]")

        current_category = None
        category_node = None

        for example in all_examples:
            if example["category"] != current_category:
                current_category = example["category"]
                category_node = tree.add(f"[yellow]{current_category}[/yellow]")

            category_node.add(
                f"[green]{example['name']}[/green]: {example['description']}"
            )

        console.print(tree)
        console.print(f"\n[cyan]Total examples: {len(all_examples)}[/cyan]")
        console.print("\n[yellow]Run an example:[/yellow]")
        console.print("  ailang examples run <category>/<name>")


@examples.command()
@click.argument("name")
@click.option("--args", help="Additional arguments to pass to the example")
@click.pass_context
def run(ctx, name, args):
    """
    Run a specific example.

    \b
    Examples:
      ailang examples run 01-foundation/hello_ollama
      ailang examples run 02-mcp/filesystem_integration --args "--path /tmp"
    """
    all_examples = get_examples()

    # Find the example
    matching = None
    for example in all_examples:
        full_name = f"{example['category']}/{example['name']}"
        if full_name == name or example['name'] == name:
            matching = example
            break

    if not matching:
        console.print(f"[red]Example '{name}' not found.[/red]")
        console.print("\n[yellow]Available examples:[/yellow]")
        for example in all_examples[:5]:
            console.print(f"  {example['category']}/{example['name']}")
        if len(all_examples) > 5:
            console.print(f"  ... and {len(all_examples) - 5} more")
        return

    console.print(f"[cyan]Running example: {matching['category']}/{matching['name']}[/cyan]")
    console.print(f"[dim]{matching['description']}[/dim]\n")

    try:
        cmd = ["python", matching["path"]]
        if args:
            cmd.extend(args.split())

        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            console.print(f"\n[green]✓ Example completed successfully[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]✗ Example failed with exit code {e.returncode}[/red]")
        raise click.Abort()
    except FileNotFoundError:
        console.print("[red]Error: Python not found in PATH[/red]")
        raise click.Abort()


@examples.command()
@click.option("--category", help="Validate specific category only")
@click.option("--fast", is_flag=True, help="Skip long-running validations")
@click.pass_context
def validate(ctx, category, fast):
    """Validate all examples can be imported and run."""
    all_examples = get_examples()

    if category:
        all_examples = [e for e in all_examples if e["category"] == category]

    if not all_examples:
        console.print("[yellow]No examples to validate.[/yellow]")
        return

    console.print(f"[cyan]Validating {len(all_examples)} examples...[/cyan]\n")

    results = {"passed": [], "failed": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=len(all_examples))

        for example in all_examples:
            example_name = f"{example['category']}/{example['name']}"
            progress.update(task, description=f"Validating {example_name}")

            try:
                # Try to import the module
                result = subprocess.run(
                    ["python", "-c", f"import sys; sys.path.insert(0, '{EXAMPLES_DIR}'); exec(open('{example['path']}').read())"],
                    capture_output=True,
                    text=True,
                    timeout=5 if fast else 30,
                )

                if result.returncode == 0:
                    results["passed"].append(example_name)
                else:
                    results["failed"].append({
                        "name": example_name,
                        "error": result.stderr[:200],
                    })

            except subprocess.TimeoutExpired:
                results["failed"].append({
                    "name": example_name,
                    "error": "Timeout exceeded",
                })
            except Exception as e:
                results["failed"].append({
                    "name": example_name,
                    "error": str(e)[:200],
                })

            progress.update(task, advance=1)

    # Display results
    console.print("\n[bold]Validation Results:[/bold]\n")

    if results["passed"]:
        console.print(f"[green]✓ Passed: {len(results['passed'])}[/green]")
        for name in results["passed"]:
            console.print(f"  [dim]{name}[/dim]")

    if results["failed"]:
        console.print(f"\n[red]✗ Failed: {len(results['failed'])}[/red]")
        for item in results["failed"]:
            console.print(f"  [red]{item['name']}[/red]")
            console.print(f"    [dim]{item['error']}[/dim]")

    # Summary
    total = len(results["passed"]) + len(results["failed"])
    pass_rate = (len(results["passed"]) / total * 100) if total > 0 else 0

    console.print(f"\n[cyan]Pass rate: {pass_rate:.1f}% ({len(results['passed'])}/{total})[/cyan]")

    if results["failed"]:
        raise click.Abort()
