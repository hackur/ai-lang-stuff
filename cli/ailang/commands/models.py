"""Model management commands."""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Model recommendations by task type
MODEL_RECOMMENDATIONS = {
    "coding": {
        "fast": "qwen3:30b-a3b",
        "balanced": "qwen3:8b",
        "lightweight": "gemma3:4b",
    },
    "reasoning": {
        "best": "qwen3:8b",
        "fast": "gemma3:12b",
    },
    "vision": {
        "best": "qwen3-vl:8b",
    },
    "multilingual": {
        "best": "gemma3:12b",
    },
    "edge": {
        "best": "gemma3:4b",
        "minimal": "gemma3:3b",
    },
}


@click.group()
def models():
    """Manage local LLM models with Ollama."""
    pass


@models.command()
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list(ctx, json_output):
    """List all available local models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )

        if json_output:
            # Parse and output as JSON
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models_data = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        models_data.append({
                            "name": parts[0],
                            "id": parts[1],
                            "size": parts[2],
                        })
            click.echo(json.dumps(models_data, indent=2))
        else:
            # Pretty table output
            table = Table(title="Available Local Models")
            table.add_column("Model", style="cyan")
            table.add_column("ID", style="magenta")
            table.add_column("Size", style="green")
            table.add_column("Modified", style="yellow")

            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split(maxsplit=3)
                    if len(parts) >= 3:
                        table.add_row(*parts)

            console.print(table)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error: Failed to list models[/red]")
        console.print(f"[red]{e.stderr}[/red]")
        raise click.Abort()
    except FileNotFoundError:
        console.print("[red]Error: Ollama not found. Is it installed?[/red]")
        console.print("Install from: https://ollama.ai")
        raise click.Abort()


@models.command()
@click.argument("name")
@click.option("--force", is_flag=True, help="Force pull even if model exists")
@click.pass_context
def pull(ctx, name, force):
    """Pull a specific model from Ollama registry."""
    if not force:
        # Check if model already exists
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
            if name in result.stdout:
                console.print(f"[yellow]Model '{name}' already exists.[/yellow]")
                if not click.confirm("Pull anyway?"):
                    return
        except subprocess.CalledProcessError:
            pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Pulling model '{name}'...", total=None)

        try:
            process = subprocess.Popen(
                ["ollama", "pull", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            for line in process.stdout:
                progress.update(task, description=f"Pulling '{name}': {line.strip()}")

            process.wait()

            if process.returncode == 0:
                console.print(f"[green]✓ Successfully pulled model '{name}'[/green]")
            else:
                console.print(f"[red]✗ Failed to pull model '{name}'[/red]")
                console.print(f"[red]{process.stderr.read()}[/red]")
                raise click.Abort()

        except FileNotFoundError:
            console.print("[red]Error: Ollama not found. Is it installed?[/red]")
            raise click.Abort()


@models.command()
@click.argument("name")
@click.pass_context
def info(ctx, name):
    """Show detailed information about a model."""
    try:
        result = subprocess.run(
            ["ollama", "show", name],
            capture_output=True,
            text=True,
            check=True,
        )

        table = Table(title=f"Model Information: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Parse output
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                table.add_row(key.strip(), value.strip())

        console.print(table)

    except subprocess.CalledProcessError:
        console.print(f"[red]Error: Model '{name}' not found[/red]")
        console.print(f"[yellow]Tip: Run 'ailang models pull {name}' to download it[/yellow]")
        raise click.Abort()
    except FileNotFoundError:
        console.print("[red]Error: Ollama not found. Is it installed?[/red]")
        raise click.Abort()


@models.command()
@click.argument("name")
@click.option("--iterations", "-n", default=5, help="Number of test iterations")
@click.pass_context
def benchmark(ctx, name, iterations):
    """Benchmark a model's performance."""
    console.print(f"[cyan]Benchmarking model '{name}' with {iterations} iterations...[/cyan]")

    test_prompt = "Explain quantum computing in one sentence."
    times = []

    with Progress(console=console) as progress:
        task = progress.add_task(f"Running benchmark...", total=iterations)

        for i in range(iterations):
            start_time = time.time()

            try:
                subprocess.run(
                    ["ollama", "run", name, test_prompt],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                elapsed = time.time() - start_time
                times.append(elapsed)

            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error during benchmark iteration {i+1}[/red]")
                raise click.Abort()

            progress.update(task, advance=1)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    table = Table(title=f"Benchmark Results: {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Average Time", f"{avg_time:.2f}s")
    table.add_row("Min Time", f"{min_time:.2f}s")
    table.add_row("Max Time", f"{max_time:.2f}s")
    table.add_row("Iterations", str(iterations))

    console.print(table)


@models.command()
@click.argument("task", type=click.Choice(list(MODEL_RECOMMENDATIONS.keys())))
@click.option("--variant", help="Specific variant (e.g., 'fast', 'best', 'lightweight')")
@click.pass_context
def recommend(ctx, task, variant):
    """Get model recommendations for specific tasks."""
    recommendations = MODEL_RECOMMENDATIONS.get(task, {})

    if not recommendations:
        console.print(f"[red]No recommendations available for task '{task}'[/red]")
        return

    table = Table(title=f"Recommended Models for: {task}")
    table.add_column("Variant", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")

    # Check which models are already pulled
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        available_models = result.stdout
    except subprocess.CalledProcessError:
        available_models = ""

    if variant:
        if variant in recommendations:
            model = recommendations[variant]
            status = "✓ Installed" if model in available_models else "✗ Not installed"
            table.add_row(variant, model, status)
        else:
            console.print(f"[red]Variant '{variant}' not available for task '{task}'[/red]")
            console.print(f"[yellow]Available variants: {', '.join(recommendations.keys())}[/yellow]")
            return
    else:
        for var, model in recommendations.items():
            status = "✓ Installed" if model in available_models else "✗ Not installed"
            table.add_row(var, model, status)

    console.print(table)

    # Show pull commands for missing models
    missing = [m for m in recommendations.values() if m not in available_models]
    if missing:
        console.print("\n[yellow]To install missing models:[/yellow]")
        for model in missing:
            console.print(f"  ailang models pull {model}")
