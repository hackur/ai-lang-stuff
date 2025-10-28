"""MCP server management commands."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

MCP_DIR = Path(__file__).parent.parent.parent.parent / "mcp-servers"
PID_DIR = Path.home() / ".ailang" / "mcp-pids"


def get_mcp_servers() -> List[dict]:
    """Scan mcp-servers directory and return list of servers."""
    servers = []

    if not MCP_DIR.exists():
        return servers

    # Check official servers
    official_dir = MCP_DIR / "official"
    if official_dir.exists():
        for server_dir in official_dir.iterdir():
            if server_dir.is_dir() and not server_dir.name.startswith("."):
                servers.append({
                    "name": server_dir.name,
                    "type": "official",
                    "path": str(server_dir),
                })

    # Check custom servers
    custom_dir = MCP_DIR / "custom"
    if custom_dir.exists():
        for server_dir in custom_dir.iterdir():
            if server_dir.is_dir() and not server_dir.name.startswith("."):
                servers.append({
                    "name": server_dir.name,
                    "type": "custom",
                    "path": str(server_dir),
                })

    return servers


def get_server_pid(name: str) -> Optional[int]:
    """Get PID of running server."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = PID_DIR / f"{name}.pid"

    if not pid_file.exists():
        return None

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Check if process is still running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, OSError):
        # Process doesn't exist, clean up stale PID file
        pid_file.unlink(missing_ok=True)
        return None


def save_server_pid(name: str, pid: int):
    """Save PID of running server."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = PID_DIR / f"{name}.pid"

    with open(pid_file, "w") as f:
        f.write(str(pid))


def remove_server_pid(name: str):
    """Remove PID file for server."""
    pid_file = PID_DIR / f"{name}.pid"
    pid_file.unlink(missing_ok=True)


@click.group()
def mcp():
    """Manage Model Context Protocol (MCP) servers."""
    pass


@mcp.command()
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list(ctx, json_output):
    """List all available MCP servers."""
    servers = get_mcp_servers()

    if not servers:
        console.print("[yellow]No MCP servers found.[/yellow]")
        console.print(f"[dim]Expected location: {MCP_DIR}[/dim]")
        return

    # Check running status
    for server in servers:
        pid = get_server_pid(server["name"])
        server["status"] = "running" if pid else "stopped"
        server["pid"] = pid

    if json_output:
        click.echo(json.dumps(servers, indent=2))
    else:
        table = Table(title="MCP Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("PID", style="magenta")

        for server in servers:
            status_color = "green" if server["status"] == "running" else "red"
            pid_str = str(server["pid"]) if server["pid"] else "-"

            table.add_row(
                server["name"],
                server["type"],
                f"[{status_color}]{server['status']}[/{status_color}]",
                pid_str,
            )

        console.print(table)
        console.print(f"\n[cyan]Total servers: {len(servers)}[/cyan]")


@mcp.command()
@click.argument("name")
@click.option("--port", type=int, help="Port to run server on")
@click.option("--background", "-b", is_flag=True, help="Run in background")
@click.pass_context
def start(ctx, name, port, background):
    """Start an MCP server."""
    servers = get_mcp_servers()
    matching = next((s for s in servers if s["name"] == name), None)

    if not matching:
        console.print(f"[red]Server '{name}' not found.[/red]")
        return

    # Check if already running
    pid = get_server_pid(name)
    if pid:
        console.print(f"[yellow]Server '{name}' is already running (PID: {pid})[/yellow]")
        if not click.confirm("Restart?"):
            return
        # Stop the existing server
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except ProcessLookupError:
            pass
        remove_server_pid(name)

    server_path = Path(matching["path"])

    # Find the main server file
    main_file = None
    for candidate in ["server.py", "main.py", "index.js", "server.js"]:
        if (server_path / candidate).exists():
            main_file = server_path / candidate
            break

    if not main_file:
        console.print(f"[red]No main file found in {server_path}[/red]")
        return

    # Build command
    if main_file.suffix == ".py":
        cmd = ["python", str(main_file)]
    elif main_file.suffix == ".js":
        cmd = ["node", str(main_file)]
    else:
        console.print(f"[red]Unsupported file type: {main_file.suffix}[/red]")
        return

    if port:
        cmd.extend(["--port", str(port)])

    console.print(f"[cyan]Starting MCP server '{name}'...[/cyan]")

    try:
        if background:
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=server_path,
            )
            save_server_pid(name, process.pid)
            console.print(f"[green]✓ Server started in background (PID: {process.pid})[/green]")
        else:
            # Run in foreground
            subprocess.run(cmd, cwd=server_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise click.Abort()


@mcp.command()
@click.argument("name")
@click.pass_context
def stop(ctx, name):
    """Stop a running MCP server."""
    pid = get_server_pid(name)

    if not pid:
        console.print(f"[yellow]Server '{name}' is not running.[/yellow]")
        return

    console.print(f"[cyan]Stopping server '{name}' (PID: {pid})...[/cyan]")

    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for process to terminate
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break

        remove_server_pid(name)
        console.print(f"[green]✓ Server stopped successfully[/green]")

    except ProcessLookupError:
        console.print(f"[yellow]Server process not found[/yellow]")
        remove_server_pid(name)
    except Exception as e:
        console.print(f"[red]Error stopping server: {e}[/red]")
        raise click.Abort()


@mcp.command()
@click.argument("name")
@click.option("--timeout", type=int, default=10, help="Test timeout in seconds")
@click.pass_context
def test(ctx, name, timeout):
    """Test an MCP server's functionality."""
    servers = get_mcp_servers()
    matching = next((s for s in servers if s["name"] == name), None)

    if not matching:
        console.print(f"[red]Server '{name}' not found.[/red]")
        return

    console.print(f"[cyan]Testing MCP server '{name}'...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running tests...", total=None)

        server_path = Path(matching["path"])
        test_file = server_path / "test.py"

        if not test_file.exists():
            progress.stop()
            console.print(f"[yellow]No test file found at {test_file}[/yellow]")
            console.print("[dim]Create test.py to enable testing[/dim]")
            return

        try:
            result = subprocess.run(
                ["python", str(test_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=server_path,
            )

            progress.stop()

            if result.returncode == 0:
                console.print("[green]✓ All tests passed[/green]")
                if ctx.obj.get("VERBOSE"):
                    console.print("\n[dim]Output:[/dim]")
                    console.print(result.stdout)
            else:
                console.print("[red]✗ Tests failed[/red]")
                console.print(f"\n[red]Error output:[/red]")
                console.print(result.stderr)
                raise click.Abort()

        except subprocess.TimeoutExpired:
            progress.stop()
            console.print(f"[red]✗ Tests timed out after {timeout}s[/red]")
            raise click.Abort()
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error running tests: {e}[/red]")
            raise click.Abort()
