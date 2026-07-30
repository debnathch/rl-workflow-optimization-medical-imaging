from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help='Launch the Streamlit dashboard')
console = Console()

@app.command()
def main(
    port: int = typer.Option(8501, '--port', '-p'),
    config_path: Path | None = typer.Option(None, '--config', '-c'),
) -> None:
    dashboard_path = Path(__file__).parents[1] / 'dashboard' / 'app.py'
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(dashboard_path), '--server.port', str(port)]
    if config_path:
        cmd += ['--', '--config', str(config_path)]
    console.print(f'[green]Launching dashboard at http://localhost:{port}[/green]')
    subprocess.run(cmd)

if __name__ == "__main__":
    app()
