from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from medimg_twin.config.settings import load_config
from medimg_twin.data_generation.generator import SyntheticDataGenerator

app = typer.Typer(help='Generate synthetic medical imaging patient dataset')
console = Console()

@app.command()
def main(
    n_patients: Optional[int] = typer.Option(None, '--n-patients', '-n', help='Number of patient encounters to generate (default: from config)'),
    seed: int = typer.Option(42, '--seed', '-s', help='Random seed for reproducibility'),
    output_dir: Optional[Path] = typer.Option(None, '--output-dir', '-o', help='Output directory for Parquet files'),
    config_path: Optional[Path] = typer.Option(None, '--config', '-c', help='Path to YAML config file'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Enable verbose logging'),
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    config = load_config(config_path)
    
    console.print(Panel.fit(
        f"Output Dir: {output_dir or config.paths.output_dir}\nSeed: {seed}",
        title="Medical Imaging Digital Twin",
        subtitle="Data Generator Config"
    ))
    
    generator = SyntheticDataGenerator(config)
    output_files = generator.generate(n_patients=n_patients, output_dir=output_dir, show_progress=True)
    
    table = Table(title="Generated Files")
    table.add_column("File Path", style="cyan")
    table.add_column("Size (MB)", justify="right", style="green")
    
    for file_path in output_files.values():
        if file_path.exists():
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            table.add_row(str(file_path), f"{size_mb:.2f}")
    
    console.print(table)
    typer.echo("Data generation completed successfully.")

if __name__ == "__main__":
    app()
