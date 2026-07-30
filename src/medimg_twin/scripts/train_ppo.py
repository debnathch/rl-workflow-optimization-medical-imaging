from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel

from medimg_twin.config.settings import load_config
from medimg_twin.training.trainer import PPOTrainer
from medimg_twin.analytics.reporting import ReportGenerator

app = typer.Typer(help='Train PPO agent for adaptive scheduling')
console = Console()

@app.command()
def main(
    timesteps: Optional[int] = typer.Option(None, '--timesteps', '-t', help='Total training timesteps (default: from config)'),
    seed: int = typer.Option(42, '--seed', '-s'),
    output_dir: Optional[Path] = typer.Option(None, '--output-dir', '-o'),
    config_path: Optional[Path] = typer.Option(None, '--config', '-c'),
    fast: bool = typer.Option(False, '--fast', help='Quick training with 10k timesteps'),
    evaluate: bool = typer.Option(True, '--evaluate/--no-evaluate', help='Evaluate trained model after training'),
    n_eval_episodes: int = typer.Option(10, '--n-eval', help='Number of evaluation episodes'),
) -> None:
    console.print(Panel.fit("Training PPO Agent", style="bold blue"))
    config = load_config(config_path)
    
    if fast:
        config.rl.total_timesteps = 10000
    elif timesteps is not None:
        config.rl.total_timesteps = timesteps
        
    trainer = PPOTrainer(config, output_dir=output_dir)
    model_path = trainer.train(seed=seed)
    console.print(f"Model saved to: {model_path}")
    
    if evaluate:
        results = trainer.evaluate(n_episodes=n_eval_episodes)
        console.print("Evaluation Results:", results)

if __name__ == "__main__":
    app()
