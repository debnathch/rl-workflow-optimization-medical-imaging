"""CLI script to run policy comparison experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medimg_twin.analytics.metrics import MetricsComputer, PolicyMetrics
from medimg_twin.analytics.reporting import ReportGenerator
from medimg_twin.config.settings import load_config
from medimg_twin.simulation.hospital import HospitalSimulation
from medimg_twin.simulation.policies import get_policy

app = typer.Typer(help="Run policy comparison experiments")
console = Console()


@app.command()
def main(
    policy: str = typer.Option(
        "all",
        "--policy", "-p",
        help="Policy to run: fifo, priority, ppo, or all",
    ),
    duration: float = typer.Option(
        480.0,
        "--duration", "-d",
        help="Simulation duration in minutes (default: 480 = 8 hours)",
    ),
    n_runs: int = typer.Option(
        5,
        "--n-runs", "-r",
        help="Number of independent simulation runs per policy",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed"),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path", "-m",
        help="Path to trained PPO model .zip (required for ppo policy)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for figures and CSVs",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML config file",
    ),
    fast: bool = typer.Option(False, "--fast", help="Run 1 run per policy for quick smoke testing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run scheduling policy comparison experiments and generate analytics reports."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if fast:
        n_runs = 1
        console.print("[yellow]Fast mode: 1 run per policy[/yellow]")

    # Load config
    config = load_config(config_path)
    config.simulation.duration_minutes = duration

    # Determine output directory
    effective_output_dir = output_dir or Path(config.analytics.output_dir)
    figures_dir = effective_output_dir.parent / "figures"
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Determine policies to run
    if policy == "all":
        policies_to_run = ["fifo", "priority"]
        if model_path:
            policies_to_run.append("ppo")
    else:
        policies_to_run = [policy]

    console.print(Panel(
        f"[bold blue]Policy Comparison Experiment[/bold blue]\n"
        f"Policies: {policies_to_run}\n"
        f"Duration: {duration:.0f} min | Runs per policy: {n_runs} | Seed: {seed}",
        title="🏥 Medical Imaging Digital Twin",
    ))

    # Run experiments
    all_metrics: list[PolicyMetrics] = []
    all_stats: dict[str, object] = {}
    all_utils: dict[str, dict[str, float]] = {}
    mc = MetricsComputer()

    for p_name in policies_to_run:
        console.print(f"\n[cyan]Running policy: {p_name.upper()}[/cyan] ({n_runs} runs)")

        run_stats_list = []
        run_utils_list: list[dict[str, float]] = []
        run_rad_list: list[dict] = []

        for i in range(n_runs):
            current_seed = seed + i
            console.print(f"  Run {i+1}/{n_runs} (seed={current_seed})...", end="")

            if p_name == "ppo" and model_path:
                pol = get_policy(p_name, model_path=model_path)
            else:
                pol = get_policy(p_name)

            sim = HospitalSimulation(config=config, seed=current_seed)
            stats = sim.run(duration=duration)

            scanner_utils = sim.scanner_utilizations()
            rad_workloads = sim.radiologist_workloads()

            run_stats_list.append(stats)
            run_utils_list.append(scanner_utils)
            run_rad_list.append(rad_workloads)
            console.print(f" ✓ completed={stats.n_completed}")

        # Average metrics across runs
        # For simplicity, compute metrics per run then average the PolicyMetrics
        run_metrics: list[PolicyMetrics] = []
        for stats, scanner_utils, rad_workloads in zip(run_stats_list, run_utils_list, run_rad_list):
            m = mc.compute(
                policy_name=p_name,
                stats=stats,
                sim_duration=duration,
                scanner_utils=scanner_utils,
                rad_workloads=rad_workloads,
            )
            run_metrics.append(m)

        # Average across runs
        avg_metrics = PolicyMetrics(
            policy_name=p_name,
            n_patients_completed=int(np.mean([m.n_patients_completed for m in run_metrics])),
            throughput_per_hour=float(np.mean([m.throughput_per_hour for m in run_metrics])),
            avg_wait_time_min=float(np.mean([m.avg_wait_time_min for m in run_metrics])),
            std_wait_time_min=float(np.mean([m.std_wait_time_min for m in run_metrics])),
            p50_wait_time_min=float(np.mean([m.p50_wait_time_min for m in run_metrics])),
            p95_wait_time_min=float(np.mean([m.p95_wait_time_min for m in run_metrics])),
            p99_wait_time_min=float(np.mean([m.p99_wait_time_min for m in run_metrics])),
            avg_emergency_tat_min=float(np.mean([m.avg_emergency_tat_min for m in run_metrics])),
            p95_emergency_tat_min=float(np.mean([m.p95_emergency_tat_min for m in run_metrics])),
            avg_total_tat_min=float(np.mean([m.avg_total_tat_min for m in run_metrics])),
            ct_utilization=float(np.mean([m.ct_utilization for m in run_metrics])),
            mri_utilization=float(np.mean([m.mri_utilization for m in run_metrics])),
            xray_utilization=float(np.mean([m.xray_utilization for m in run_metrics])),
            avg_scanner_utilization=float(np.mean([m.avg_scanner_utilization for m in run_metrics])),
            radiologist_workload_std=float(np.mean([m.radiologist_workload_std for m in run_metrics])),
            radiologist_workload_gini=float(np.mean([m.radiologist_workload_gini for m in run_metrics])),
            avg_queue_length=float(np.mean([m.avg_queue_length for m in run_metrics])),
            max_queue_length=float(np.mean([m.max_queue_length for m in run_metrics])),
            sim_duration_min=duration,
        )
        all_metrics.append(avg_metrics)
        all_stats[p_name] = run_stats_list[-1]  # Last run for figures
        all_utils[p_name] = run_utils_list[-1]

    # Print comparison table
    table = Table(title="Policy Comparison Results", show_header=True, header_style="bold magenta")
    table.add_column("Policy", style="cyan", min_width=12)
    table.add_column("Avg Wait (min)", justify="right")
    table.add_column("P95 Wait (min)", justify="right")
    table.add_column("Emg TAT (min)", justify="right")
    table.add_column("Avg Util %", justify="right")
    table.add_column("Throughput /hr", justify="right")
    table.add_column("Workload Std", justify="right")
    table.add_column("Completed", justify="right")

    for m in all_metrics:
        table.add_row(
            m.policy_name.upper(),
            f"{m.avg_wait_time_min:.1f}",
            f"{m.p95_wait_time_min:.1f}",
            f"{m.avg_emergency_tat_min:.1f}",
            f"{m.avg_scanner_utilization*100:.1f}%",
            f"{m.throughput_per_hour:.2f}",
            f"{m.radiologist_workload_std:.3f}",
            str(m.n_patients_completed),
        )

    console.print("\n")
    console.print(table)

    # Generate reports
    rg = ReportGenerator(
        output_dir=effective_output_dir,
        figures_dir=figures_dir,
        dpi=config.analytics.figure_dpi,
        palette=config.analytics.color_palette,
    )

    csv_path = rg.export_csv_summary(all_metrics)
    console.print(f"\n[green]✓ CSV summary:[/green] {csv_path}")

    try:
        fig_path = rg.generate_policy_comparison(all_metrics)
        console.print(f"[green]✓ Comparison figure:[/green] {fig_path}")
    except Exception as e:
        console.print(f"[yellow]Figure generation skipped: {e}[/yellow]")

    try:
        util_fig = rg.generate_scanner_utilization_heatmap(all_utils)
        console.print(f"[green]✓ Utilization heatmap:[/green] {util_fig}")
    except Exception as e:
        console.print(f"[yellow]Heatmap skipped: {e}[/yellow]")

    console.print("\n[bold green]Experiment complete![/bold green]")


if __name__ == "__main__":
    app()
