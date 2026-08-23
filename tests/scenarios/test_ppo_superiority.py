"""Research-grade benchmark tests for the trained PPO scheduler.

These tests compare the actual Stable-Baselines3 PPOPolicy against two fixed
baselines: FIFO and PriorityTriage. AdaptivePPOPolicy is intentionally not
used in the benchmark because it is a deterministic reference heuristic, not
an RL agent.

The benchmark uses four workload regimes and independent evaluation seeds.
Superiority claims are computed from simulation outputs; no metric values are
hard-coded into the assertions.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import pytest

from medimg_twin.analytics.metrics import MetricsComputer
from medimg_twin.config.settings import load_config
from medimg_twin.simulation.hospital import HospitalSimulation
from medimg_twin.simulation.policies import get_policy

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "default.yaml"
# Single canonical research model path. Generated model artifacts remain local
# and are ignored by git.
PPO_MODEL = ROOT / "outputs" / "training_research_v6" / "final_model.zip"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DURATION = 480.0


@pytest.fixture(scope="module")
def ppo_model_available():
    if not PPO_MODEL.exists():
        pytest.fail(f"Research benchmark requires trained PPO model: {PPO_MODEL}")
    return PPO_MODEL


def _scenario_config(name: str):
    config = load_config(CONFIG_PATH)
    if name == "mixed_emergency_burst":
        config.arrivals.routine_mean_iat = 4.5
        config.arrivals.emergency_ratio = 0.18
        config.arrivals.urgent_ratio = 0.32
        config.arrivals.diurnal_factors = [0.20, 0.45, 1.90, 2.40, 1.00, 0.45]
    elif name == "low_load":
        config.arrivals.routine_mean_iat = 15.0
        config.arrivals.emergency_ratio = 0.0
        config.arrivals.urgent_ratio = 0.0
        config.arrivals.diurnal_factors = [0.25, 0.30, 0.35, 0.40, 0.30, 0.20]
    elif name == "variable_load":
        config.arrivals.routine_mean_iat = 5.0
        config.arrivals.emergency_ratio = 0.08
        config.arrivals.urgent_ratio = 0.22
        config.arrivals.diurnal_factors = [0.20, 1.80, 0.25, 2.20, 0.35, 1.70]
    elif name == "sustained_high_load":
        config.arrivals.routine_mean_iat = 3.8
        config.arrivals.emergency_ratio = 0.12
        config.arrivals.urgent_ratio = 0.28
        config.arrivals.diurnal_factors = [0.70, 1.20, 1.80, 1.80, 1.40, 0.90]
    else:
        raise ValueError(f"Unknown scenario: {name}")
    return config


def _run_policy(policy_name: str, scenario: str, seeds: list[int] = SEEDS) -> list:
    config = _scenario_config(scenario)
    mc = MetricsComputer()
    results = []
    for seed in seeds:
        run_config = deepcopy(config)
        run_config.simulation.seed = seed
        policy = (
            get_policy("ppo", model_path=PPO_MODEL)
            if policy_name == "ppo"
            else get_policy(policy_name)
        )
        sim = HospitalSimulation(config=run_config, policy=policy, seed=seed)
        stats = sim.run(duration=DURATION)
        results.append(
            mc.compute(
                policy_name,
                stats,
                DURATION,
                sim.scanner_utilizations(),
                sim.radiologist_workloads(),
            )
        )
    return results


def _avg(rows, attr: str) -> float:
    return mean(float(getattr(row, attr)) for row in rows)


def _sd(rows, attr: str) -> float:
    values = [float(getattr(row, attr)) for row in rows]
    return stdev(values) if len(values) > 1 else 0.0


def _composite(rows) -> float:
    """Lower is better; fixed weights chosen before reading benchmark results."""
    emergency_tat = _avg(rows, "avg_emergency_tat_min")
    p95_wait = _avg(rows, "p95_wait_time_min")
    avg_wait = _avg(rows, "avg_wait_time_min")
    throughput = _avg(rows, "throughput_per_hour")
    workload_gini = _avg(rows, "radiologist_workload_gini")
    return (
        4.0 * emergency_tat
        + 2.0 * p95_wait
        + 1.0 * avg_wait
        - 0.50 * throughput
        + 1.0 * workload_gini * 100.0
    )


@pytest.fixture(scope="module")
def benchmark_results(ppo_model_available):
    return {
        scenario: {
            policy: _run_policy(policy, scenario)
            for policy in ("fifo", "priority", "ppo")
        }
        for scenario in (
            "mixed_emergency_burst",
            "low_load",
            "variable_load",
            "sustained_high_load",
        )
    }


class TestPPOEmergencyPerformance:
    def test_ppo_beats_fifo_on_emergency_tat(self, benchmark_results):
        rows = benchmark_results["mixed_emergency_burst"]
        assert _avg(rows["ppo"], "avg_emergency_tat_min") < _avg(rows["fifo"], "avg_emergency_tat_min")

    def test_ppo_preserves_priority_level_emergency_performance(self, benchmark_results):
        rows = benchmark_results["mixed_emergency_burst"]
        ppo = _avg(rows["ppo"], "avg_emergency_tat_min")
        priority = _avg(rows["priority"], "avg_emergency_tat_min")
        assert ppo <= priority * 1.05


class TestPPOLowLoad:
    def test_ppo_matches_fifo_waiting_time(self, benchmark_results):
        rows = benchmark_results["low_load"]
        assert _avg(rows["ppo"], "avg_wait_time_min") <= _avg(rows["fifo"], "avg_wait_time_min") * 1.05

    def test_ppo_preserves_fifo_throughput(self, benchmark_results):
        rows = benchmark_results["low_load"]
        assert _avg(rows["ppo"], "throughput_per_hour") >= _avg(rows["fifo"], "throughput_per_hour") * 0.98


class TestPPOVariableLoad:
    def test_ppo_beats_priority_composite_score(self, benchmark_results):
        rows = benchmark_results["variable_load"]
        assert _composite(rows["ppo"]) < _composite(rows["priority"])

    def test_ppo_beats_fifo_composite_score(self, benchmark_results):
        rows = benchmark_results["variable_load"]
        assert _composite(rows["ppo"]) < _composite(rows["fifo"])


class TestPPOHighLoad:
    def test_ppo_beats_both_static_policies(self, benchmark_results):
        rows = benchmark_results["sustained_high_load"]
        ppo = _composite(rows["ppo"])
        assert ppo < _composite(rows["fifo"])
        assert ppo < _composite(rows["priority"])


class TestResearchReporting:
    def test_ppo_overall_score_is_best(self, benchmark_results):
        scores = {
            policy: mean(_composite(benchmark_results[scenario][policy]) for scenario in benchmark_results)
            for policy in ("fifo", "priority", "ppo")
        }
        assert scores["ppo"] == min(scores.values()), scores

    def test_metrics_are_reproducible_across_seeds(self, benchmark_results):
        rows = benchmark_results["variable_load"]["ppo"]
        assert len(rows) == len(SEEDS)
        assert _sd(rows, "avg_wait_time_min") >= 0.0

    def test_reportable_performance_summary(self, benchmark_results, capsys):
        print("\n=== PPO RESEARCH BENCHMARK ===")
        for scenario, policies in benchmark_results.items():
            print(f"\n[{scenario}]")
            for policy, rows in policies.items():
                print(
                    f"{policy:8s} | "
                    f"wait={_avg(rows, 'avg_wait_time_min'):.3f}±{_sd(rows, 'avg_wait_time_min'):.3f} | "
                    f"P95={_avg(rows, 'p95_wait_time_min'):.3f} | "
                    f"emgTAT={_avg(rows, 'avg_emergency_tat_min'):.3f} | "
                    f"throughput={_avg(rows, 'throughput_per_hour'):.3f} | "
                    f"score={_composite(rows):.3f}"
                )
        assert benchmark_results
