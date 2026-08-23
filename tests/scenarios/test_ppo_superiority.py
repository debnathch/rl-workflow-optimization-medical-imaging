"""
Scenario-level acceptance tests for PPO superiority.

The purpose of these tests is NOT to make PPO look better by changing the
assertions after observing a result.  Each scenario defines a workload where
an adaptive policy should have a defensible advantage over a fixed heuristic.

Required PPO behaviours:
1. Mixed load + emergency burst -> aggressive emergency-first scheduling.
2. Low load + no emergencies -> fall back to FIFO-like throughput behaviour.
3. Variable load -> change scheduling behaviour as workload pressure changes.
4. Repeated load shifts -> outperform static Priority on the aggregate
   multi-objective score rather than only on one cherry-picked KPI.

If a test fails, the policy/training behaviour should be fixed; the test should
not be weakened simply to make the PPO result pass.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean

import pytest

from medimg_twin.analytics.metrics import MetricsComputer
from medimg_twin.config.settings import load_config
from medimg_twin.simulation.hospital import HospitalSimulation
from medimg_twin.simulation.policies import get_policy


CONFIG_PATH = Path(__file__).parents[2] / "config" / "default.yaml"
SEEDS = [42, 43, 44, 45, 46]
DURATION = 480.0


def _scenario_config(name: str):
    """Build deterministic workload variants from the validated base config."""
    config = load_config(CONFIG_PATH)

    if name == "emergency_burst":
        # Mixed load with a concentrated emergency burst during the middle
        # of the shift.  PPO should react more aggressively than static
        # Priority when emergency pressure rises.
        config.arrivals.routine_mean_iat = 3.0
        config.arrivals.emergency_ratio = 0.20
        config.arrivals.urgent_ratio = 0.45
        config.arrivals.diurnal_factors = [0.25, 0.50, 2.20, 1.80, 0.60, 0.30]
    elif name == "low_load":
        # No emergencies and low arrival pressure.  FIFO should be the
        # natural baseline; PPO should learn to avoid unnecessary priority
        # scheduling overhead and behave FIFO-like.
        config.arrivals.routine_mean_iat = 15.0
        config.arrivals.emergency_ratio = 0.0
        config.arrivals.urgent_ratio = 0.05
        config.arrivals.diurnal_factors = [0.25, 0.30, 0.35, 0.40, 0.30, 0.20]
    elif name == "variable_load":
        # Alternating low/high demand blocks.  Static Priority sees the same
        # rules in every block; PPO should adapt its action to queue pressure.
        config.arrivals.routine_mean_iat = 5.0
        config.arrivals.emergency_ratio = 0.08
        config.arrivals.urgent_ratio = 0.22
        config.arrivals.diurnal_factors = [0.20, 1.80, 0.30, 2.00, 0.40, 1.60]
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return config


def _run_policy(policy_name: str, scenario: str, seeds: list[int] = SEEDS) -> list:
    """Run one policy over all seeds for one workload scenario."""
    config = _scenario_config(scenario)
    mc = MetricsComputer()
    results = []
    for seed in seeds:
        # Avoid mutating the shared fixture/config between replications.
        run_config = deepcopy(config)
        run_config.simulation.seed = seed
        policy = get_policy(policy_name)
        sim = HospitalSimulation(config=run_config, policy=policy, seed=seed)
        stats = sim.run(duration=DURATION)
        metrics = mc.compute(
            policy_name,
            stats,
            DURATION,
            sim.scanner_utilizations(),
            sim.radiologist_workloads(),
        )
        results.append(metrics)
    return results


def _avg(metrics_list, attr: str) -> float:
    return mean(getattr(m, attr) for m in metrics_list)


def _normalised_improvement(baseline: float, candidate: float, lower_is_better: bool = True) -> float:
    """Return relative improvement, guarding against zero denominators."""
    if abs(baseline) < 1e-9:
        return 0.0
    if lower_is_better:
        return (baseline - candidate) / baseline
    return (candidate - baseline) / baseline


def _multi_objective_score(metrics_list) -> float:
    """Lower is better; combines the clinically important queue KPIs.

    Emergency TAT receives the largest weight because clinical urgency must
    dominate.  P95 wait and average wait capture queue/fairness behaviour.
    Throughput is rewarded as a small negative term.
    """
    emergency_tat = _avg(metrics_list, "avg_emergency_tat_min")
    p95_wait = _avg(metrics_list, "p95_wait_time_min")
    avg_wait = _avg(metrics_list, "avg_wait_time_min")
    throughput = _avg(metrics_list, "throughput_per_hour")
    return (
        3.0 * emergency_tat
        + 1.5 * p95_wait
        + 1.0 * avg_wait
        - 0.30 * throughput
    )


@pytest.fixture(scope="module")
def emergency_burst_results():
    return {
        policy: _run_policy(policy, "emergency_burst")
        for policy in ("fifo", "priority", "adaptive_ppo")
    }


@pytest.fixture(scope="module")
def low_load_results():
    return {
        policy: _run_policy(policy, "low_load")
        for policy in ("fifo", "priority", "adaptive_ppo")
    }


@pytest.fixture(scope="module")
def variable_load_results():
    return {
        policy: _run_policy(policy, "variable_load")
        for policy in ("fifo", "priority", "adaptive_ppo")
    }


class TestMixedLoadEmergencyBurst:
    """PPO must exploit the emergency-first action during burst pressure."""

    def test_ppo_emergency_tat_beats_priority(self, emergency_burst_results):
        priority = _avg(emergency_burst_results["priority"], "avg_emergency_tat_min")
        ppo = _avg(emergency_burst_results["adaptive_ppo"], "avg_emergency_tat_min")
        assert ppo < priority, (
            f"Emergency-burst requirement violated: PPO={ppo:.2f} min, "
            f"Priority={priority:.2f} min. PPO must react more aggressively "
            "when emergency arrivals burst."
        )

    def test_ppo_emergency_tat_beats_fifo(self, emergency_burst_results):
        fifo = _avg(emergency_burst_results["fifo"], "avg_emergency_tat_min")
        ppo = _avg(emergency_burst_results["adaptive_ppo"], "avg_emergency_tat_min")
        assert ppo < fifo, (
            f"PPO emergency TAT ({ppo:.2f}) must beat FIFO ({fifo:.2f}) "
            "during an emergency burst."
        )


class TestLowLoadNoEmergencies:
    """PPO should exploit FIFO-like behaviour when priority pressure is absent."""

    def test_ppo_matches_or_beats_fifo_wait(self, low_load_results):
        fifo = _avg(low_load_results["fifo"], "avg_wait_time_min")
        ppo = _avg(low_load_results["adaptive_ppo"], "avg_wait_time_min")
        assert ppo <= fifo, (
            f"Low-load requirement violated: PPO avg wait={ppo:.2f}, "
            f"FIFO={fifo:.2f}. With no emergencies, PPO should fall back "
            "toward FIFO behaviour rather than imposing priority ordering."
        )

    def test_ppo_throughput_not_below_fifo(self, low_load_results):
        fifo = _avg(low_load_results["fifo"], "throughput_per_hour")
        ppo = _avg(low_load_results["adaptive_ppo"], "throughput_per_hour")
        assert ppo >= fifo * 0.99, (
            f"PPO throughput={ppo:.2f}/h fell below FIFO={fifo:.2f}/h "
            "in the low-load/no-emergency scenario."
        )


class TestVariableLoadAdaptation:
    """PPO must improve the aggregate objective under changing demand."""

    def test_ppo_beats_static_priority(self, variable_load_results):
        priority_score = _multi_objective_score(variable_load_results["priority"])
        ppo_score = _multi_objective_score(variable_load_results["adaptive_ppo"])
        assert ppo_score < priority_score, (
            f"Variable-load requirement violated: PPO score={ppo_score:.2f}, "
            f"Priority score={priority_score:.2f}. PPO should adapt to changing "
            "queue pressure instead of applying one fixed rule."
        )

    def test_ppo_beats_both_static_baselines(self, variable_load_results):
        fifo_score = _multi_objective_score(variable_load_results["fifo"])
        priority_score = _multi_objective_score(variable_load_results["priority"])
        ppo_score = _multi_objective_score(variable_load_results["adaptive_ppo"])
        assert ppo_score < fifo_score and ppo_score < priority_score, (
            "PPO must be the best policy on the aggregate variable-load "
            f"objective: FIFO={fifo_score:.2f}, Priority={priority_score:.2f}, "
            f"PPO={ppo_score:.2f}."
        )


class TestRepeatedLoadShifts:
    """Four independent shifts protect against a single favourable seed."""

    @pytest.mark.parametrize(
        "scenario",
        ["emergency_burst", "low_load", "variable_load"],
    )
    def test_ppo_is_not_worse_on_composite_objective(self, scenario):
        fifo = _run_policy("fifo", scenario)
        priority = _run_policy("priority", scenario)
        ppo = _run_policy("adaptive_ppo", scenario)

        fifo_score = _multi_objective_score(fifo)
        priority_score = _multi_objective_score(priority)
        ppo_score = _multi_objective_score(ppo)
        best_static = min(fifo_score, priority_score)

        # PPO must beat the better static baseline, not merely the worse one.
        assert ppo_score < best_static, (
            f"PPO is not best in {scenario}: FIFO={fifo_score:.2f}, "
            f"Priority={priority_score:.2f}, PPO={ppo_score:.2f}"
        )


class TestPPOClaimGuardrails:
    """Prevent tests from silently accepting a weaker PPO result."""

    def test_emergency_improvement_is_measurable(self, emergency_burst_results):
        priority = _avg(emergency_burst_results["priority"], "avg_emergency_tat_min")
        ppo = _avg(emergency_burst_results["adaptive_ppo"], "avg_emergency_tat_min")
        improvement = _normalised_improvement(priority, ppo)
        assert improvement > 0.0, (
            f"PPO emergency improvement must be positive; observed {improvement:.2%}."
        )

    def test_variable_load_p95_wait_improves(self, variable_load_results):
        priority = _avg(variable_load_results["priority"], "p95_wait_time_min")
        ppo = _avg(variable_load_results["adaptive_ppo"], "p95_wait_time_min")
        assert ppo < priority, (
            f"PPO P95 wait={ppo:.2f} must be below Priority={priority:.2f} "
            "under variable load."
        )
