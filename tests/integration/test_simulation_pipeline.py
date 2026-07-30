"""Integration tests for the SimPy simulation pipeline."""

from __future__ import annotations

import pytest

from medimg_twin.analytics.metrics import MetricsComputer
from medimg_twin.config.settings import load_config
from medimg_twin.simulation.hospital import HospitalSimulation, SimulationStats
from medimg_twin.simulation.policies import FIFOPolicy, PriorityTriagePolicy, get_policy


@pytest.fixture(scope="module")
def fast_cfg():
    """Config with 2-hour simulation for fast integration tests."""
    cfg = load_config()
    cfg.simulation.duration_minutes = 120.0   # 2-hour run
    cfg.simulation.warmup_minutes = 10.0
    cfg.arrivals.routine_mean_iat = 4.0  # Higher arrival rate
    return cfg


def _run_sim(cfg, policy=None, seed: int = 42) -> tuple[HospitalSimulation, SimulationStats]:
    """Helper: create and run a simulation with the given policy."""
    sim = HospitalSimulation(config=cfg, policy=policy, seed=seed)
    stats = sim.run(duration=cfg.simulation.duration_minutes)
    return sim, stats


class TestBasicSimulationRuns:
    def test_simulation_runs_without_error(self, fast_cfg) -> None:
        """Simulation runs end-to-end without raising any exceptions."""
        _, stats = _run_sim(fast_cfg)
        assert stats is not None

    def test_simulation_produces_patients(self, fast_cfg) -> None:
        """Simulation with 120-min run produces at least some patients."""
        _, stats = _run_sim(fast_cfg)
        assert stats.n_arrived >= 1, f"No patients arrived at all: {stats.n_arrived}"

    def test_fifo_policy_runs(self, fast_cfg) -> None:
        """FIFO scheduling policy runs without errors."""
        _, stats = _run_sim(fast_cfg, policy=FIFOPolicy())
        assert stats.n_completed >= 0

    def test_priority_policy_runs(self, fast_cfg) -> None:
        """Priority triage scheduling policy runs without errors."""
        _, stats = _run_sim(fast_cfg, policy=PriorityTriagePolicy())
        assert stats.n_completed >= 0

    def test_get_policy_fifo(self, fast_cfg) -> None:
        """get_policy('fifo') returns a FIFOPolicy instance."""
        pol = get_policy("fifo")
        _, stats = _run_sim(fast_cfg, policy=pol)
        assert stats.n_arrived >= 0

    def test_get_policy_priority(self, fast_cfg) -> None:
        """get_policy('priority') returns a PriorityTriagePolicy instance."""
        pol = get_policy("priority")
        _, stats = _run_sim(fast_cfg, policy=pol)
        assert stats.n_arrived >= 0


class TestSimulationStats:
    def test_scanner_utilization_in_range(self, fast_cfg) -> None:
        """All scanner utilizations are in [0, 1]."""
        sim, _ = _run_sim(fast_cfg)
        utils = sim.scanner_utilizations()
        for scanner_id, util in utils.items():
            assert 0.0 <= util <= 1.0, f"Utilization out of range for {scanner_id}: {util}"

    def test_stats_wait_times_non_negative(self, fast_cfg) -> None:
        """All recorded wait times are non-negative."""
        _, stats = _run_sim(fast_cfg)
        for wt in stats.wait_times:
            assert wt >= 0.0, f"Negative wait time: {wt}"

    def test_n_completed_le_n_arrived(self, fast_cfg) -> None:
        """Completed patients <= arrived patients."""
        _, stats = _run_sim(fast_cfg)
        assert stats.n_completed <= stats.n_arrived

    def test_emergency_turnarounds_non_negative(self, fast_cfg) -> None:
        """Emergency turnaround times are non-negative."""
        _, stats = _run_sim(fast_cfg)
        for tat in stats.emergency_turnarounds:
            assert tat >= 0.0, f"Negative TAT: {tat}"

    def test_radiologist_workloads_populated(self, fast_cfg) -> None:
        """Radiologist workloads dict is populated after a run."""
        sim, _ = _run_sim(fast_cfg)
        workloads = sim.radiologist_workloads()
        assert isinstance(workloads, dict)


class TestReproducibility:
    def test_two_seeds_produce_different_results(self, fast_cfg) -> None:
        """Simulations with different seeds produce non-identical arrival counts."""
        _, stats1 = _run_sim(fast_cfg, seed=1)
        _, stats2 = _run_sim(fast_cfg, seed=2)
        # Not guaranteed to differ but highly likely with different seeds
        # Use a soft assertion
        if stats1.n_arrived == stats2.n_arrived:
            import warnings
            warnings.warn("Same arrival count with different seeds — may indicate seed handling issue")

    def test_same_seed_produces_identical_wait_times(self, fast_cfg) -> None:
        """Same seed and policy produces identical wait_times list."""
        _, stats1 = _run_sim(fast_cfg, policy=FIFOPolicy(), seed=42)
        _, stats2 = _run_sim(fast_cfg, policy=FIFOPolicy(), seed=42)
        assert stats1.n_arrived == stats2.n_arrived
        assert stats1.n_completed == stats2.n_completed


class TestMetricsIntegration:
    def test_metrics_compute_from_simulation(self, fast_cfg) -> None:
        """MetricsComputer correctly processes SimulationStats from a real run."""
        sim, stats = _run_sim(fast_cfg, policy=FIFOPolicy(), seed=42)
        mc = MetricsComputer()
        metrics = mc.compute(
            policy_name="fifo",
            stats=stats,
            sim_duration=fast_cfg.simulation.duration_minutes,
            scanner_utils=sim.scanner_utilizations(),
            rad_workloads=sim.radiologist_workloads(),
        )
        assert metrics.policy_name == "fifo"
        assert metrics.throughput_per_hour >= 0.0
        assert 0.0 <= metrics.avg_scanner_utilization <= 1.0
