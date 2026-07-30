"""Unit tests for MetricsComputer and PolicyMetrics."""

from __future__ import annotations

import pytest
import pandas as pd
from medimg_twin.analytics.metrics import MetricsComputer, PolicyMetrics
from medimg_twin.simulation.hospital import SimulationStats


@pytest.fixture
def sample_stats() -> SimulationStats:
    """Build a SimulationStats object with realistic test data."""
    stats = SimulationStats()
    stats.wait_times = [10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0]
    stats.emergency_turnarounds = [15.0, 20.0, 25.0]
    stats.total_turnarounds = [60.0, 80.0, 100.0, 120.0]
    stats.scan_durations = [20.0, 25.0, 30.0]
    stats.report_durations = [15.0, 20.0, 25.0]
    stats.modality_throughput = {"CT": 3, "MRI": 2, "XRAY": 4}
    stats.n_arrived = 10
    stats.n_completed = 9
    stats.queue_snapshots = [(i * 5.0, 2, 1, 3, 0) for i in range(10)]
    return stats


@pytest.fixture
def sample_scanner_utils() -> dict[str, float]:
    return {
        "CT_SCANNER_01": 0.85,
        "CT_SCANNER_02": 0.78,
        "MRI_SCANNER_01": 0.70,
        "XRAY_ROOM_01": 0.60,
    }


@pytest.fixture
def sample_rad_workloads() -> dict[str, dict]:
    return {
        "RAD001": {"radiologist_id": "RAD001", "total_reads": 20, "workload_score": 0.33},
        "RAD002": {"radiologist_id": "RAD002", "total_reads": 25, "workload_score": 0.50},
        "RAD003": {"radiologist_id": "RAD003", "total_reads": 15, "workload_score": 0.33},
    }


def test_compute_returns_policy_metrics(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """compute() returns a PolicyMetrics instance."""
    comp = MetricsComputer()
    result = comp.compute(
        policy_name="TestPolicy",
        stats=sample_stats,
        sim_duration=480.0,
        scanner_utils=sample_scanner_utils,
        rad_workloads=sample_rad_workloads,
    )
    assert isinstance(result, PolicyMetrics)


def test_avg_wait_time_computed_correctly(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """avg_wait_time_min matches manual mean of wait_times."""
    import numpy as np
    comp = MetricsComputer()
    result = comp.compute(
        policy_name="test",
        stats=sample_stats,
        sim_duration=480.0,
        scanner_utils=sample_scanner_utils,
        rad_workloads=sample_rad_workloads,
    )
    expected = float(np.mean(sample_stats.wait_times))
    assert result.avg_wait_time_min == pytest.approx(expected, rel=1e-5)


def test_p95_wait_time_greater_than_p50(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """p95 wait time >= p50 wait time."""
    comp = MetricsComputer()
    result = comp.compute(
        policy_name="test",
        stats=sample_stats,
        sim_duration=480.0,
        scanner_utils=sample_scanner_utils,
        rad_workloads=sample_rad_workloads,
    )
    assert result.p95_wait_time_min >= result.p50_wait_time_min


def test_throughput_per_hour_computed(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """throughput_per_hour = n_completed / (duration / 60)."""
    comp = MetricsComputer()
    result = comp.compute(
        policy_name="test",
        stats=sample_stats,
        sim_duration=60.0,  # 1 hour
        scanner_utils=sample_scanner_utils,
        rad_workloads=sample_rad_workloads,
    )
    # n_completed=9, duration=60 min -> 9/hr
    assert result.throughput_per_hour == pytest.approx(9.0, rel=1e-4)


def test_gini_equal_loads() -> None:
    """Gini coefficient of equal loads is 0.0."""
    comp = MetricsComputer()
    gini = comp._gini_coefficient([1.0, 1.0, 1.0, 1.0])
    assert gini == pytest.approx(0.0, abs=1e-6)


def test_gini_max_inequality() -> None:
    """Gini coefficient with all load on one radiologist is close to 0.75."""
    comp = MetricsComputer()
    gini = comp._gini_coefficient([0.01, 0.01, 0.01, 10.0])
    assert gini > 0.6  # Should be close to 0.75


def test_compare_returns_dataframe(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """compare() with 2 PolicyMetrics returns DataFrame with 2 rows."""
    comp = MetricsComputer()
    m1 = comp.compute("Policy1", sample_stats, 480.0, sample_scanner_utils, sample_rad_workloads)
    m2 = comp.compute("Policy2", sample_stats, 480.0, sample_scanner_utils, sample_rad_workloads)
    df = comp.compare([m1, m2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_policy_name_preserved(
    sample_stats: SimulationStats,
    sample_scanner_utils: dict,
    sample_rad_workloads: dict,
) -> None:
    """Policy name is preserved in returned PolicyMetrics."""
    comp = MetricsComputer()
    result = comp.compute("MyPolicy", sample_stats, 480.0, sample_scanner_utils, sample_rad_workloads)
    assert result.policy_name == "MyPolicy"


def test_empty_stats_returns_zero_metrics() -> None:
    """Empty stats returns zeros without errors."""
    comp = MetricsComputer()
    stats = SimulationStats()  # all lists empty, n_completed=0
    result = comp.compute("empty", stats, 480.0, {}, {})
    assert result.avg_wait_time_min == 0.0
    assert result.throughput_per_hour == 0.0
