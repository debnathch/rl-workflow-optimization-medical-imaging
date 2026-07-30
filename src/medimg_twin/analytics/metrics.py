from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np
import pandas as pd
from medimg_twin.simulation.hospital import SimulationStats

logger = logging.getLogger(__name__)

@dataclass
class PolicyMetrics:
    policy_name: str
    n_patients_completed: int = 0
    throughput_per_hour: float = 0.0
    avg_wait_time_min: float = 0.0
    std_wait_time_min: float = 0.0
    p50_wait_time_min: float = 0.0
    p95_wait_time_min: float = 0.0
    p99_wait_time_min: float = 0.0
    avg_emergency_tat_min: float = 0.0
    p95_emergency_tat_min: float = 0.0
    avg_total_tat_min: float = 0.0
    ct_utilization: float = 0.0
    mri_utilization: float = 0.0
    xray_utilization: float = 0.0
    avg_scanner_utilization: float = 0.0
    radiologist_workload_std: float = 0.0  # std dev of reads across radiologists
    radiologist_workload_gini: float = 0.0  # Gini coefficient for fairness
    avg_queue_length: float = 0.0
    max_queue_length: float = 0.0
    sim_duration_min: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class MetricsComputer:
    def compute(
        self,
        policy_name: str,
        stats: SimulationStats,
        sim_duration: float,
        scanner_utils: dict[str, float],
        rad_workloads: dict[str, dict[str, Any]],
    ) -> PolicyMetrics:
        """Compute all KPIs from a completed simulation run.

        Args:
            policy_name: Name of the scheduling policy.
            stats: SimulationStats from a completed simulation run.
            sim_duration: Total simulation duration in minutes.
            scanner_utils: Dict of scanner_id -> utilization fraction.
            rad_workloads: Dict of rad_id -> workload summary dict.

        Returns:
            PolicyMetrics dataclass with all computed KPIs.
        """
        logger.info("Computing metrics for policy: %s", policy_name)

        # Use SimulationStats lists directly (no completed_patients attribute)
        n_completed = stats.n_completed
        throughput = (n_completed / sim_duration) * 60.0 if sim_duration > 0 else 0.0

        wait_times = stats.wait_times
        emergency_tats = stats.emergency_turnarounds
        total_tats = stats.total_turnarounds

        if wait_times:
            avg_wait = float(np.mean(wait_times))
            std_wait = float(np.std(wait_times))
            p50_wait = float(np.percentile(wait_times, 50))
            p95_wait = float(np.percentile(wait_times, 95))
            p99_wait = float(np.percentile(wait_times, 99))
        else:
            avg_wait = std_wait = p50_wait = p95_wait = p99_wait = 0.0

        if emergency_tats:
            avg_em_tat = float(np.mean(emergency_tats))
            p95_em_tat = float(np.percentile(emergency_tats, 95))
        else:
            avg_em_tat = p95_em_tat = 0.0

        avg_tot_tat = float(np.mean(total_tats)) if total_tats else 0.0

        # Aggregate scanner utilization by modality (average across scanners of same type)
        ct_utils = [v for k, v in scanner_utils.items() if "CT" in k.upper()]
        mri_utils = [v for k, v in scanner_utils.items() if "MRI" in k.upper()]
        xray_utils = [v for k, v in scanner_utils.items() if "XRAY" in k.upper() or "X-RAY" in k.upper() or "XRAY" in k]
        ct_util = float(np.mean(ct_utils)) if ct_utils else 0.0
        mri_util = float(np.mean(mri_utils)) if mri_utils else 0.0
        xray_util = float(np.mean(xray_utils)) if xray_utils else 0.0

        utils_vals = list(scanner_utils.values())
        avg_util = float(np.mean(utils_vals)) if utils_vals else 0.0

        # Radiologist workload: use 'total_reads' key from rad.to_dict()
        reads = [float(w.get("total_reads", w.get("reads", 0))) for w in rad_workloads.values()] if rad_workloads else []
        if reads:
            rad_std = np.std(reads)
            rad_gini = self._gini_coefficient(reads)
        else:
            rad_std = rad_gini = 0.0
            
        # queue_snapshots: list of (time, ct_q, mri_q, xray_q, emg_q) tuples from SimulationStats
        all_q_lengths = []
        if stats.queue_snapshots:
            for snap in stats.queue_snapshots:
                if isinstance(snap, tuple) and len(snap) >= 2:
                    # Sum CT + MRI + XRAY + emergency queues (indices 1-4)
                    total_q = sum(snap[1:])
                    all_q_lengths.append(float(total_q))
                elif isinstance(snap, (int, float)):
                    all_q_lengths.append(float(snap))
                    
        avg_q = np.mean(all_q_lengths) if all_q_lengths else 0.0
        max_q = np.max(all_q_lengths) if all_q_lengths else 0.0
        
        return PolicyMetrics(
            policy_name=policy_name,
            n_patients_completed=n_completed,
            throughput_per_hour=float(throughput),
            avg_wait_time_min=float(avg_wait),
            std_wait_time_min=float(std_wait),
            p50_wait_time_min=float(p50_wait),
            p95_wait_time_min=float(p95_wait),
            p99_wait_time_min=float(p99_wait),
            avg_emergency_tat_min=float(avg_em_tat),
            p95_emergency_tat_min=float(p95_em_tat),
            avg_total_tat_min=float(avg_tot_tat),
            ct_utilization=float(ct_util),
            mri_utilization=float(mri_util),
            xray_utilization=float(xray_util),
            avg_scanner_utilization=float(avg_util),
            radiologist_workload_std=float(rad_std),
            radiologist_workload_gini=float(rad_gini),
            avg_queue_length=float(avg_q),
            max_queue_length=float(max_q),
            sim_duration_min=float(sim_duration)
        )

    def compare(self, metrics_list: list[PolicyMetrics]) -> pd.DataFrame:
        """Create comparison DataFrame from multiple policy metrics."""
        data = [m.to_dict() for m in metrics_list]
        df = pd.DataFrame(data)
        df.set_index('policy_name', inplace=True)
        return df

    def _gini_coefficient(self, values: list[float]) -> float:
        """Compute Gini coefficient for workload fairness."""
        if not values:
            return 0.0
        x = np.array(values, dtype=np.float64)
        if np.amin(x) < 0:
            x -= np.amin(x)
        x += 1e-10
        x = np.sort(x)
        index = np.arange(1, x.shape[0] + 1)
        n = x.shape[0]
        return float((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))
