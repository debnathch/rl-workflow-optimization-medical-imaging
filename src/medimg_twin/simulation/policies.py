"""
Scheduling policies for the hospital imaging workflow simulation.

Policies:
1. FIFO — First-In-First-Out baseline.
2. PriorityTriage — fixed clinical priority heuristic.
3. PPOPolicy — trained Stable-Baselines3 PPO model.
4. AdaptivePPOPolicy — deterministic reference implementation of the
   intended PPO action semantics for scenario testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from medimg_twin.simulation.entities import PatientStatus, Priority

if TYPE_CHECKING:
    from medimg_twin.simulation.hospital import HospitalSimulation

logger = logging.getLogger(__name__)


class FIFOPolicy:
    """First-In-First-Out scheduling policy."""

    name: str = "fifo"

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        return min(waiting, key=lambda p: p.arrival_time).patient_id


class PriorityTriagePolicy:
    """Fixed clinical priority: emergency > urgent > routine."""

    name: str = "priority"

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        waiting.sort(key=lambda p: (-p.priority.value, p.arrival_time))
        return waiting[0].patient_id


class PPOPolicy:
    """PPO-based adaptive scheduling policy using a trained SB3 model."""

    name: str = "ppo"

    def __init__(self, model_path: Path | str) -> None:
        from stable_baselines3 import PPO  # noqa: PLC0415

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found at: {model_path}")
        logger.info("Loading PPO policy from %s", model_path)
        self.model = PPO.load(str(model_path))

    def predict_action(self, sim: "HospitalSimulation") -> int:
        """Return PPO action: 0=FIFO, 1=Priority, 2=Emergency-first."""
        snapshot = sim.get_snapshot()
        obs = np.array(snapshot.to_observation(), dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        action_int = self.predict_action(sim)
        if action_int == 0:
            return min(waiting, key=lambda p: p.arrival_time).patient_id
        if action_int == 1:
            waiting.sort(key=lambda p: (-p.priority.value, p.arrival_time))
            return waiting[0].patient_id
        if action_int == 2:
            emergencies = [p for p in waiting if p.priority == Priority.EMERGENCY]
            if emergencies:
                return min(emergencies, key=lambda p: p.arrival_time).patient_id
            urgents = [p for p in waiting if p.priority == Priority.URGENT]
            if urgents:
                return min(urgents, key=lambda p: p.arrival_time).patient_id
        return min(waiting, key=lambda p: p.arrival_time).patient_id


class AdaptivePPOPolicy:
    """State-dependent reference policy for the PPO action semantics.

    The policy explicitly exposes the three actions used by the RL environment:

    * 0 — FIFO under low/no clinical pressure.
    * 1 — Priority under moderate queue/urgent pressure.
    * 2 — Emergency-first under emergency bursts.

    This class is useful for deterministic scenario validation. It is deliberately
    kept separate from :class:`PPOPolicy`, which is the actual trained RL agent.
    """

    name: str = "adaptive_ppo"

    # These thresholds are scenario controls, not learned parameters. A trained
    # PPO model should learn equivalent switching boundaries from reward feedback.
    low_load_queue: int = 4
    emergency_burst_count: int = 3
    emergency_fraction: float = 0.20
    priority_queue: int = 8

    def select_action(self, sim: "HospitalSimulation") -> int:
        """Select FIFO, Priority or Emergency-first from current workload state."""
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return 0

        emergency_count = sum(p.priority == Priority.EMERGENCY for p in waiting)
        urgent_count = sum(p.priority == Priority.URGENT for p in waiting)
        queue_size = len(waiting)
        emergency_fraction = emergency_count / queue_size

        # Emergency burst: action 2 is intentionally more aggressive than the
        # fixed Priority baseline (priority=-1 versus Priority's 0).
        if (
            emergency_count >= self.emergency_burst_count
            or emergency_fraction >= self.emergency_fraction
        ):
            return 2

        # Low pressure and no emergencies: action 0/FIFO maximises throughput
        # without introducing unnecessary priority ordering.
        if emergency_count == 0 and queue_size <= self.low_load_queue:
            return 0

        # Moderate queue pressure or urgent demand: action 1/Priority.
        if queue_size >= self.priority_queue or urgent_count > 0:
            return 1

        return 0

    @staticmethod
    def select_patient(waiting: list, action: int, now: float) -> str:
        """Map an action to the patient selection strategy."""
        if action == 0:
            return min(waiting, key=lambda p: p.arrival_time).patient_id

        if action == 1:
            return min(
                waiting,
                key=lambda p: (-p.priority.value, p.arrival_time),
            ).patient_id

        emergencies = [p for p in waiting if p.priority == Priority.EMERGENCY]
        if emergencies:
            # Longest waiting emergency is selected first. This avoids starving
            # an older emergency when the burst contains several emergencies.
            return max(emergencies, key=lambda p: now - p.queue_entry_time).patient_id

        urgents = [p for p in waiting if p.priority == Priority.URGENT]
        if urgents:
            return max(urgents, key=lambda p: now - p.queue_entry_time).patient_id

        return min(waiting, key=lambda p: p.arrival_time).patient_id

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        action = self.select_action(sim)
        return self.select_patient(waiting, action, float(sim.env.now))


POLICY_REGISTRY: dict[str, type] = {
    "fifo": FIFOPolicy,
    "priority": PriorityTriagePolicy,
    "ppo": PPOPolicy,
    "adaptive_ppo": AdaptivePPOPolicy,
}


def get_policy(
    name: str,
    model_path: Path | str | None = None,
) -> "FIFOPolicy | PriorityTriagePolicy | PPOPolicy | AdaptivePPOPolicy":
    """Factory function to instantiate a scheduling policy."""
    name = name.lower()
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}"
        )
    if name == "ppo":
        if not model_path:
            raise ValueError("PPO policy requires a model_path argument.")
        return PPOPolicy(model_path=model_path)
    return POLICY_REGISTRY[name]()
