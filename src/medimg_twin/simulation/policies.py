"""Scheduling policies for the hospital imaging workflow simulation.

Policies:
1. FIFO — First-In-First-Out baseline.
2. PriorityTriage — fixed clinical priority heuristic.
3. PPOPolicy — trained Stable-Baselines3 PPO model.
4. ActionSchedulingPolicy — applies an already-selected RL action.

AdaptivePPOPolicy remains available as a deterministic diagnostic reference,
but is not used by the research benchmark.
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


class _TrainingActionModel:
    """Tiny adapter used only by the Gym training environment.

    HospitalSimulation historically re-queried ``policy.model.predict`` when a
    policy was named ``ppo``. During PPO training the environment intentionally
    supplies a selected action, so this adapter makes that action the single
    source of truth instead of falling back to Priority scheduling.
    """

    def __init__(self, owner: "FIFOPolicy") -> None:
        self.owner = owner

    def predict(self, _obs, deterministic: bool = True):
        return np.asarray(self.owner._training_action, dtype=np.int64), None


class FIFOPolicy:
    """FIFO baseline plus a compatibility adapter for RL environment actions."""

    def __init__(self) -> None:
        self._name = "fifo"
        self._training_action = 0
        self.model = _TrainingActionModel(self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        # The RL environment uses this existing compatibility object to pass
        # its selected action into HospitalSimulation without a second PPO call.
        self._training_action = {"fifo": 0, "priority": 1, "ppo": 2}.get(value, 0)

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [p for p in sim.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        if not waiting:
            return None
        return min(waiting, key=lambda p: p.arrival_time).patient_id


class PriorityTriagePolicy:
    name: str = "priority"

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [p for p in sim.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        if not waiting:
            return None
        return min(waiting, key=lambda p: (-p.priority.value, p.arrival_time)).patient_id


class ActionSchedulingPolicy:
    """Apply an already-selected RL action to the simulation."""

    name: str = "rl_action"

    def __init__(self, action: int = 0) -> None:
        self.action = int(action)

    def set_action(self, action: int) -> None:
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError(f"Unsupported scheduling action: {action}")
        self.action = action

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [p for p in sim.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        if not waiting:
            return None
        if self.action == 0:
            return min(waiting, key=lambda p: p.arrival_time).patient_id
        if self.action == 1:
            return min(waiting, key=lambda p: (-p.priority.value, p.arrival_time)).patient_id
        emergencies = [p for p in waiting if p.priority == Priority.EMERGENCY]
        if emergencies:
            return min(emergencies, key=lambda p: p.queue_entry_time).patient_id
        urgents = [p for p in waiting if p.priority == Priority.URGENT]
        if urgents:
            return min(urgents, key=lambda p: p.queue_entry_time).patient_id
        return min(waiting, key=lambda p: p.arrival_time).patient_id


class PPOPolicy:
    """PPO-based adaptive scheduling policy using a trained SB3 model."""

    name: str = "ppo"

    def __init__(self, model_path: Path | str) -> None:
        from stable_baselines3 import PPO
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found at: {model_path}")
        logger.info("Loading PPO policy from %s", model_path)
        self.model = PPO.load(str(model_path))

    def predict_action(self, sim: "HospitalSimulation") -> int:
        snapshot = sim.get_snapshot()
        obs = np.asarray(snapshot.to_observation(), dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        waiting = [p for p in sim.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        if not waiting:
            return None
        action = self.predict_action(sim)
        return ActionSchedulingPolicy(action)(sim)


class AdaptivePPOPolicy:
    """Deterministic reference for diagnostics; not used in PPO benchmarks."""

    name: str = "adaptive_ppo"

    def __init__(self) -> None:
        self.low_load_queue = 4
        self.emergency_burst_count = 3
        self.emergency_fraction = 0.20
        self.priority_queue = 8

    def select_action(self, sim: "HospitalSimulation") -> int:
        waiting = [p for p in sim.patients.values() if p.status == PatientStatus.WAITING_SCAN]
        if not waiting:
            return 0
        emergency_count = sum(p.priority == Priority.EMERGENCY for p in waiting)
        urgent_count = sum(p.priority == Priority.URGENT for p in waiting)
        queue_size = len(waiting)
        emergency_fraction = emergency_count / queue_size
        if emergency_count >= self.emergency_burst_count or emergency_fraction >= self.emergency_fraction:
            return 2
        if emergency_count == 0 and queue_size <= self.low_load_queue:
            return 0
        if queue_size >= self.priority_queue or urgent_count > 0:
            return 1
        return 0

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        return ActionSchedulingPolicy(self.select_action(sim))(sim)


POLICY_REGISTRY: dict[str, type] = {
    "fifo": FIFOPolicy,
    "priority": PriorityTriagePolicy,
    "ppo": PPOPolicy,
    "adaptive_ppo": AdaptivePPOPolicy,
}


def get_policy(name: str, model_path: Path | str | None = None):
    name = name.lower()
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}")
    if name == "ppo":
        if not model_path:
            raise ValueError("PPO policy requires a model_path argument.")
        return PPOPolicy(model_path=model_path)
    return POLICY_REGISTRY[name]()
