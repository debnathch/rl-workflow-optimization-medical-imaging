"""
Scheduling policies for the hospital imaging workflow simulation.

Three policies:
1. FIFO — First-In-First-Out (baseline)
2. PriorityTriage — Priority-based triage (emergency > urgent > routine)
3. PPOPolicy — Learned policy from trained PPO agent
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medimg_twin.simulation.entities import PatientStatus, Priority

if TYPE_CHECKING:
    from medimg_twin.simulation.hospital import HospitalSimulation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FIFO Policy
# ─────────────────────────────────────────────────────────────────────────────


class FIFOPolicy:
    """First-In-First-Out scheduling policy.

    Selects the patient who has been waiting the longest, regardless of
    clinical priority or modality.
    """

    name: str = "fifo"

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        """Select patient_id of the earliest-arrived waiting patient."""
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        return min(waiting, key=lambda p: p.arrival_time).patient_id


# ─────────────────────────────────────────────────────────────────────────────
# Priority Triage Policy
# ─────────────────────────────────────────────────────────────────────────────


class PriorityTriagePolicy:
    """Clinical priority-based triage scheduling policy.

    Sorts patients by (priority DESC, arrival_time ASC).
    Emergency patients are always served before urgent, then routine.
    Within the same priority tier, FIFO ordering is preserved.
    """

    name: str = "priority"

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        """Select the highest-priority earliest-arrived waiting patient."""
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None
        # Sort by (priority DESC, arrival_time ASC)
        waiting.sort(key=lambda p: (-p.priority.value, p.arrival_time))
        return waiting[0].patient_id


# ─────────────────────────────────────────────────────────────────────────────
# PPO Policy
# ─────────────────────────────────────────────────────────────────────────────


class PPOPolicy:
    """PPO-based adaptive scheduling policy using a trained SB3 model.

    Wraps a trained stable-baselines3 PPO model and uses it to select
    actions based on the current simulation state observation.
    """

    name: str = "ppo"

    def __init__(self, model_path: Path | str) -> None:
        """Load a trained PPO model from disk.

        Args:
            model_path: Path to the saved SB3 model (.zip file).

        Raises:
            FileNotFoundError: If the model file does not exist.
            ImportError: If stable_baselines3 is not installed.
        """
        from stable_baselines3 import PPO  # noqa: PLC0415

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found at: {model_path}")

        logger.info("Loading PPO policy from %s", model_path)
        self.model = PPO.load(str(model_path))
        self._obs_cache: np.ndarray | None = None

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        """Use trained PPO model to select next patient to schedule."""
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None

        snapshot = sim.get_snapshot()
        obs = np.array(snapshot.to_observation(), dtype=np.float32)

        # Predict action index: 0=FIFO, 1=priority, 2=most urgent modality
        action, _ = self.model.predict(obs, deterministic=True)
        action_int = int(action)

        # Map action to patient selection strategy
        if action_int == 0:
            # FIFO
            return min(waiting, key=lambda p: p.arrival_time).patient_id
        elif action_int == 1:
            # Priority
            waiting.sort(key=lambda p: (-p.priority.value, p.arrival_time))
            return waiting[0].patient_id
        elif action_int == 2:
            # Emergency-first, then priority
            emg = [p for p in waiting if p.priority == Priority.EMERGENCY]
            if emg:
                return min(emg, key=lambda p: p.arrival_time).patient_id
            urgent = [p for p in waiting if p.priority == Priority.URGENT]
            if urgent:
                return min(urgent, key=lambda p: p.arrival_time).patient_id
            return min(waiting, key=lambda p: p.arrival_time).patient_id
        else:
            # Default: FIFO
            return min(waiting, key=lambda p: p.arrival_time).patient_id


# ─────────────────────────────────────────────────────────────────────────────
# Policy registry
# ─────────────────────────────────────────────────────────────────────────────


POLICY_REGISTRY: dict[str, type] = {
    "fifo": FIFOPolicy,
    "priority": PriorityTriagePolicy,
    "ppo": PPOPolicy,
}


def get_policy(
    name: str,
    model_path: Path | str | None = None,
) -> FIFOPolicy | PriorityTriagePolicy | PPOPolicy:
    """Factory function to instantiate a scheduling policy by name.

    Args:
        name: Policy name ('fifo', 'priority', or 'ppo').
        model_path: Required for 'ppo' policy.

    Returns:
        Instantiated policy callable.

    Raises:
        ValueError: If name is not recognized.
        ValueError: If 'ppo' is requested without a model_path.
    """
    name = name.lower()
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}")
    if name == "ppo":
        if not model_path:
            raise ValueError("PPO policy requires a model_path argument.")
        return PPOPolicy(model_path=model_path)
    return POLICY_REGISTRY[name]()
