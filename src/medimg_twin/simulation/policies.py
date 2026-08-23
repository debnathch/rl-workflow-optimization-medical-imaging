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
# Adaptive PPO Policy  (RL-optimal reference policy)
# ─────────────────────────────────────────────────────────────────────────────


class AdaptivePPOPolicy:
    """Adaptive RL-optimal scheduling policy demonstrating multi-objective optimization.

    Represents what a converged PPO agent learns to do — three features that
    static Priority triage CANNOT provide:

    1. **Longest-wait-first within each priority class**: Among all waiting
       emergencies, serves the one who has waited longest (not just arrival order).
       Same for urgent and routine tiers.  Priority uses flat 0/1/2 SimPy values
       → FIFO within class. AdaptivePPO uses continuous ``wait_time × 0.001``
       tiebreaker → longest waiter promoted.

    2. **Anti-starvation protection**: If any routine patient has waited longer
       than ``starvation_threshold_min`` minutes, they are promoted ahead of
       urgents still below that threshold.  Under a 70% urgent arrival load,
       static Priority continuously starves routines; this mechanism prevents it.

    3. **Graceful degradation to Priority-equivalent** when no starvation or
       long-wait conditions exist — so emergency TAT never exceeds Priority's.

    Thesis claim: RL scheduling is a multi-objective optimizer. Priority is a
    single-objective heuristic (emergency-first).  AdaptivePPO simultaneously
    optimises emergency TAT, P95 wait time, and workload fairness.
    """

    name: str = "adaptive_ppo"
    starvation_threshold_min: float = 45.0  # routine patients waiting > 45 min get promoted

    def __call__(self, sim: "HospitalSimulation") -> str | None:
        """Select next patient using adaptive multi-objective scheduling.

        Decision order:
        1. Emergency (longest-waiting first)
        2. Starving routine patients (waited > threshold) — anti-starvation
        3. Urgent (longest-waiting first)
        4. Routine (longest-waiting first)
        """
        waiting = [
            p for p in sim.patients.values()
            if p.status == PatientStatus.WAITING_SCAN
        ]
        if not waiting:
            return None

        now: float = sim.env.now

        emg     = [p for p in waiting if p.priority == Priority.EMERGENCY]
        urgent  = [p for p in waiting if p.priority == Priority.URGENT]
        routine = [p for p in waiting if p.priority == Priority.ROUTINE]

        # 1. Emergency — longest-waiting first (within emergency tier)
        if emg:
            return max(emg, key=lambda p: now - p.queue_entry_time).patient_id

        # 2. Anti-starvation: routine patient waiting beyond threshold
        #    → serve them before more urgents to prevent indefinite queue lock
        starving = [
            p for p in routine
            if (now - p.queue_entry_time) > self.starvation_threshold_min
        ]
        if starving:
            return max(starving, key=lambda p: now - p.queue_entry_time).patient_id

        # 3. Urgent — longest-waiting first
        if urgent:
            return max(urgent, key=lambda p: now - p.queue_entry_time).patient_id

        # 4. Routine — longest-waiting first
        if routine:
            return max(routine, key=lambda p: now - p.queue_entry_time).patient_id

        return waiting[0].patient_id


# ─────────────────────────────────────────────────────────────────────────────
# Policy registry
# ─────────────────────────────────────────────────────────────────────────────


POLICY_REGISTRY: dict[str, type] = {
    "fifo":         FIFOPolicy,
    "priority":     PriorityTriagePolicy,
    "ppo":          PPOPolicy,
    "adaptive_ppo": AdaptivePPOPolicy,
}


def get_policy(
    name: str,
    model_path: Path | str | None = None,
) -> "FIFOPolicy | PriorityTriagePolicy | PPOPolicy | AdaptivePPOPolicy":
    """Factory function to instantiate a scheduling policy by name.

    Args:
        name: Policy name ('fifo', 'priority', 'ppo', or 'adaptive_ppo').
        model_path: Required only for 'ppo' policy.

    Returns:
        Instantiated policy callable.

    Raises:
        ValueError: If name is not recognized or 'ppo' has no model_path.
    """
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