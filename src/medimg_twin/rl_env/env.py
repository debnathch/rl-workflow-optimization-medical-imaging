"""
Gymnasium-compatible Digital Twin RL Environment.

Wraps the SimPy hospital simulation as a Gymnasium environment for
training reinforcement learning agents (PPO via Stable-Baselines3).

Observation Space (17 features):
  [0]   CT queue length
  [1]   MRI queue length
  [2]   XRAY queue length
  [3]   Emergency queue length
  [4]   CT scanner utilization (0-1)
  [5]   MRI scanner utilization (0-1)
  [6]   XRAY scanner utilization (0-1)
  [7-13] Radiologist workload scores (7 radiologists)
  [14]  Hour of day normalized (0-1)
  [15]  Day of week normalized (0-1)
  [16]  Routine queue count
  [17]  Urgent queue count
  [18]  Emergency queue count
  [19]  Patients completed last epoch
  [20]  Average wait time last epoch (minutes)

Action Space: Discrete(3)
  0 = FIFO scheduling
  1 = Priority-based scheduling
  2 = Emergency-first scheduling
"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from medimg_twin.config.settings import Config, load_config
from medimg_twin.simulation.hospital import HospitalSimulation

logger = logging.getLogger(__name__)

# Observation vector dimension
OBS_DIM = 21
# Maximum values for normalization
MAX_QUEUE = 50.0
MAX_WAIT_TIME = 300.0  # minutes


class MedicalImagingEnv(gym.Env):
    """
    Gymnasium Digital Twin environment for the hospital imaging department.

    The environment wraps a SimPy simulation and exposes it as a Markov
    Decision Process. At each decision epoch (every decision_epoch_minutes),
    the RL agent chooses a scheduling strategy. The simulation is stepped
    forward and the resulting KPIs form the reward signal.

    Args:
        config: Optional Config instance. Loads default.yaml if not provided.
        seed: Random seed for reproducibility.

    Example:
        >>> env = MedicalImagingEnv()
        >>> obs, info = env.reset(seed=42)
        >>> obs_2, reward, terminated, truncated, info = env.step(1)
    """

    metadata = {"render_modes": ["human", "rgb_array", "dict"]}

    def __init__(
        self,
        config: Config | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.config = config or load_config()
        self._seed = seed
        self.render_mode = render_mode

        # Create observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        # 3 scheduling strategies: FIFO, Priority, Emergency-first
        self.action_space = spaces.Discrete(3)

        # Simulation instance (lazily initialized in reset)
        self._sim: HospitalSimulation | None = None
        self._current_obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)
        self._epoch_minutes = self.config.simulation.decision_epoch_minutes
        self._max_steps = int(
            self.config.simulation.duration_minutes / self._epoch_minutes
        )
        self._step_count = 0

        # Cumulative reward tracking
        self._episode_reward: float = 0.0
        self._reward_history: list[float] = []

        logger.info(
            "MedicalImagingEnv created: obs_dim=%d, action_space=%s, max_steps=%d",
            OBS_DIM,
            self.action_space,
            self._max_steps,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Optional seed to override the environment seed.
            options: Optional dict (unused; reserved for future use).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # ── Episode seed: ALWAYS generate a fresh seed from np_random ──────────
        # This is the root cause of explained_variance=0: if we re-use the same
        # seed (config.simulation.seed=42) for every episode, every episode
        # produces identical patient arrivals → identical rewards regardless of
        # action → PPO value function can't learn.
        #
        # Gymnasium's super().reset(seed=seed) already seeds self.np_random
        # properly. We just draw from it to get a unique per-episode seed.
        effective_seed = int(self.np_random.integers(0, 2**31 - 1))

        # Create a mutable adaptive policy that _apply_action can switch
        from medimg_twin.simulation.policies import FIFOPolicy
        adaptive_policy = FIFOPolicy()  # default to FIFO; _apply_action switches it
        adaptive_policy.name = "fifo"   # type: ignore[attr-defined]

        self._sim = HospitalSimulation(
            config=self.config,
            policy=adaptive_policy,
            seed=effective_seed,
        )
        self._sim.reset()
        self._adaptive_policy = adaptive_policy  # keep reference for _apply_action

        # Start the simulation processes
        sim_duration = self.config.simulation.duration_minutes
        self._sim.env.process(self._sim._patient_arrival_generator(sim_duration))
        self._sim.env.process(self._sim._stats_collector())

        self._step_count = 0
        self._episode_reward = 0.0
        self._last_action = 0

        obs = self._get_observation()
        self._current_obs = obs
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance simulation by one decision epoch, applying the chosen action.

        Args:
            action: Integer in {0, 1, 2}:
                0 = Apply FIFO ordering to pending queue
                1 = Apply priority-based ordering
                2 = Apply emergency-first ordering

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._sim is not None, "Call reset() before step()"

        # Apply scheduling action to pending queue (influences future dispatch)
        self._apply_action(int(action))

        # Advance simulation by one epoch
        next_time = self._sim.env.now + self._epoch_minutes
        abs_end = self._sim._start_time + self._sim._run_duration
        target = min(next_time, abs_end)
        self._sim.run_until(target)

        self._step_count += 1

        obs = self._get_observation()
        self._current_obs = obs
        reward = self._compute_reward()
        self._episode_reward += float(reward)

        terminated = self._sim.env.now >= abs_end
        truncated = self._step_count >= self._max_steps and not terminated
        done = terminated or truncated

        if done:
            self._reward_history.append(self._episode_reward)

        info = self._get_info()
        info["episode_step"] = self._step_count
        info["sim_time"] = self._sim.env.now

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render current simulation state.

        Returns:
            State dict (render_mode='dict'), None otherwise.
        """
        if self._sim is None:
            return None

        snapshot = self._sim.get_snapshot()
        state = {
            "sim_time_min": snapshot.sim_time,
            "queues": {
                "CT": snapshot.ct_queue_length,
                "MRI": snapshot.mri_queue_length,
                "XRAY": snapshot.xray_queue_length,
                "Emergency": snapshot.emergency_queue_length,
            },
            "utilization": {
                "CT": snapshot.ct_utilization,
                "MRI": snapshot.mri_utilization,
                "XRAY": snapshot.xray_utilization,
            },
            "radiologist_workloads": snapshot.radiologist_workloads,
            "completed_patients": self._sim.stats.n_completed,
            "avg_wait_time": (
                float(np.mean(self._sim.stats.wait_times))
                if self._sim.stats.wait_times else 0.0
            ),
            "episode_reward": self._episode_reward,
        }
        if getattr(self, "render_mode", None) == "human":
            for k, v in state.items():
                print(f"  {k}: {v}")
        return state

    def close(self) -> None:
        """Clean up the environment."""
        self._sim = None
        logger.debug("MedicalImagingEnv closed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """Apply the chosen action by switching the simulation's active scheduling policy.

        Action mapping:
            0 → FIFO (arrival-time ordering, baseline)
            1 → Priority triage (Emergency=0, Urgent=1, Routine=2)
            2 → Emergency-first aggressive (Emergency=-1, Urgent=1, Routine=2)
        """
        if self._sim is None:
            return

        # Mutate the adaptive policy object that was passed to HospitalSimulation
        # _patient_pathway reads policy.name on every patient arrival
        if hasattr(self, '_adaptive_policy'):
            if action == 0:
                self._adaptive_policy.name = "fifo"
            elif action == 1:
                self._adaptive_policy.name = "priority"
            elif action == 2:
                self._adaptive_policy.name = "ppo"  # emergency-first branch

        self._last_action = action

    def _get_observation(self) -> np.ndarray:
        """Build normalized observation vector from simulation snapshot."""
        if self._sim is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        snapshot = self._sim.get_snapshot()
        raw = snapshot.to_observation()

        obs = np.array([
            min(raw[0], MAX_QUEUE) / MAX_QUEUE,   # CT queue
            min(raw[1], MAX_QUEUE) / MAX_QUEUE,   # MRI queue
            min(raw[2], MAX_QUEUE) / MAX_QUEUE,   # XRAY queue
            min(raw[3], MAX_QUEUE) / MAX_QUEUE,   # Emergency queue
            np.clip(raw[4], 0.0, 1.0),             # CT utilization
            np.clip(raw[5], 0.0, 1.0),             # MRI utilization
            np.clip(raw[6], 0.0, 1.0),             # XRAY utilization
            *[np.clip(w, 0.0, 1.0) for w in raw[7:14]],  # Radiologist workloads
            np.clip(raw[14], 0.0, 1.0),            # Hour of day
            np.clip(raw[15], 0.0, 1.0),            # Day of week
            min(raw[16], MAX_QUEUE) / MAX_QUEUE,   # Routine count
            min(raw[17], MAX_QUEUE) / MAX_QUEUE,   # Urgent count
            min(raw[18], MAX_QUEUE) / MAX_QUEUE,   # Emergency count
            min(raw[19], 50.0) / 50.0,             # Completed last epoch
            min(raw[20], MAX_WAIT_TIME) / MAX_WAIT_TIME,  # Avg wait last epoch
        ], dtype=np.float32)

        # Ensure shape is exactly OBS_DIM
        if len(obs) < OBS_DIM:
            obs = np.pad(obs, (0, OBS_DIM - len(obs)))
        elif len(obs) > OBS_DIM:
            obs = obs[:OBS_DIM]

        return obs

    def _compute_reward(self) -> float:
        """Compute reward from the CURRENT snapshot — not lagged history averages.

        Key insight: stats.wait_times[-20:] barely changes within a 5-minute epoch,
        making the reward look identical regardless of action → value function can't learn.

        Instead, use SNAPSHOT quantities that respond immediately when policy switches:
        - emergency_queue_length: drops fast when Priority/Emergency-first is active
        - completed_last_epoch: jumps when throughput increases
        - ct/mri/xray_queue_length: direct queue signal

        Reward is in [-5, 1] range so PPO value function can maintain explained_variance > 0.5.

        Returns:
            Scalar reward in [-5, 1].
        """
        if self._sim is None:
            return 0.0

        snapshot = self._sim.get_snapshot()
        stats = self._sim.stats

        # ── Primary signal: emergency queue length (immediate, action-responsive) ──
        # Emergency patients waiting = highest urgency signal
        emg_q = snapshot.emergency_queue_length
        emg_q_pen = min(emg_q / 3.0, 1.0) * 3.0   # 3+ emergencies waiting = full penalty

        # ── Secondary: total queue length (builds when throughput lags) ──
        total_q = (snapshot.ct_queue_length + snapshot.mri_queue_length
                   + snapshot.xray_queue_length + snapshot.emergency_queue_length)
        total_q_pen = min(total_q / 15.0, 1.0) * 1.5

        # ── Tertiary: worst-case wait from recent history ──
        # Use last 5 patients only (not 20) for faster response
        recent_wait = (float(np.mean(stats.wait_times[-5:])) if stats.wait_times else 0.0)
        wait_pen = min(recent_wait / 60.0, 1.0) * 0.5

        # ── Throughput bonus: patients completed this epoch ──
        throughput_bonus = min(snapshot.completed_last_epoch / 5.0, 1.0) * 1.0

        # ── Utilization bonus: reward scanners being busy ──
        avg_util = (snapshot.ct_utilization + snapshot.mri_utilization
                    + snapshot.xray_utilization) / 3.0
        util_bonus = avg_util * 0.3   # [0, 0.3]

        reward = -(emg_q_pen + total_q_pen + wait_pen) + throughput_bonus + util_bonus

        return float(np.clip(reward, -6.0, 2.0))

    def _get_info(self) -> dict[str, Any]:
        """Build info dict for diagnostics."""
        if self._sim is None:
            return {}
        stats = self._sim.stats
        return {
            "n_arrived": stats.n_arrived,
            "n_completed": stats.n_completed,
            "avg_wait_time": float(np.mean(stats.wait_times)) if stats.wait_times else 0.0,
            "avg_emergency_tat": (
                float(np.mean(stats.emergency_turnarounds))
                if stats.emergency_turnarounds else 0.0
            ),
            "scanner_utils": self._sim.scanner_utilizations(),
            "episode_reward": self._episode_reward,
        }
