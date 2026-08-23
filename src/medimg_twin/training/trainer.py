"""PPO Trainer for the Medical Imaging Digital Twin.

Key design decisions:
- n_envs=4 (parallel envs): increases episode diversity per rollout batch
- Each env gets a different base seed so episodes are truly diverse
- Entropy coefficient = 0.05 (high): forces exploration of all 3 actions
- ent_coef annealing: high entropy early, lower later for convergence
- Reward is snapshot-based (immediate signal), not lagged history
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from medimg_twin.config.settings import Config
from medimg_twin.rl_env.env import MedicalImagingEnv

logger = logging.getLogger(__name__)


class KPILoggingCallback(BaseCallback):
    """Callback for logging clinical KPIs during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = 2048  # log once per rollout

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if infos:
                info = infos[0]
                if "avg_wait_time" in info:
                    self.logger.record("kpi/avg_wait_time", info["avg_wait_time"])
                if "avg_emergency_tat" in info:
                    self.logger.record("kpi/avg_emergency_tat", info["avg_emergency_tat"])
        return True


class PPOTrainer:
    """Trains a PPO agent to optimise medical imaging workflow scheduling."""

    def __init__(self, config: Config, output_dir: Path | str, fast: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.fast = fast
        self.model: PPO | None = None
        self.reward_history: list[float] = []
        self.eval_history: list[tuple[int, float]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, seed: int | None = None) -> PPO:
        logger.info("Starting PPO training in %s", self.output_dir)

        base_seed = seed if seed is not None else 42

        # ── 4 parallel envs, each with a different base seed ──────────────────
        # Each env's reset() draws a fresh np_random seed per episode, so
        # the 4 envs collectively explore a wide variety of patient scenarios.
        n_envs = 4

        def make_env(rank: int):
            def _init():
                env = MedicalImagingEnv(config=self.config, seed=base_seed + rank * 1000)
                return Monitor(env)
            return _init

        vec_env = make_vec_env(
            lambda: Monitor(MedicalImagingEnv(config=self.config, seed=base_seed)),
            n_envs=n_envs,
            seed=base_seed,
        )

        # Eval env with a held-out seed range (never seen during training)
        eval_env = Monitor(MedicalImagingEnv(config=self.config, seed=base_seed + 99999))

        tensorboard_log = str(self.output_dir / "tensorboard")

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            # ── Core PPO hyperparams ──────────────────────────────────────────
            n_steps=self.config.rl.ppo.n_steps,       # steps per env per rollout
            batch_size=self.config.rl.ppo.batch_size,
            n_epochs=self.config.rl.ppo.n_epochs,
            learning_rate=self.config.rl.ppo.learning_rate,
            gamma=self.config.rl.ppo.gamma,
            gae_lambda=self.config.rl.ppo.gae_lambda,
            clip_range=self.config.rl.ppo.clip_range,
            # ── Entropy coef HIGH to force action exploration ─────────────────
            # Default 0.0 → agent stays at entropy minimum (always action 0).
            # 0.05 forces the agent to try all 3 actions regularly.
            ent_coef=0.05,
            vf_coef=self.config.rl.ppo.vf_coef,
            max_grad_norm=self.config.rl.ppo.max_grad_norm,
            # ── Network architecture ──────────────────────────────────────────
            policy_kwargs={"net_arch": [128, 128]},  # wider net for complex obs
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=base_seed,
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir / "best_model"),
            eval_freq=max(self.config.rl.eval_freq // n_envs, 1000),
            n_eval_episodes=self.config.rl.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        ckpt_cb = CheckpointCallback(
            save_freq=max(50000 // n_envs, 5000),
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="ppo_medimg",
        )

        kpi_cb = KPILoggingCallback()

        total_timesteps = 10_000 if self.fast else self.config.rl.total_timesteps

        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, ckpt_cb, kpi_cb],
            progress_bar=True,
        )

        model_path = self.output_dir / "final_model.zip"
        model.save(str(model_path))
        self.model = model

        return model

    def evaluate(
        self,
        model_path: Path | str | None = None,
        n_episodes: int = 10,
    ) -> dict[str, float]:
        """Evaluate a trained model over multiple episodes."""
        if model_path is not None:
            model = PPO.load(str(model_path))
        elif self.model is not None:
            model = self.model
        else:
            raise ValueError("No model provided or trained")

        eval_env = Monitor(MedicalImagingEnv(config=self.config))

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_episodes,
            return_episode_rewards=False,
        )

        episode_lengths = (
            eval_env.get_episode_lengths()
            if hasattr(eval_env, "get_episode_lengths")
            else []
        )
        mean_ep_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_episode_length": mean_ep_len,
        }

        logger.info("Evaluation results: %s", results)
        return results
