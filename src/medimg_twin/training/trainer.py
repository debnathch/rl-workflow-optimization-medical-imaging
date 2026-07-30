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
from gymnasium.utils.env_checker import check_env
from medimg_twin.config.settings import Config
from medimg_twin.rl_env.env import MedicalImagingEnv

logger = logging.getLogger(__name__)

class KPILoggingCallback(BaseCallback):
    """Callback for logging KPIs during training."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = 1000
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                ep_info = info.get('episode', {})
                if 'r' in ep_info:
                    mean_reward = ep_info['r']
                    logger.info(f"Step: {self.num_timesteps} - Mean Reward: {mean_reward}")
                    self.logger.record('rollout/ep_rew_mean', mean_reward)
                
                if 'avg_wait_time' in info:
                    logger.info(f"Step: {self.num_timesteps} - Avg Wait Time: {info['avg_wait_time']}")
                    self.logger.record('kpi/avg_wait_time', info['avg_wait_time'])
                    
                if 'avg_emergency_tat' in info:
                    logger.info(f"Step: {self.num_timesteps} - Avg Emergency TAT: {info['avg_emergency_tat']}")
                    self.logger.record('kpi/avg_emergency_tat', info['avg_emergency_tat'])
                    
        return True

class PPOTrainer:
    def __init__(self, config: Config, output_dir: Path | str, fast: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.fast = fast
        self.model: PPO | None = None
        self.reward_history: list[float] = []
        self.eval_history: list[tuple[int, float]] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, seed: int | None = None) -> PPO:
        logger.info(f"Starting PPO training in {self.output_dir}")
        env = MedicalImagingEnv(config=self.config, seed=seed)
        check_env(env)
        
        vec_env = make_vec_env(
            lambda: Monitor(MedicalImagingEnv(config=self.config, seed=seed)), 
            n_envs=1
        )
        
        eval_seed = (seed or 0) + 999
        eval_env = Monitor(MedicalImagingEnv(config=self.config, seed=eval_seed))
        
        tensorboard_log = str(self.output_dir / 'tensorboard')
        
        model = PPO(
            policy='MlpPolicy',
            env=vec_env,
            n_steps=self.config.rl.ppo.n_steps,
            batch_size=self.config.rl.ppo.batch_size,
            n_epochs=self.config.rl.ppo.n_epochs,
            learning_rate=self.config.rl.ppo.learning_rate,
            gamma=self.config.rl.ppo.gamma,
            gae_lambda=self.config.rl.ppo.gae_lambda,
            clip_range=self.config.rl.ppo.clip_range,
            ent_coef=self.config.rl.ppo.ent_coef,
            vf_coef=self.config.rl.ppo.vf_coef,
            max_grad_norm=self.config.rl.ppo.max_grad_norm,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed
        )
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir / 'best_model'),
            eval_freq=self.config.rl.eval_freq,
            n_eval_episodes=self.config.rl.n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        ckpt_cb = CheckpointCallback(
            save_freq=50000,
            save_path=str(self.output_dir / 'checkpoints'),
            name_prefix='ppo_medimg'
        )
        
        kpi_cb = KPILoggingCallback()
        
        total_timesteps = 10_000 if self.fast else self.config.rl.total_timesteps
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, ckpt_cb, kpi_cb],
            progress_bar=True
        )
        
        model_path = self.output_dir / 'final_model.zip'
        model.save(str(model_path))
        self.model = model
        
        return model

    def evaluate(self, model_path: Path | str | None = None, n_episodes: int = 10) -> dict[str, float]:
        if model_path is not None:
            model = PPO.load(str(model_path))
        elif self.model is not None:
            model = self.model
        else:
            raise ValueError("No model provided or trained")
            
        eval_env = Monitor(MedicalImagingEnv(config=self.config))
        
        # evaluate_policy returns (mean_reward, std_reward)
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=n_episodes, 
            return_episode_rewards=False
        )
        
        episode_rewards = eval_env.get_episode_rewards() if hasattr(eval_env, 'get_episode_rewards') else []
        episode_lengths = eval_env.get_episode_lengths() if hasattr(eval_env, 'get_episode_lengths') else []
        
        mean_ep_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        
        results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_episode_length': mean_ep_len
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
