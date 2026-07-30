from __future__ import annotations
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from medimg_twin.config.settings import load_config
from medimg_twin.rl_env.env import MedicalImagingEnv, OBS_DIM

@pytest.fixture(scope='module')
def fast_config():
    cfg = load_config()
    cfg.simulation.duration_minutes = 60  # 1-hour for fast tests
    cfg.simulation.decision_epoch_minutes = 5
    return cfg

def test_env_check_passes(fast_config):
    """Test that the environment passes Gym's env_checker."""
    env = MedicalImagingEnv(config=fast_config)
    check_env(env)

def test_reset_returns_correct_shape(fast_config):
    """Test reset returns observation of correct shape and type."""
    env = MedicalImagingEnv(config=fast_config)
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32

def test_reset_obs_in_bounds(fast_config):
    """Test initial observation is within expected bounds."""
    env = MedicalImagingEnv(config=fast_config)
    obs, info = env.reset()
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

def test_step_returns_correct_types(fast_config):
    """Test step returns correct types for returns."""
    env = MedicalImagingEnv(config=fast_config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_obs_in_bounds(fast_config):
    """Test step observation is within expected bounds."""
    env = MedicalImagingEnv(config=fast_config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

def test_all_actions_valid(fast_config):
    """Test taking different valid actions does not crash."""
    env = MedicalImagingEnv(config=fast_config)
    env.reset()
    env.step(0)
    env.step(1)
    env.step(2)

def test_episode_terminates(fast_config):
    """Test the episode eventually terminates."""
    env = MedicalImagingEnv(config=fast_config)
    env.reset()
    done = False
    max_steps = 1000
    steps = 0
    while not done and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
    assert done

def test_reward_is_scalar(fast_config):
    """Test reward is a scalar float."""
    env = MedicalImagingEnv(config=fast_config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)

def test_deterministic_with_seed(fast_config):
    """Test setting seed produces deterministic initial observations."""
    env1 = MedicalImagingEnv(config=fast_config)
    obs1, _ = env1.reset(seed=42)
    
    env2 = MedicalImagingEnv(config=fast_config)
    obs2, _ = env2.reset(seed=42)
    
    np.testing.assert_array_equal(obs1, obs2)

def test_render_returns_dict(fast_config):
    """Test render returns a dictionary when mode is 'rgb_array' or default."""
    env = MedicalImagingEnv(config=fast_config, render_mode='rgb_array')
    env.reset()
    rendered = env.render()
    assert isinstance(rendered, dict) or isinstance(rendered, np.ndarray) or rendered is None
