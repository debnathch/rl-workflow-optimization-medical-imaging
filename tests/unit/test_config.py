"""Unit tests for configuration system."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medimg_twin.config.settings import (
    ArrivalConfig,
    Config,
    ModalityDistribution,
    load_config,
)


def test_default_config_instantiates() -> None:
    """Config() creates successfully with all defaults."""
    cfg = Config()
    assert cfg is not None


def test_load_config_default_file() -> None:
    """load_config() loads from default.yaml without error."""
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_modality_distribution_sums_to_one() -> None:
    """Default modality distribution sums to 1.0."""
    dist = ModalityDistribution()
    total = dist.CT + dist.MRI + dist.XRAY
    assert total == pytest.approx(1.0, abs=1e-6)


def test_modality_distribution_invalid_raises() -> None:
    """Distribution not summing to 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        ModalityDistribution(CT=0.5, MRI=0.5, XRAY=0.5)


def test_arrival_config_diurnal_wrong_length_raises() -> None:
    """Diurnal factor list of wrong length raises ValidationError."""
    with pytest.raises(ValidationError):
        ArrivalConfig(diurnal_factors=[1.0] * 5)  # Must be 6


def test_config_simulation_seed_default() -> None:
    """Default simulation seed is 42."""
    cfg = Config()
    assert cfg.simulation.seed == 42


def test_config_scanner_counts_positive() -> None:
    """All scanner counts are >= 1."""
    cfg = Config()
    sc = cfg.modalities.scanner_count
    assert sc.CT >= 1
    assert sc.MRI >= 1
    assert sc.XRAY >= 1


def test_config_emergency_ratio_in_range() -> None:
    """Emergency ratio is between 0 and 1."""
    cfg = load_config()
    assert 0.0 <= cfg.arrivals.emergency_ratio <= 1.0


def test_config_radiologist_roster_not_empty() -> None:
    """Radiologist roster has at least one entry after loading YAML."""
    cfg = load_config()
    assert len(cfg.radiologists.roster) > 0


def test_config_simulation_duration_positive() -> None:
    """Simulation duration is positive."""
    cfg = load_config()
    assert cfg.simulation.duration_minutes > 0


def test_config_rl_ppo_learning_rate_positive() -> None:
    """PPO learning rate is positive."""
    cfg = load_config()
    assert cfg.rl.ppo.learning_rate > 0
