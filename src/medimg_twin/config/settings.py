"""Configuration loading and validation using Pydantic Settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Sub-config models
# ─────────────────────────────────────────────────────────────────────────────


class OperatingHours(BaseModel):
    start: int = Field(6, ge=0, le=23)
    end: int = Field(22, ge=1, le=24)


class SimulationConfig(BaseModel):
    seed: int = 42
    duration_minutes: float = 480.0
    warmup_minutes: float = 60.0
    decision_epoch_minutes: float = 5.0
    operating_hours: OperatingHours = Field(default_factory=OperatingHours)
    weekend_multiplier: float = Field(0.6, gt=0.0, le=1.0)


class ArrivalConfig(BaseModel):
    routine_mean_iat: float = Field(8.0, gt=0.0)
    emergency_ratio: float = Field(0.08, ge=0.0, le=1.0)
    urgent_ratio: float = Field(0.15, ge=0.0, le=1.0)
    diurnal_factors: list[float] = Field(default_factory=lambda: [0.1, 0.15, 1.0, 0.9, 0.85, 0.6])

    @field_validator("diurnal_factors")
    @classmethod
    def validate_diurnal_length(cls, v: list[float]) -> list[float]:
        if len(v) != 6:
            raise ValueError("diurnal_factors must have exactly 6 elements (one per 4-hour block)")
        return v


class DurationParams(BaseModel):
    mean: float = Field(gt=0.0)
    sigma: float = Field(gt=0.0)


class ScannerCount(BaseModel):
    CT: int = Field(3, ge=1)
    MRI: int = Field(2, ge=1)
    XRAY: int = Field(4, ge=1)


class SetupTime(BaseModel):
    mean: float = Field(5.0, gt=0.0)
    sigma: float = Field(1.5, gt=0.0)


class ModalityDistribution(BaseModel):
    CT: float = Field(0.45, ge=0.0, le=1.0)
    MRI: float = Field(0.30, ge=0.0, le=1.0)
    XRAY: float = Field(0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "ModalityDistribution":
        total = self.CT + self.MRI + self.XRAY
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Modality distribution must sum to 1.0, got {total:.4f}")
        return self


class ModalitiesConfig(BaseModel):
    distribution: ModalityDistribution = Field(default_factory=ModalityDistribution)
    scan_duration: dict[str, DurationParams] = Field(
        default_factory=lambda: {
            "CT": DurationParams(mean=25.0, sigma=0.3),
            "MRI": DurationParams(mean=45.0, sigma=0.35),
            "XRAY": DurationParams(mean=12.0, sigma=0.25),
        }
    )
    scanner_count: ScannerCount = Field(default_factory=ScannerCount)
    setup_time: SetupTime = Field(default_factory=SetupTime)


class RadiologistEntry(BaseModel):
    id: str
    specialty: str
    shift_start: int = Field(ge=0, le=23)
    shift_end: int = Field(ge=1, le=24)
    max_daily_reads: int = Field(gt=0)


class RadiologistsConfig(BaseModel):
    roster: list[RadiologistEntry] = Field(default_factory=list)
    reporting_duration: dict[str, DurationParams] = Field(default_factory=dict)
    emergency_priority_speedup: float = Field(0.6, gt=0.0, le=1.0)


class PPOConfig(BaseModel):
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class RewardWeights(BaseModel):
    avg_wait_time: float = 1.0
    emergency_tat: float = 3.0
    scanner_utilization: float = 0.5
    utilization_target: float = 0.85
    workload_imbalance: float = 0.8
    throughput: float = 0.3


class RLConfig(BaseModel):
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)


class DatasetConfig(BaseModel):
    n_patients: int = Field(100_000, ge=1)
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    output_dir: str = "outputs/dataset"
    body_parts: dict[str, list[list[Any]]] = Field(default_factory=dict)


class AnalyticsConfig(BaseModel):
    output_dir: str = "outputs/analytics"
    figures_dir: str = "outputs/figures"
    figure_dpi: int = 300
    figure_format: str = "png"
    color_palette: str = "Set2"


class DashboardConfig(BaseModel):
    refresh_interval_seconds: int = 2
    max_history_points: int = 500
    port: int = 8501
    theme: str = "dark"


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────


class Config(BaseModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    arrivals: ArrivalConfig = Field(default_factory=ArrivalConfig)
    modalities: ModalitiesConfig = Field(default_factory=ModalitiesConfig)
    radiologists: RadiologistsConfig = Field(default_factory=RadiologistsConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to YAML config. Defaults to config/default.yaml.

    Returns:
        Validated Config instance.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file not found at %s — using defaults.", config_path)
        return Config()

    logger.info("Loading configuration from %s", config_path)
    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return Config.model_validate(raw)
