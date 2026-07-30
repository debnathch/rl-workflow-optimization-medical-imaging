# Experiments Guide: Medical Imaging Digital Twin

This document provides a comprehensive guide for configuring, executing, and analyzing experiments using the Medical Imaging Digital Twin Workflow Optimizer locally on macOS.

---

## 1. Prerequisites

Before running experiments, ensure your local macOS environment meets the following requirements:
- **macOS Version**: Ventura (13.0) or later (Apple Silicon M1/M2/M3 highly recommended for hardware acceleration).
- **Python**: Version 3.12 is strictly required.
- **Package Manager**: `uv` (by Astral) for fast dependency resolution.
- **Memory**: Minimum 16GB RAM recommended. Generating large datasets (e.g., >500k patients) may require 32GB+.

---

## 2. Environment Setup

We use `uv` to manage the virtual environment and dependencies efficiently.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project root
cd /path/to/RL_WFOPT_MI

# Sync dependencies (this creates a .venv and installs all packages from uv.lock)
uv sync

# Verify installation by invoking the CLI
uv run medimg-twin --help
```

---

## 3. Dataset Generation

The simulation requires a synthetic dataset representing patient arrivals, HL7 events, and DICOM metadata.

```bash
# Generate a standard dataset (100,000 patients)
uv run medimg-generate --n-patients 100000 --seed 42 --output data/processed/patients.parquet
```

**Configurations & Expectations:**
- `n-patients=10,000`: ~10 seconds, ~2MB Parquet file. Good for testing code logic.
- `n-patients=100,000`: ~1-2 minutes, ~20MB Parquet file. Standard for most experiments.
- `n-patients=1,000,000`: ~10-15 minutes, ~200MB Parquet file. Used for final thesis results; requires significant RAM.

---

## 4. Running Policy Experiments

You can evaluate the three implemented scheduling policies using the generated dataset.

```bash
# Run all policies and compare
uv run medimg-experiment --policy all --n-runs 5 --dataset data/processed/patients.parquet
```

- **`--policy`**: Choices are `fifo`, `priority`, `ppo`, or `all`.
- **`--n-runs`**: Number of distinct simulation seeds to average results over (reduces stochastic variance).
- **Output**: Summary metrics are printed to stdout, while detailed logs are saved to `experiments/logs/`.

---

## 5. PPO Training

To train the Proximal Policy Optimization (PPO) agent, use the training module.

```bash
# Fast training for code verification (fewer timesteps, smaller network)
uv run medimg-train --fast

# Full training run
uv run medimg-train --config configs/train_ppo.yaml
```

**Training Time Expectations:**
- **CPU**: ~4 hours for 1 million timesteps.
- **MPS (Apple Silicon GPU)**: PyTorch MPS backend is supported automatically. Training 1 million timesteps drops to ~1.5 hours.

**Monitoring:**
Monitor training in real-time using TensorBoard:
```bash
uv run tensorboard --logdir runs/ppo_logs/
```

---

## 6. Analyzing Results

Post-experiment analysis involves processing the output CSVs and generating plots.

- **CSV Outputs**: Located in `experiments/results/`.
  - `metrics_summary.csv`: Aggregated KPIs per policy.
  - `wait_times.csv`: Raw wait times per patient for distribution plotting.
- **Figures**: Automatically generated and saved to `experiments/figures/` (e.g., `wait_time_distributions.png`, `scanner_utilization_heatmap.png`).

**Regenerate Figures:**
```bash
uv run medimg-analyze --results-dir experiments/results/ --output-dir experiments/figures/
```

---

## 7. Dashboard

The interactive Streamlit dashboard allows for dynamic exploration of the digital twin.

```bash
uv run medimg-dashboard
```

**Dashboard Tabs:**
- **Overview**: High-level KPIs and architecture diagram.
- **Live Simulation**: Step through the simulation day-by-day and watch queues evolve.
- **Policy Comparison**: Interactive bar charts and box plots comparing FIFO, Priority, and PPO.
- **RL Training Analytics**: Embedding of TensorBoard logs and reward curves.

*Performance Tip*: If the dashboard lags, downsample the visualized dataset using the slider in the sidebar.

---

## 8. Reproducibility

Ensuring deterministic results is critical for academic research.

- **Seeds**: The `--seed` flag controls random number generation across `numpy`, `SimPy`, `torch`, and `gymnasium`.
- **Thesis Results**: To exactly reproduce the figures in the thesis, use the provided script:
  ```bash
  ./scripts/reproduce_thesis_results.sh
  ```
  *(Note: This process takes approximately 12 hours on an M2 Max).*

---

## 9. Troubleshooting

### PyTorch on MPS
If you encounter `NotImplementedError` regarding MPS backend operations, fallback to CPU by setting the environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Streamlit Port Conflicts
If port 8501 is in use, start the dashboard on a different port:
```bash
uv run streamlit run src/medimg_twin/dashboard/app.py --server.port 8502
```

### Memory Errors (OOM)
When generating 1,000,000+ patients, if the process is killed by macOS due to memory pressure, increase your swap space or process the generation in chunks using the `--chunk-size` parameter.

---

## 10. Configuration Reference

All experiments are driven by YAML configurations in the `configs/` directory.

| Key | Default | Description |
|---|---|---|
| `simulation.sim_time` | `86400` | Duration of simulation in seconds (1 day). |
| `simulation.ct_scanners` | `2` | Number of available CT scanners. |
| `simulation.mri_scanners` | `1` | Number of available MRI scanners. |
| `rl.algorithm` | `"PPO"` | RL algorithm to use via Stable-Baselines3. |
| `rl.learning_rate` | `0.0003` | Learning rate for the PPO optimizer. |
| `rl.total_timesteps`| `1000000`| Total environmental steps for RL training. |
| `data.arrival_rate` | `0.15` | Base Poisson arrival rate (patients per minute). |
| `env.random_seed` | `42` | Master seed for reproducibility. |
