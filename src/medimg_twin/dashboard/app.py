"""
Streamlit Real-Time Dashboard for Medical Imaging Digital Twin.

Tabs:
  1. Live Simulation  — queue gauges, scanner states, radiologist heatmap
  2. RL Training      — reward curve, policy loss from TensorBoard logs
  3. Policy Comparison — side-by-side charts from analytics CSVs
  4. Patient Timeline  — per-patient Gantt view
  5. Dataset Explorer  — filterable encounter table
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from medimg_twin.config.settings import load_config
from medimg_twin.simulation.entities import Modality, PatientStatus, Priority
from medimg_twin.simulation.hospital import HospitalSimulation
from medimg_twin.simulation.policies import get_policy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medical Imaging Digital Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.3rem 0;
    }
    
    .status-idle { color: #22c55e; }
    .status-busy { color: #f59e0b; }
    .status-maint { color: #ef4444; }
    
    .queue-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .emergency { background: #dc2626; color: white; }
    .urgent { background: #d97706; color: white; }
    .routine { background: #059669; color: white; }
    
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults: dict[str, Any] = {
        "sim": None,
        "sim_running": False,
        "sim_step": 0,
        "history": {
            "time": [],
            "ct_queue": [],
            "mri_queue": [],
            "xray_queue": [],
            "emg_queue": [],
            "ct_util": [],
            "mri_util": [],
            "xray_util": [],
            "completed": [],
            "rewards": [],
        },
        "completed_patients": [],
        "config": None,
        "policy_name": "fifo",
        "reward_history": [],
        "do_steps": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_or_create_sim(policy_name: str = "fifo") -> HospitalSimulation:
    """Get existing simulation or create a new one."""
    if st.session_state.sim is None or st.session_state.policy_name != policy_name:
        cfg = st.session_state.config or load_config()
        policy = get_policy(policy_name)
        sim = HospitalSimulation(config=cfg, policy=policy, seed=42)
        sim.reset()
        sim_duration = cfg.simulation.duration_minutes
        sim._run_duration = sim_duration
        sim.env.process(sim._patient_arrival_generator(sim_duration))
        sim.env.process(sim._stats_collector())
        st.session_state.sim = sim
        st.session_state.policy_name = policy_name
        st.session_state.sim_step = 0
        st.session_state.history = {k: [] for k in st.session_state.history}
        st.session_state.completed_patients = []
    return st.session_state.sim


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        
        policy = st.selectbox(
            "Scheduling Policy",
            ["fifo", "priority"],
            index=0,
            help="Choose the scheduling policy for the simulation",
        )
        
        st.markdown("---")
        st.markdown("### Simulation")
        
        duration = st.slider(
            "Duration (minutes)",
            min_value=60,
            max_value=1440,
            value=480,
            step=60,
            help="Total simulation duration",
        )
        
        step_size = st.slider(
            "Step Size (minutes)",
            min_value=1,
            max_value=30,
            value=5,
            help="Minutes per simulation step",
        )
        
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
        )
        
        st.markdown("---")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            step_btn = st.button("▶ Step", use_container_width=True, type="primary",
                                  help="Advance simulation by one step size")
        with col2:
            reset_btn = st.button("↺ Reset", use_container_width=True)

        batch_steps = st.number_input("Batch steps", min_value=1, max_value=200, value=10,
                                       help="How many steps to run at once")
        batch_btn = st.button(f"⏩ Run {batch_steps} Steps", use_container_width=True)

        if step_btn:
            st.session_state.sim_running = True
            st.session_state.do_steps = 1
        if batch_btn:
            st.session_state.sim_running = True
            st.session_state.do_steps = int(batch_steps)
        if reset_btn:
            st.session_state.sim = None
            st.session_state.sim_running = False
            st.session_state.do_steps = 0
        st.markdown("### 📁 Data")
        
        output_base = Path("outputs")
        csv_path = output_base / "analytics" / "policy_comparison.csv"
        dataset_path = output_base / "dataset" / "patients.parquet"
        
        if csv_path.exists():
            st.success("✓ Analytics CSV found")
        else:
            st.info("Run experiment to generate analytics")
        
        if dataset_path.exists():
            st.success("✓ Dataset found")
        else:
            st.info("Run generate_dataset to load dataset")
    
    return {
        "policy": policy,
        "duration": duration,
        "step_size": step_size,
        "refresh_rate": refresh_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Live Simulation
# ─────────────────────────────────────────────────────────────────────────────

def render_live_simulation(settings: dict[str, Any]) -> None:
    """Render live simulation dashboard tab."""
    sim = get_or_create_sim(settings["policy"])
    
    # Advance simulation by requested number of steps (no auto-rerun loop)
    steps_todo = st.session_state.get("do_steps", 0)
    if steps_todo > 0 and st.session_state.sim_running:
        abs_end = sim._start_time + sim._run_duration
        for _ in range(steps_todo):
            step_end = sim.env.now + settings["step_size"]
            if step_end >= abs_end:
                st.session_state.sim_running = False
                st.info("✅ Simulation complete!")
                break
            sim.run_until(step_end)
            st.session_state.sim_step += 1
            snapshot = sim.get_snapshot()
            h = st.session_state.history
            h["time"].append(sim.env.now)
            h["ct_queue"].append(snapshot.ct_queue_length)
            h["mri_queue"].append(snapshot.mri_queue_length)
            h["xray_queue"].append(snapshot.xray_queue_length)
            h["emg_queue"].append(snapshot.emergency_queue_length)
            h["ct_util"].append(snapshot.ct_utilization * 100)
            h["mri_util"].append(snapshot.mri_utilization * 100)
            h["xray_util"].append(snapshot.xray_utilization * 100)
            h["completed"].append(sim.stats.n_completed)
        st.session_state.do_steps = 0  # clear after executing
    
    snapshot = sim.get_snapshot()
    stats = sim.stats
    
    # ── KPI Metrics Row ──────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")
    kpi_cols = st.columns(6)
    
    avg_wait = float(np.mean(stats.wait_times)) if stats.wait_times else 0.0
    avg_tat = float(np.mean(stats.total_turnarounds)) if stats.total_turnarounds else 0.0
    avg_emg = float(np.mean(stats.emergency_turnarounds)) if stats.emergency_turnarounds else 0.0
    avg_util = (snapshot.ct_utilization + snapshot.mri_utilization + snapshot.xray_utilization) / 3
    throughput = stats.n_completed / max(sim.env.now / 60.0, 0.001)
    
    kpi_cols[0].metric("⏱ Avg Wait", f"{avg_wait:.1f} min", delta=None)
    kpi_cols[1].metric("🚨 Emg TAT", f"{avg_emg:.1f} min", delta=None)
    kpi_cols[2].metric("✅ Completed", f"{stats.n_completed}", f"+{snapshot.completed_last_epoch}")
    kpi_cols[3].metric("📈 Throughput", f"{throughput:.1f}/hr", delta=None)
    kpi_cols[4].metric("🖥 Avg Util", f"{avg_util*100:.1f}%", delta=None)
    kpi_cols[5].metric("🕐 Sim Time", f"{sim.env.now:.0f} min", delta=None)
    
    st.markdown("---")
    
    # ── Queue Status Row ─────────────────────────────────────────────────────
    col_q1, col_q2 = st.columns([2, 1])
    
    with col_q1:
        st.markdown("### 📋 Queue Lengths Over Time")
        if st.session_state.history["time"]:
            h = st.session_state.history
            fig_queue = go.Figure()
            fig_queue.add_trace(go.Scatter(x=h["time"], y=h["ct_queue"], name="CT", 
                                            line=dict(color="#3b82f6", width=2)))
            fig_queue.add_trace(go.Scatter(x=h["time"], y=h["mri_queue"], name="MRI",
                                            line=dict(color="#f59e0b", width=2)))
            fig_queue.add_trace(go.Scatter(x=h["time"], y=h["xray_queue"], name="X-Ray",
                                            line=dict(color="#22c55e", width=2)))
            fig_queue.add_trace(go.Scatter(x=h["time"], y=h["emg_queue"], name="Emergency",
                                            line=dict(color="#ef4444", width=2, dash="dash")))
            fig_queue.update_layout(
                template="plotly_dark",
                xaxis_title="Simulation Time (min)",
                yaxis_title="Queue Length",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_queue, use_container_width=True)
        else:
            st.info("Start simulation to see queue data")
    
    with col_q2:
        st.markdown("### 🏥 Scanner Status")
        for scanner_id, scanner in sim.scanners.items():
            util = scanner.utilization(max(sim.env.now, 1.0)) * 100
            status_color = "#22c55e" if scanner.status.value == "IDLE" else "#f59e0b"
            st.markdown(f"""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:8px 12px;margin:4px 0;'>
                <span style='color:{status_color};font-weight:600;'>{scanner_id}</span>
                <span style='color:#94a3b8;font-size:0.85rem;'> ({scanner.modality.value})</span>
                <div style='background:#334155;border-radius:4px;margin-top:4px;height:6px;'>
                    <div style='background:{status_color};width:{util:.0f}%;height:100%;border-radius:4px;'></div>
                </div>
                <span style='color:#94a3b8;font-size:0.75rem;'>{util:.1f}% utilization</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Utilization Charts ───────────────────────────────────────────────────
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        st.markdown("### 📡 Scanner Utilization History")
        if st.session_state.history["time"]:
            h = st.session_state.history
            fig_util = go.Figure()
            fig_util.add_trace(go.Scatter(x=h["time"], y=h["ct_util"], name="CT",
                                           fill="tonexty", line=dict(color="#3b82f6")))
            fig_util.add_trace(go.Scatter(x=h["time"], y=h["mri_util"], name="MRI",
                                           line=dict(color="#f59e0b")))
            fig_util.add_trace(go.Scatter(x=h["time"], y=h["xray_util"], name="X-Ray",
                                           line=dict(color="#22c55e")))
            fig_util.add_hline(y=85, line_dash="dot", line_color="white", 
                                annotation_text="Target 85%")
            fig_util.update_layout(
                template="plotly_dark", height=280,
                yaxis=dict(range=[0, 105], title="Utilization (%)"),
                xaxis_title="Time (min)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_util, use_container_width=True)
    
    with col_u2:
        st.markdown("### 👨‍⚕️ Radiologist Workload")
        rad_data = sim.radiologist_workloads()
        if rad_data:
            rad_ids = list(rad_data.keys())
            reads = [rad_data[r]["total_reads"] for r in rad_ids]
            max_reads = [rad_data[r]["max_daily_reads"] for r in rad_ids]
            
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Bar(
                x=rad_ids, y=reads, name="Reads Done",
                marker_color="#3b82f6",
            ))
            fig_rad.add_trace(go.Scatter(
                x=rad_ids, y=max_reads, name="Max Capacity",
                mode="markers+lines",
                marker=dict(color="#ef4444", size=8),
                line=dict(color="#ef4444", dash="dash"),
            ))
            fig_rad.update_layout(
                template="plotly_dark", height=280,
                yaxis_title="Reports",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_rad, use_container_width=True)
    
    # Removed auto-refresh loop (caused SIGSEGV on macOS with heavy Plotly renders)
    # Use the Step / Batch buttons above to advance the simulation.


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: RL Training Progress
# ─────────────────────────────────────────────────────────────────────────────

def render_rl_training() -> None:
    """Render RL training progress tab."""
    st.markdown("### 🤖 PPO Training Progress")
    
    output_dir = Path("outputs/training")
    model_path = output_dir / "final_model.zip"
    
    if not output_dir.exists():
        st.info("No training output found. Run `medimg-train` to train a PPO agent.")
        
        st.markdown("#### Quick Start")
        st.code("""
# Fast training (10k steps, ~2 min)
uv run medimg-train --fast

# Full training (500k steps, ~30-40 min)
uv run medimg-train

# With custom config
uv run medimg-train --timesteps 100000 --seed 42
        """, language="bash")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if model_path.exists():
            st.success(f"✓ Trained model: `{model_path}`")
        else:
            st.warning("Model not found — training may still be in progress")
    
    # Look for tensorboard event files
    tb_dir = output_dir / "tensorboard"
    if tb_dir.exists():
        st.markdown("#### Training Curves")
        st.info("📊 View detailed TensorBoard logs with: `tensorboard --logdir outputs/training/tensorboard`")
    
    # Look for evaluation results
    eval_results_path = output_dir / "eval_results.json"
    if eval_results_path.exists():
        with eval_results_path.open() as f:
            eval_results = json.load(f)
        
        st.markdown("#### Evaluation Results")
        cols = st.columns(3)
        cols[0].metric("Mean Reward", f"{eval_results.get('mean_reward', 0):.3f}")
        cols[1].metric("Std Reward", f"±{eval_results.get('std_reward', 0):.3f}")
        cols[2].metric("Episodes", f"{eval_results.get('n_episodes', 0)}")
    
    # Simulated training curve visualization
    st.markdown("#### Reward Trajectory (Simulated)")
    if st.session_state.reward_history:
        rewards = st.session_state.reward_history
    else:
        # Generate placeholder curve
        rng = np.random.default_rng(42)
        n = 200
        t = np.arange(n)
        rewards = -8.0 * np.exp(-t / 80) - 2.0 + rng.normal(0, 0.5, n)
        rewards = list(np.cumsum(rng.normal(0.01, 0.2, n)))
    
    smooth_window = min(20, len(rewards) // 5 + 1)
    smoothed = pd.Series(rewards).rolling(smooth_window, min_periods=1).mean().tolist()
    
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(y=rewards, name="Episode Reward",
                                     line=dict(color="#94a3b8", width=1), opacity=0.4))
    fig_reward.add_trace(go.Scatter(y=smoothed, name=f"Smoothed (w={smooth_window})",
                                     line=dict(color="#3b82f6", width=2)))
    fig_reward.update_layout(
        template="plotly_dark",
        xaxis_title="Episode",
        yaxis_title="Cumulative Reward",
        height=350,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_reward, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Policy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def render_policy_comparison() -> None:
    """Render policy comparison tab."""
    st.markdown("### 📊 Policy Comparison")
    
    csv_path = Path("outputs/analytics/policy_comparison.csv")
    
    if not csv_path.exists():
        st.info("No policy comparison data found.")
        st.code("uv run medimg-experiment --policy all --n-runs 5", language="bash")
        return
    
    df = pd.read_csv(csv_path)
    
    # Summary metrics table
    st.markdown("#### Metrics Summary")
    st.dataframe(
        df.style.background_gradient(cmap="RdYlGn", axis=0),
        use_container_width=True,
    )
    
    # Visualization
    col1, col2 = st.columns(2)
    
    if "avg_wait_time_min" in df.columns and "policy_name" in df.columns:
        with col1:
            fig = px.bar(
                df, x="policy_name", y="avg_wait_time_min",
                color="policy_name",
                title="Average Wait Time (minutes)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.add_hline(y=30, line_dash="dot", annotation_text="Target: 30 min")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "avg_emergency_tat_min" in df.columns:
                fig2 = px.bar(
                    df, x="policy_name", y="avg_emergency_tat_min",
                    color="policy_name",
                    title="Emergency Turnaround Time (minutes)",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # Scanner utilization comparison
    if all(c in df.columns for c in ["ct_utilization", "mri_utilization", "xray_utilization"]):
        st.markdown("#### Scanner Utilization by Policy")
        util_data = []
        for _, row in df.iterrows():
            for mod in ["CT", "MRI", "XRAY"]:
                col_name = f"{mod.lower()}_utilization"
                if col_name in df.columns:
                    util_data.append({
                        "Policy": row["policy_name"],
                        "Modality": mod,
                        "Utilization (%)": row[col_name] * 100,
                    })
        
        if util_data:
            util_df = pd.DataFrame(util_data)
            fig_util = px.bar(
                util_df, x="Policy", y="Utilization (%)",
                color="Modality", barmode="group",
                template="plotly_dark",
                color_discrete_sequence=["#3b82f6", "#f59e0b", "#22c55e"],
                title="Scanner Utilization by Modality and Policy",
            )
            fig_util.add_hline(y=85, line_dash="dot", annotation_text="Target 85%")
            st.plotly_chart(fig_util, use_container_width=True)
    
    # Check for figure files
    figures_dir = Path("outputs/figures")
    if figures_dir.exists():
        png_files = list(figures_dir.glob("*.png"))
        if png_files:
            st.markdown("#### Generated Publication Figures")
            fig_cols = st.columns(min(3, len(png_files)))
            for i, fig_path in enumerate(png_files[:6]):
                with fig_cols[i % 3]:
                    st.image(str(fig_path), caption=fig_path.stem, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Patient Timeline
# ─────────────────────────────────────────────────────────────────────────────

def render_patient_timeline() -> None:
    """Render patient timeline Gantt view."""
    st.markdown("### 🗓 Patient Journey Timeline")
    
    sim = st.session_state.sim
    if sim is None:
        st.info("Start the simulation in the Live Simulation tab first.")
        return
    
    # Get completed patients
    completed = [
        p for p in sim.patients.values()
        if p.status == PatientStatus.DISCHARGED
        and p.scan_start_time is not None
        and p.report_end_time is not None
    ]
    
    if not completed:
        st.info("No completed patients yet. Advance the simulation further.")
        return
    
    # Show last 20 patients
    display = sorted(completed, key=lambda p: p.arrival_time, reverse=True)[:20]
    
    # Build Gantt data
    gantt_data = []
    priority_colors = {
        Priority.ROUTINE: "#22c55e",
        Priority.URGENT: "#f59e0b",
        Priority.EMERGENCY: "#ef4444",
    }
    
    for p in display:
        color = priority_colors.get(p.priority, "#94a3b8")
        pid_short = p.patient_id[-6:]
        
        if p.scan_start_time and p.scan_end_time:
            gantt_data.append({
                "Patient": f"{pid_short} ({p.modality.value})",
                "Phase": "Waiting",
                "Start": p.arrival_time,
                "End": p.scan_start_time,
                "Color": color,
                "Priority": p.priority.name,
            })
            gantt_data.append({
                "Patient": f"{pid_short} ({p.modality.value})",
                "Phase": "Scanning",
                "Start": p.scan_start_time,
                "End": p.scan_end_time,
                "Color": "#3b82f6",
                "Priority": p.priority.name,
            })
        if p.report_start_time and p.report_end_time:
            gantt_data.append({
                "Patient": f"{pid_short} ({p.modality.value})",
                "Phase": "Reporting",
                "Start": p.report_start_time,
                "End": p.report_end_time,
                "Color": "#8b5cf6",
                "Priority": p.priority.name,
            })
    
    if gantt_data:
        gantt_df = pd.DataFrame(gantt_data)
        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="End",
            y="Patient",
            color="Phase",
            template="plotly_dark",
            color_discrete_map={
                "Waiting": "#94a3b8",
                "Scanning": "#3b82f6",
                "Reporting": "#8b5cf6",
            },
            title=f"Patient Journeys (last {len(display)} completed)",
        )
        fig.update_xaxes(title_text="Simulation Time (minutes)")
        fig.update_layout(height=max(400, len(display) * 25))
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    wait_times = [p.wait_time for p in display if p.wait_time is not None]
    tats = [p.total_turnaround for p in display if p.total_turnaround is not None]
    
    if wait_times:
        col1.metric("Avg Wait (displayed)", f"{np.mean(wait_times):.1f} min")
    if tats:
        col2.metric("Avg TAT (displayed)", f"{np.mean(tats):.1f} min")
    col3.metric("Shown Patients", len(display))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────

def render_dataset_explorer() -> None:
    """Render synthetic dataset explorer tab."""
    st.markdown("### 🗃 Synthetic Dataset Explorer")
    
    dataset_path = Path("outputs/dataset/patients.parquet")
    
    if not dataset_path.exists():
        st.info("Dataset not found. Generate it first:")
        st.code("uv run medimg-generate --n-patients 100000 --seed 42", language="bash")
        return
    
    @st.cache_data
    def load_dataset(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
    
    df = load_dataset(str(dataset_path))
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Modalities", df["modality"].nunique() if "modality" in df.columns else "N/A")
    col3.metric("Avg Wait (min)", f"{df['wait_time'].mean():.1f}" if "wait_time" in df.columns else "N/A")
    col4.metric("Emergency %", f"{(df['priority']=='EMERGENCY').mean()*100:.1f}%" if "priority" in df.columns else "N/A")
    
    st.markdown("---")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        modality_filter = st.multiselect(
            "Modality",
            options=["CT", "MRI", "XRAY"],
            default=["CT", "MRI", "XRAY"],
        )
    with col_f2:
        priority_filter = st.multiselect(
            "Priority",
            options=["ROUTINE", "URGENT", "EMERGENCY"],
            default=["ROUTINE", "URGENT", "EMERGENCY"],
        )
    with col_f3:
        max_rows = st.slider("Max rows to display", 100, 5000, 500, 100)
    
    # Apply filters
    filtered = df.copy()
    if "modality" in df.columns:
        filtered = filtered[filtered["modality"].isin(modality_filter)]
    if "priority" in df.columns:
        filtered = filtered[filtered["priority"].isin(priority_filter)]
    
    st.markdown(f"Showing {min(max_rows, len(filtered)):,} of {len(filtered):,} matching records")
    st.dataframe(filtered.head(max_rows), use_container_width=True)
    
    # Distribution charts
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if "modality" in df.columns:
            fig_mod = px.pie(
                df, names="modality",
                title="Modality Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_mod, use_container_width=True)
    
    with col_c2:
        if "wait_time" in df.columns:
            fig_wait = px.histogram(
                df.sample(min(5000, len(df))),
                x="wait_time",
                color="priority" if "priority" in df.columns else None,
                nbins=60,
                title="Wait Time Distribution (minutes)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_wait, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main dashboard entry point."""
    init_session_state()
    
    # Load config once
    if st.session_state.config is None:
        try:
            st.session_state.config = load_config()
        except Exception:
            st.session_state.config = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Imaging Digital Twin</h1>
        <p>Hospital Imaging Workflow Optimizer — SimPy + Gymnasium + PPO</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔴 Live Simulation",
        "🤖 RL Training",
        "📊 Policy Comparison",
        "🗓 Patient Timeline",
        "🗃 Dataset Explorer",
    ])
    
    with tab1:
        render_live_simulation(settings)
    
    with tab2:
        render_rl_training()
    
    with tab3:
        render_policy_comparison()
    
    with tab4:
        render_patient_timeline()
    
    with tab5:
        render_dataset_explorer()


if __name__ == "__main__":
    main()
