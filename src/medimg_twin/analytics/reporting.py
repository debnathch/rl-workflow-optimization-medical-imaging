from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from medimg_twin.analytics.metrics import PolicyMetrics
from medimg_twin.simulation.hospital import SimulationStats

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: Path | str, figures_dir: Path | str, dpi: int = 300, palette: str = 'Set2'):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.dpi = dpi
        self.palette = palette
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(palette)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_policy_comparison(self, metrics_list: list[PolicyMetrics]) -> Path:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        df = pd.DataFrame([m.to_dict() for m in metrics_list])
        
        # 1. Bar chart: avg wait time
        sns.barplot(data=df, x='policy_name', y='avg_wait_time_min', ax=axes[0])
        axes[0].set_title('Avg Wait Time by Policy')
        axes[0].set_ylabel('Minutes')
        # add error bars for std if possible (we have std but seaborn barplot from aggregated data doesn't easily show custom error bars, so we plot value)
        
        # 2. Bar chart: emergency TAT
        sns.barplot(data=df, x='policy_name', y='avg_emergency_tat_min', ax=axes[1])
        axes[1].set_title('Emergency TAT by Policy')
        axes[1].set_ylabel('Minutes')
        
        # 3. Bar chart: scanner utilization
        util_cols = ['ct_utilization', 'mri_utilization', 'xray_utilization']
        df_util = df[['policy_name'] + util_cols].melt(id_vars='policy_name', var_name='Scanner', value_name='Utilization')
        sns.barplot(data=df_util, x='policy_name', y='Utilization', hue='Scanner', ax=axes[2])
        axes[2].set_title('Scanner Utilization')
        axes[2].set_ylabel('Utilization (0-1)')
        
        # 4. Bar chart: throughput per hour
        sns.barplot(data=df, x='policy_name', y='throughput_per_hour', ax=axes[3])
        axes[3].set_title('Throughput per Hour')
        axes[3].set_ylabel('Patients / Hr')
        
        # 5. Bar chart: radiologist workload std
        sns.barplot(data=df, x='policy_name', y='radiologist_workload_std', ax=axes[4])
        axes[4].set_title('Radiologist Workload Std (Balance)')
        axes[4].set_ylabel('Std Dev')
        
        # 6. Bar chart: avg queue length
        sns.barplot(data=df, x='policy_name', y='avg_queue_length', ax=axes[5])
        axes[5].set_title('Avg Queue Length')
        axes[5].set_ylabel('Patients')
        
        # Value labels
        for ax in axes:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
                
        plt.tight_layout()
        
        png_path = self.figures_dir / 'policy_comparison.png'
        pdf_path = self.figures_dir / 'policy_comparison.pdf'
        fig.savefig(png_path, dpi=self.dpi)
        fig.savefig(pdf_path, dpi=self.dpi)
        plt.close(fig)
        
        return png_path

    def generate_wait_time_cdf(self, stats_by_policy: dict[str, SimulationStats]) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy_name, stats in stats_by_policy.items():
            if hasattr(stats, 'completed_patients') and stats.completed_patients:
                wait_times = [p.wait_time for p in stats.completed_patients]
                if wait_times:
                    sns.ecdfplot(wait_times, ax=ax, label=policy_name)
                    
                    p50 = np.percentile(wait_times, 50)
                    p95 = np.percentile(wait_times, 95)
                    ax.axvline(p50, linestyle='--', alpha=0.5)
                    ax.axvline(p95, linestyle=':', alpha=0.5)
                
        ax.set_title('Wait Time CDF')
        ax.set_xlabel('Wait Time (minutes)')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        
        png_path = self.figures_dir / 'wait_time_cdf.png'
        pdf_path = self.figures_dir / 'wait_time_cdf.pdf'
        fig.savefig(png_path, dpi=self.dpi)
        fig.savefig(pdf_path, dpi=self.dpi)
        plt.close(fig)
        
        return png_path

    def generate_queue_time_series(self, stats_by_policy: dict[str, SimulationStats]) -> Path:
        n_policies = len(stats_by_policy)
        if n_policies == 0:
            return self.figures_dir / 'queue_time_series.png'
            
        fig, axes = plt.subplots(n_policies, 1, figsize=(12, 4 * n_policies), sharex=True)
        if n_policies == 1:
            axes = [axes]
            
        for ax, (policy_name, stats) in zip(axes, stats_by_policy.items()):
            if hasattr(stats, 'queue_snapshots') and stats.queue_snapshots:
                times = [s.get('time', i) if isinstance(s, dict) else i for i, s in enumerate(stats.queue_snapshots)]
                for k, color in [('CT', 'blue'), ('MRI', 'orange'), ('XRAY', 'green'), ('Emergency', 'red')]:
                    vals = [s.get(k, 0) if isinstance(s, dict) else s for s in stats.queue_snapshots]
                    ls = '--' if k == 'Emergency' else '-'
                    ax.plot(times, vals, label=k, color=color, linestyle=ls)
            
            ax.set_title(f'Queue Lengths - {policy_name}')
            ax.set_ylabel('Queue Length')
            ax.legend()
            
        axes[-1].set_xlabel('Simulation Time (minutes)')
        plt.tight_layout()
        
        png_path = self.figures_dir / 'queue_time_series.png'
        pdf_path = self.figures_dir / 'queue_time_series.pdf'
        fig.savefig(png_path, dpi=self.dpi)
        fig.savefig(pdf_path, dpi=self.dpi)
        plt.close(fig)
        
        return png_path

    def generate_scanner_utilization_heatmap(self, utils_by_policy: dict[str, dict[str, float]]) -> Path:
        df = pd.DataFrame(utils_by_policy).T
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
        
        ax.set_title('Scanner Utilization Heatmap')
        ax.set_xlabel('Scanner')
        ax.set_ylabel('Policy')
        
        png_path = self.figures_dir / 'scanner_util_heatmap.png'
        pdf_path = self.figures_dir / 'scanner_util_heatmap.pdf'
        fig.savefig(png_path, dpi=self.dpi)
        fig.savefig(pdf_path, dpi=self.dpi)
        plt.close(fig)
        
        return png_path

    def generate_rl_training_curve(self, reward_history: list[float], eval_history: list[tuple[int, float]]) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if reward_history:
            s = pd.Series(reward_history)
            ax1.plot(s, alpha=0.3, color='blue')
            ax1.plot(s.rolling(20).mean(), color='blue', label='Moving Avg (20)')
            ax1.set_title('Episode Reward')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            
        if eval_history:
            steps, rews = zip(*eval_history)
            ax2.plot(steps, rews, marker='o', color='green')
            ax2.set_title('Evaluation Mean Reward')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Mean Reward')
            
        plt.tight_layout()
        
        png_path = self.figures_dir / 'rl_training_curve.png'
        pdf_path = self.figures_dir / 'rl_training_curve.pdf'
        fig.savefig(png_path, dpi=self.dpi)
        fig.savefig(pdf_path, dpi=self.dpi)
        plt.close(fig)
        
        return png_path

    def export_csv_summary(self, metrics_list: list[PolicyMetrics]) -> Path:
        df = pd.DataFrame([m.to_dict() for m in metrics_list])
        if 'policy_name' in df.columns:
            df.set_index('policy_name', inplace=True)
        
        csv_path = self.output_dir / 'policy_summary.csv'
        df.to_csv(csv_path)
        
        tex_path = self.output_dir / 'policy_summary.tex'
        with open(tex_path, 'w') as f:
            f.write(df.to_latex())
            
        return csv_path
