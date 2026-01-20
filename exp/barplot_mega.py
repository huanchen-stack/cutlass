#!/usr/bin/env python3
"""
Create a MEGA bar plot where all group counts are merged into bar groups,
with each group sharing the same batch size (M value) on the x-axis.
Each bar shows stacked contributions: baseline (no_tma_no_mma), +MMA, +TMA.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Experiment parameters (from prof_group_batch.py)
M_VALUES = [1, 8, 16, 32, 64, 128, 256, 512]
GROUP_VALUES = [8, 16, 32, 64, 128, 256]
N = 768
K = 2048

def load_results():
    """Load all three result files."""
    with open("exp_results_nvfp4_no_tma_no_mma.json", "r") as f:
        no_tma_no_mma = json.load(f)
    with open("exp_results_nvfp4_no_tma.json", "r") as f:
        no_tma = json.load(f)
    with open("exp_results_nvfp4.json", "r") as f:
        full = json.load(f)
    return no_tma_no_mma, no_tma, full

def create_mega_barplot():
    """Create a mega bar plot with all groups merged into bar groups per M value."""
    no_tma_no_mma, no_tma, full = load_results()

    fig, ax = plt.subplots(figsize=(18, 8))

    n_groups = len(GROUP_VALUES)
    n_m_values = len(M_VALUES)

    # Width of each bar and spacing
    bar_width = 0.12
    group_spacing = 0.15  # Extra space between M value clusters

    # Colors for the stacked components
    color_base = '#2ecc71'  # Green - baseline
    color_mma = '#3498db'   # Blue - +MMA
    color_tma = '#e74c3c'   # Red - +TMA

    # For each M value, we place n_groups bars side by side
    for m_idx, m_val in enumerate(M_VALUES):
        # Calculate the center position for this M value's bar group
        group_center = m_idx * (n_groups * bar_width + group_spacing)

        for group_idx, groups in enumerate(GROUP_VALUES):
            # Get runtimes
            base_time = no_tma_no_mma[group_idx][m_idx]
            with_mma_time = no_tma[group_idx][m_idx]
            full_time = full[group_idx][m_idx]

            # Calculate incremental overhead
            mma_overhead = with_mma_time - base_time
            tma_overhead = full_time - with_mma_time

            # Position for this specific bar
            x_pos = group_center + (group_idx - n_groups/2 + 0.5) * bar_width

            # Create stacked bar (only add labels for first bar to avoid legend duplicates)
            label_base = 'Baseline (no MMA, no TMA)' if m_idx == 0 and group_idx == 0 else None
            label_mma = '+MMA' if m_idx == 0 and group_idx == 0 else None
            label_tma = '+TMA' if m_idx == 0 and group_idx == 0 else None

            ax.bar(x_pos, base_time, bar_width, label=label_base, color=color_base, edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, mma_overhead, bar_width, bottom=base_time, label=label_mma, color=color_mma, edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, tma_overhead, bar_width, bottom=base_time + mma_overhead, label=label_tma, color=color_tma, edgecolor='white', linewidth=0.5)

    # Set x-axis ticks at the center of each M value group
    x_tick_positions = [m_idx * (n_groups * bar_width + group_spacing) for m_idx in range(n_m_values)]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f'M={m}' for m in M_VALUES], fontsize=11)

    # Add a secondary legend for group counts
    # Create proxy artists for group legend
    from matplotlib.patches import Patch
    group_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_groups))

    ax.set_xlabel('Batch Size (M)', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title(f'NVFP4 Grouped GEMM Performance Breakdown\n(n={N}, k={K}) - All Groups Combined', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation for group ordering within each cluster
    ax.text(0.98, 0.02, f'Within each cluster: groups={GROUP_VALUES}',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('barplot_mega.png', dpi=150)
    print("Saved: barplot_mega.png")

def create_mega_barplot_by_groups():
    """Create a mega bar plot with all M values merged into bar groups per group count."""
    no_tma_no_mma, no_tma, full = load_results()

    fig, ax = plt.subplots(figsize=(18, 8))

    n_groups = len(GROUP_VALUES)
    n_m_values = len(M_VALUES)

    # Width of each bar and spacing
    bar_width = 0.10
    group_spacing = 0.15  # Extra space between group count clusters

    # Colors for the stacked components
    color_base = '#2ecc71'  # Green - baseline
    color_mma = '#3498db'   # Blue - +MMA
    color_tma = '#e74c3c'   # Red - +TMA

    # For each group count, we place n_m_values bars side by side
    for group_idx, groups in enumerate(GROUP_VALUES):
        # Calculate the center position for this group count's bar cluster
        cluster_center = group_idx * (n_m_values * bar_width + group_spacing)

        for m_idx, m_val in enumerate(M_VALUES):
            # Get runtimes
            base_time = no_tma_no_mma[group_idx][m_idx]
            with_mma_time = no_tma[group_idx][m_idx]
            full_time = full[group_idx][m_idx]

            # Calculate incremental overhead
            mma_overhead = with_mma_time - base_time
            tma_overhead = full_time - with_mma_time

            # Position for this specific bar
            x_pos = cluster_center + (m_idx - n_m_values/2 + 0.5) * bar_width

            # Create stacked bar (only add labels for first bar to avoid legend duplicates)
            label_base = 'Baseline (no MMA, no TMA)' if group_idx == 0 and m_idx == 0 else None
            label_mma = '+MMA' if group_idx == 0 and m_idx == 0 else None
            label_tma = '+TMA' if group_idx == 0 and m_idx == 0 else None

            ax.bar(x_pos, base_time, bar_width, label=label_base, color=color_base, edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, mma_overhead, bar_width, bottom=base_time, label=label_mma, color=color_mma, edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, tma_overhead, bar_width, bottom=base_time + mma_overhead, label=label_tma, color=color_tma, edgecolor='white', linewidth=0.5)

    # Set x-axis ticks at the center of each group count cluster
    x_tick_positions = [group_idx * (n_m_values * bar_width + group_spacing) for group_idx in range(n_groups)]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f'Groups={g}' for g in GROUP_VALUES], fontsize=11)

    ax.set_xlabel('Number of Groups', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title(f'NVFP4 Grouped GEMM Performance Breakdown\n(n={N}, k={K}) - Grouped by Group Count', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation for M ordering within each cluster
    ax.text(0.98, 0.02, f'Within each cluster: M={M_VALUES}',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('barplot_mega_by_groups.png', dpi=150)
    print("Saved: barplot_mega_by_groups.png")

if __name__ == "__main__":
    create_mega_barplot()
    create_mega_barplot_by_groups()