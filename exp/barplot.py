#!/usr/bin/env python3
"""
Create a stacked bar plot showing the incremental contributions of MMA and TMA
optimizations on top of the baseline (no_tma_no_mma).
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

def create_barplot():
    """Create stacked bar plot showing incremental contributions."""
    no_tma_no_mma, no_tma, full = load_results()

    # Create a figure with subplots for each group count
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for group_idx, groups in enumerate(GROUP_VALUES):
        ax = axes[group_idx]

        # Get runtimes for this group count
        base_times = np.array(no_tma_no_mma[group_idx])  # baseline (no_tma_no_mma)
        with_mma_times = np.array(no_tma[group_idx])     # no_tma (has MMA)
        full_times = np.array(full[group_idx])           # full (has MMA + TMA)

        # Calculate the incremental additions (overhead from each feature)
        # base: no_tma_no_mma runtime
        # +mma: additional time from enabling MMA (no_tma - no_tma_no_mma)
        # +tma: additional time from enabling TMA (full - no_tma)
        mma_overhead = with_mma_times - base_times
        tma_overhead = full_times - with_mma_times

        x = np.arange(len(M_VALUES))
        width = 0.6

        # Create stacked bars
        bars1 = ax.bar(x, base_times, width, label='Baseline (no MMA, no TMA)', color='#2ecc71')
        bars2 = ax.bar(x, mma_overhead, width, bottom=base_times, label='+MMA', color='#3498db')
        bars3 = ax.bar(x, tma_overhead, width, bottom=base_times + mma_overhead, label='+TMA', color='#e74c3c')

        ax.set_xlabel('M (batch size)')
        ax.set_ylabel('Runtime (ms)')
        ax.set_title(f'Groups = {groups}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in M_VALUES])
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'NVFP4 Grouped GEMM Performance Breakdown\n(n={N}, k={K})', fontsize=14)
    plt.tight_layout()
    plt.savefig('barplot_nvfp4_breakdown.png', dpi=150)
    print("Saved: barplot_nvfp4_breakdown.png")

def create_single_barplot():
    """Create a single combined bar plot with all configurations side by side."""
    no_tma_no_mma, no_tma, full = load_results()

    # Select a representative group count (e.g., 64) or aggregate
    # Let's create one plot per M value, showing all groups

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create x positions for grouped bars
    # Each M value will have a cluster of bars for different group counts
    n_groups = len(GROUP_VALUES)
    n_m_values = len(M_VALUES)

    # Use group_idx=3 (groups=64) as representative
    group_idx = 3
    groups = GROUP_VALUES[group_idx]

    base_times = np.array(no_tma_no_mma[group_idx])
    with_mma_times = np.array(no_tma[group_idx])
    full_times = np.array(full[group_idx])

    mma_overhead = with_mma_times - base_times
    tma_overhead = full_times - with_mma_times

    x = np.arange(len(M_VALUES))
    width = 0.6

    # Create stacked bars
    bars1 = ax.bar(x, base_times, width, label='Baseline (no MMA, no TMA)', color='#2ecc71')
    bars2 = ax.bar(x, mma_overhead, width, bottom=base_times, label='+MMA', color='#3498db')
    bars3 = ax.bar(x, tma_overhead, width, bottom=base_times + mma_overhead, label='+TMA', color='#e74c3c')

    ax.set_xlabel('M (batch size)', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title(f'NVFP4 Grouped GEMM Performance Breakdown (Groups={groups}, n={N}, k={K})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('barplot_nvfp4_single.png', dpi=150)
    print("Saved: barplot_nvfp4_single.png")

if __name__ == "__main__":
    create_barplot()
    create_single_barplot()