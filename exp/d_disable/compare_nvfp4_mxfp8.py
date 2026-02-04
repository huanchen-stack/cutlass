#!/usr/bin/env python3
"""
Compare NVFP4 vs MXFP8 grouped GEMM performance at groups=128.
8 bars per batch size: 4 schemes x 2 precisions.
NVFP4 = solid fill, MXFP8 = "/" hatching. Same color per scheme.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
M_VALUES = [32, 64, 128, 256, 512]
GROUPS = "128"

# 4 schemes in order
SCHEMES = [
    ("mma=1,tma=1,epi=1", "All disabled"),
    ("mma=0,tma=1,epi=1", "MMA only"),
    ("mma=1,tma=0,epi=1", "TMA only"),
    ("mma=0,tma=0,epi=0", "All enabled"),
]

# Colors from analyze.py
COLORS = ["#d62728", "#ff9896", "#98df8a", "#1f77b4"]


def load_sorted(subdir):
    with open(SCRIPT_DIR / subdir / "results_sorted.json") as f:
        return json.load(f)


def make_plot(nvfp4_dir, mxfp8_dir, title_suffix, out_name):
    nvfp4 = load_sorted(nvfp4_dir)
    mxfp8 = load_sorted(mxfp8_dir)

    fig, ax = plt.subplots(figsize=(12, 6))

    n_m = len(M_VALUES)
    n_bars = 8  # 4 schemes x 2 precisions
    bar_width = 0.09
    x = np.arange(n_m)

    for i, (key, label) in enumerate(SCHEMES):
        # NVFP4 (solid)
        nvfp4_times = [nvfp4[GROUPS][str(m)][key] for m in M_VALUES]
        offset_nv = (2 * i - (n_bars - 1) / 2) * bar_width
        ax.bar(
            x + offset_nv,
            nvfp4_times,
            bar_width,
            label=f"NVFP4 {label}",
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.5,
        )

        # MXFP8 (hatched)
        mxfp8_times = [mxfp8[GROUPS][str(m)][key] for m in M_VALUES]
        offset_mx = (2 * i + 1 - (n_bars - 1) / 2) * bar_width
        ax.bar(
            x + offset_mx,
            mxfp8_times,
            bar_width,
            label=f"MXFP8 {label}",
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
            alpha=0.7,
        )

    ax.set_xlabel("Batch Size (m)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(
        f"NVFP4 vs MXFP8 Grouped GEMM Performance{title_suffix} (groups=128, n=768, k=2048)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(M_VALUES)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = SCRIPT_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_plot("ori", "mxfp8_ori", "", "compare_nvfp4_mxfp8.png")
    make_plot("n32", "mxfp8_n32", " [N32]", "compare_nvfp4_mxfp8_n32.png")
