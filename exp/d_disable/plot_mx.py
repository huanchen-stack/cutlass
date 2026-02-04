#!/usr/bin/env python3
"""
Plot MXFP8 activations with different weight precisions (MXFP8, MXFP6, MXFP4).
12 bars per batch size: 4 schemes x 3 precisions.
MXFP8 = solid, MXFP6 = "//" hatch, MXFP4 = "xx" hatch. Same color per scheme.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
M_VALUES = [32, 64, 128, 256, 512]

# 4 schemes in order
SCHEMES = [
    (1, 1, 1, "All disabled"),
    (0, 1, 1, "MMA only"),
    (1, 0, 1, "TMA only"),
    (0, 0, 0, "All enabled"),
]

# Colors from original script - one per scheme
COLORS = ["#d62728", "#ff9896", "#98df8a", "#1f77b4"]

# Format B precisions
FORMATS = ["MXFP8", "MXFP6", "MXFP4"]
HATCHES = [None, "//", "xx"]  # solid, diagonal, cross


def load_data():
    """Load and organize data from results_mx.log"""
    data = {}
    with open(SCRIPT_DIR / "results_mx.log") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Create key from disable flags
            disable_key = (
                record["disable_mma"],
                record["disable_tma"],
                record["disable_epilogue"],
            )
            format_b = record["format_B"]
            m = record["m"]
            time_ms = record["time_ms"]

            if disable_key not in data:
                data[disable_key] = {}
            if format_b not in data[disable_key]:
                data[disable_key][format_b] = {}
            data[disable_key][format_b][m] = time_ms

    return data


def make_plot():
    data = load_data()

    fig, ax = plt.subplots(figsize=(14, 6))

    n_m = len(M_VALUES)
    n_bars = 12  # 4 schemes x 3 formats
    bar_width = 0.06
    x = np.arange(n_m)

    bar_idx = 0
    for scheme_idx, (disable_mma, disable_tma, disable_epi, label) in enumerate(SCHEMES):
        disable_key = (disable_mma, disable_tma, disable_epi)

        for fmt_idx, (fmt, hatch) in enumerate(zip(FORMATS, HATCHES)):
            # Get times for this scheme + format combination
            times = []
            for m in M_VALUES:
                try:
                    times.append(data[disable_key][fmt][m])
                except KeyError:
                    times.append(0)  # Missing data

            offset = (bar_idx - (n_bars - 1) / 2) * bar_width

            ax.bar(
                x + offset,
                times,
                bar_width,
                label=f"{fmt} {label}",
                color=COLORS[scheme_idx],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
                alpha=0.85 if hatch else 1.0,
            )

            bar_idx += 1

    ax.set_xlabel("Batch Size (m)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(
        "MXFP8 Activations with Different Weight Precisions (groups=128, n=768, k=2048)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(M_VALUES)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = SCRIPT_DIR / "plot_mx.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_plot()
