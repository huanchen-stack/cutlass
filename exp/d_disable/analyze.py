"""
read from results.log

1. provide a re-sorted version of the results:
    Currently, resutls with the same cmake flags are grouped together.
    Instead, we want to group by groups and then batch sizes (value of m).

2. create a mega plot that shows the results per groups size
    (so three plots for the three groups sizes: 8, 32, 128)
    And the x-axis is the batch size (m)
    Each bar is a compounded bar with multiple values from cmake configuations.
        the base is the value for disabling everything
        the neck should be the value of disabling one of the components (mma, tma, epilogue)
            there will be three such necks in the bar
        the top is the value of disabling nothing (all components enabled)
    use savefig to save the three plots as png files.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Constants
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "results.log"
SORTED_FILE = SCRIPT_DIR / "results_sorted.json"

M_VALUES = [32, 64, 128, 256, 512]
GROUPS_VALUES = [8, 32, 128]

# Config keys: (disable_mma, disable_tma, disable_epilogue)
# Order: base (all disabled) -> enable one -> disable one -> top (all enabled)
CONFIG_ORDER = [
    # All disabled (base)
    (1, 1, 1),
    # Enable only ONE component (two disabled)
    (0, 1, 1),  # MMA enabled only
    (1, 0, 1),  # TMA enabled only
    (1, 1, 0),  # Epilogue enabled only
    # Disable only ONE component (two enabled)
    (1, 0, 0),  # MMA disabled only
    (0, 1, 0),  # TMA disabled only
    (0, 0, 1),  # Epilogue disabled only
    # All enabled (top)
    (0, 0, 0),
]

CONFIG_LABELS = {
    (1, 1, 1): "All disabled",
    (0, 1, 1): "MMA only",
    (1, 0, 1): "TMA only",
    (1, 1, 0): "Epilogue only",
    (1, 0, 0): "MMA disabled",
    (0, 1, 0): "TMA disabled",
    (0, 0, 1): "Epilogue disabled",
    (0, 0, 0): "All enabled",
}

# Colors for the bars (8 distinct colors)
CONFIG_COLORS = {
    (1, 1, 1): "#d62728",  # red - all disabled
    (0, 1, 1): "#ff9896",  # light red - MMA only
    (1, 0, 1): "#98df8a",  # light green - TMA only
    (1, 1, 0): "#c5b0d5",  # light purple - Epilogue only
    (1, 0, 0): "#ff7f0e",  # orange - MMA disabled
    (0, 1, 0): "#2ca02c",  # green - TMA disabled
    (0, 0, 1): "#9467bd",  # purple - Epilogue disabled
    (0, 0, 0): "#1f77b4",  # blue - all enabled
}


def load_results() -> list[dict]:
    """Load results from results.log (JSON lines format)."""
    results = []
    if not LOG_FILE.exists():
        print(f"[ERROR] Log file not found: {LOG_FILE}")
        return results

    with open(LOG_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line {line_num}: {e}")

    print(f"[INFO] Loaded {len(results)} results from {LOG_FILE}")
    return results


def resort_results(results: list[dict]) -> dict:
    """
    Re-sort results: group by groups -> m -> config

    Returns: {
        groups: {
            m: {
                (disable_mma, disable_tma, disable_epilogue): time_ms,
                ...
            },
            ...
        },
        ...
    }
    """
    sorted_data = defaultdict(lambda: defaultdict(dict))

    for r in results:
        groups = r.get("groups")
        m = r.get("m")
        time_ms = r.get("time_ms")

        if groups is None or m is None:
            continue

        config_key = (
            r.get("disable_mma"),
            r.get("disable_tma"),
            r.get("disable_epilogue"),
        )

        sorted_data[groups][m][config_key] = time_ms

    # Convert defaultdicts to regular dicts for JSON serialization
    return {
        groups: {m: dict(configs) for m, configs in m_data.items()}
        for groups, m_data in sorted_data.items()
    }


def save_sorted_results(sorted_data: dict):
    """Save re-sorted results to results_sorted.json."""
    # Convert tuple keys to strings for JSON compatibility
    json_compatible = {}
    for groups, m_data in sorted_data.items():
        json_compatible[groups] = {}
        for m, configs in m_data.items():
            json_compatible[groups][m] = {}
            for config_key, time_ms in configs.items():
                # Convert tuple key to string: "(1, 1, 1)" -> "mma=1,tma=1,epi=1"
                key_str = f"mma={config_key[0]},tma={config_key[1]},epi={config_key[2]}"
                json_compatible[groups][m][key_str] = time_ms

    with open(SORTED_FILE, "w") as f:
        json.dump(json_compatible, f, indent=2)

    print(f"[INFO] Saved re-sorted results to {SORTED_FILE}")


def get_time(sorted_data: dict, groups: int, m: int, config: tuple) -> Optional[float]:
    """Safely get time value from sorted_data."""
    try:
        return sorted_data.get(groups, {}).get(m, {}).get(config)
    except (KeyError, TypeError):
        return None


def create_plots(sorted_data: dict):
    """
    Create 3 grouped bar charts (one per groups value).

    For each groups size:
    - X-axis: m values (32, 64, 128, 256, 512)
    - Y-axis: Time (ms)
    - 8 bars per m value (one per config), side by side
    """
    for groups in GROUPS_VALUES:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(M_VALUES))  # x locations for the groups
        width = 0.10  # width of each bar (smaller for 8 bars)
        num_configs = len(CONFIG_ORDER)

        # Plot bars for each configuration
        for i, config_key in enumerate(CONFIG_ORDER):
            times = []
            for m in M_VALUES:
                time_val = get_time(sorted_data, groups, m, config_key)
                times.append(time_val if time_val is not None else 0)

            # Calculate offset to center the group of bars
            offset = (i - (num_configs - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                times,
                width,
                label=CONFIG_LABELS[config_key],
                color=CONFIG_COLORS[config_key],
            )

            # Add value labels on top of bars
            for bar, time_val in zip(bars, times):
                if time_val > 0:
                    height = bar.get_height()
                    ax.annotate(
                        f"{time_val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=60,
                    )

        ax.set_xlabel("Batch Size (m)", fontsize=12)
        ax.set_ylabel("Cuda Graph Avg Time (ms)", fontsize=12)
        ax.set_title(
            f"CUTLASS Benchmark: Pingpong Schedule (groups={groups})", fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(M_VALUES)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        output_file = SCRIPT_DIR / f"groups_{groups}.png"
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"[INFO] Saved plot to {output_file}")


def print_summary(sorted_data: dict):
    """Print a summary table to console."""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 120)

    for groups in GROUPS_VALUES:
        print(f"\n[Groups = {groups}]")
        print("-" * 110)

        # Header
        header = f"{'m':>6} | "
        header += " | ".join(f"{CONFIG_LABELS[c][:11]:>11}" for c in CONFIG_ORDER)
        print(header)
        print("-" * 110)

        # Data rows
        for m in M_VALUES:
            row = f"{m:>6} | "
            values = []
            for config in CONFIG_ORDER:
                time_val = get_time(sorted_data, groups, m, config)
                if time_val is not None:
                    values.append(f"{time_val:>11.5f}")
                else:
                    values.append(f"{'N/A':>11}")
            row += " | ".join(values)
            print(row)

    print("\n" + "=" * 120)


def main():
    # Load results
    results = load_results()
    if not results:
        print("[ERROR] No results to analyze")
        return

    # Re-sort results
    sorted_data = resort_results(results)

    # Save sorted results
    save_sorted_results(sorted_data)

    # Print summary table
    print_summary(sorted_data)

    # Create plots
    create_plots(sorted_data)

    print(f"\n[DONE] Analysis complete!")
    print(f"  - Sorted results: {SORTED_FILE}")
    print(f"  - Plots: groups_8.png, groups_32.png, groups_128.png")


if __name__ == "__main__":
    main()
