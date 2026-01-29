"""
(main) root@C.30240005:/workspace/cutlass/build$ ./79d_blackwell_geforce_nvfp4_grouped_gemm --no_verif --alpha=4 --beta=4 --m=128 --n=256 --k=256 --groups=64
Running kernel with Cooperative kernel schedule:
  Problem Sizes, Alpha, Beta 
    (128,256,256), 4, 4
  Groups      : 64
  Cuda Graph Avg Time : 0.00940153 ms
Running kernel with Pingpong kernel schedule:
  Problem Sizes, Alpha, Beta 
    (128,256,256), 4, 4
  Groups      : 64
  Cuda Graph Avg Time : 0.00970194 ms


run experiments by running the command line with these arguments, 
there are two iterations
one is m choose from [1, 32, 64, 128, 256, 512, 1024]
the other is groups choose from [8, 16, 32, 64, 128] 
n is always 768
k is always 2048
always use no verification flag
always use alpha == beta == 5

extract the average cuda runtime after Cuda Graph Avg Time : and before ms
extract the float number

there are two kinds of schedules (Cooperative and Pingpong)
pick the number that is the faster

the result should be list of list
where is inner list is the result for each fixed group with varying m
and the outer list is for varying groups

your ultimate task is to create a plot of #groups lines in a lineplot
the x axis is m, the y axis is the average cuda runtime in ms
the y axis should be in log scale, where the x axis is treated is distributed datapoints
the batch sizes should have same separation
"""

#!/usr/bin/env python3
"""
Run CUTLASS FP4 grouped GEMM experiments and plot results.
"""

import subprocess
import re
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Experiment parameters
M_VALUES = [1, 8, 16, 32, 64, 128, 256, 512]
GROUP_VALUES = [8, 16, 32, 64, 128, 256]
N = 768
K = 2048
ALPHA = 1
BETA = 0

# Set this to your build directory
BUILD_DIR = "/workspace/cutlass/build"
EXECUTABLE = os.path.join(BUILD_DIR, "examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm")
# EXECUTABLE = os.path.join(BUILD_DIR, "examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise")

# Or use the symlink if it exists
if os.path.exists(os.path.join(BUILD_DIR, "79d_blackwell_geforce_nvfp4_grouped_gemm")):
    EXECUTABLE = os.path.join(BUILD_DIR, "79d_blackwell_geforce_nvfp4_grouped_gemm")

def run_experiment(m, groups):
    """Run a single experiment and return the faster runtime."""
    cmd = [
        EXECUTABLE,
        "--no_verif",
        f"--alpha={ALPHA}",
        f"--beta={BETA}",
        f"--m={m}",
        f"--n={N}",
        f"--k={K}",
        f"--groups={groups}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Extract all "Cuda Graph Avg Time : X.XXX ms" values
        times = re.findall(r'Cuda Graph Avg Time : ([\d.]+) ms', output)
        
        if len(times) >= 2:
            cooperative_time = float(times[0])
            pingpong_time = float(times[1])
            faster_time = min(cooperative_time, pingpong_time)
            schedule = "Cooperative" if cooperative_time <= pingpong_time else "Pingpong"
            print(f"  m={m:4d}, groups={groups:3d}: Coop={cooperative_time:.6f}ms, Ping={pingpong_time:.6f}ms -> {schedule} ({faster_time:.6f}ms)")
            return faster_time
        elif len(times) == 1:
            print(f"  m={m:4d}, groups={groups:3d}: Only one result={times[0]}ms")
            return float(times[0])
        else:
            print(f"  m={m:4d}, groups={groups:3d}: No timing found in output")
            print(f"    stdout: {output[:500]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  m={m:4d}, groups={groups:3d}: Timeout")
        return None
    except Exception as e:
        print(f"  m={m:4d}, groups={groups:3d}: Error - {e}")
        return None

def main():
    print(f"Running experiments with n={N}, k={K}, alpha={ALPHA}, beta={BETA}")
    print(f"M values: {M_VALUES}")
    print(f"Group values: {GROUP_VALUES}")
    print()
    
    # results[group_idx][m_idx] = runtime

    import json
    if False:
        with open("exp_results.json", "r") as f:
            results = json.load(f)
        
    else:
        results = []
        for groups in GROUP_VALUES:
            print(f"\n=== Groups = {groups} ===")
            group_results = []
            for m in M_VALUES:
                runtime = run_experiment(m, groups)
                group_results.append(runtime)
            results.append(group_results)
        # results = [
        #     [0.0172368,0.0164106,0.0177529,0.0168358,0.0167851,0.016073,0.016215,0.024899],
        #     [0.0182318,0.0173825,0.017492,0.017205,0.0168517,0.0166092,0.0248135,0.0376613],
        #     [0.0290702,0.0251519,0.0265288,0.0260726,0.0254053,0.0246857,0.0374072,0.0653436],
        #     [0.0369502,0.0364439,0.0376161,0.0367134,0.0367046,0.0376737,0.0662833,0.153606],
        #     [0.0654666,0.0669451,0.0671094,0.0711952,0.0854322,0.119648,0.177957,0.29837],
        #     [0.186107,0.188368,0.193616,0.201496,0.212762,0.236069,0.347982,0.588868]
        # ]
        with open("exp_results.json", "w") as f:
            json.dump(results, f, indent=2)
    # Create plot
    plt.figure(figsize=(6, 12))
    
    for i, groups in enumerate(GROUP_VALUES):
        runtimes = results[i]
        # Filter out None values for plotting
        valid_m = [M_VALUES[j] for j in range(len(M_VALUES)) if runtimes[j] is not None]
        valid_times = [runtimes[j] for j in range(len(M_VALUES)) if runtimes[j] is not None]
        
        if valid_times:
            plt.plot(range(len(valid_m)), valid_times, marker='o', label=f'groups={groups}')
    
    # X-axis with equal spacing
    plt.xticks(range(len(M_VALUES)), [str(m) for m in M_VALUES])
    plt.xlabel('M (batch size)')
    plt.ylabel('Average CUDA Runtime (ms)')
    # plt.yscale('log')
    plt.title(f'FP4 Grouped GEMM Performance\n(n={N}, k={K}, alpha={ALPHA}, beta={BETA})')
    plt.legend(title='#Groups')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    output_path = 'gemm_performance.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    main()