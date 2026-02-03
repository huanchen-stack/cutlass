"""
in /workspace/cutlass/build, there are three instructions you should know:

1. how to compile with "disable" flags
cmake .. -DCUTLASS_NVCC_ARCHS=120a -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=0

2. how to compile executable
make 79e_blackwell_geforce_nvfp4_grouped_gemm

3. how to run experiments
./examples/79_blackwell_geforce_gemm/79e_blackwell_geforce_nvfp4_grouped_gemm --alpha=1 --beta=0 --m=32 --n=768 --k=2048 --groups=128

Configurations for -D flags:
{
    -DCUTLASS_DISABLE_MMA=1 -DCUTLASS_DISABLE_TMA=1 -DCUTLASS_DISABLE_EPILOGUE=1 ,
    -DCUTLASS_DISABLE_MMA=1 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=0 ,
    -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=1 -DCUTLASS_DISABLE_EPILOGUE=0 ,
    -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=1 ,
    -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=0 ,
}

Configurations for experiments:
{
    --alpha=1 --beta=0 --m=32  --n=768 --k=2048 --groups=8 ,
    --alpha=1 --beta=0 --m=64  --n=768 --k=2048 --groups=8 ,
    --alpha=1 --beta=0 --m=128 --n=768 --k=2048 --groups=8 ,
    --alpha=1 --beta=0 --m=256 --n=768 --k=2048 --groups=8 ,
    --alpha=1 --beta=0 --m=512 --n=768 --k=2048 --groups=8 ,

    --alpha=1 --beta=0 --m=32  --n=768 --k=2048 --groups=32 ,
    --alpha=1 --beta=0 --m=64  --n=768 --k=2048 --groups=32 ,
    --alpha=1 --beta=0 --m=128 --n=768 --k=2048 --groups=32 ,
    --alpha=1 --beta=0 --m=256 --n=768 --k=2048 --groups=32 ,
    --alpha=1 --beta=0 --m=512 --n=768 --k=2048 --groups=32 ,

    --alpha=1 --beta=0 --m=32  --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=64  --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=128 --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=256 --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=512 --n=768 --k=2048 --groups=128 ,
}

so in total there will be 5x15 = 75 experiments.

each experiment will generate an output like this:
(base) huanchen@HS:~/cutlass/build$ ./examples/79_blackwell_geforce_gemm/79e_blackwell_geforce_nvfp4_grouped_gemm --alpha=1 --beta=0 --m=32 --n=768 --k=2048 --groups=16
L2 Cache Size    : 48 MB
Workspace Count  : 11
Running Pingpong kernel with L2 cache busting:
  Problem Sizes, Alpha, Beta 
    (32,768,2048), 1, 0
  Groups         : 16
  Workspace Count: 11
  Graph iterations: 22
  Total graph time: 0.76192 ms
  Avg kernel time : 0.0346327 ms
  TFLOPS          : 46.5055

you should extract the Avg kernel time as a float number.
store them in a .log file where each line is a json string.
print results to the .log file as soon as the experiment is done.
"""

import subprocess
import json
import re
import os
from pathlib import Path
from typing import Optional

# Constants
BUILD_DIR = "/workspace/cutlass/build"
LOG_FILE = Path(__file__).parent / "results.log"
EXECUTABLE = (
    "./examples/79_blackwell_geforce_gemm/79e_blackwell_geforce_nvfp4_grouped_gemm"
)

# 5 CMake configurations (in order from docstring)
CMAKE_CONFIGS = [
    {"mma": 1, "tma": 1, "epilogue": 1},
    {"mma": 1, "tma": 0, "epilogue": 0},
    {"mma": 0, "tma": 1, "epilogue": 0},
    {"mma": 0, "tma": 0, "epilogue": 1},
    {"mma": 0, "tma": 1, "epilogue": 1},
    {"mma": 1, "tma": 0, "epilogue": 1},
    {"mma": 1, "tma": 1, "epilogue": 0},
    {"mma": 0, "tma": 0, "epilogue": 0},
]

# 15 experiment configurations (m x groups = 5 x 3)
M_VALUES = [32, 64, 128, 256, 512]
GROUPS_VALUES = [8, 32, 128]
FIXED_PARAMS = {"alpha": 1, "beta": 0, "n": 768, "k": 2048}


def run_cmake(config: dict) -> bool:
    """Run cmake with disable flags. Returns True on success."""
    cmd = [
        "cmake",
        "..",
        "-DCUTLASS_NVCC_ARCHS=120a",
        f"-DCUTLASS_DISABLE_MMA={config['mma']}",
        f"-DCUTLASS_DISABLE_TMA={config['tma']}",
        f"-DCUTLASS_DISABLE_EPILOGUE={config['epilogue']}",

        f"-DCUTLASS_TB_N32=1",

    ]
    print(f"[CMAKE] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for cmake
        )
        if result.returncode != 0:
            print(f"[CMAKE ERROR] Return code: {result.returncode}")
            print(f"[CMAKE STDERR] {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[CMAKE ERROR] Timeout expired")
        return False
    except Exception as e:
        print(f"[CMAKE ERROR] {e}")
        return False


def run_make() -> bool:
    """Compile the target. Returns True on success."""
    cmd = ["make", "79e_blackwell_geforce_nvfp4_grouped_gemm"]
    print(f"[MAKE] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for make
        )
        if result.returncode != 0:
            print(f"[MAKE ERROR] Return code: {result.returncode}")
            print(f"[MAKE STDERR] {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[MAKE ERROR] Timeout expired")
        return False
    except Exception as e:
        print(f"[MAKE ERROR] {e}")
        return False


def parse_pingpong_time(output: str) -> Optional[float]:
    """
    Extract Avg kernel time from 79e output.

    The output format is:
        Running Pingpong kernel with L2 cache busting:
          Problem Sizes, Alpha, Beta 
            (32,768,2048), 1, 0
          Groups         : 16
          Workspace Count: 11
          Graph iterations: 22
          Total graph time: 0.76192 ms
          Avg kernel time : 0.0346327 ms
          TFLOPS          : 46.5055
    """
    # Find the section marker
    marker = "Running Pingpong kernel with L2 cache busting:"
    if marker not in output:
        return None

    # Get the section after the marker
    section = output.split(marker)[1]

    # Find "Avg kernel time : X.XXX ms"
    pattern = r"Avg kernel time\s*:\s*([\d.]+)\s*ms"
    match = re.search(pattern, section)

    if match:
        return float(match.group(1))
    return None


def run_experiment(m: int, groups: int) -> Optional[float]:
    """Run experiment and return Pingpong Cuda Graph Avg Time."""
    cmd = [
        EXECUTABLE,
        f"--alpha={FIXED_PARAMS['alpha']}",
        f"--beta={FIXED_PARAMS['beta']}",
        # f"--m={m}",
        # f"--n={FIXED_PARAMS['n']}",

        f"--n={m}",
        f"--m={FIXED_PARAMS['n']}",
        
        f"--k={FIXED_PARAMS['k']}",
        f"--groups={groups}",
    ]
    print(f"[RUN] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min timeout for experiment
        )
        if result.returncode != 0:
            print(f"[RUN ERROR] Return code: {result.returncode}")
            print(f"[RUN STDERR] {result.stderr[:500]}")
            return None

        time_ms = parse_pingpong_time(result.stdout)
        if time_ms is None:
            print(f"[PARSE ERROR] Could not extract Pingpong time from output")
            print(f"[OUTPUT] {result.stdout[:500]}")
        return time_ms

    except subprocess.TimeoutExpired:
        print("[RUN ERROR] Timeout expired")
        return None
    except Exception as e:
        print(f"[RUN ERROR] {e}")
        return None


def main():
    total = len(CMAKE_CONFIGS) * len(M_VALUES) * len(GROUPS_VALUES)  # 75
    count = 0

    # Ensure build directory exists
    if not os.path.isdir(BUILD_DIR):
        print(f"[ERROR] Build directory does not exist: {BUILD_DIR}")
        print("[ERROR] Please create it with: mkdir -p /workspace/cutlass/build")
        return

    print(f"[INFO] Starting {total} experiments")
    print(f"[INFO] Log file: {LOG_FILE}")
    print(f"[INFO] Build directory: {BUILD_DIR}")
    print()

    with open(LOG_FILE, "a") as log:
        for config_idx, config in enumerate(CMAKE_CONFIGS):
            config_name = f"MMA={config['mma']}, TMA={config['tma']}, EPILOGUE={config['epilogue']}"
            print(f"\n{'=' * 60}")
            print(f"[CONFIG {config_idx + 1}/{len(CMAKE_CONFIGS)}] {config_name}")
            print(f"{'=' * 60}\n")

            # Run cmake
            cmake_success = run_cmake(config)
            if not cmake_success:
                # Log errors for all experiments with this config
                for groups in GROUPS_VALUES:
                    for m in M_VALUES:
                        count += 1
                        result = {
                            "disable_mma": config["mma"],
                            "disable_tma": config["tma"],
                            "disable_epilogue": config["epilogue"],
                            "m": m,
                            "n": FIXED_PARAMS["n"],
                            "k": FIXED_PARAMS["k"],
                            "groups": groups,
                            "time_ms": None,
                            "error": "cmake_failed",
                        }
                        log.write(json.dumps(result) + "\n")
                        log.flush()
                continue

            # Run make
            make_success = run_make()
            if not make_success:
                # Log errors for all experiments with this config
                for groups in GROUPS_VALUES:
                    for m in M_VALUES:
                        count += 1
                        result = {
                            "disable_mma": config["mma"],
                            "disable_tma": config["tma"],
                            "disable_epilogue": config["epilogue"],
                            "m": m,
                            "n": FIXED_PARAMS["n"],
                            "k": FIXED_PARAMS["k"],
                            "groups": groups,
                            "time_ms": None,
                            "error": "make_failed",
                        }
                        log.write(json.dumps(result) + "\n")
                        log.flush()
                continue

            # Run experiments for this configuration
            for groups in GROUPS_VALUES:
                for m in M_VALUES:
                    count += 1
                    print(
                        f"\n[{count}/{total}] m={m}, n={FIXED_PARAMS['n']}, k={FIXED_PARAMS['k']}, groups={groups}"
                    )

                    time_ms = run_experiment(m, groups)

                    result = {
                        "disable_mma": config["mma"],
                        "disable_tma": config["tma"],
                        "disable_epilogue": config["epilogue"],
                        "m": m,
                        "n": FIXED_PARAMS["n"],
                        "k": FIXED_PARAMS["k"],
                        "groups": groups,
                        "time_ms": time_ms,
                        "error": None if time_ms is not None else "run_failed",
                    }

                    log.write(json.dumps(result) + "\n")
                    log.flush()

                    if time_ms is not None:
                        print(f"[RESULT] Pingpong Cuda Graph Avg Time: {time_ms} ms")

    print(f"\n{'=' * 60}")
    print(f"[DONE] Completed {count} experiments")
    print(f"[DONE] Results saved to: {LOG_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
