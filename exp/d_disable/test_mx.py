"""
in /home/huanchen/cutlass/build, there are three instructions you should know:

1. how to compile with "disable" flags and format flags
cmake .. -DCUTLASS_NVCC_ARCHS=120a {DISABLE_FLAGS} {CUDA_FLAGS}

2. how to compile executable
make 79f_blackwell_geforce_mxfp8_grouped_gemm

3. how to run experiments
./examples/79_blackwell_geforce_gemm/79f_blackwell_geforce_mxfp8_grouped_gemm --alpha=1 --beta=0 --m=32 --n=768 --k=2048 --groups=128

Configurations for -D flags:
{
    -DCUTLASS_DISABLE_MMA=1 -DCUTLASS_DISABLE_TMA=1 -DCUTLASS_DISABLE_EPILOGUE=1 ,
    -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=1 -DCUTLASS_DISABLE_EPILOGUE=1 ,
    -DCUTLASS_DISABLE_MMA=1 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=1 ,
    -DCUTLASS_DISABLE_MMA=0 -DCUTLASS_DISABLE_TMA=0 -DCUTLASS_DISABLE_EPILOGUE=0 ,
}

Configuration for -DCMAKE_CUDA_FLAGS:
{
    -DCMAKE_CUDA_FLAGS="-DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=1" ,
    -DCMAKE_CUDA_FLAGS="-DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=3" ,
    -DCMAKE_CUDA_FLAGS="-DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=5" ,
    -DCMAKE_CUDA_FLAGS="-DELEMENT_A_FORMAT=5 -DELEMENT_B_FORMAT=5" ,
}
Here's the mapping between format and number:
#define MXFP8_E4M3  1
#define MXFP6_E3M2  3
#define MXFP4       5

Configurations for experiments:
{
    --alpha=1 --beta=0 --m=32  --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=64  --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=128 --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=256 --n=768 --k=2048 --groups=128 ,
    --alpha=1 --beta=0 --m=512 --n=768 --k=2048 --groups=128 ,
}

so in total there will be 4x4x5 = 80 experiments.
The experiment is extremely compile time heavy (4x4 compiles), and each compile may take up to a few minutes.
Each compile takes 1 cpu core, so it is recommended to run this script on a machine with multiple cpu cores.
Use multiprocessing so speedup compile time and name the executables differently to avoid conflicts.
For example you can name them like: 79f_blackwell_geforce_mxfp8_grouped_gemm_mma1_tma0_epi1_A_MXFP8_B_MXFP6

However, each GPU experiments should be run sequentially for proper profiling.
The simplist way is to first compile all the executables and then run the experiments sequentially.

each experiment will generate an output like this:
(base) huanchen@HS:~/cutlass/build$ ./examples/79_blackwell_geforce_gemm/79f_blackwell_geforce_mxfp8_grouped_gemm --alpha=1 --beta=0 --m=32 --n=768 --k=2048 --groups=128
L2 Cache Size    : 48 MB
Workspace Count  : 2
Element A Format : MXFP8_E4M3
Element B Format : MXFP6_E3M2
Running MX-format Grouped GEMM with L2 cache busting:
  Problem Sizes, Alpha, Beta 
    (128,768,2048), 1, 0
  Groups         : 128
  Workspace Count: 2
  Graph iterations: 4
  Total graph time: 1.47312 ms
  Avg kernel time : 0.36828 ms
  TFLOPS          : 139.947

you should extract the Avg kernel time as a float number.
store them in a .log file where each line is a json string.
print results to the .log file as soon as the experiment is done.
"""

import subprocess
import json
import re
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

# Constants
CUTLASS_ROOT = Path("/workspace/cutlass")
BUILD_DIR = CUTLASS_ROOT / "build"
TEMP_BUILD_ROOT = CUTLASS_ROOT / "build_temp"
LOG_FILE = Path(__file__).parent / "results_mx.log"
BASE_EXECUTABLE_NAME = "79f_blackwell_geforce_mxfp8_grouped_gemm"
EXECUTABLE_DIR = BUILD_DIR / "examples" / "79_blackwell_geforce_gemm"

# Maximum parallel compilations (each uses ~1 CPU core + GPU memory for ptxas)
# Limited to 4 to reduce compile-time burden
MAX_PARALLEL_COMPILES = min(cpu_count(), 4)

# 4 disable configurations
CMAKE_DISABLE_CONFIGS = [
    {"mma": 1, "tma": 1, "epilogue": 1},  # All disabled (baseline)
    {"mma": 0, "tma": 1, "epilogue": 1},  # MMA enabled only
    {"mma": 1, "tma": 0, "epilogue": 1},  # TMA enabled only
    {"mma": 0, "tma": 0, "epilogue": 0},  # All enabled (full kernel)
]

# 4 format configurations (A_format, B_format, A_name, B_name)
CUDA_FLAGS_CONFIGS = [
    {"A": 1, "B": 1, "A_name": "MXFP8", "B_name": "MXFP8"},
    {"A": 1, "B": 3, "A_name": "MXFP8", "B_name": "MXFP6"},
    {"A": 1, "B": 5, "A_name": "MXFP8", "B_name": "MXFP4"},
    # {"A": 5, "B": 5, "A_name": "MXFP4", "B_name": "MXFP4"},
]

# 5 experiment configurations (m values with fixed groups=128)
M_VALUES = [
    32, 
    64, 128, 256, 512
]
FIXED_PARAMS = {"alpha": 1, "beta": 0, "n": 768, "k": 2048, "groups": 128}


@dataclass
class CompileConfig:
    """Represents a unique compilation configuration."""
    disable_mma: int
    disable_tma: int
    disable_epilogue: int
    format_A: int
    format_B: int
    format_A_name: str
    format_B_name: str
    
    @property
    def config_id(self) -> str:
        """Short ID for build directory naming."""
        return (
            f"mma{self.disable_mma}_tma{self.disable_tma}_epi{self.disable_epilogue}_"
            f"A{self.format_A}_B{self.format_B}"
        )
    
    @property
    def executable_name(self) -> str:
        """Generate unique executable name."""
        return (
            f"{BASE_EXECUTABLE_NAME}_"
            f"mma{self.disable_mma}_tma{self.disable_tma}_epi{self.disable_epilogue}_"
            f"A_{self.format_A_name}_B_{self.format_B_name}"
        )
    
    @property
    def temp_build_dir(self) -> Path:
        """Path to temporary build directory for this config."""
        return TEMP_BUILD_ROOT / f"build_{self.config_id}"
    
    @property
    def final_executable_path(self) -> Path:
        """Final path in main build directory."""
        return EXECUTABLE_DIR / self.executable_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "disable_mma": self.disable_mma,
            "disable_tma": self.disable_tma,
            "disable_epilogue": self.disable_epilogue,
            "format_A": self.format_A_name,
            "format_B": self.format_B_name,
        }


def generate_all_configs() -> List[CompileConfig]:
    """Generate all 16 compile configurations."""
    configs = []
    for disable in CMAKE_DISABLE_CONFIGS:
        for fmt in CUDA_FLAGS_CONFIGS:
            configs.append(CompileConfig(
                disable_mma=disable["mma"],
                disable_tma=disable["tma"],
                disable_epilogue=disable["epilogue"],
                format_A=fmt["A"],
                format_B=fmt["B"],
                format_A_name=fmt["A_name"],
                format_B_name=fmt["B_name"],
            ))
    return configs


def compile_one_config(config: CompileConfig) -> Tuple[str, bool, str]:
    """
    Compile a single configuration in its own build directory.
    
    This function is designed to be called by multiprocessing.Pool.
    Each config gets its own build directory to allow parallel cmake/make.
    
    Returns: (executable_name, success, error_message)
    """
    build_dir = config.temp_build_dir
    
    try:
        # Create build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Build cmake command - note we use CUTLASS_ROOT as source dir
        cmake_cmd = [
            "cmake",
            str(CUTLASS_ROOT),
            "-DCUTLASS_NVCC_ARCHS=120a",
            f"-DCUTLASS_DISABLE_MMA={config.disable_mma}",
            f"-DCUTLASS_DISABLE_TMA={config.disable_tma}",
            f"-DCUTLASS_DISABLE_EPILOGUE={config.disable_epilogue}",
            "-DCUTLASS_TB_N32=1",
            # f"-DCMAKE_CUDA_FLAGS=-DELEMENT_A_FORMAT={config.format_A} -DELEMENT_B_FORMAT={config.format_B}",
            f"-DCMAKE_CUDA_FLAGS=-DELEMENT_A_FORMAT={config.format_B} -DELEMENT_B_FORMAT={config.format_A}",
        ]
        
        print(f"[CMAKE] {config.config_id}: Starting cmake...")
        print(f"\t\tCommand: {' '.join(cmake_cmd)}")
        
        # Run cmake in temp build directory
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return (config.executable_name, False, f"cmake failed: {result.stderr[:300]}")
        
        # Run make (single-threaded since we're parallelizing at config level)
        make_cmd = ["make", BASE_EXECUTABLE_NAME, "-j1"]
        print(f"[MAKE] {config.config_id}: Starting make...")
        
        result = subprocess.run(
            make_cmd,
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return (config.executable_name, False, f"make failed: {result.stderr[:300]}")
        
        # Copy executable to main build directory with unique name
        src_executable = build_dir / "examples" / "79_blackwell_geforce_gemm" / BASE_EXECUTABLE_NAME
        dst_executable = config.final_executable_path
        
        if not src_executable.exists():
            return (config.executable_name, False, f"Executable not found: {src_executable}")
        
        # Ensure destination directory exists
        dst_executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_executable, dst_executable)
        
        print(f"[DONE] {config.config_id}: Compiled -> {config.executable_name}")
        return (config.executable_name, True, "")
        
    except subprocess.TimeoutExpired:
        return (config.executable_name, False, "Timeout expired")
    except Exception as e:
        return (config.executable_name, False, str(e))


def compile_batch(configs: List[CompileConfig]) -> Dict[str, bool]:
    """
    Compile a batch of configurations in parallel using separate build directories.
    
    Each config gets its own temp build directory under TEMP_BUILD_ROOT,
    allowing cmake and make to run independently in parallel.
    
    Returns dict mapping executable_name -> success.
    """
    print(f"\n[INFO] Compiling {len(configs)} configurations in parallel")
    print(f"[INFO] Temp build root: {TEMP_BUILD_ROOT}")
    print()
    
    # Create temp build root
    TEMP_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Compile in parallel
    results = {}
    
    with Pool(processes=len(configs)) as pool:
        compile_results = pool.map(compile_one_config, configs)
    
    # Process results
    for executable_name, success, error in compile_results:
        results[executable_name] = success
        if not success:
            print(f"[ERROR] {executable_name}: {error}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\n[COMPILE SUMMARY] {successful}/{len(configs)} compiled successfully")
    
    return results


def parse_kernel_time(output: str) -> Optional[float]:
    """Extract Avg kernel time from 79f output."""
    marker = "with L2 cache busting:"
    if marker not in output:
        return None
    
    section = output.split(marker)[1]
    pattern = r"Avg kernel time\s*:\s*([\d.]+)\s*ms"
    match = re.search(pattern, section)
    
    if match:
        return float(match.group(1))
    return None


def run_experiment(executable_path: Path, m: int) -> Optional[float]:
    """Run a single experiment and return kernel time in ms."""
    cmd = [
        str(executable_path),
        f"--alpha={FIXED_PARAMS['alpha']}",
        f"--beta={FIXED_PARAMS['beta']}",
        # f"--m={m}",
        # f"--n={FIXED_PARAMS['n']}",
        f"--n={m}",
        f"--m={FIXED_PARAMS['n']}",

        f"--k={FIXED_PARAMS['k']}",
        f"--groups={FIXED_PARAMS['groups']}",
    ]
    print(f"\t\tCommand: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BUILD_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"    [RUN ERROR] Return code: {result.returncode}")
            print(f"    [STDERR] {result.stderr[:200]}")
            return None
        
        time_ms = parse_kernel_time(result.stdout)
        if time_ms is None:
            print(f"    [PARSE ERROR] Could not extract kernel time")
            print(f"    [OUTPUT] {result.stdout[:300]}")
        return time_ms
        
    except subprocess.TimeoutExpired:
        print("    [RUN ERROR] Timeout expired")
        return None
    except Exception as e:
        print(f"    [RUN ERROR] {e}")
        return None


def run_experiments_for_configs(configs: List[CompileConfig], compile_results: Dict[str, bool], batch_num: int, total_batches: int):
    """Run experiments for a batch of configs sequentially for proper GPU profiling."""
    total = len(configs) * len(M_VALUES)
    count = 0
    
    print(f"\n{'='*60}")
    print(f"[BATCH {batch_num}/{total_batches}] Running {total} experiments sequentially")
    print(f"[INFO] Log file: {LOG_FILE}")
    print(f"{'='*60}\n")
    
    with open(LOG_FILE, "a") as log:
        for config in configs:
            executable_name = config.executable_name
            
            if not compile_results.get(executable_name, False):
                # Log failures for all m values
                print(f"[SKIP] {executable_name} (compile failed)")
                for m in M_VALUES:
                    count += 1
                    result = {
                        **config.to_dict(),
                        "m": m,
                        "n": FIXED_PARAMS["n"],
                        "k": FIXED_PARAMS["k"],
                        "groups": FIXED_PARAMS["groups"],
                        "time_ms": None,
                        "error": "compile_failed",
                    }
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                continue
            
            executable_path = config.final_executable_path
            if not executable_path.exists():
                print(f"[SKIP] {executable_name} (executable not found)")
                for m in M_VALUES:
                    count += 1
                    result = {
                        **config.to_dict(),
                        "m": m,
                        "n": FIXED_PARAMS["n"],
                        "k": FIXED_PARAMS["k"],
                        "groups": FIXED_PARAMS["groups"],
                        "time_ms": None,
                        "error": "executable_not_found",
                    }
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                continue
            
            print(f"\n[CONFIG] {executable_name}")
            
            for m in M_VALUES:
                count += 1
                print(f"  [{count}/{total}] m={m}, groups={FIXED_PARAMS['groups']}")
                
                time_ms = run_experiment(executable_path, m)
                
                result = {
                    **config.to_dict(),
                    "m": m,
                    "n": FIXED_PARAMS["n"],
                    "k": FIXED_PARAMS["k"],
                    "groups": FIXED_PARAMS["groups"],
                    "time_ms": time_ms,
                    "error": None if time_ms is not None else "run_failed",
                }
                
                log.write(json.dumps(result) + "\n")
                log.flush()
                
                if time_ms is not None:
                    print(f"    -> Avg kernel time: {time_ms:.4f} ms")


def main():
    # Ensure main build directory exists (for storing final executables)
    if not BUILD_DIR.is_dir():
        print(f"[ERROR] Build directory does not exist: {BUILD_DIR}")
        print(f"[ERROR] Please create it with: mkdir -p {BUILD_DIR} && cd {BUILD_DIR} && cmake ..")
        return
    
    # Ensure executable directory exists
    EXECUTABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    configs = generate_all_configs()
    total_configs = len(configs)
    total_experiments = total_configs * len(M_VALUES)
    
    # Split configs into batches of 4
    batch_size = MAX_PARALLEL_COMPILES
    batches = [configs[i:i + batch_size] for i in range(0, len(configs), batch_size)]
    
    print(f"{'='*60}")
    print(f"MX-format Grouped GEMM Benchmark")
    print(f"{'='*60}")
    print(f"  Disable configs  : {len(CMAKE_DISABLE_CONFIGS)}")
    print(f"  Format configs   : {len(CUDA_FLAGS_CONFIGS)}")
    print(f"  Total compiles   : {total_configs}")
    print(f"  M values         : {M_VALUES}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Total batches    : {len(batches)}")
    print(f"  Log file         : {LOG_FILE}")
    print(f"{'='*60}")
    
    # Process each batch: compile 4 in parallel, then run their experiments
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n{'#'*60}")
        print(f"# BATCH {batch_num}/{len(batches)}: {len(batch)} configs")
        print(f"{'#'*60}")
        
        # Compile this batch in parallel
        compile_results = compile_batch(batch)
        
        # Run experiments for this batch sequentially
        run_experiments_for_configs(batch, compile_results, batch_num, len(batches))
    
    print(f"\n{'='*60}")
    print(f"[DONE] Completed all experiments")
    print(f"[DONE] Results saved to: {LOG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
