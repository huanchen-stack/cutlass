/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


/*! \file
    \brief MX-format Grouped GEMM with BF16 output and L2 cache busting using workspace rotation.

    This example supports configurable input formats for A and B matrices via compile-time defines.
    Supported formats: MXFP8 (e4m3, e5m2), MXFP6 (e3m2, e2m3), MXFP4

    Features:
    - Compile-time configurable input formats for A and B matrices
    - BF16 output with simple LinearCombination epilogue
    - Multiple workspace sets to avoid L2 cache camping during profiling
    - Inline CUDA graph profiling with workspace rotation
    - Auto-calculation of workspace count based on L2 cache size

    Build Instructions:
    
    Default build (MXFP8 e4m3 for both A and B):
      $ cmake .. -DCUTLASS_NVCC_ARCHS=120a
      $ cmake --build . --target 79f_blackwell_geforce_mxfp8_grouped_gemm

    Custom format build (e.g., MXFP8 A, MXFP6 B):
      $ cmake .. -DCUTLASS_NVCC_ARCHS=120a -DCMAKE_CUDA_FLAGS="-DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=3"
      $ cmake --build . --target 79f_blackwell_geforce_mxfp8_grouped_gemm

    Format codes:
      1 = MXFP8_E4M3  (mx_float8_t<float_e4m3_t>, 8-bit)
      2 = MXFP8_E5M2  (mx_float8_t<float_e5m2_t>, 8-bit)
      3 = MXFP6_E3M2  (mx_float6_t<float_e3m2_t>, 6-bit)
      4 = MXFP6_E2M3  (mx_float6_t<float_e2m3_t>, 6-bit)
      5 = MXFP4       (mx_float4_t<float_e2m1_t>, 4-bit)

    Example format combinations:
      -DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=1  # MXFP8 x MXFP8 (default)
      -DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=3  # MXFP8 x MXFP6
      -DELEMENT_A_FORMAT=1 -DELEMENT_B_FORMAT=5  # MXFP8 x MXFP4
      -DELEMENT_A_FORMAT=3 -DELEMENT_B_FORMAT=3  # MXFP6 x MXFP6

    To run this example:

      $ ./examples/79_blackwell_geforce_gemm/79f_blackwell_geforce_mxfp8_grouped_gemm --m=2048 --n=2048 --k=2048 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes.
      Use --workspace_count=N to override the auto-calculated workspace count.
*/

#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <float.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "helper.h"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Format selection via preprocessor defines
/////////////////////////////////////////////////////////////////////////////////////////////////

// Format enum values for preprocessor selection
#define MXFP8_E4M3  1
#define MXFP8_E5M2  2
#define MXFP6_E3M2  3
#define MXFP6_E2M3  4
#define MXFP4       5

// Default formats if not specified at compile time
#ifndef ELEMENT_A_FORMAT
#define ELEMENT_A_FORMAT MXFP8_E4M3
#endif

#ifndef ELEMENT_B_FORMAT
#define ELEMENT_B_FORMAT MXFP8_E4M3
#endif

// Helper macros for format name strings
#define FORMAT_NAME_1 "MXFP8_E4M3"
#define FORMAT_NAME_2 "MXFP8_E5M2"
#define FORMAT_NAME_3 "MXFP6_E3M2"
#define FORMAT_NAME_4 "MXFP6_E2M3"
#define FORMAT_NAME_5 "MXFP4"

#define FORMAT_NAME_HELPER(x) FORMAT_NAME_##x
#define FORMAT_NAME(x) FORMAT_NAME_HELPER(x)

#define ELEMENT_A_FORMAT_STR FORMAT_NAME(ELEMENT_A_FORMAT)
#define ELEMENT_B_FORMAT_STR FORMAT_NAME(ELEMENT_B_FORMAT)

#if 1 or defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration - selected via ELEMENT_A_FORMAT
#if ELEMENT_A_FORMAT == MXFP8_E4M3
  using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignmentA  = 16;   // 128 bits / 8 bits = 16 elements
#elif ELEMENT_A_FORMAT == MXFP8_E5M2
  using         ElementA    = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignmentA  = 16;   // 128 bits / 8 bits = 16 elements
#elif ELEMENT_A_FORMAT == MXFP6_E3M2
  using         ElementA    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
  constexpr int AlignmentA  = 128;  // Special alignment for 6-bit formats
#elif ELEMENT_A_FORMAT == MXFP6_E2M3
  using         ElementA    = cutlass::mx_float6_t<cutlass::float_e2m3_t>;
  constexpr int AlignmentA  = 128;  // Special alignment for 6-bit formats
#elif ELEMENT_A_FORMAT == MXFP4
  using         ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignmentA  = 32;   // 128 bits / 4 bits = 32 elements
#else
  #error "Unknown ELEMENT_A_FORMAT. Use 1-5 (MXFP8_E4M3, MXFP8_E5M2, MXFP6_E3M2, MXFP6_E2M3, MXFP4)"
#endif
using         LayoutATag  = cutlass::layout::RowMajor;

// B matrix configuration - selected via ELEMENT_B_FORMAT
#if ELEMENT_B_FORMAT == MXFP8_E4M3
  using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignmentB  = 16;   // 128 bits / 8 bits = 16 elements
#elif ELEMENT_B_FORMAT == MXFP8_E5M2
  using         ElementB    = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignmentB  = 16;   // 128 bits / 8 bits = 16 elements
#elif ELEMENT_B_FORMAT == MXFP6_E3M2
  using         ElementB    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
  constexpr int AlignmentB  = 128;  // Special alignment for 6-bit formats
#elif ELEMENT_B_FORMAT == MXFP6_E2M3
  using         ElementB    = cutlass::mx_float6_t<cutlass::float_e2m3_t>;
  constexpr int AlignmentB  = 128;  // Special alignment for 6-bit formats
#elif ELEMENT_B_FORMAT == MXFP4
  using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignmentB  = 32;   // 128 bits / 4 bits = 32 elements
#else
  #error "Unknown ELEMENT_B_FORMAT. Use 1-5 (MXFP8_E4M3, MXFP8_E5M2, MXFP6_E3M2, MXFP6_E2M3, MXFP4)"
#endif
using         LayoutBTag  = cutlass::layout::ColumnMajor;

// C/D matrix configuration - BF16
using         ElementD    = cutlass::bfloat16_t;                          // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                    // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                    // Layout type for D matrix operand

constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;  // Alignment of D matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                        // Element type for internal accumulation
using ElementCompute      = float;                                        // Element type for internal computation
using ArchTag             = cutlass::arch::Sm120;                         // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;    // Operator class tag

// Kernel Perf config
// Cluster Shape fixed to 1x1x1
#ifndef CUTLASS_TB_N32
#define CUTLASS_TB_N32 0
#endif
#if CUTLASS_TB_N32
using ThreadBlockShape    = Shape<_128,_32,_128>;
#else
using ThreadBlockShape    = Shape<_128,_128,_128>;
#endif

using ClusterShape        = Shape<_1,_1,_1>;

// Simple linear combination epilogue: D = alpha * acc + beta * C
using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute>;

// Cooperative kernel schedule
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutDTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementA, LayoutATag *, AlignmentA,
  ElementB, LayoutBTag *, AlignmentB,
  ElementAccumulator,
  ThreadBlockShape, ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<
  static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


// Grouped GEMM kernel schedule - use KernelScheduleAuto for MXFP8
// (KernelPtrArrayTmaWarpSpecializedPingpong is not supported for MXFP8 in select_instr)
using CollectiveMainloopPingpong = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementA, LayoutATag *, AlignmentA,
  ElementB, LayoutBTag *, AlignmentB,
  ElementAccumulator,
  ThreadBlockShape, ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<
  static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernelPingpong = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopPingpong,
    CollectiveEpilogue
>;

using GemmPingpong = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelPingpong>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Host tensor types
using HostTensorA = cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout>;
using HostTensorB = cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout>;
using HostTensorSF = cutlass::HostTensor<typename Gemm::GemmKernel::CollectiveMainloop::ElementSF, cutlass::layout::PackedVectorLayout>;
using HostTensorC = cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout>;
using HostTensorD = cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GemmWorkspace: Encapsulates one complete workspace set for L2 cache busting
/////////////////////////////////////////////////////////////////////////////////////////////////
struct GemmWorkspace {
  // Host tensors (one per group)
  std::vector<HostTensorA> block_A;
  std::vector<HostTensorB> block_B;
  std::vector<HostTensorSF> block_SFA;
  std::vector<HostTensorSF> block_SFB;
  std::vector<HostTensorC> block_C;
  std::vector<HostTensorD> block_D;

  // Device pointer arrays
  cutlass::DeviceAllocation<const typename ElementA::DataType *> ptr_A;
  cutlass::DeviceAllocation<const typename ElementB::DataType *> ptr_B;
  cutlass::DeviceAllocation<const typename Gemm::GemmKernel::CollectiveMainloop::ElementSF *> ptr_SFA;
  cutlass::DeviceAllocation<const typename Gemm::GemmKernel::CollectiveMainloop::ElementSF *> ptr_SFB;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementD *> ptr_D;

  // Gemm instance and its workspace
  GemmPingpong gemm;
  typename GemmPingpong::Arguments arguments;
  cutlass::device_memory::allocation<uint8_t> device_workspace;
};

// Multiple workspace sets for L2 cache busting
std::vector<GemmWorkspace> workspaces;

// Shared data (same across all workspaces)
std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<LayoutSFA> layout_SFA_host;
std::vector<LayoutSFA> layout_SFB_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

// Device-side shared allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

// Reference tensors for verification (only need one set)
std::vector<HostTensorD> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help = false;
  bool verification = false;  // Disabled by default for profiling focus
  bool use_pdl = false;

  float alpha = std::numeric_limits<float>::max();
  float beta  = std::numeric_limits<float>::max();
  int iterations = 10;
  int m = 1024, n = 2048, k = 512, groups = 10;
  int workspace_count = 0;  // 0 = auto-calculate based on L2 cache size
  RasterOrderOptions raster_order = RasterOrderOptions::AlongN;
  int max_sm_count = INT_MAX;
  std::string benchmark_path;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  // Use the maximum alignment required by A, B, C, D for random problem generation
  static constexpr int alignment = []() constexpr {
    int align = AlignmentA > AlignmentB ? AlignmentA : AlignmentB;
    constexpr int alignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    return align > alignC ? align : alignC;
  }();

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    if (cmd.check_cmd_line_flag("verification")) {
      verification = true;
    }
    if (cmd.check_cmd_line_flag("use_pdl")) {
      use_pdl = true;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, std::numeric_limits<float>::max());
    cmd.get_cmd_line_argument("beta",  beta,  std::numeric_limits<float>::max());
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);
    cmd.get_cmd_line_argument("max_sm_count", max_sm_count, INT_MAX);
    cmd.get_cmd_line_argument("workspace_count", workspace_count, 0);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster_order = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster_order = RasterOrderOptions::AlongM;
    }
  }

  void randomize_problems(cutlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_host.reserve(groups);

    for (int i = groups; i > 0; i--) {
      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;
      if (m < 1) {
        m = alignment * ((rand() % 64) + 1);
      }
      if (n < 1) {
        n = alignment * ((rand() % 64) + 1);
      }
      if (k < 1) {
        k = alignment * ((rand() % 64) + 1);
      }
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
      }
    }
    groups = static_cast<int>(problem_sizes_host.size());

    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "79f_blackwell_geforce_mxfp8_grouped_gemm\n\n"
      << "  Blackwell MXFP8 Grouped GEMM with BF16 output and L2 cache busting.\n\n"
      << "Options:\n\n"
      << "  --help                                                       If specified, displays this usage statement\n\n"
      << "  --m=<int>                                                    Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                                                    Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                                                    Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>                                               Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>                                                Epilogue scalar alpha\n"
      << "  --beta=<f32>                                                 Epilogue scalar beta\n"
      << "  --raster=<char>                                              CTA Rasterization direction (N for along N, M for along M)\n\n"
      << "  --iterations=<int>                                           Number of profiling iterations to perform\n\n"
      << "  --benchmark=<str>                                            Executes a benchmark problem size\n"
      << "  --max_sm_count=<int>                                         Run kernels using only these number of SMs\n"
      << "  --workspace_count=<int>                                      Number of workspaces for L2 cache busting (0=auto)\n"
      << "  --verification                                               Enable host-side verification\n"
      << "  --use_pdl                                                    Launch kernel with PDL (Programmatic Dependent Launch) enabled\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "79f_blackwell_geforce_mxfp8_grouped_gemm" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

#if 1 or defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Calculate workspace count based on L2 cache size
int calculate_workspace_count(const Options& options, const cudaDeviceProp& props) {
  if (options.workspace_count > 0) {
    return options.workspace_count;
  }

  // Calculate bits per element for A and B based on compile-time format selection
  constexpr int BitsPerElementA = cutlass::sizeof_bits<typename ElementA::DataType>::value;
  constexpr int BitsPerElementB = cutlass::sizeof_bits<typename ElementB::DataType>::value;

  // Calculate total bytes for one workspace (all groups)
  int64_t total_bytes = 0;
  for (const auto& problem : options.problem_sizes_host) {
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);
    // A(M×K), B(K×N) - size depends on format
    // C, D are bfloat16 (2 bytes/element)
    total_bytes += (int64_t(M) * K * BitsPerElementA) / 8;  // A
    total_bytes += (int64_t(K) * N * BitsPerElementB) / 8;  // B
    total_bytes += int64_t(M) * N * 2;    // C (BF16, 2 bytes)
    total_bytes += int64_t(M) * N * 2;    // D (BF16, 2 bytes)
  }

  int64_t l2_size = props.l2CacheSize;
  // Want total workspace >= 3x L2 to ensure eviction
  int count = std::max(1, (int)((3 * l2_size) / std::max(total_bytes, int64_t(1))) + 1);
  return std::min(count + 1, 64);  // Cap at 64
}

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data for all workspaces
void allocate(const Options &options, int workspace_count) {
  workspaces.resize(workspace_count);

  // Compute and store strides/layouts (shared across workspaces)
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto stride_A_val = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B_val = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C_val = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D_val = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_A = make_layout(make_shape(M, K, 1), stride_A_val);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B_val);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C_val);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D_val);
    auto layout_SFA_val = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB_val = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    stride_A_host.push_back(stride_A_val);
    stride_B_host.push_back(stride_B_val);
    layout_SFA_host.push_back(layout_SFA_val);
    layout_SFB_host.push_back(layout_SFB_val);
    stride_C_host.push_back(stride_C_val);
    stride_D_host.push_back(stride_D_val);

    // Reference tensors for verification (only one set needed)
    block_ref_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
  }

  // Allocate per-workspace tensors
  for (int ws = 0; ws < workspace_count; ++ws) {
    auto& w = workspaces[ws];

    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      auto stride_A_val = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
      auto stride_B_val = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
      auto stride_C_val = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
      auto stride_D_val = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

      auto layout_A = make_layout(make_shape(M, K, 1), stride_A_val);
      auto layout_B = make_layout(make_shape(N, K, 1), stride_B_val);
      auto layout_C = make_layout(make_shape(M, N, 1), stride_C_val);
      auto layout_D = make_layout(make_shape(M, N, 1), stride_D_val);
      auto layout_SFA_val = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
      auto layout_SFB_val = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

      w.block_A.push_back(HostTensorA(cutlass::make_Coord(size(layout_A))));
      w.block_B.push_back(HostTensorB(cutlass::make_Coord(size(layout_B))));
      w.block_SFA.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFA_val)))));
      w.block_SFB.push_back(HostTensorSF(cutlass::make_Coord(size(filter_zeros(layout_SFB_val)))));
      w.block_C.push_back(HostTensorC(cutlass::make_Coord(size(layout_C))));
      w.block_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
    }
  }

  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands for all workspaces
void initialize(const Options &options, int workspace_count) {
  uint64_t seed = 2020;

  // Shared data
  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  layout_SFA.reset(options.groups);
  layout_SFA.copy_from_host(layout_SFA_host.data());

  layout_SFB.reset(options.groups);
  layout_SFB.copy_from_host(layout_SFB_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  // Alpha/beta (shared across workspaces)
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    alpha_host.push_back((options.alpha == std::numeric_limits<float>::max())
        ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == std::numeric_limits<float>::max())
        ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  // Per-workspace initialization
  for (int ws = 0; ws < workspace_count; ++ws) {
    auto& w = workspaces[ws];

    std::vector<typename ElementA::DataType *> ptr_A_host(options.groups);
    std::vector<typename ElementB::DataType *> ptr_B_host(options.groups);
    std::vector<typename Gemm::GemmKernel::CollectiveMainloop::ElementSF *> ptr_SFA_host(options.groups);
    std::vector<typename Gemm::GemmKernel::CollectiveMainloop::ElementSF *> ptr_SFB_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementD *> ptr_D_host(options.groups);

    for (int32_t i = 0; i < options.groups; ++i) {
      // Different seed per workspace to get different data patterns
      initialize_block(w.block_A.at(i).host_view(), seed + ws * 1000 + 2021);
      initialize_block(w.block_B.at(i).host_view(), seed + ws * 1000 + 2022);
      initialize_block(w.block_C.at(i).host_view(), seed + ws * 1000 + 2023);
      initialize_block(w.block_SFA.at(i).host_view(), seed + ws * 1000 + 2024);
      initialize_block(w.block_SFB.at(i).host_view(), seed + ws * 1000 + 2025);

      w.block_A.at(i).sync_device();
      w.block_B.at(i).sync_device();
      w.block_C.at(i).sync_device();
      w.block_SFA.at(i).sync_device();
      w.block_SFB.at(i).sync_device();

      ptr_A_host.at(i) = w.block_A.at(i).device_data();
      ptr_B_host.at(i) = w.block_B.at(i).device_data();
      ptr_SFA_host.at(i) = w.block_SFA.at(i).device_data();
      ptr_SFB_host.at(i) = w.block_SFB.at(i).device_data();
      ptr_C_host.at(i) = w.block_C.at(i).device_data();
      ptr_D_host.at(i) = w.block_D.at(i).device_data();
    }

    w.ptr_A.reset(options.groups);
    w.ptr_A.copy_from_host(ptr_A_host.data());

    w.ptr_B.reset(options.groups);
    w.ptr_B.copy_from_host(ptr_B_host.data());

    w.ptr_SFA.reset(options.groups);
    w.ptr_SFA.copy_from_host(ptr_SFA_host.data());

    w.ptr_SFB.reset(options.groups);
    w.ptr_SFB.copy_from_host(ptr_SFB_host.data());

    w.ptr_C.reset(options.groups);
    w.ptr_C.copy_from_host(ptr_C_host.data());

    w.ptr_D.reset(options.groups);
    w.ptr_D.copy_from_host(ptr_D_host.data());
  }
}

/// Initialize gemm arguments and workspace for a specific workspace index
void initialize_gemm_for_workspace(int ws, Options &options) {
  auto& w = workspaces[ws];

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = std::min(
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id),
      options.max_sm_count);

  typename GemmPingpong::Arguments arguments_for_type;
  decltype(arguments_for_type.epilogue.thread) fusion_args;
  (void)arguments_for_type;  // Suppress unused variable warning

  // Alpha/beta setup
  if (options.alpha != std::numeric_limits<float>::max()) {
    fusion_args.alpha = options.alpha;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.alpha = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.dAlpha = {_0{}, _0{}, 1};
  }

  if (options.beta != std::numeric_limits<float>::max()) {
    fusion_args.beta = options.beta;
    fusion_args.beta_ptr = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.beta = 0;
    fusion_args.beta_ptr = nullptr;
    fusion_args.beta_ptr_array = beta_device.get();
    fusion_args.dBeta = {_0{}, _0{}, 1};
  }

  typename GemmPingpong::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = options.raster_order;

  w.arguments = typename GemmPingpong::Arguments {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), nullptr},
    {w.ptr_A.get(), stride_A.get(), w.ptr_B.get(), stride_B.get(),
     w.ptr_SFA.get(), layout_SFA.get(), w.ptr_SFB.get(), layout_SFB.get()},
    {fusion_args, w.ptr_C.get(), stride_C.get(), w.ptr_D.get(), stride_D.get()},
    hw_info, scheduler
  };

  size_t workspace_size = GemmPingpong::get_workspace_size(w.arguments);
  w.device_workspace.reset(workspace_size);

  CUTLASS_CHECK(w.gemm.can_implement(w.arguments));
  CUTLASS_CHECK(w.gemm.initialize(w.arguments, w.device_workspace.get()));
}

/// Execute profiling with L2 cache busting via workspace rotation
int run(Options &options, int workspace_count) {
  std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
  for (int32_t i = 0; i < options.groups && i < 1; ++i) {
    std::cout << "    " << options.problem_sizes_host.at(i);
    std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
  }
  std::cout << "  Groups         : " << options.groups << std::endl;
  std::cout << "  Workspace Count: " << workspace_count << std::endl;

  // Initialize all gemm instances
  for (int ws = 0; ws < workspace_count; ++ws) {
    initialize_gemm_for_workspace(ws, options);
  }

  // Correctness run with workspace 0
  CUTLASS_CHECK(workspaces[0].gemm.run(nullptr, nullptr, options.use_pdl));
  cudaDeviceSynchronize();

  //
  // Inline profiling with workspace rotation
  //
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Pre-warmup: run each workspace once
  for (int ws = 0; ws < workspace_count; ++ws) {
    CUTLASS_CHECK(workspaces[ws].gemm.run(stream, nullptr, options.use_pdl));
  }
  cudaStreamSynchronize(stream);

  // Capture graph: 2 rounds through all workspaces
  int graph_iterations = 2 * workspace_count;

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < graph_iterations; ++i) {
    int ws = i % workspace_count;
    CUTLASS_CHECK(workspaces[ws].gemm.run(stream, nullptr, options.use_pdl));
  }

  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
    std::cerr << "Graph capture failed" << std::endl;
    return -1;
  }

  if (cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    std::cerr << "Graph instantiate failed" << std::endl;
    return -1;
  }

  // Warmup: launch graph once
  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);

  // Timed: launch graph once
  cudaEventRecord(start, stream);
  cudaGraphLaunch(graphExec, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);

  // Time per kernel = total_time / graph_iterations
  float avg_time_ms = total_ms / graph_iterations;

  std::cout << "  Graph iterations: " << graph_iterations << std::endl;
  std::cout << "  Total graph time: " << total_ms << " ms" << std::endl;
  std::cout << "  Avg kernel time : " << avg_time_ms << " ms" << std::endl;

  // Compute TFLOPS
  double runtime_s = avg_time_ms / 1000.0;
  double gflops_val = options.gflops(runtime_s, options.problem_sizes_host);
  std::cout << "  TFLOPS          : " << gflops_val / 1000.0 << std::endl;

  // Cleanup
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit for SM120 support,
  // or CUDA 12.9 or higher for SM121 support.
#if 1 or defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 ||
       ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
       )
     ) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Blackwell Architecture (compute capability 120 or 121).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if 1 or defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  // Calculate workspace count for L2 cache busting
  int workspace_count = calculate_workspace_count(options, props);

  std::cout << "L2 Cache Size    : " << (props.l2CacheSize >> 20) << " MB" << std::endl;
  std::cout << "Workspace Count  : " << workspace_count << std::endl;
  std::cout << "Element A Format : " << ELEMENT_A_FORMAT_STR << std::endl;
  std::cout << "Element B Format : " << ELEMENT_B_FORMAT_STR << std::endl;

  allocate(options, workspace_count);
  initialize(options, workspace_count);

  //
  // Evaluate CUTLASS kernels
  //

  std::cout << "Running MX-format Grouped GEMM with L2 cache busting:" << std::endl;
  run(options, workspace_count);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
