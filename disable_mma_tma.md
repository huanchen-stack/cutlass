`cutlass/build$ cmake .. -DCUTLASS_NVCC_ARCHS="120a" -DCUTLASS_DISABLE_TMA=1 -DCUTLASS_DISABLE_MMA=1`

Prelim results on m=n=k=2048 (Schedule: PingPong always better)
regular: 0.451428 ms
no_tma_no_mma: 0.16158 ms
no_mma: 0.328722
no_tma: 0.38159 ms