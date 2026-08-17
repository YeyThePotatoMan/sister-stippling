# Manual run notes (work log, not formal docs)

## Environment
- Python 3.10, Pillow 9.0.1, numba (GPU via cu11 toolkit wheels).
- 16 CPU cores, NVIDIA RTX 3050 (compute 8.6).

## Commands tried
- sequential: works, ~7s for 300 pts @120px / 20 iters.
- cpu --workers 8: ~3.5s, ~2x speedup over sequential.
- gpu: ~0.2s, ~30x speedup over sequential.
- benchmark: prints table + speedup vs sequential.
- animation: `convergence.gif` has distinct frames (no aliasing bug).
- simd vs scalar: identical results (max diff 0.0).

## Bugs hit during dev (already fixed in history)
- transpose x/y in render (commit a7b5f44).
- sum-vs-max for epsilon (f3f4b17).
- threading instead of multiprocessing (ebab3de).
- gpu race without atomic.add (c45aa50 -> 7f1daba).
- snapshot aliasing (a6539a8 -> 8baba77).
- ctypes interleaved-array garbage (9953ce2 -> 0278c7a).
- simd tie-break differences vs scalar (fixed in simd_kernel.c).
