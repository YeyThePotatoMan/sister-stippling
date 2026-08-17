# sister-stippling

## Requirements

```bash
pip install Pillow numba
```

- **GPU mode** additionally needs a CUDA toolkit + NVIDIA driver. If no GPU is
  available, the program prints a clear message and exits instead of crashing.
- **SIMD bonus** needs `gcc` and an AVX2-capable CPU. Run `bash native/build.sh`
  once to compile the shared libraries.

## Usage

```bash
python main.py --input IMG --points N --iters K --epsilon E --output OUT --mode MODE
```

| Argument     | Meaning                                   | Default     |
|--------------|-------------------------------------------|-------------|
| `--input`    | input image path                          | *(required)*|
| `--points`   | number of stipple dots                    | `500`       |
| `--iters`    | max Lloyd iterations                      | `25`        |
| `--epsilon`  | stop when max dot movement drops below this | `0.5`     |
| `--output`   | output PNG path                           | `out.png`   |
| `--mode`     | `sequential` / `cpu` / `gpu` / `benchmark`| `sequential`|

Handy extras: `--workers N` (cpu processes), `--max-side N` (resize longest side,
default `150`, `--scale N` (upscale the rendered dots), `--animate --gif out.gif`
(make a convergence animation), and `--interactive` (answer prompts instead of
using flags).

The CLI lives in `stippling/main.py`; run it from inside the `stippling/` folder.

## Examples

```bash
# Baseline on a single core
python main.py --input ../test/sample_small.png --points 400 --iters 25 \
               --epsilon 0.5 --output out_seq.png --mode sequential

# Use all CPU cores
python main.py --input ../test/sample_small.png --points 400 --iters 25 \
               --epsilon 0.5 --output out_cpu.png --mode cpu --workers 8

# GPU (needs CUDA)
python main.py --input ../test/sample_small.png --points 400 --iters 25 \
               --epsilon 0.5 --output out_gpu.png --mode gpu

# Run all three with identical parameters and print a speedup table
python main.py --input ../test/sample_small.png --points 500 --iters 25 \
               --epsilon 0.5 --mode benchmark --workers 8

# Save an animation of the dots relaxing into place
python main.py --input ../test/sample_small.png --points 300 --iters 20 \
               --epsilon 0.3 --output out.png --mode cpu --workers 4 \
               --animate --gif convergence.gif
```

## Project layout

```
stippling/
├── main.py                 # CLI + pipeline orchestration
├── image_io.py             # Pillow: load, resize, density map, render, gif
├── init_points.py          # rejection sampling (pure python)
├── lloyd_sequential.py     # one process, one thread
├── lloyd_cpu_parallel.py   # multiprocessing, pixel rows split per worker
├── lloyd_gpu.py            # numba.cuda kernel, one thread per pixel
├── benchmark.py            # runs all three modes and prints a speedup table
├── interactive.py          # optional TUI prompt
├── lloyd_simd.py           # optional AVX2 SIMD wrapper
├── native/                 # C kernels (scalar + AVX2) and build script
└── test/                   # sample image + manual run notes
```
