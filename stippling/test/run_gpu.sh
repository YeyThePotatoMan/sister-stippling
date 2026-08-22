#!/bin/bash
# manual benchmark: gpu vs sequential vs cpu on the small sample image
# requires a working CUDA toolkit + numba for the gpu mode
set -e
cd "$(dirname "$0")/.."
python3 main.py --input test/sample_small.png --points 400 --iters 20 --epsilon 0.5 --mode benchmark --workers 8
