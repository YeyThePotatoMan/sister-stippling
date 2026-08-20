#!/bin/bash
# manual comparison: sequential vs cpu parallel on the small sample image
set -e
cd "$(dirname "$0")/.."
python3 main.py --input test/sample_small.png --points 300 --iters 20 --epsilon 0.5 --output /tmp/seq.png --mode sequential
python3 main.py --input test/sample_small.png --points 300 --iters 20 --epsilon 0.5 --output /tmp/cpu.png --mode cpu --workers 8
echo "done; compare /tmp/seq.png and /tmp/cpu.png"
