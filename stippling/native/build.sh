#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "compiling scalar kernel (no simd)..."
gcc -shared -fPIC -O2 -o scalar_kernel.so scalar_kernel.c

echo "compiling avx2 simd kernel..."
gcc -shared -fPIC -O2 -mavx2 -o simd_kernel.so simd_kernel.c

echo "done"
