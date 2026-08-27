import sys
import time
import image_io
import init_points
import lloyd_sequential
import lloyd_simd

IMG = sys.argv[1] if len(sys.argv) > 1 else "test/sample_small.png"
POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20

img = image_io.load_image(IMG)
img = image_io.resize(img, 200)
density, w, h = image_io.to_density_map(img)
pts = init_points.rejection_sampling(density, w, h, POINTS)

t0 = time.perf_counter()
fpy, hpy = lloyd_sequential.run_sequential(density, pts, w, h, ITERS, 0.01)
tpy = time.perf_counter() - t0

t0 = time.perf_counter()
fsc, hsc = lloyd_simd.run_simd(density, pts, w, h, ITERS, 0.01, use_simd=False)
tsc = time.perf_counter() - t0

t0 = time.perf_counter()
fsi, hsi = lloyd_simd.run_simd(density, pts, w, h, ITERS, 0.01, use_simd=True)
tsi = time.perf_counter() - t0

maxdiff = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(fpy, fsi))
print("pure python : %.4f s  final shift %.4f" % (tpy, hpy[-1]))
print("scalar c    : %.4f s  final shift %.4f" % (tsc, hsc[-1]))
print("avx2 simd   : %.4f s  final shift %.4f" % (tsi, hsi[-1]))
print("simd speedup vs scalar: %.2fx" % (tsc / tsi))
print("max point diff simd vs python: %.6f" % maxdiff)
