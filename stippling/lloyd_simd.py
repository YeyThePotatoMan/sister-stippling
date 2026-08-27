import os
import ctypes
import glob

from types import SimpleNamespace

_NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")


def avx2_supported():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx2" in line.split()
    except Exception:
        return False
    return False


def _load(libname):
    path = os.path.join(_NATIVE_DIR, libname)
    if not os.path.isfile(path):
        raise FileNotFoundError("native lib not found: %s (run native/build.sh)" % path)
    lib = ctypes.CDLL(path)
    proto = (ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
             ctypes.c_int,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double))
    if hasattr(lib, "lloyd_assign_scalar"):
        lib.lloyd_assign_scalar.argtypes = proto
    if hasattr(lib, "lloyd_assign_simd"):
        lib.lloyd_assign_simd.argtypes = proto
    return lib


def run_simd(density_map, points, width, height, max_iter, epsilon, use_simd=True):
    if use_simd and avx2_supported():
        lib = _load("simd_kernel.so")
        fn = lib.lloyd_assign_simd
    else:
        lib = _load("scalar_kernel.so")
        fn = lib.lloyd_assign_scalar

    n = len(points)
    density_arr = (ctypes.c_double * len(density_map))(*density_map)
    sum_x = (ctypes.c_double * n)()
    sum_y = (ctypes.c_double * n)()
    sum_w = (ctypes.c_double * n)()

    current = points
    history = []
    for it in range(max_iter):
        px = [p[0] for p in current]
        py = [p[1] for p in current]
        px_arr = (ctypes.c_double * n)(*px)
        py_arr = (ctypes.c_double * n)(*py)
        fn(density_arr, width, height, px_arr, py_arr, n, sum_x, sum_y, sum_w)
        new_points = []
        max_shift = 0.0
        for i in range(n):
            if sum_w[i] > 0.0:
                nx = sum_x[i] / sum_w[i]
                ny = sum_y[i] / sum_w[i]
            else:
                nx, ny = current[i]
            dx = nx - current[i][0]
            dy = ny - current[i][1]
            shift = (dx * dx + dy * dy) ** 0.5
            if shift > max_shift:
                max_shift = shift
            new_points.append((nx, ny))
        history.append(max_shift)
        if max_shift < epsilon:
            current = new_points
            break
        current = new_points
    return current, history
