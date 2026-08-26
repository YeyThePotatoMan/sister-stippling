import os
import ctypes

_NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")


def run_simd(density_map, points, width, height, max_iter, epsilon, use_simd=True):
    libname = "simd_kernel.so" if use_simd else "scalar_kernel.so"
    path = os.path.join(_NATIVE_DIR, libname)
    lib = ctypes.CDLL(path)
    proto = (ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
             ctypes.c_int,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double))
    lib.lloyd_assign_simd.argtypes = proto

    n = len(points)
    density_arr = (ctypes.c_double * len(density_map))(*density_map)
    flat = []
    for (x, y) in points:
        flat.append(x)
        flat.append(y)
    pts_arr = (ctypes.c_double * (2 * n))(*flat)
    sum_x = (ctypes.c_double * n)()
    sum_y = (ctypes.c_double * n)()
    sum_w = (ctypes.c_double * n)()

    current = points
    history = []
    for it in range(max_iter):
        lib.lloyd_assign_simd(density_arr, width, height, pts_arr, pts_arr, n, sum_x, sum_y, sum_w)
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
