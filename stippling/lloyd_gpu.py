import numpy as np
from numba import cuda


def gpu_available():
    try:
        if not cuda.is_available():
            return False
        _ = cuda.get_current_device()
        return True
    except Exception:
        return False


@cuda.jit
def assign_kernel(density, px, py, n, width, height, sum_x, sum_y, sum_w):
    idx = cuda.grid(1)
    total = width * height
    if idx >= total:
        return
    x = idx % width
    y = idx // width
    w = density[idx]
    if w == 0.0:
        return
    best = 0
    best_d = -1.0
    for i in range(n):
        dx = x - px[i]
        dy = y - py[i]
        d = dx * dx + dy * dy
        if best_d < 0.0 or d < best_d:
            best_d = d
            best = i
    # atomic wajib, banyak thread nulis ke index sama
    cuda.atomic.add(sum_x, best, x * w)
    cuda.atomic.add(sum_y, best, y * w)
    cuda.atomic.add(sum_w, best, w)


def run_gpu(density_map, points, width, height, max_iter, epsilon, snapshots=None, progress=None):
    n = len(points)
    density_np = np.array(density_map, dtype=np.float32)
    density_dev = cuda.to_device(density_np)
    history = []
    current = points
    total = width * height
    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block
    sum_x_dev = cuda.to_device(np.zeros(n, dtype=np.float32))
    sum_y_dev = cuda.to_device(np.zeros(n, dtype=np.float32))
    sum_w_dev = cuda.to_device(np.zeros(n, dtype=np.float32))
    for it in range(max_iter):
        px = np.array([p[0] for p in current], dtype=np.float32)
        py = np.array([p[1] for p in current], dtype=np.float32)
        px_dev = cuda.to_device(px)
        py_dev = cuda.to_device(py)
        sum_x_dev.copy_to_device(np.zeros(n, dtype=np.float32))
        sum_y_dev.copy_to_device(np.zeros(n, dtype=np.float32))
        sum_w_dev.copy_to_device(np.zeros(n, dtype=np.float32))
        assign_kernel[blocks, threads_per_block](
            density_dev, px_dev, py_dev, n, width, height,
            sum_x_dev, sum_y_dev, sum_w_dev,
        )
        cuda.synchronize()
        sx = sum_x_dev.copy_to_host()
        sy = sum_y_dev.copy_to_host()
        sw = sum_w_dev.copy_to_host()
        new_points = []
        max_shift = 0.0
        for i in range(n):
            if sw[i] > 0.0:
                nx = float(sx[i]) / float(sw[i])
                ny = float(sy[i]) / float(sw[i])
            else:
                nx, ny = current[i]
            dx = nx - current[i][0]
            dy = ny - current[i][1]
            shift = (dx * dx + dy * dy) ** 0.5
            if shift > max_shift:
                max_shift = shift
            new_points.append((nx, ny))
        history.append(max_shift)
        if snapshots is not None:
            snapshots.append(list(new_points))
        if progress is not None:
            progress(it, max_shift)
        if max_shift < epsilon:
            current = new_points
            break
        current = new_points
    return current, history
