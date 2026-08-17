import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

_shm = None
_density = None
_width = 0
_height = 0

def _init_worker(shm_name, shape, width, height):
    global _shm, _density, _width, _height
    _shm = shared_memory.SharedMemory(name=shm_name)
    _density = np.ndarray(shape, dtype=np.float64, buffer=_shm.buf)
    _width, _height = width, height

def _process_chunk(args):
    start, end, points = args
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    sub = _density[start:end]
    ys_local, xs = np.nonzero(sub)
    if xs.size == 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    ys = ys_local + start
    w = sub[ys_local, xs]
    dx = xs[:, None] - pts[None, :, 0]
    dy = ys[:, None] - pts[None, :, 1]
    best = np.argmin(dx * dx + dy * dy, axis=1)
    return (np.bincount(best, weights=xs * w, minlength=n),
            np.bincount(best, weights=ys * w, minlength=n),
            np.bincount(best, weights=w, minlength=n))

def run_hybrid(density_map, points, width, height, max_iter, epsilon,
                n_workers, snapshots=None, progress=None):
    density = np.ascontiguousarray(density_map, dtype=np.float64).reshape(height, width)
    shm = shared_memory.SharedMemory(create=True, size=density.nbytes)
    shm_arr = np.ndarray(density.shape, dtype=np.float64, buffer=shm.buf)
    shm_arr[:] = density[:]

    chunks = _split_rows(height, n_workers)  # reuse fungsi kamu
    history, current = [], points
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(n_workers, initializer=_init_worker,
                       initargs=(shm.name, density.shape, width, height)) as pool:
            for it in range(max_iter):
                results = pool.map(_process_chunk, [(s, e, current) for s, e in chunks])
                n = len(current)
                sum_x = sum(r[0] for r in results)
                sum_y = sum(r[1] for r in results)
                sum_w = sum(r[2] for r in results)
                new_points, max_shift = [], 0.0
                for i in range(n):
                    if sum_w[i] > 0:
                        nx, ny = sum_x[i] / sum_w[i], sum_y[i] / sum_w[i]
                    else:
                        nx, ny = current[i]
                    shift = ((nx - current[i][0])**2 + (ny - current[i][1])**2) ** 0.5
                    max_shift = max(max_shift, shift)
                    new_points.append((nx, ny))
                history.append(max_shift)
                if snapshots is not None: snapshots.append(list(new_points))
                if progress is not None: progress(it, max_shift)
                current = new_points
                if max_shift < epsilon:
                    break
    finally:
        shm.close()
        shm.unlink()
    return current, history