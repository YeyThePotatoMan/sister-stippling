import multiprocessing as mp


_density = None
_width = 0
_height = 0
_chunks = []


def _init_worker(density, width, height, chunks):
    global _density, _width, _height, _chunks
    _density = density
    _width = width
    _height = height
    _chunks = chunks


def _process_chunk(args):
    start, end, points = args
    n = len(points)
    sum_x = [0.0] * n
    sum_y = [0.0] * n
    sum_w = [0.0] * n
    for y in range(start, end):
        row = y * _width
        for x in range(_width):
            w = _density[row + x]
            if w == 0.0:
                continue
            best = 0
            best_d = -1.0
            for i in range(n):
                px, py = points[i]
                dx = x - px
                dy = y - py
                d = dx * dx + dy * dy
                if best_d < 0.0 or d < best_d:
                    best_d = d
                    best = i
            sum_x[best] += x * w
            sum_y[best] += y * w
            sum_w[best] += w
    return (sum_x, sum_y, sum_w)


def _split_rows(height, n_workers):
    n_workers = max(1, min(n_workers, height))
    base = height // n_workers
    rem = height % n_workers
    chunks = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        if size > 0:
            chunks.append((start, start + size))
            start += size
    return chunks


def run_cpu_parallel(density_map, points, width, height, max_iter, epsilon, n_workers, snapshots=None, progress=None):
    chunks = _split_rows(height, n_workers)
    history = []
    current = points
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(density_map, width, height, chunks)) as pool:
        for it in range(max_iter):
            tasks = [(s, e, current) for (s, e) in chunks]
            results = pool.map(_process_chunk, tasks)
            n = len(current)
            sum_x = [0.0] * n
            sum_y = [0.0] * n
            sum_w = [0.0] * n
            for (sx, sy, sw) in results:
                for i in range(n):
                    sum_x[i] += sx[i]
                    sum_y[i] += sy[i]
                    sum_w[i] += sw[i]
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
            if snapshots is not None:
                snapshots.append(list(new_points))
            if progress is not None:
                progress(it, max_shift)
            if max_shift < epsilon:
                current = new_points
                break
            current = new_points
    return current, history
