import multiprocessing as mp


def _worker(args):
    start, end, density, points, width, height = args
    n = len(points)
    sum_x = [0.0] * n
    sum_y = [0.0] * n
    sum_w = [0.0] * n
    for y in range(start, end):
        row = y * width
        for x in range(width):
            w = density[row + x]
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


def run_cpu_parallel(density_map, points, width, height, max_iter, epsilon, n_workers):
    history = []
    current = points
    chunk = max(1, height // n_workers)
    with mp.Pool(n_workers) as pool:
        for it in range(max_iter):
            n = len(current)
            tasks = []
            for w in range(n_workers):
                start = w * chunk
                end = height if w == n_workers - 1 else (w + 1) * chunk
                tasks.append((start, end, density_map, current, width, height))
            results = pool.map(_worker, tasks)
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
            if max_shift < epsilon:
                current = new_points
                break
            current = new_points
    return current, history
