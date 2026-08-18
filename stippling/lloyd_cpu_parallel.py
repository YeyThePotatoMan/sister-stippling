import threading


def _worker(rows, density, points, width, height, sum_x, sum_y, sum_w):
    n = len(points)
    for y in rows:
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


def run_cpu_parallel(density_map, points, width, height, max_iter, epsilon, n_workers):
    history = []
    current = points
    for it in range(max_iter):
        n = len(current)
        sum_x = [0.0] * n
        sum_y = [0.0] * n
        sum_w = [0.0] * n
        threads = []
        chunk = max(1, height // n_workers)
        for w in range(n_workers):
            start = w * chunk
            end = height if w == n_workers - 1 else (w + 1) * chunk
            rows = list(range(start, end))
            t = threading.Thread(target=_worker, args=(rows, density_map, current, width, height, sum_x, sum_y, sum_w))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
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
