def lloyd_iteration(density_map, points, width, height):
    n = len(points)
    sum_x = [0.0] * n
    sum_y = [0.0] * n
    sum_w = [0.0] * n
    for y in range(height):
        row = y * width
        for x in range(width):
            w = density_map[row + x]
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
    new_points = []
    for i in range(n):
        if sum_w[i] > 0.0:
            new_points.append((sum_x[i] / sum_w[i], sum_y[i] / sum_w[i]))
        else:
            new_points.append(points[i])
    return new_points
