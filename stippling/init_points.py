import random


def uniform_random(width, height, num_points, rng):
    points = []
    for _ in range(num_points):
        x = rng.random() * (width - 1)
        y = rng.random() * (height - 1)
        points.append((x, y))
    return points


def rejection_sampling(density_map, width, height, num_points, max_tries_per_point=2000):
    rng = random.Random(12345)
    total_density = sum(density_map)
    points = []
    for _ in range(num_points):
        placed = False
        for _ in range(max_tries_per_point):
            x = int(rng.random() * width)
            y = int(rng.random() * height)
            idx = y * width + x
            d = density_map[idx]
            if total_density > 0:
                if rng.random() < d:
                    points.append((float(x), float(y)))
                    placed = True
                    break
            else:
                break
        if not placed:
            x = rng.random() * (width - 1)
            y = rng.random() * (height - 1)
            points.append((x, y))
    return points
