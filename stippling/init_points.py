import random


def uniform_random(width, height, num_points, rng):
    points = []
    for _ in range(num_points):
        x = rng.random() * (width - 1)
        y = rng.random() * (height - 1)
        points.append((x, y))
    return points


def rejection_sampling(density_map, width, height, num_points):
    rng = random.Random(12345)
    total_density = sum(density_map)
    points = []
    for _ in range(num_points):
        while True:
            x = int(rng.random() * width)
            y = int(rng.random() * height)
            idx = y * width + x
            d = density_map[idx]
            if total_density > 0:
                if rng.random() < d:
                    points.append((float(x), float(y)))
                    break
            else:
                break
    return points
