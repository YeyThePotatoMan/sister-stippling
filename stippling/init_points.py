import random


def uniform_random(width, height, num_points, rng):
    points = []
    for _ in range(num_points):
        x = rng.random() * (width - 1)
        y = rng.random() * (height - 1)
        points.append((x, y))
    return points
