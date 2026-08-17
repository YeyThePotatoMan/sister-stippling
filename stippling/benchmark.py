import time
import os
import image_io
import init_points
import lloyd_sequential


def run_benchmark(input_path, num_points, max_iter, epsilon):
    t0 = time.perf_counter()
    img = image_io.load_image(input_path)
    img = image_io.resize(img, 150)
    density, width, height = image_io.to_density_map(img)
    points = init_points.rejection_sampling(density, width, height, num_points)
    final_points, history = lloyd_sequential.run_sequential(
        density, points, width, height, max_iter, epsilon)
    t1 = time.perf_counter()
    print("")
    print("Benchmark (input=%s, points=%d, iters=%d, epsilon=%.3f)"
          % (os.path.basename(input_path), num_points, max_iter, epsilon))
    print("Resolution: %dx%d" % (width, height))
    print("")
    print("sequential time: %.4f s" % (t1 - t0))
    print("")
    return final_points
