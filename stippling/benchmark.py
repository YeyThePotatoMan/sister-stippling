import time
import os
import image_io
import init_points
import lloyd_sequential
import lloyd_cpu_parallel
import lloyd_gpu


def run_benchmark(input_path, num_points, max_iter, epsilon, workers, max_side=150):
    results = {}

    t0 = time.perf_counter()
    density, width, height = image_io.to_density_map(
        image_io.resize(image_io.load_image(input_path), max_side))
    pts = init_points.rejection_sampling(density, width, height, num_points)
    fin, _ = lloyd_sequential.run_sequential(density, pts, width, height, max_iter, epsilon)
    results["sequential"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    density, width, height = image_io.to_density_map(
        image_io.resize(image_io.load_image(input_path), max_side))
    ptc = init_points.rejection_sampling(density, width, height, num_points)
    finc, _ = lloyd_cpu_parallel.run_cpu_parallel(density, ptc, width, height, max_iter, epsilon, workers)
    results["cpu"] = time.perf_counter() - t0

    if lloyd_gpu.gpu_available():
        t0 = time.perf_counter()
        density, width, height = image_io.to_density_map(
            image_io.resize(image_io.load_image(input_path), max_side))
        ptg = init_points.rejection_sampling(density, width, height, num_points)
        fing, _ = lloyd_gpu.run_gpu(density, ptg, width, height, max_iter, epsilon)
        results["gpu"] = time.perf_counter() - t0

    print("Benchmark (input=%s, points=%d, iters=%d, epsilon=%.3f, workers=%d)"
          % (os.path.basename(input_path), num_points, max_iter, epsilon, workers))
    print("Resolution: %dx%d" % (width, height))
    print("")
    print("sequential: %.4f s" % results["sequential"])
    print("cpu:        %.4f s" % results["cpu"])
    if "gpu" in results:
        print("gpu:        %.4f s" % results["gpu"])
    print("")
    return results
