import time
import os
import image_io
import init_points
import lloyd_sequential
import lloyd_cpu_parallel
import lloyd_gpu


def _build_density(input_path, max_side):
    img = image_io.load_image(input_path)
    img = image_io.resize(img, max_side)
    return image_io.to_density_map(img)


def run_benchmark(input_path, num_points, max_iter, epsilon, workers, max_side=150):
    density, width, height = _build_density(input_path, max_side)

    results = {}

    t0 = time.perf_counter()
    pts = init_points.rejection_sampling(density, width, height, num_points)
    fin, _ = lloyd_sequential.run_sequential(density, pts, width, height, max_iter, epsilon)
    t1 = time.perf_counter()
    results["sequential"] = (t1 - t0, fin)

    t0 = time.perf_counter()
    ptc = init_points.rejection_sampling(density, width, height, num_points)
    finc, _ = lloyd_cpu_parallel.run_cpu_parallel(density, ptc, width, height, max_iter, epsilon, workers)
    t1 = time.perf_counter()
    results["cpu"] = (t1 - t0, finc)

    gpu_time = None
    if lloyd_gpu.gpu_available():
        t0 = time.perf_counter()
        ptg = init_points.rejection_sampling(density, width, height, num_points)
        fing, _ = lloyd_gpu.run_gpu(density, ptg, width, height, max_iter, epsilon)
        t1 = time.perf_counter()
        gpu_time = t1 - t0
        results["gpu"] = (gpu_time, fing)

    seq_time = results["sequential"][0]
    print("")
    print("Benchmark (input=%s, points=%d, iters=%d, epsilon=%.3f, workers=%d, max_side=%d)"
          % (os.path.basename(input_path), num_points, max_iter, epsilon, workers, max_side))
    print("Resolution: %dx%d" % (width, height))
    print("")
    header = "%-12s %12s %12s" % ("mode", "time (s)", "speedup")
    print(header)
    print("-" * len(header))
    print("%-12s %12.4f %12s" % ("sequential", seq_time, "1.00x"))
    print("%-12s %12.4f %11.2fx" % ("cpu", results["cpu"][0], seq_time / results["cpu"][0]))
    if gpu_time is not None:
        print("%-12s %12.4f %11.2fx" % ("gpu", gpu_time, seq_time / gpu_time))
    else:
        print("%-12s %12s %12s" % ("gpu", "N/A", "n/a"))
    print("")
    return results
