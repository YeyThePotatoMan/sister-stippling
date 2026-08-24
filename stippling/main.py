import os
import sys
import argparse
import time

import image_io
import init_points
import lloyd_sequential
import lloyd_cpu_parallel
import lloyd_gpu
import benchmark


def run_pipeline(args, mode):
    if not os.path.isfile(args.input):
        sys.exit("error: input file not found: %s" % args.input)

    if mode == "gpu" and not lloyd_gpu.gpu_available():
        sys.exit("error: GPU not available (numba.cuda.is_available() is False).")

    t0 = time.perf_counter()
    img = image_io.load_image(args.input)
    img = image_io.resize(img, args.max_side)
    density, width, height = image_io.to_density_map(img)
    t1 = time.perf_counter()
    print("image io (load/resize/density): %.3f s  resolution: %dx%d"
          % (t1 - t0, width, height))

    points = init_points.rejection_sampling(density, width, height, args.points)
    snapshots = [] if args.animate else None

    run_t0 = time.perf_counter()
    if mode == "sequential":
        final_points, history = lloyd_sequential.run_sequential(
            density, points, width, height, args.iters, args.epsilon, snapshots)
    elif mode == "cpu":
        final_points, history = lloyd_cpu_parallel.run_cpu_parallel(
            density, points, width, height, args.iters, args.epsilon, args.workers, snapshots)
    elif mode == "gpu":
        final_points, history = lloyd_gpu.run_gpu(
            density, points, width, height, args.iters, args.epsilon, snapshots)
    run_t1 = time.perf_counter()
    print("%s lloyd loop: %.3f s  iterations: %d  final max_shift: %.4f"
          % (mode, run_t1 - run_t0, len(history), history[-1] if history else 0.0))

    out = image_io.render_points_to_image(final_points, width, height)
    out.save(args.output)
    print("saved output: %s (%d points)" % (args.output, len(final_points)))

    if args.animate and snapshots:
        frames = [image_io.render_points_to_image(s, width, height) for s in snapshots]
        gif_path = args.gif if args.gif else (os.path.splitext(args.output)[0] + ".gif")
        image_io.save_gif(frames, gif_path, duration=args.gif_duration)
        print("saved animation: %s (%d frames)" % (gif_path, len(frames)))


def build_parser():
    p = argparse.ArgumentParser(description="Stippling via Lloyd's Algorithm")
    p.add_argument("--input", required=False, help="path to input image")
    p.add_argument("--points", type=int, default=500, help="number of stipple points")
    p.add_argument("--iters", type=int, default=25, help="max iterations")
    p.add_argument("--epsilon", type=float, default=0.5, help="convergence threshold")
    p.add_argument("--output", default="out.png", help="output image path")
    p.add_argument("--mode", choices=["sequential", "cpu", "gpu", "benchmark"], default="sequential")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="cpu worker processes")
    p.add_argument("--max-side", type=int, default=150, help="resize longest side to this")
    p.add_argument("--scale", type=int, default=1, help="upscale factor for rendered dots")
    p.add_argument("--animate", action="store_true", help="save per-iteration snapshots")
    p.add_argument("--gif", default=None, help="gif output path for animation")
    p.add_argument("--gif-duration", type=int, default=200, dest="gif_duration", help="ms per frame")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required")
    if args.mode == "benchmark":
        benchmark.run_benchmark(args.input, args.points, args.iters, args.epsilon, args.workers, args.max_side)
        return
    run_pipeline(args, args.mode)


if __name__ == "__main__":
    main()
