import os
import sys
import argparse

import image_io
import init_points
import lloyd_sequential


def build_parser():
    p = argparse.ArgumentParser(description="Stippling via Lloyd's Algorithm")
    p.add_argument("--input", required=True, help="path to input image")
    p.add_argument("--points", type=int, default=500, help="number of stipple points")
    p.add_argument("--iters", type=int, default=25, help="max iterations")
    p.add_argument("--epsilon", type=float, default=0.5, help="convergence threshold")
    p.add_argument("--output", default="out.png", help="output image path")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit("error: input file not found: %s" % args.input)

    img = image_io.load_image(args.input)
    img = image_io.resize(img, 150)
    density, width, height = image_io.to_density_map(img)

    points = init_points.rejection_sampling(density, width, height, args.points)
    final_points, history = lloyd_sequential.run_sequential(
        density, points, width, height, args.iters, args.epsilon)

    out = image_io.render_points_to_image(final_points, width, height)
    out.save(args.output)
    print("saved output: %s (%d points)" % (args.output, len(final_points)))


if __name__ == "__main__":
    main()
