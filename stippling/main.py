import argparse


def build_parser():
    p = argparse.ArgumentParser(description="Stippling via Lloyd's Algorithm")
    p.add_argument("--input", help="path to input image")
    p.add_argument("--points", type=int, default=500, help="number of stipple points")
    p.add_argument("--iters", type=int, default=25, help="max iterations")
    p.add_argument("--epsilon", type=float, default=0.5, help="convergence threshold")
    p.add_argument("--output", default="out.png", help="output image path")
    p.add_argument("--mode", choices=["sequential", "cpu", "gpu", "benchmark"], default="sequential")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    print("parsed args:", args)


if __name__ == "__main__":
    main()
