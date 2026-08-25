import os
from types import SimpleNamespace


def prompt():
    print("=== Stippling interactive mode ===")
    input_path = input("input image path: ").strip()
    while not os.path.isfile(input_path):
        print("  file not found, try again")
        input_path = input("input image path: ").strip()

    points = _ask_int("number of points", 500)
    iters = _ask_int("max iterations", 25)
    epsilon = _ask_float("epsilon (convergence)", 0.5)
    mode = _ask_choice("mode", ["sequential", "cpu", "gpu", "benchmark"], "sequential")
    workers = _ask_int("cpu workers", os.cpu_count())
    max_side = _ask_int("resize longest side", 150)
    animate = _ask_yes("export animation gif? (y/n)", False)
    scale = _ask_int("render upscale factor", 1)

    output = input("output image path [out.png]: ").strip() or "out.png"
    gif = None
    if animate:
        gif = input("gif path [out.gif]: ").strip() or "out.gif"

    return SimpleNamespace(
        input=input_path,
        points=points,
        iters=iters,
        epsilon=epsilon,
        output=output,
        mode=mode,
        workers=workers,
        max_side=max_side,
        scale=scale,
        animate=animate,
        gif=gif,
        gif_duration=200,
        interactive=True,
    )


def _ask_int(label, default):
    raw = input("%s [%d]: " % (label, default)).strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print("  invalid, using default %d" % default)
        return default


def _ask_float(label, default):
    raw = input("%s [%.3f]: " % (label, default)).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _ask_choice(label, choices, default):
    raw = input("%s %s [%s]: " % (label, choices, default)).strip()
    if raw == "" or raw not in choices:
        return default
    return raw


def _ask_yes(label, default):
    raw = input("%s: " % label).strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1", "true")
