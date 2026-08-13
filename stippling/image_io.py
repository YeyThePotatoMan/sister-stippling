from PIL import Image


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def resize(img, max_side):
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def to_density_map(img):
    gray = img.convert("L")
    w, h = gray.size
    px = gray.load()
    density = [0.0] * (w * h)
    for y in range(h):
        row = y * w
        for x in range(w):
            g = px[x, y]
            density[row + x] = (255 - g) / 255.0
    return density, w, h
