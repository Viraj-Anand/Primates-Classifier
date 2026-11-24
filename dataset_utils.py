"""Everything about the photo folder that does not need TensorFlow:
class discovery, validation, duplicate detection, image loading, demo data."""
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (256, 256)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PER_CLASS = 3        # below this a train/val/test split is impossible
WARN_PER_CLASS = 30      # below this the test accuracy is mostly noise
NEAR_DUP_BITS = 5        # perceptual-hash distance that counts as "the same photo"
DEMO_CLASSES = {"chimpanzee": (70, 50, 40), "gorilla": (60, 60, 65), "orangutan": (200, 110, 40)}


def discover_classes(data_dir):
    """Every sub-folder of data_dir is a class; the folder name is the class name."""
    root = Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"Data folder not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def is_image(path):
    """True if the file really decodes as an image (checked by content, not by name)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def scan(data_dir):
    """Return (classes, photos, skipped). photos is [(path, label)], skipped is [(path, reason)]."""
    classes = discover_classes(data_dir)
    photos, skipped = [], []
    for label, name in enumerate(classes):
        for p in sorted((Path(data_dir) / name).iterdir()):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in (".heic", ".heif"):
                skipped.append((str(p), "HEIC is not supported: convert to JPEG (or pip install pillow-heif)"))
            elif ext not in EXTENSIONS:
                skipped.append((str(p), f"unsupported file type {ext or '(none)'}"))
            elif not is_image(p):
                skipped.append((str(p), "unreadable or corrupt image"))
            else:
                photos.append((str(p), label))
    return classes, photos, skipped


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average_hash(path, size=8):
    """64-bit perceptual hash: shrink to 8x8 grey, one bit per pixel above the mean."""
    with Image.open(path) as im:
        px = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((size, size), Image.LANCZOS), dtype=float)
    return (px > px.mean()).flatten()


def duplicate_clusters(paths, max_bits=NEAR_DUP_BITS):
    """Group photos that are byte-identical or look the same (hash distance <= max_bits).
    Returns clusters with more than one photo. O(n^2): fine for a few thousand photos."""
    hashes = [average_hash(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if np.count_nonzero(hashes[i] != hashes[j]) <= max_bits:
                parent[find(i)] = find(j)
    groups = {}
    for i, p in enumerate(paths):
        groups.setdefault(find(i), []).append(p)
    return [g for g in groups.values() if len(g) > 1]


def check_data(data_dir):
    """Print a preflight report. Returns True if training is possible."""
    classes, photos, skipped = scan(data_dir)
    print(f"Data folder: {data_dir}")
    if len(classes) < 2:
        print(f"  ERROR: need at least 2 class folders, found {len(classes)}: {classes}")
        return False
    ok = True
    counts = {c: sum(1 for _, l in photos if l == i) for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}, alphabetical = label order):")
    for c, n in counts.items():
        note = ""
        if n < MIN_PER_CLASS:
            note, ok = f"  ERROR: need at least {MIN_PER_CLASS}", False
        elif n < WARN_PER_CLASS:
            note = f"  warning: under {WARN_PER_CLASS} photos, test accuracy will be noisy"
        print(f"  {c:<14}{n:>5} usable photos{note}")
    print(f"Total usable photos: {len(photos)}")
    if skipped:
        print(f"Skipped files ({len(skipped)}):")
        for path, why in skipped:
            print(f"  {path}: {why}")
    exact = {}
    for p, _ in photos:
        exact.setdefault(file_hash(p), []).append(p)
    exact = [g for g in exact.values() if len(g) > 1]
    near = duplicate_clusters([p for p, _ in photos])
    print(f"Exact duplicate groups: {len(exact)}")
    print(f"Near-duplicate groups (hash distance <= {NEAR_DUP_BITS}): {len(near)}")
    for g in near[:10]:
        print("  " + "  ==  ".join(Path(p).name for p in g))
    if near:
        print("  train.py keeps each group inside one split, so burst photos cannot leak between sets.")
    print("Result: " + ("OK to train." if ok else "NOT ready, fix the errors above."))
    return ok


def split(photos, seed, groups=()):
    """About 70% train / 20% validation / 10% test, made once inside each class.
    Photos in the same duplicate group always land in the same set."""
    rng = random.Random(seed)
    cluster = {p: i for i, g in enumerate(groups) for p in g}
    train, val, test = [], [], []
    for label in sorted({l for _, l in photos}):
        units = {}
        for p, l in photos:
            if l == label:
                units.setdefault(cluster.get(p, p), []).append((p, l))
        units = list(units.values())
        rng.shuffle(units)
        total = sum(len(u) for u in units)
        n_test, n_val = max(1, int(total * 0.1)), max(1, int(total * 0.2))
        a, b = [], []
        for u in units:
            if len(a) < n_test:
                a += u
            elif len(b) < n_val:
                b += u
            else:
                train += u
        test += a
        val += b
    return train, val, test


def load_array(path, size=IMG_SIZE, scale=True):
    """Photo -> array (H, W, 3): float32 0-1 if scale, else uint8. Handles EXIF rotation and webp."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    a = np.asarray(im, dtype=np.uint8)
    return a.astype(np.float32) / 255.0 if scale else a


def synthetic_image(base_rgb, size=256, rng=None):
    """Plain coloured noise. Not a primate: it only exercises the code."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, 35, (size, size, 3))
    return Image.fromarray(np.clip(np.array(base_rgb) + noise, 0, 255).astype(np.uint8))


def make_demo_dataset(root, per_class=100, size=256, seed=0):
    """Write per_class synthetic JPEGs for each demo class into root/<class>/."""
    rng = np.random.default_rng(seed)
    for name, rgb in DEMO_CLASSES.items():
        folder = Path(root) / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            synthetic_image(rgb, size, rng).save(folder / f"{name}_{i:03d}.jpg", quality=90)
    return Path(root)
