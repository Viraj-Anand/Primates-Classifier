"""Pure-Python helpers shared by train.py (no TensorFlow / numpy needed)."""
import random

CLASSES = ["chimpanzee", "gorilla", "orangutan"]  # labels 0, 1, 2
IMG_SIZE = (256, 256)
BATCH = 30
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_PER_CLASS = 3  # each class needs at least one photo in train, validation and test


def split(photos, seed):
    """About 70% train, 20% validation, 10% test. The split is made once, inside each
    class, so every set has all three animals and no photo is in two sets."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in range(len(CLASSES)):
        group = [p for p in photos if p[1] == label]
        if len(group) < MIN_PER_CLASS:
            raise SystemExit(f"{CLASSES[label]} has {len(group)} photo(s); "
                             f"it needs at least {MIN_PER_CLASS}")
        rng.shuffle(group)
        n_test = max(1, int(len(group) * 0.1))
        n_val = max(1, int(len(group) * 0.2))
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    return train, val, test


def format_matrix(matrix):
    """Text table of a confusion matrix: rows are the true animal, columns the prediction."""
    lines = [" " * 12 + "".join(f"{name:>12}" for name in CLASSES)]
    for name, row in zip(CLASSES, matrix):
        lines.append(f"{name:<12}" + "".join(f"{int(count):>12}" for count in row))
    return "\n".join(lines)
