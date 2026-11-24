"""Train the primate classifier with a fixed train / validation / test split.

    python train.py --demo                  # 1 epoch on generated stand-in images (not real primates)
    python train.py --data data --check-data
    python train.py --data data --epochs 20 --save-report reports/run1

Every sub-folder of --data is a class (folder name = class name, alphabetical order).
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import dataset_utils as du

BATCH = 30


def build_model(num_classes, arch="flatten", img_size=du.IMG_SIZE):
    """arch='flatten' is the original network (about 9.2M parameters, 99.9% of them in the
    first Dense layer). arch='gap' swaps Flatten for GlobalAveragePooling2D (15,039 parameters).
    arch='transfer' = frozen ImageNet MobileNetV2 + a small trained head (needs a download)."""
    import tensorflow as tf
    L = tf.keras.layers
    if arch == "transfer":
        base = tf.keras.applications.MobileNetV2(input_shape=(*img_size, 3), include_top=False,
                                                 weights="imagenet", name="last_conv")
        base.trainable = False
        return tf.keras.Sequential([
            tf.keras.Input(shape=(*img_size, 3)),
            L.RandomFlip("horizontal"), L.RandomRotation(0.1), L.RandomZoom(0.1),
            L.Rescaling(2.0, offset=-1.0),  # MobileNetV2 expects -1..1, our data is 0..1
            base, L.GlobalAveragePooling2D(), L.Dropout(0.2), L.Dense(num_classes, activation="softmax"),
        ])
    head = [L.GlobalAveragePooling2D()] if arch == "gap" else [L.Flatten()]
    return tf.keras.Sequential([
        tf.keras.Input(shape=(*img_size, 3)),
        L.RandomFlip("horizontal"), L.RandomRotation(0.1), L.RandomZoom(0.1),  # training only
        L.Conv2D(10, (3, 3), activation="relu"), L.AveragePooling2D((2, 2)),
        L.Conv2D(20, (3, 3), activation="relu", name="last_conv"), L.AveragePooling2D((2, 2)),
        *head,
        L.Dense(120, activation="relu"), L.Dense(84, activation="relu"),
        L.Dense(num_classes, activation="softmax"),
    ])


def make_dataset(photos, training):
    import tensorflow as tf
    x = np.stack([du.load_array(p, scale=False) for p, _ in photos])
    y = np.array([l for _, l in photos])
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        data = data.shuffle(len(photos))  # only the training photos are shuffled
    return data.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b)).batch(BATCH)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def save_report(folder, metrics, matrix, classes):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="folder with one sub-folder per class")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="drives the split and the training")
    ap.add_argument("--model-out", default="primate_model.keras")
    ap.add_argument("--arch", choices=["flatten", "gap", "transfer"], default="flatten", help="gap = far fewer parameters; transfer = pretrained MobileNetV2")
    ap.add_argument("--check-data", action="store_true", help="validate the data folder and exit")
    ap.add_argument("--save-report", metavar="DIR", help="write metrics.json, confusion_matrix.png, train_log.csv")
    ap.add_argument("--export-tflite", action="store_true", help="also write <model>.tflite")
    ap.add_argument("--demo", action="store_true", help="train 1 epoch on generated stand-in images")
    args = ap.parse_args()

    if args.demo:
        args.data = tempfile.mkdtemp(prefix="primates_demo_")
        du.make_demo_dataset(args.data, per_class=100)
        args.epochs = 1
        print(f"DEMO: synthetic coloured-noise images in {args.data}. These are NOT primates;")
        print("      the numbers below only show that the code runs.\n")
    if args.check_data:
        sys.exit(0 if du.check_data(args.data) else 1)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    print(f"seed={args.seed} python={sys.version.split()[0]} tensorflow={tf.__version__} "
          f"keras={tf.keras.__version__} numpy={np.__version__} commit={git_commit()}")

    classes, photos, skipped = du.scan(args.data)
    for path, why in skipped:
        print(f"skipped {path}: {why}")
    counts = [sum(1 for _, l in photos if l == i) for i in range(len(classes))]
    if len(classes) < 2 or min(counts) < du.MIN_PER_CLASS:
        raise SystemExit(f"Need at least 2 classes with {du.MIN_PER_CLASS}+ photos each. "
                         f"Found {dict(zip(classes, counts))}. Run with --check-data.")
    if min(counts) < du.WARN_PER_CLASS:
        print(f"warning: a class has under {du.WARN_PER_CLASS} photos, so the test accuracy will be noisy")
    groups = du.duplicate_clusters([p for p, _ in photos])
    train, val, test = du.split(photos, args.seed, groups)
    print(f"classes: {classes}")
    print(f"{len(photos)} photos: {len(train)} train, {len(val)} validation, {len(test)} test "
          f"({len(groups)} near-duplicate groups kept together)")

    train_ds, val_ds, test_ds = (make_dataset(s, t) for s, t in ((train, True), (val, False), (test, False)))
    model = build_model(len(classes), args.arch)
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"architecture={args.arch} parameters={model.count_params():,}")
    callbacks = []
    if args.save_report:
        Path(args.save_report).mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(str(Path(args.save_report) / "train_log.csv")))
    t0 = time.time()
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, shuffle=False, callbacks=callbacks)
    seconds = time.time() - t0

    # The test photos are used once, here, after training is finished.
    _, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {accuracy:.3f} on {len(test)} photos the model never saw")
    y_true = np.concatenate([b.numpy() for _, b in test_ds])
    y_pred = np.concatenate([model.predict(a, verbose=0).argmax(axis=1) for a, _ in test_ds])
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(classes)).numpy()
    print("Rows are the true class, columns are the predicted class:")
    print(" " * 14 + "".join(f"{c[:11]:>12}" for c in classes))
    for c, row in zip(classes, matrix):
        print(f"{c[:13]:<14}" + "".join(f"{n:>12}" for n in row))

    model.save(args.model_out)
    sidecar = Path(args.model_out).with_suffix(".classes.json")
    sidecar.write_text(json.dumps({"classes": classes, "img_size": list(du.IMG_SIZE),
                                   "normalize": "divide by 255"}, indent=2))
    print(f"\nModel saved to {args.model_out} (class names in {sidecar})")

    if args.export_tflite:
        out = Path(args.model_out).with_suffix(".tflite")
        out.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
        print(f"TFLite model saved to {out}")
    if args.save_report:
        save_report(args.save_report, {
            "test_accuracy": float(accuracy), "test_photos": len(test), "seed": args.seed,
            "epochs": args.epochs, "arch": args.arch, "parameters": int(model.count_params()),
            "classes": classes, "train_seconds": round(seconds, 1), "tensorflow": tf.__version__,
            "commit": git_commit(), "synthetic_demo": args.demo,
            "confusion_matrix": matrix.tolist()}, matrix.tolist(), classes)
        print(f"Report written to {args.save_report}")


if __name__ == "__main__":
    main()
