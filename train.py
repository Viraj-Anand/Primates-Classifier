"""Train the primate classifier with a fixed train / validation / test split.

Usage:  python train.py --data data --epochs 20

Expects three folders inside the data folder: chimpanzee, gorilla and orangutan,
with the photos inside them.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASSES = ["chimpanzee", "gorilla", "orangutan"]  # labels 0, 1, 2
IMG_SIZE = (256, 256)
BATCH = 30
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_photos(data_dir):
    """Return (path, label) for every photo. Files that are not images are skipped."""
    photos = []
    for label, name in enumerate(CLASSES):
        folder = Path(data_dir) / name
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise SystemExit(f"No photos found in {folder}")
        photos += [(str(p), label) for p in files]
    return photos


def split(photos, seed):
    """About 70% train, 20% validation, 10% test. The split is made once, inside each
    class, so every set has all three animals and no photo is in two sets."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in range(len(CLASSES)):
        group = [p for p in photos if p[1] == label]
        rng.shuffle(group)
        n_test = max(1, int(len(group) * 0.1))
        n_val = max(1, int(len(group) * 0.2))
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    return train, val, test


def load_image(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    return image, label


def make_dataset(photos, training):
    paths = [p for p, _ in photos]
    labels = [label for _, label in photos]
    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        data = data.shuffle(len(photos))  # only the training photos are shuffled
    return data.map(load_image).batch(BATCH)


def build_model():
    # Same layers as the first notebook, plus random flips, turns and zoom
    # that are only applied while training.
    return tf.keras.Sequential([
        tf.keras.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(10, (3, 3), activation="relu"),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Conv2D(20, (3, 3), activation="relu"),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(84, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="folder with one subfolder per animal")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="primate_model.keras", help="where to save the model")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    photos = list_photos(args.data)
    train, val, test = split(photos, args.seed)
    print(f"{len(photos)} photos: {len(train)} train, {len(val)} validation, {len(test)} test")

    train_ds = make_dataset(train, training=True)
    val_ds = make_dataset(val, training=False)
    test_ds = make_dataset(test, training=False)

    model = build_model()
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, shuffle=False)  # train_ds is already shuffled

    # The test photos are used once, here, after training is finished.
    _, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {accuracy:.3f} on {len(test)} photos the model never saw")

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = np.concatenate([model.predict(images, verbose=0).argmax(axis=1) for images, _ in test_ds])
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASSES)).numpy()
    print("Rows are the true animal, columns are the predicted animal:")
    print(" " * 12 + "".join(f"{name[:10]:>12}" for name in CLASSES))
    for name, row in zip(CLASSES, matrix):
        print(f"{name:<12}" + "".join(f"{count:>12}" for count in row))

    model.save(args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()
