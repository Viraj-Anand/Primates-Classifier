"""Say which primate is in one photo.

Usage:  python predict.py photo.jpg --model primate_model.keras
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import CLASSES, prepare_image


def predict_probabilities(model, photo):
    """Return the model's probability for each class, in CLASSES order."""
    try:
        image = prepare_image(photo)
    except tf.errors.OpError:
        raise SystemExit(f"Could not read {photo} as an image")
    return model.predict(image[tf.newaxis], verbose=0)[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--model", default="primate_model.keras")
    args = parser.parse_args(argv)

    if not Path(args.photo).is_file():
        raise SystemExit(f"Photo not found: {args.photo}")
    if not Path(args.model).is_file():
        raise SystemExit(f"Model not found: {args.model} (train one with train.py first)")

    model = tf.keras.models.load_model(args.model)
    probabilities = predict_probabilities(model, args.photo)

    best = int(np.argmax(probabilities))
    print(f"{CLASSES[best]} ({probabilities[best]:.0%})")
    for name, p in zip(CLASSES, probabilities):
        print(f"  {name}: {p:.1%}")
    return CLASSES[best], probabilities


if __name__ == "__main__":
    main()
