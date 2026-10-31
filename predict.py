"""Say which primate is in one photo.

Usage:  python predict.py photo.jpg --model primate_model.keras
"""
import argparse

import numpy as np
import tensorflow as tf

from train import CLASSES, IMG_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--model", default="primate_model.keras")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    image = tf.io.decode_image(tf.io.read_file(args.photo), channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    probabilities = model.predict(image[tf.newaxis], verbose=0)[0]

    best = int(np.argmax(probabilities))
    print(f"{CLASSES[best]} ({probabilities[best]:.0%})")
    for name, p in zip(CLASSES, probabilities):
        print(f"  {name}: {p:.1%}")


if __name__ == "__main__":
    main()
