"""Say which class is in a photo.

    python predict.py photo.jpg
    python predict.py photo.jpg --top-k 2 --min-confidence 0.70 --json
    python predict.py photo.jpg --explain      # saves a Grad-CAM heat-map next to the photo
    python predict.py --source 0               # webcam, press q to quit
    python predict.py --demo                   # train a demo model, predict on a generated image
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import dataset_utils as du


def load_model(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    side = Path(path).with_suffix(".classes.json")
    if not side.exists():
        raise SystemExit(f"{side} not found: train.py writes it next to the model.")
    meta = json.loads(side.read_text())
    return model, meta["classes"], tuple(meta["img_size"])


def describe(probs, classes, top_k, min_conf):
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    unknown = probs[best] < min_conf
    label = (f"unknown (best guess {classes[best]} - {probs[best]:.0%}, below threshold)" if unknown
             else f"{classes[best]} ({probs[best]:.0%})")
    top = [{"class": classes[i], "probability": round(float(probs[i]), 4)} for i in order[:top_k]]
    return {"prediction": "unknown" if unknown else classes[best], "label": label, "top": top}


def grad_cam(model, arr):
    """Heat-map (0-1) of where the last conv layer looked for the predicted class."""
    import tensorflow as tf
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(arr[None])
        for layer in model.layers:  # manual forward pass: a loaded Sequential has no functional graph
            x = layer(x, training=False)
            if layer.name == "last_conv":
                maps = x
        score = tf.reduce_max(x[0])
    grads = tape.gradient(score, maps)
    cam = tf.nn.relu(tf.reduce_sum(maps[0] * tf.reduce_mean(grads, axis=(0, 1, 2)), axis=-1)).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def save_overlay(arr, cam, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(arr.shape[1::-1], Image.BILINEAR)) / 255
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr), ax.imshow(heat, cmap="jet", alpha=0.4), ax.axis("off")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def webcam(model, classes, size, min_conf, index):
    import cv2  # opencv-python, only needed for this mode
    cap = cv2.VideoCapture(int(index))
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (size[1], size[0])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        text = describe(model.predict(rgb[None], verbose=0)[0], classes, 1, min_conf)["label"]
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("predict (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--model", default="primate_model.keras")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="below this the answer is 'unknown' (try 0.70 for photos that may not be a primate)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--explain", action="store_true", help="save a Grad-CAM heat-map")
    ap.add_argument("--source", help="webcam index, e.g. 0")
    ap.add_argument("--demo", action="store_true", help="train a 1-epoch demo model, predict on a generated image")
    args = ap.parse_args()

    if args.demo:
        tmp = Path(tempfile.mkdtemp(prefix="primates_demo_"))
        args.model = str(tmp / "demo_model.keras")
        subprocess.check_call([sys.executable, str(Path(__file__).parent / "train.py"),
                               "--demo", "--model-out", args.model])
        args.photo = str(tmp / "stand_in.jpg")
        du.synthetic_image(du.DEMO_CLASSES["gorilla"], 256, np.random.default_rng(1)).save(args.photo)
        print("\nDEMO: predicting on a generated image that is NOT a primate.")
    model, classes, size = load_model(args.model)
    if args.source is not None:
        return webcam(model, classes, size, args.min_confidence, args.source)
    if not args.photo:
        ap.error("give a photo, --source, or --demo")
    arr = du.load_array(args.photo, size)
    result = describe(model.predict(arr[None], verbose=0)[0], classes, args.top_k, args.min_confidence)
    if args.json:
        print(json.dumps({k: result[k] for k in ("prediction", "top")}))
    else:
        print(result["label"])
        for t in result["top"]:
            print(f"  {t['class']}: {t['probability']:.1%}")
    if args.explain:
        out = Path(args.photo).with_suffix(".gradcam.png")
        save_overlay(arr, grad_cam(model, arr), out)
        print(f"Grad-CAM saved to {out}")


if __name__ == "__main__":
    main()
