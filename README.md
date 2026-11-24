# Primates-Classifier

A small CNN that tells chimpanzees, gorillas and orangutans apart from photos (TensorFlow/Keras), plus the tooling to check your data, train honestly and see what the model looks at.

[![tests](https://github.com/Viraj-Anand/Primates-Classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/Viraj-Anand/Primates-Classifier/actions/workflows/tests.yml)

**Status:** a learning project. The code is tested; the model has **not** been evaluated on real photos (see below).

## Can I trust it?

- **The notebook's 100% accuracy is not trustworthy.** Its split reshuffles on every pass, so training and validation photos mix (details in [Limitations](#limitations)).
- **`train.py` is the honest pipeline**: one fixed seeded split, near-duplicates kept together, test photos used once at the end.
- **It has never been run on real photos.** The original 291 photos are gone. Every number below that comes from a run is from *synthetic* coloured images, and says so. The results table stays empty until someone (me, or you) runs it on real data.

## Quick start

```
pip install -r requirements.txt
python predict.py --demo
```

`--demo` trains for one epoch on generated stand-in images (coloured noise, **not primates**), then predicts one. It proves the code runs; it says nothing about accuracy. Takes about 12 s on my CPU.

### Use your own photos

Put photos in one folder per class, check them, train, predict:

```
python train.py --data data --check-data
python train.py --data data --epochs 20 --save-report reports/run1
python predict.py some_photo.jpg --model primate_model.keras
```

No photos? `python scripts/build_dataset.py --per-class 100` downloads Creative Commons–licensed, research-grade photos from iNaturalist into `data/` and writes credits to `data/LICENSES.csv`. It takes one photo per observation so the split cannot leak. (Tested live with 3 photos per class; I have not trained on a full download.)

### What you should see

Real output from a synthetic `--demo` run (1 epoch; the numbers mean nothing about real accuracy):

```
Classes: ['chimpanzee', 'gorilla', 'orangutan']
300 photos: 210 train, 60 validation, 30 test (0 near-duplicate groups kept together)
architecture=flatten parameters=9,238,239
Test accuracy: 0.667 on 30 photos the model never saw
Rows are the true class, columns are the predicted class:
                chimpanzee     gorilla   orangutan
chimpanzee               0           0          10
gorilla                  0          10           0
orangutan                0           0          10
```

and `--check-data` on the same folder:

```
Classes (3, alphabetical = label order):
  chimpanzee      100 usable photos
  gorilla         100 usable photos
  orangutan       100 usable photos
Total usable photos: 300
Exact duplicate groups: 0
Near-duplicate groups (hash distance <= 5): 0
Result: OK to train.
```

`predict.py` on a synthetic image: `gorilla (88%)`. With `--min-confidence 0.99 --top-k 2` it prints `unknown (best guess gorilla - 88%, below threshold)`. `--json` prints `{"prediction": "gorilla", "top": [...]}`. `--explain` saves a Grad-CAM heat-map next to the photo.

## Flags

`train.py`

| Flag | Meaning |
|---|---|
| `--data DIR` | folder with one sub-folder per class (default `data`) |
| `--epochs N` | default 20 |
| `--seed N` | drives split and training (default 42); the run header prints seed, versions and git commit |
| `--model-out FILE` | default `primate_model.keras`; also writes `<model>.classes.json` |
| `--arch` | `flatten` (original), `gap` (15,039 params), `transfer` (frozen MobileNetV2) |
| `--check-data` | validate the folder and exit |
| `--save-report DIR` | write `metrics.json`, `confusion_matrix.png`, `train_log.csv` |
| `--export-tflite` | also write `<model>.tflite` |
| `--demo` | 1 epoch on generated stand-in images |

`predict.py`

| Flag | Meaning |
|---|---|
| `--model FILE` | default `primate_model.keras` |
| `--top-k N` | show the N most likely classes |
| `--min-confidence P` | below P, answer `unknown` (e.g. `0.70`) |
| `--json` | machine-readable output |
| `--explain` | save a Grad-CAM overlay |
| `--source 0` | live webcam (needs `opencv-python`; **written but untested**) |
| `--demo` | train on stand-in images, then predict one |

## Data format

- Folder name = class name. Classes are discovered **alphabetically** and that order is the label order (saved in the `.classes.json` sidecar, so `predict.py` always agrees with training).
- Formats: `.jpg .jpeg .png .bmp .webp`. HEIC is not supported. Files are checked by content, so a renamed text file is skipped.
- At least 3 photos per class (hard minimum); under 30 triggers a warning. Add a 4th folder and retrain: it is picked up automatically.

**How much data do I need?** The original run used 291 photos (about 100 per class) and that is a floor, not a target. With ~100 per class a 10% test split is ~10 photos per class, so one wrong answer moves accuracy by several points. Aim for a few hundred per class, and treat any result on fewer as noisy.

## Results

| Architecture | Photos | Test accuracy (mean of 3 seeds) |
|---|---|---|
| flatten | – | *pending: no real-photo run yet* |
| gap | – | *pending* |
| transfer | – | *pending* |

Filled in only from real runs.

## Photos that aren't a primate

A softmax always picks one of the three classes, so a photo of a cat is still called "gorilla" with some confidence. `--min-confidence 0.70` turns low-confidence answers into `unknown`. This reduces the problem; it does not solve it, and a confident wrong answer on a non-primate is still possible.

## The model, and why `--arch gap` exists

Conv2D(10) → AvgPool → Conv2D(20) → AvgPool → Flatten → Dense(120) → Dense(84) → Dense(3, softmax); 256 × 256 input scaled to 0–1; 9,238,239 parameters.

**99.9% of those parameters sit in one place**: the Flatten output (76,880 values) feeding Dense(120). That is a lot of capacity to memorise 291 photos. `--arch gap` replaces Flatten with GlobalAveragePooling2D and drops the model to **15,039 parameters**. `--arch transfer` uses a frozen ImageNet MobileNetV2 (downloads weights on first use). I ran both end-to-end on synthetic data only.

## Limitations

- Small data, and the original model was trained from scratch.
- The notebook takes batches from a dataset that is reshuffled on every pass, so training and validation photos can mix between epochs. Its 100% validation accuracy is probably far too optimistic, and it never computed a test accuracy.
- The original 291 photos are gone, so the notebook's results cannot be reproduced.
- Photos from one animal or one session can still look alike; near-duplicate detection catches copies and crops, not "same gorilla, different day".
- Scope: a hobby classifier for three genera. Don't use it for anything that matters (conservation, species ID, law enforcement).

## FAQ

**Why only three classes?** That is what the first dataset had. Add a folder to add a class.
**Can I use it on a phone photo?** If it is jpg/png/webp, yes. Convert HEIC first.
**Why is the demo accuracy 0.667?** One epoch on coloured noise: it learned two of three colours. Ignore it.
**Does it run on GPU?** TensorFlow uses one if it is available; I only tested on CPU.

## Testing

See [TESTING.md](TESTING.md). 10 tests; CI runs them on Python 3.11 and 3.12.

## Tested with

Python 3.11, TensorFlow 2.21, numpy 2.4, Pillow 13, Windows 11, CPU only. `requirements.txt` allows TensorFlow 2.16–2.21; CI covers Python 3.11 and 3.12 on Ubuntu (the first CI run is the first time I will know that holds).

## The notebook

[notebooks/01_first_experiment.ipynb](notebooks/01_first_experiment.ipynb) is the original experiment, kept for history. It needs Python ≤ 3.12 (`imghdr`), and its last cells load files from my old machine, so skip them.

## Repo map

| Path | What |
|---|---|
| `train.py`, `predict.py` | training and prediction CLIs |
| `dataset_utils.py` | scanning, validation, splitting, demo data (no TensorFlow needed) |
| `scripts/build_dataset.py` | iNaturalist downloader |
| `tests/` | pytest suite |
| `notebooks/` | the first experiment |

## Licence

Code: MIT, see [LICENSE](LICENSE). The licence covers the code only. No photos are included or redistributed; photos from `build_dataset.py` stay under their own Creative Commons licences (listed in `data/LICENSES.csv`).
