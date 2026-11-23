# Progress report

## Latest pass: review fixes

Review findings and what was done:

| # | Problem | Fix |
|---|---------|-----|
| 1 | Tests failed to import (`numpy` missing) | Installed requirements in a Python 3.11 venv (see *Environment* below). |
| 2 | Bare `pytest` could not `import train` | `tests/test_train_utils.py` adds the project root to `sys.path`. |
| 3 | Overlap test did not check disjointness | Test now checks the three sets are pairwise disjoint and cover every photo. |
| 4 | Corrupt or renamed "photos" crashed training later; README said they were skipped | `list_photos` checks each file's content with TensorFlow and skips unreadable files, with a message. |
| 5 | A class with 1-2 photos gave an empty training set | `split` stops with a clear message if a class has fewer than `MIN_PER_CLASS` (3) photos. |
| 6 | Missing folder gave a confusing error | Clear "Folder not found" message. |
| 7 | `predict.py` duplicated preprocessing, and gave a raw TensorFlow error for bad input | Shares `prepare_image` from `train.py`; clear errors for a missing photo, missing model, or unreadable photo. |
| 8 | Test helper used deprecated `keras.preprocessing.image.save_img` | Images are written with `tf.io`. |
| 9 | `test_predict_main` monkeypatched `argparse` in confusing ways | `main(argv)` takes arguments directly; tests pass them in. |
| 10 | No test of the training script | Added an end-to-end test that trains for 1 epoch on tiny generated data and checks the confusion matrix agrees with the reported accuracy. |
| 11 | Model output could in theory change between calls | Test checks that augmentation is off at inference time. |
| 12 | Output of `model.fit` warned about `shuffle` on a dataset | Passed `shuffle=False`, since the training dataset is already shuffled. |
| 13 | `requirements.txt` had no minimum version; no dev requirements | Added `tensorflow>=2.16` (needed for `.keras` saving) and `requirements-dev.txt` with `pytest`. |
| 14 | `data/` and the saved model could be committed by accident | Added both to `.gitignore`. |
| 15 | Test-only path bug, and `str()` on a symbolic tensor in `tf.data.map` (introduced and caught while editing) | `prepare_image` only converts real `Path` objects. |

## Test results

- `pytest` (bare, from the project root): **19 passed**.
- Command-line check on generated stand-in images: `train.py` ran, skipped a renamed text file, saved the model; `predict.py` gave a prediction; missing-model error shown.
- The only warning is a NumPy `DeprecationWarning` raised inside Keras, caused by library versions.

## Environment

- The system Python is the Microsoft Store build, and pip failed to install TensorFlow there because Windows long paths are disabled (the TensorFlow headers exceed the 260-character limit).
- The tests were run in a virtualenv at `C:\pv` (short path). Nothing outside the project was changed except that venv.

## Not done yet

- The notebook `img_classification.ipynb` was not edited. It still has hard-coded paths in its last cells, as the README says.
- No real photos were available, so accuracy on real data is still unknown.
- Changes are not committed.
