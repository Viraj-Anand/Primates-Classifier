# Testing

```
pip install -r requirements-dev.txt
pytest -q
```

10 tests (last run: 10 passed in 5.0 s, Python 3.11, TensorFlow 2.21).

- `tests/test_dataset_utils.py` (7): classes are alphabetical, non-images are skipped by content, a 4th folder is picked up, the seeded split is disjoint, near-duplicates stay in one split, `--check-data` fails on too few photos, loaded arrays have the right shape and range.
- `tests/test_model_and_predict.py` (3): the original network has 9,238,239 parameters, the `gap` network has 15,039, and the `--min-confidence` threshold turns a weak prediction into `unknown`.

The model tests are skipped when TensorFlow is not installed. **Green-but-skipped is not green**: check the summary line says `10 passed`, not `7 passed, 3 skipped`.

These tests check that the code works. None of them says anything about accuracy on real primate photos.

CI runs the same command on Python 3.11 and 3.12 ([workflow](.github/workflows/tests.yml)).
