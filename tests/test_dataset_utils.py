"""Tests for dataset_utils.py. These need only the standard library and pytest,
so they run even when TensorFlow / numpy are not installed."""
import subprocess
import sys
from pathlib import Path

import pytest

# Make `import dataset_utils` work with a plain `pytest` run from any folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataset_utils as du  # noqa: E402


def fake_photos(counts):
    """Build (path, label) pairs without touching the disk, e.g. counts=(10, 7, 3)."""
    return [(f"{du.CLASSES[label]}_{i}.jpg", label)
            for label, n in enumerate(counts) for i in range(n)]


def test_constants():
    assert du.CLASSES == ["chimpanzee", "gorilla", "orangutan"]
    assert du.MIN_PER_CLASS == 3
    assert du.IMG_SIZE == (256, 256)


@pytest.mark.parametrize("counts", [(10, 7, 3), (3, 3, 3), (50, 50, 50), (4, 9, 21)])
def test_split_counts_per_class(counts):
    photos = fake_photos(counts)
    train, val, test = du.split(photos, seed=5)
    for label, n in enumerate(counts):
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.2))
        assert sum(1 for _, l in test if l == label) == n_test
        assert sum(1 for _, l in val if l == label) == n_val
        assert sum(1 for _, l in train if l == label) == n - n_test - n_val


def test_split_sets_are_disjoint_and_cover_everything():
    photos = fake_photos((12, 8, 5))
    train, val, test = du.split(photos, seed=7)
    sets = [{p for p, _ in s} for s in (train, val, test)]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert len(train) + len(val) + len(test) == len(photos)
    assert sets[0] | sets[1] | sets[2] == {p for p, _ in photos}


def test_split_every_set_has_every_class():
    train, val, test = du.split(fake_photos((10, 10, 10)), seed=3)
    for s in (train, val, test):
        assert {label for _, label in s} == {0, 1, 2}


def test_split_is_reproducible_for_a_seed():
    photos = fake_photos((10, 10, 10))
    assert du.split(photos, seed=1) == du.split(photos, seed=1)
    assert du.split(photos, seed=1) != du.split(photos, seed=2)


def test_split_does_not_modify_its_input():
    photos = fake_photos((6, 6, 6))
    before = list(photos)
    du.split(photos, seed=0)
    assert photos == before


@pytest.mark.parametrize("bad_class, counts", [(0, (2, 5, 5)), (1, (5, 2, 5)), (2, (5, 5, 2))])
def test_split_rejects_class_with_too_few_photos(bad_class, counts):
    with pytest.raises(SystemExit) as excinfo:
        du.split(fake_photos(counts), seed=0)
    message = str(excinfo.value)
    assert du.CLASSES[bad_class] in message
    assert f"at least {du.MIN_PER_CLASS}" in message


def test_split_rejects_empty_class():
    with pytest.raises(SystemExit) as excinfo:
        du.split(fake_photos((5, 0, 5)), seed=0)
    assert "gorilla has 0 photo(s)" in str(excinfo.value)


def test_format_matrix_layout():
    text = du.format_matrix([[1, 2, 0], [0, 3, 0], [0, 0, 4]])
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[0].split() == du.CLASSES
    assert lines[1].split() == ["chimpanzee", "1", "2", "0"]
    assert lines[2].split() == ["gorilla", "0", "3", "0"]
    assert lines[3].split() == ["orangutan", "0", "0", "4"]


def test_format_matrix_accepts_float_counts():
    # Counts from a numpy array can arrive as floats; they are shown as whole numbers.
    lines = du.format_matrix([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 4.0]]).splitlines()
    assert lines[1].split() == ["chimpanzee", "2", "0", "1"]
    assert lines[3].split() == ["orangutan", "1", "0", "4"]


def test_dataset_utils_does_not_import_tensorflow_or_numpy():
    # Run in a fresh interpreter so modules loaded by other tests do not affect the result.
    code = ("import sys, dataset_utils; "
            "assert 'tensorflow' not in sys.modules and 'numpy' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
