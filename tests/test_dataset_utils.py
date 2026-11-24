import numpy as np
import pytest
from PIL import Image

import dataset_utils as du


@pytest.fixture
def folder(tmp_path):
    du.make_demo_dataset(tmp_path, per_class=10, size=64)
    return tmp_path


def test_classes_are_alphabetical(folder):
    assert du.discover_classes(folder) == ["chimpanzee", "gorilla", "orangutan"]


def test_scan_skips_bad_files_by_content(folder):
    (folder / "gorilla" / "fake.jpg").write_text("not an image")
    (folder / "gorilla" / "notes.txt").write_text("hi")
    (folder / "gorilla" / "phone.heic").write_bytes(b"x")
    _, photos, skipped = du.scan(folder)
    assert len(photos) == 30
    assert len(skipped) == 3


def test_fourth_class_is_picked_up(folder):
    Image.new("RGB", (8, 8)).save(folder / "extra.png")
    (folder / "bonobo").mkdir()
    Image.new("RGB", (8, 8)).save(folder / "bonobo" / "a.png")
    assert du.discover_classes(folder)[0] == "bonobo"


def test_split_is_disjoint_seeded_and_per_class(folder):
    _, photos, _ = du.scan(folder)
    a, b = du.split(photos, 42)[0], du.split(photos, 42)[0]
    assert a == b
    tr, va, te = du.split(photos, 42)
    assert not ({p for p, _ in tr} & {p for p, _ in te})
    assert not ({p for p, _ in va} & {p for p, _ in te})
    assert len(tr) + len(va) + len(te) == 30
    assert {l for _, l in te} == {0, 1, 2}


def test_duplicates_stay_in_one_split(tmp_path):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
    rng = np.random.default_rng(0)
    base = du.synthetic_image((100, 100, 100), 64, rng)
    for i in range(10):
        (base if i < 4 else du.synthetic_image((i * 20, 255 - i * 20, 90), 64, rng)).save(tmp_path / "a" / f"{i}.png")
        du.synthetic_image((10 * i, 30, 200 - i * 10), 64, rng).save(tmp_path / "b" / f"{i}.png")
    _, photos, _ = du.scan(tmp_path)
    groups = du.duplicate_clusters([p for p, _ in photos])
    assert any(len(g) >= 4 for g in groups)
    sets = du.split(photos, 1, groups)
    for g in groups:
        assert sum(any(p in {q for q, _ in s} for p in g) for s in sets) == 1


def test_check_data_flags_too_few(tmp_path, capsys):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
        Image.new("RGB", (8, 8)).save(tmp_path / c / "1.png")
    assert du.check_data(tmp_path) is False
    assert "NOT ready" in capsys.readouterr().out


def test_load_array_range_and_shape(folder):
    p = next((folder / "gorilla").iterdir())
    a = du.load_array(p)
    assert a.shape == (256, 256, 3) and 0 <= a.min() and a.max() <= 1
