import sys
from pathlib import Path

# Make `import train` / `import predict` work with a plain `pytest` run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

# These tests need the real ML stack (pip install -r requirements-dev.txt).
# If it is missing, the whole module is reported as SKIPPED with this reason,
# and tests/test_dataset_utils.py still runs. The importorskip calls must come
# before `import predict` / `import train`, because those import TensorFlow.
np = pytest.importorskip("numpy", reason="needs numpy: pip install -r requirements-dev.txt")
tf = pytest.importorskip("tensorflow", reason="needs TensorFlow: pip install -r requirements-dev.txt")

import predict  # noqa: E402
import train  # noqa: E402


def write_image(path, size=(10, 12, 3)):
    """Write a small random image with TensorFlow (no Keras preprocessing helpers)."""
    img = tf.constant(np.random.randint(0, 256, size, dtype=np.uint8))
    data = tf.io.encode_png(img) if path.suffix.lower() == ".png" else tf.io.encode_jpeg(img)
    tf.io.write_file(str(path), data)


def make_data_dir(root, per_class):
    for cls in train.CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(per_class):
            write_image(cls_dir / f"{cls}_{i}.jpg")
    return root


@pytest.fixture
def temp_data_dir(tmp_path):
    """A data folder with 5 dummy photos per class."""
    return make_data_dir(tmp_path / "data", per_class=5)


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Run the real training script once (1 epoch, tiny data) and share the saved model."""
    root = tmp_path_factory.mktemp("trained")
    data = make_data_dir(root / "data", per_class=6)
    out = root / "model.keras"
    accuracy, matrix = train.main(["--data", str(data), "--epochs", "1", "--out", str(out)])
    return out, accuracy, matrix


# ---------- finding photos ----------

def test_list_photos(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    assert len(photos) == 15
    for path, label in photos:
        assert Path(path).parent.name in train.CLASSES
        assert label == train.CLASSES.index(Path(path).parent.name)


def test_list_photos_no_files(tmp_path):
    for cls in train.CLASSES:
        (tmp_path / cls).mkdir()
    with pytest.raises(SystemExit) as excinfo:
        train.list_photos(tmp_path)
    assert "No photos found" in str(excinfo.value)


def test_list_photos_missing_folder(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        train.list_photos(tmp_path)
    assert "Folder not found" in str(excinfo.value)


def test_list_photos_skips_non_images(temp_data_dir):
    # A text file renamed to .jpg, and a file with a non-image extension.
    (temp_data_dir / "chimpanzee" / "fake.jpg").write_text("this is not a picture")
    (temp_data_dir / "gorilla" / "notes.txt").write_text("hello")
    photos = train.list_photos(temp_data_dir)
    assert len(photos) == 15
    assert not any(Path(p).name in {"fake.jpg", "notes.txt"} for p, _ in photos)


# ---------- splitting ----------

def test_split_distribution(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    train_set, val_set, test_set = train.split(photos, seed=123)
    for label, cls in enumerate(train.CLASSES):
        n = sum(1 for _, l in photos if l == label)
        assert sum(1 for _, l in test_set if l == label) == max(1, int(n * 0.1))
        assert sum(1 for _, l in val_set if l == label) == max(1, int(n * 0.2))
        assert sum(1 for _, l in train_set if l == label) == n - max(1, int(n * 0.1)) - max(1, int(n * 0.2))


def test_split_sets_are_disjoint_and_cover_everything(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    train_set, val_set, test_set = train.split(photos, seed=7)
    train_paths, val_paths, test_paths = ({p for p, _ in s} for s in (train_set, val_set, test_set))
    assert not (train_paths & val_paths)
    assert not (train_paths & test_paths)
    assert not (val_paths & test_paths)
    assert len(train_set) + len(val_set) + len(test_set) == len(photos)
    assert train_paths | val_paths | test_paths == {p for p, _ in photos}


def test_split_is_reproducible(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    assert train.split(photos, seed=1) == train.split(photos, seed=1)
    assert train.split(photos, seed=1) != train.split(photos, seed=2)


def test_split_rejects_class_with_too_few_photos(tmp_path):
    data = make_data_dir(tmp_path / "data", per_class=5)
    for p in (data / "orangutan").glob("*.jpg"):
        p.unlink()
    for i in range(train.MIN_PER_CLASS - 1):
        write_image(data / "orangutan" / f"o_{i}.jpg")
    with pytest.raises(SystemExit) as excinfo:
        train.split(train.list_photos(data), seed=0)
    assert "orangutan" in str(excinfo.value)


# ---------- images and datasets ----------

def test_load_image(temp_data_dir):
    path, label = train.list_photos(temp_data_dir)[0]
    image, lbl = train.load_image(path, label)
    assert image.shape == train.IMG_SIZE + (3,)
    assert image.dtype == tf.float32
    assert lbl == label
    assert float(tf.reduce_min(image)) >= 0.0
    assert float(tf.reduce_max(image)) <= 1.0


def test_load_png(tmp_path):
    path = tmp_path / "photo.png"
    write_image(path)
    image = train.prepare_image(path)
    assert image.shape == train.IMG_SIZE + (3,)


def test_make_dataset_batches_every_photo(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    ds = train.make_dataset(photos, training=True)
    total = 0
    for images, labels in ds:
        assert images.shape[1:] == train.IMG_SIZE + (3,)
        assert labels.shape[0] == images.shape[0] <= train.BATCH
        total += labels.shape[0]
    assert total == len(photos)


# ---------- model ----------

def test_build_model_output_shape():
    model = train.build_model()
    out = model(tf.zeros((1, *train.IMG_SIZE, 3), dtype=tf.float32))
    assert out.shape == (1, len(train.CLASSES))
    assert np.allclose(tf.reduce_sum(out, axis=-1).numpy(), 1.0, atol=1e-5)


def test_augmentation_only_while_training():
    # Random flips/turns/zoom must not change predictions at inference time.
    model = train.build_model()
    x = tf.random.uniform((2, *train.IMG_SIZE, 3))
    a = model(x, training=False).numpy()
    b = model(x, training=False).numpy()
    assert np.allclose(a, b)


def test_format_matrix():
    text = train.format_matrix(np.array([[1, 2, 0], [0, 3, 0], [0, 0, 4]]))
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[1].split() == ["chimpanzee", "1", "2", "0"]
    assert lines[2].split() == ["gorilla", "0", "3", "0"]
    assert lines[3].split() == ["orangutan", "0", "0", "4"]


# ---------- end to end ----------

def test_train_main_end_to_end(trained_model):
    out, accuracy, matrix = trained_model
    assert out.is_file()
    assert 0.0 <= accuracy <= 1.0
    # Test set: 1 photo per class (see split()), so 3 photos in total.
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 3
    # The confusion matrix and the reported accuracy must agree.
    assert abs(np.trace(matrix) / matrix.sum() - accuracy) < 1e-4


def test_train_main_missing_data_folder(tmp_path):
    with pytest.raises(SystemExit):
        train.main(["--data", str(tmp_path / "nope"), "--epochs", "1",
                    "--out", str(tmp_path / "m.keras")])


def test_predict_main(trained_model, tmp_path, capsys):
    model_path, _, _ = trained_model
    photo = tmp_path / "test.jpg"
    write_image(photo)

    label, probabilities = predict.main([str(photo), "--model", str(model_path)])

    assert label in train.CLASSES
    assert abs(float(np.sum(probabilities)) - 1.0) < 1e-4
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(label)
    for name in train.CLASSES:
        assert any(line.strip().startswith(f"{name}:") for line in out[1:])


def test_predict_rejects_unreadable_photo(trained_model, tmp_path):
    model_path, _, _ = trained_model
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(SystemExit) as excinfo:
        predict.main([str(bad), "--model", str(model_path)])
    assert "Could not read" in str(excinfo.value)


def test_predict_rejects_missing_model(tmp_path):
    photo = tmp_path / "p.jpg"
    write_image(photo)
    with pytest.raises(SystemExit) as excinfo:
        predict.main([str(photo), "--model", str(tmp_path / "missing.keras")])
    assert "Model not found" in str(excinfo.value)
