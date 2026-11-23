import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import pytest

from pathlib import Path

# Import functions from train.py
import train

# Helper to create dummy image files

def create_dummy_image(path, size=(10, 10, 3)):
    img = np.random.randint(0, 256, size, dtype=np.uint8)
    tf.keras.preprocessing.image.save_img(str(path), img)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with subfolders for each class and dummy images."""
    for cls in train.CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        # create 5 dummy images per class
        for i in range(5):
            create_dummy_image(cls_dir / f"{cls}_{i}.jpg")
    return tmp_path


def test_list_photos(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    # Expect 3 classes * 5 images = 15 entries
    assert len(photos) == 15
    # Check labels correspond to class index
    for path, label in photos:
        assert Path(path).parent.name in train.CLASSES
        assert label == train.CLASSES.index(Path(path).parent.name)


def test_list_photos_no_files(tmp_path):
    # Create empty class directories
    for cls in train.CLASSES:
        (tmp_path / cls).mkdir()
    with pytest.raises(SystemExit) as excinfo:
        train.list_photos(tmp_path)
    assert "No photos found" in str(excinfo.value)


def test_split_distribution(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    train_set, val_set, test_set = train.split(photos, seed=123)
    # Each class should have at least 1 in test, 1 in val
    for cls in train.CLASSES:
        cls_photos = [p for p, l in photos if l == train.CLASSES.index(cls)]
        n = len(cls_photos)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.2))
        # Count in splits
        test_count = sum(1 for p, l in test_set if l == train.CLASSES.index(cls))
        val_count = sum(1 for p, l in val_set if l == train.CLASSES.index(cls))
        train_count = sum(1 for p, l in train_set if l == train.CLASSES.index(cls))
        assert test_count == n_test
        assert val_count == n_val
        assert train_count == n - n_test - n_val
    # Ensure no overlap
    all_paths = set(p for p, _ in photos)
    assert len(set(p for p, _ in train_set + val_set + test_set)) == len(all_paths)


def test_load_image(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    path, label = photos[0]
    image, lbl = train.load_image(path, label)
    # Image should be a tensor of shape (256,256,3) and dtype float32
    assert image.shape == train.IMG_SIZE + (3,)
    assert image.dtype == tf.float32
    assert lbl == label


def test_make_dataset_and_iterate(temp_data_dir):
    photos = train.list_photos(temp_data_dir)
    ds = train.make_dataset(photos, training=True)
    # Should be batched
    for batch in ds.take(1):
        images, labels = batch
        # images shape: (batch_size, 256,256,3)
        assert images.shape[1:] == train.IMG_SIZE + (3,)
        assert labels.shape[0] == images.shape[0]


def test_build_model_output_shape():
    model = train.build_model()
    # Build dummy input
    dummy = tf.zeros((1, *train.IMG_SIZE, 3), dtype=tf.float32)
    out = model(dummy)
    assert out.shape == (1, len(train.CLASSES))
    # Softmax output sums to 1
    assert np.allclose(tf.reduce_sum(out, axis=-1).numpy(), 1.0)


def test_predict_main(tmp_path, monkeypatch, capsys):
    # Create a dummy model and save
    model = train.build_model()
    dummy_input = tf.zeros((1, *train.IMG_SIZE, 3), dtype=tf.float32)
    model(dummy_input)  # build
    model_path = tmp_path / "dummy_model.keras"
    model.save(str(model_path))

    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    create_dummy_image(img_path)

    # Monkeypatch argparse to provide arguments
    import argparse
    def fake_parse_args():
        class Args:
            photo = str(img_path)
            model = str(model_path)
        return Args()
    monkeypatch.setattr(train, "argparse", argparse)
    monkeypatch.setattr(train, "argparse", argparse)
    # Instead of calling main, we will import predict and run its main
    import predict
    monkeypatch.setattr(predict, "argparse", argparse)
    # Patch argparse.ArgumentParser.parse_args to return our fake args
    monkeypatch.setattr(predict.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args())
    # Run main
    predict.main()
    captured = capsys.readouterr()
    # Should contain class names and probabilities
    for cls in train.CLASSES:
        assert cls in captured.out
    # Ensure a probability line exists
    assert any(f"{cls}:" in captured.out for cls in train.CLASSES)
