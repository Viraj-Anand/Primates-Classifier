import numpy as np
import pytest

import predict

tf = pytest.importorskip("tensorflow")
import train  # noqa: E402


def test_original_architecture_parameter_count():
    assert train.build_model(3).count_params() == 9_238_239


def test_gap_architecture_is_tiny():
    assert train.build_model(3, "gap").count_params() == 15_039


def test_describe_threshold():
    probs = np.array([0.5, 0.3, 0.2])
    assert predict.describe(probs, ["a", "b", "c"], 2, 0.0)["prediction"] == "a"
    r = predict.describe(probs, ["a", "b", "c"], 2, 0.7)
    assert r["prediction"] == "unknown" and len(r["top"]) == 2
