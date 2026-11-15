import numpy as np
import pytest

from zk2.models.calibration import OneVsRestCalibrator


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_returns_normalized_probabilities(method):
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.05, 0.9, size=(120, 3))
    raw /= raw.sum(axis=1, keepdims=True)
    y = np.asarray(["H", "D", "A"] * 40)
    calibrator = OneVsRestCalibrator(method).fit(raw, y, ("H", "D", "A"))
    result = calibrator.predict_proba(raw[:20])
    assert result.shape == (20, 3)
    assert np.isfinite(result).all()
    assert np.all(result >= 0)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_calibrator_requires_fit_before_prediction():
    with pytest.raises(RuntimeError):
        OneVsRestCalibrator("platt").predict_proba(np.asarray([[0.5, 0.5]]))


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        OneVsRestCalibrator("temperature")


def test_calibrator_rejects_missing_class_in_fit_data():
    raw = np.asarray([[0.7, 0.3], [0.6, 0.4]] * 6)
    y = np.asarray([0, 1] * 6)
    with pytest.raises(ValueError):
        OneVsRestCalibrator("platt").fit(raw, y, (0, 1, 2))
