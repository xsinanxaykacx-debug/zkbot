import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss

from zk2.models.metrics import evaluate_probabilities


def test_1x2_metrics_match_sklearn_without_class_order_warning():
    y_true = np.asarray(["H", "D", "A", "H"])
    probabilities = np.asarray([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
        [0.5, 0.3, 0.2],
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_probabilities(
            y_true, probabilities, class_order=("H", "D", "A")
        )

    expected = log_loss(
        np.asarray([0, 1, 2, 0]), probabilities, labels=[0, 1, 2]
    )
    assert result["log_loss"] == pytest.approx(expected)
    assert result["n"] == 4
    assert result["brier"] >= 0


def test_binary_metrics_match_sklearn_for_positive_class():
    y_true = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.9, 0.1],
    ])

    result = evaluate_probabilities(y_true, probabilities, class_order=(0, 1))

    assert result["log_loss"] == pytest.approx(log_loss(y_true, probabilities, labels=[0, 1]))
    assert result["brier"] == pytest.approx(np.mean((probabilities[:, 1] - y_true) ** 2))


def test_rejects_duplicate_or_too_short_class_order():
    with pytest.raises(ValueError, match="at least two unique"):
        evaluate_probabilities([0], np.asarray([[0.5, 0.5]]), class_order=(0, 0))
    with pytest.raises(ValueError, match="at least two unique"):
        evaluate_probabilities([0], np.asarray([[1.0]]), class_order=(0,))


def test_rejects_wrong_probability_shape_and_empty_targets():
    with pytest.raises(ValueError, match="shape"):
        evaluate_probabilities([0], np.asarray([0.5, 0.5]), class_order=(0, 1))
    with pytest.raises(ValueError, match="At least one"):
        evaluate_probabilities([], np.empty((0, 2)), class_order=(0, 1))


def test_rejects_nonfinite_and_out_of_range_probabilities():
    with pytest.raises(ValueError, match="finite"):
        evaluate_probabilities([0], np.asarray([[np.nan, 0.0]]), class_order=(0, 1))
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_probabilities([0], np.asarray([[1.1, -0.1]]), class_order=(0, 1))


def test_rejects_unknown_target_labels():
    with pytest.raises(ValueError, match="outside class_order"):
        evaluate_probabilities(["X"], np.asarray([[0.5, 0.5]]), class_order=("H", "A"))
