import numpy as np
import pytest

from zk2.models.market import implied_probabilities


def test_implied_probabilities_remove_proportional_overround():
    odds = np.asarray([[2.0, 3.0, 4.0], [1.8, 3.2, 4.5]])
    probabilities = implied_probabilities(odds)
    assert probabilities.shape == odds.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities > 0)


@pytest.mark.parametrize("odds", [
    np.asarray([2.0, 3.0]),
    np.asarray([[2.0]]),
    np.asarray([[2.0, 1.0]]),
    np.asarray([[2.0, np.nan]]),
    np.asarray([[2.0, np.inf]]),
])
def test_implied_probabilities_reject_invalid_odds(odds):
    with pytest.raises(ValueError):
        implied_probabilities(odds)
