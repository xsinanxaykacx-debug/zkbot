"""De-vigged market probabilities from decimal odds."""
from __future__ import annotations

import numpy as np


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Normalize inverse decimal odds row-wise to remove proportional overround."""
    values = np.asarray(odds, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("odds must be a 2D array with at least two outcomes.")
    if not np.isfinite(values).all() or (values <= 1.0).any():
        raise ValueError("All decimal odds must be finite and greater than 1.")
    inverse = 1.0 / values
    probabilities = inverse / inverse.sum(axis=1, keepdims=True)
    return probabilities
