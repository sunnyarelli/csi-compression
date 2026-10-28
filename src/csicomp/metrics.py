"""CSI compression metrics used in 3GPP evaluations."""
from __future__ import annotations

import numpy as np


def sgcs(W: np.ndarray, W_hat: np.ndarray) -> np.ndarray:
    """Squared generalized cosine similarity per sample, averaged over subbands."""
    num = np.abs(np.sum(W.conj() * W_hat, axis=-1)) ** 2
    den = np.sum(np.abs(W) ** 2, axis=-1) * np.sum(np.abs(W_hat) ** 2, axis=-1)
    return (num / den).mean(axis=-1)


def nmse_db(W: np.ndarray, W_hat: np.ndarray) -> float:
    err = np.sum(np.abs(W - W_hat) ** 2, axis=(-2, -1))
    ref = np.sum(np.abs(W) ** 2, axis=(-2, -1))
    return float(10 * np.log10(np.mean(err / ref)))
