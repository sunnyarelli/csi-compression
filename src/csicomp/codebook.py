"""Rule-based CSI codebooks used as baselines (Type-I / Type-II-like, single polarization)."""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, ceil, log2

import numpy as np


def dft_beams(n_tx: int, oversampling: int) -> np.ndarray:
    """Oversampled DFT beam matrix, shape (n_tx, n_tx * oversampling), unit-norm columns."""
    n = np.arange(n_tx)[:, None]
    k = np.arange(n_tx * oversampling)[None, :]
    return np.exp(2j * np.pi * n * k / (n_tx * oversampling)) / np.sqrt(n_tx)


@dataclass
class TypeICodebook:
    """Per-subband selection of the best single oversampled DFT beam."""

    n_tx: int
    oversampling: int = 4

    def __post_init__(self):
        self.beams = dft_beams(self.n_tx, self.oversampling)

    def payload_bits(self, n_subbands: int) -> int:
        return n_subbands * ceil(log2(self.n_tx * self.oversampling))

    def compress_reconstruct(self, W: np.ndarray) -> np.ndarray:
        corr = np.abs(W.conj() @ self.beams) ** 2           # (..., sb, beams)
        idx = corr.argmax(axis=-1)
        return self.beams.T[idx]


@dataclass
class TypeIICodebook:
    """Wideband selection of L orthogonal beams + per-subband quantized amplitude/phase."""

    n_tx: int
    n_beams: int = 4
    oversampling: int = 4
    amp_bits: int = 3
    phase_bits: int = 3

    def __post_init__(self):
        self.beams = dft_beams(self.n_tx, self.oversampling)
        # Amplitude levels in 3 dB steps below the strongest coefficient (incl. zero).
        n_lvl = 2 ** self.amp_bits
        self.amp_levels = np.concatenate([[0.0], 10 ** (-1.5 * np.arange(n_lvl - 2, -1, -1) / 10)])

    def payload_bits(self, n_subbands: int) -> int:
        wideband = ceil(log2(self.oversampling)) + ceil(log2(comb(self.n_tx, self.n_beams)))
        per_sb = (self.n_beams - 1) * (self.amp_bits + self.phase_bits) + ceil(log2(self.n_beams))
        return wideband + n_subbands * per_sb

    def _select(self, W: np.ndarray) -> np.ndarray:
        power = (np.abs(W.conj() @ self.beams) ** 2).sum(axis=0)    # (beams,)
        best, best_p = None, -1.0
        for q in range(self.oversampling):
            group = q + self.oversampling * np.arange(self.n_tx)
            top = group[np.argsort(power[group])[::-1][: self.n_beams]]
            p = power[top].sum()
            if p > best_p:
                best, best_p = top, p
        return best

    def _quantize(self, c: np.ndarray) -> np.ndarray:
        amp = np.abs(c) / np.abs(c).max(axis=-1, keepdims=True)
        amp_q = self.amp_levels[np.abs(amp[..., None] - self.amp_levels).argmin(axis=-1)]
        step = 2 * np.pi / 2 ** self.phase_bits
        ph_q = np.round(np.angle(c) / step) * step
        return amp_q * np.exp(1j * ph_q)

    def compress_reconstruct(self, W: np.ndarray) -> np.ndarray:
        """W: (samples, subbands, n_tx) -> reconstruction of the same shape."""
        out = np.empty_like(W)
        for s in range(W.shape[0]):
            B = self.beams[:, self._select(W[s])]                # (n_tx, L)
            coeff = W[s] @ B.conj()                              # (sb, L)
            w_hat = self._quantize(coeff) @ B.T
            out[s] = w_hat / np.linalg.norm(w_hat, axis=-1, keepdims=True)
        return out
