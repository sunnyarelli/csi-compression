"""Learned linear autoencoder for CSI compression (two-sided model).

A linear autoencoder trained with MSE converges to the PCA subspace, so the
encoder/decoder pair is fitted in closed form with an SVD. Latents are scalar
quantized with per-dimension ranges learned on the training set.
"""
from __future__ import annotations

import numpy as np


def _to_real(W: np.ndarray) -> np.ndarray:
    flat = W.reshape(W.shape[0], -1)
    return np.concatenate([flat.real, flat.imag], axis=1)


def _from_real(X: np.ndarray, shape: tuple) -> np.ndarray:
    half = X.shape[1] // 2
    return (X[:, :half] + 1j * X[:, half:]).reshape((X.shape[0],) + shape)


class LinearAutoencoder:
    def __init__(self, latent_dim: int = 64, bits_per_latent: int = 4):
        self.latent_dim = latent_dim
        self.bits = bits_per_latent

    def payload_bits(self, n_subbands: int | None = None) -> int:
        return self.latent_dim * self.bits

    def fit(self, W_train: np.ndarray) -> "LinearAutoencoder":
        self.shape = W_train.shape[1:]
        X = _to_real(W_train)
        self.mean = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.basis = vt[: self.latent_dim]                       # decoder rows / encoder cols
        Z = self.encode(W_train, quantize=False)
        self.lo = np.percentile(Z, 0.5, axis=0)
        self.hi = np.percentile(Z, 99.5, axis=0)
        return self

    def encode(self, W: np.ndarray, quantize: bool = True) -> np.ndarray:
        Z = (_to_real(W) - self.mean) @ self.basis.T
        if not quantize:
            return Z
        levels = 2 ** self.bits - 1
        q = np.round((np.clip(Z, self.lo, self.hi) - self.lo) / (self.hi - self.lo) * levels)
        return q.astype(np.int64)

    def decode(self, q: np.ndarray) -> np.ndarray:
        levels = 2 ** self.bits - 1
        Z = self.lo + q / levels * (self.hi - self.lo)
        W = _from_real(Z @ self.basis + self.mean, self.shape)
        return W / np.linalg.norm(W, axis=-1, keepdims=True)

    def compress_reconstruct(self, W: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(W))
