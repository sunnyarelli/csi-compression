"""Clustered multipath channel generator (simplified CDL-style, TR 38.901 flavoured)."""
from __future__ import annotations

import numpy as np

C = 299_792_458.0


def ula_response(n_ant: int, angles_rad: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Uniform linear array response, shape (len(angles), n_ant), unit-norm rows."""
    n = np.arange(n_ant)
    phase = 2j * np.pi * spacing * np.outer(np.sin(np.atleast_1d(angles_rad)), n)
    return np.exp(phase) / np.sqrt(n_ant)


def generate_channels(
    n_samples: int,
    n_tx: int = 32,
    n_rx: int = 4,
    n_subbands: int = 13,
    n_clusters: int = 6,
    rays_per_cluster: int = 10,
    angle_spread_deg: float = 5.0,
    delay_spread_s: float = 300e-9,
    scs_hz: float = 30e3,
    prb_per_subband: int = 4,
    site: int | None = None,
    site_jitter_deg: float = 8.0,
    rng=None,
) -> np.ndarray:
    """Return frequency-domain channels H with shape (samples, subbands, n_rx, n_tx).

    With ``site`` set, cluster departure angles are anchored to a fixed set of
    site-specific scatterer directions (seeded by ``site``) plus per-UE jitter.
    This mimics the site-/scenario-specific datasets used in 3GPP learned-CSI studies.
    With ``site=None`` every sample draws fully random cluster directions.
    """
    rng = np.random.default_rng(rng)
    if site is not None:
        site_aod = np.random.default_rng(site).uniform(-np.pi / 3, np.pi / 3, n_clusters)
    sb_bw = scs_hz * 12 * prb_per_subband
    freqs = (np.arange(n_subbands) - (n_subbands - 1) / 2) * sb_bw
    spread = np.deg2rad(angle_spread_deg)

    H = np.zeros((n_samples, n_subbands, n_rx, n_tx), dtype=complex)
    for s in range(n_samples):
        if site is None:
            aod_c = rng.uniform(-np.pi / 3, np.pi / 3, n_clusters)
        else:
            aod_c = site_aod + np.deg2rad(site_jitter_deg) * rng.standard_normal(n_clusters)
        aoa_c = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters)
        delays = np.sort(rng.exponential(delay_spread_s, n_clusters))
        delays -= delays[0]
        powers = np.exp(-delays / delay_spread_s) * 10 ** (-rng.normal(0, 3, n_clusters) / 10)
        powers /= powers.sum()
        for c in range(n_clusters):
            aod = aod_c[c] + spread * rng.laplace(0, 1 / np.sqrt(2), rays_per_cluster)
            aoa = aoa_c[c] + 2 * spread * rng.laplace(0, 1 / np.sqrt(2), rays_per_cluster)
            g = (rng.standard_normal(rays_per_cluster) + 1j * rng.standard_normal(rays_per_cluster))
            g *= np.sqrt(powers[c] / (2 * rays_per_cluster))
            a_rx = ula_response(n_rx, aoa)            # (rays, n_rx)
            a_tx = ula_response(n_tx, aod).conj()     # (rays, n_tx)
            cluster = np.einsum("r,ri,rj->ij", g, a_rx, a_tx) * np.sqrt(n_rx * n_tx)
            phase = np.exp(-2j * np.pi * freqs * delays[c])
            H[s] += phase[:, None, None] * cluster[None]
    return H


def dominant_eigenvectors(H: np.ndarray) -> np.ndarray:
    """Rank-1 precoder per subband: dominant right singular vector, shape (..., sb, n_tx).

    Each subband's eigenvector has an arbitrary common phase. The first subband is
    rotated so its strongest element is real-positive, and every following subband
    is rotated to be phase-continuous with its predecessor. SGCS is invariant to
    these rotations, but they make the data far more compressible.
    """
    _, _, vh = np.linalg.svd(H, full_matrices=False)
    w = vh[..., 0, :].conj()
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    first = w[..., 0, :]
    peak = np.take_along_axis(first, np.abs(first).argmax(axis=-1)[..., None], axis=-1)
    w[..., 0, :] = first * np.exp(-1j * np.angle(peak))
    for sb in range(1, w.shape[-2]):
        inner = np.sum(w[..., sb - 1, :].conj() * w[..., sb, :], axis=-1, keepdims=True)
        w[..., sb, :] *= np.exp(-1j * np.angle(inner))
    return w
