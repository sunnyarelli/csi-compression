# csi-compression

[![CI](https://github.com/sunnyarelli/csi-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyarelli/csi-compression/actions/workflows/ci.yml)

Benchmark for **learned (data-driven) CSI compression** (two-sided model) against rule-based NR codebooks.
Learned CSI compression has been a 3GPP air-interface study topic since Rel-18. It continues in Rel-19/20 and the
6G study, where the open questions are **generalization, dataset/site dependence, and the payload vs. accuracy trade-off**.

Pure NumPy. No GPU or deep-learning framework is needed.

## What it does

| Module | Contents |
|---|---|
| `csicomp.channel` | Clustered multipath channel generator (CDL-style), site-specific datasets, rank-1 eigenvector extraction with phase-continuous subbands |
| `csicomp.codebook` | Type-I (oversampled DFT beam per subband) and Type-II-like (L orthogonal beams + quantized amplitude/phase) baselines with payload bit counts |
| `csicomp.autoencoder` | Learned linear autoencoder (closed-form PCA encoder/decoder + per-latent scalar quantization) |
| `csicomp.metrics` | SGCS (squared generalized cosine similarity, the 3GPP KPI) and NMSE |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
csi-compression             # or: python -m csicomp --tx 32 --subbands 13 --train 1500
```

Sample output (32 Tx, 4 Rx, 13 subbands):

```
scheme                  payload [bits]  SGCS same site  SGCS unseen site
Type-I (O=4)                        91           0.460             0.481
Type-II-like L=2                   102           0.601             0.619
Type-II-like L=4                   278           0.759             0.786
Linear AE d=32 b=3                  96           0.477             0.439
Linear AE d=64 b=4                 256           0.620             0.577
Linear AE d=128 b=4                512           0.736             0.683
```

## Reading the results

- **Payload vs. accuracy:** at a similar payload (~96–102 bits), the learned linear model beats Type-I but trails the
  Type-II-like codebook.
- **Generalization gap:** the learned model loses about 0.04–0.05 SGCS on a site it was not trained on. The rule-based
  codebooks have no such dependency. This is the Rel-19/20 *model generalization / monitoring* problem in miniature.
- **Linear is not enough:** a linear AE spans one fixed subspace. Beating eType-II needs nonlinear (neural)
  encoders, which is why 3GPP evaluations use CNN/Transformer models. Any object with `compress_reconstruct(W)` and
  `payload_bits(n_subbands)` can be dropped into the benchmark, so a PyTorch model plugs in easily.

## Assumptions and limitations

- Single-polarization ULA, rank 1, ideal (noise-free) channel estimation at the UE.
- The channel generator is CDL-*style* (clusters, Laplacian rays, exponential power-delay profile). It is not a
  calibrated TR 38.901 implementation.
- The Type-II-like codebook is simplified (no polarization co-phasing or frequency-domain compression as in Rel-16 eType-II).

## References

- 3GPP TR 38.843 (Rel-18 study, CSI compression use case)
- 3GPP TS 38.214 §5.2.2.2 (Type-I / Type-II codebooks)
- 3GPP TR 38.901, *Channel model for frequencies from 0.5 to 100 GHz*

## License

MIT
