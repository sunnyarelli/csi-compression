"""Benchmark: rule-based codebooks vs. a learned two-sided CSI compressor."""
from __future__ import annotations

import argparse

from .autoencoder import LinearAutoencoder
from .channel import dominant_eigenvectors, generate_channels
from .codebook import TypeICodebook, TypeIICodebook
from .metrics import sgcs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Learned CSI feedback benchmark (Rel-18..20 style)")
    p.add_argument("--train", type=int, default=1500)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--tx", type=int, default=32)
    p.add_argument("--subbands", type=int, default=13)
    p.add_argument("--site", type=int, default=11, help="training/test site id")
    p.add_argument("--other-site", type=int, default=99, help="unseen site for generalization")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)

    def dataset(n, site, seed):
        H = generate_channels(n, a.tx, n_subbands=a.subbands, site=site, rng=seed)
        return dominant_eigenvectors(H)

    W_train = dataset(a.train, a.site, a.seed)
    W_test = dataset(a.test, a.site, a.seed + 1)
    W_unseen = dataset(a.test, a.other_site, a.seed + 2)

    schemes = [
        ("Type-I (O=4)", TypeICodebook(a.tx)),
        ("Type-II-like L=2", TypeIICodebook(a.tx, n_beams=2)),
        ("Type-II-like L=4", TypeIICodebook(a.tx, n_beams=4)),
        ("Linear AE d=32 b=3", LinearAutoencoder(32, 3).fit(W_train)),
        ("Linear AE d=64 b=4", LinearAutoencoder(64, 4).fit(W_train)),
        ("Linear AE d=128 b=4", LinearAutoencoder(128, 4).fit(W_train)),
    ]
    print(f"{'scheme':<22}{'payload [bits]':>16}{'SGCS same site':>16}{'SGCS unseen site':>18}")
    for name, s in schemes:
        same = sgcs(W_test, s.compress_reconstruct(W_test)).mean()
        unseen = sgcs(W_unseen, s.compress_reconstruct(W_unseen)).mean()
        print(f"{name:<22}{s.payload_bits(a.subbands):>16}{same:>16.3f}{unseen:>18.3f}")


if __name__ == "__main__":
    main()
