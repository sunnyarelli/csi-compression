import numpy as np
import pytest

from csicomp.autoencoder import LinearAutoencoder
from csicomp.channel import dominant_eigenvectors, generate_channels
from csicomp.codebook import TypeICodebook, TypeIICodebook
from csicomp.metrics import sgcs


@pytest.fixture(scope="module")
def data():
    def make(n, site, seed):
        return dominant_eigenvectors(generate_channels(n, n_tx=16, n_subbands=8, site=site, rng=seed))

    return make(400, 5, 1), make(100, 5, 2), make(100, 77, 3)


def test_channel_shape_and_unit_norm(data):
    W = data[0]
    assert W.shape == (400, 8, 16)
    np.testing.assert_allclose(np.linalg.norm(W, axis=-1), 1.0, atol=1e-12)


def test_sgcs_identity_is_one(data):
    W = data[0]
    np.testing.assert_allclose(sgcs(W, W), 1.0)


def test_phase_continuity_across_subbands(data):
    W = data[0]
    inner = np.sum(W[:, :-1].conj() * W[:, 1:], axis=-1)
    np.testing.assert_allclose(np.angle(inner), 0.0, atol=1e-9)


def test_type2_beats_type1(data):
    W = data[1]
    s1 = sgcs(W, TypeICodebook(16).compress_reconstruct(W)).mean()
    s2 = sgcs(W, TypeIICodebook(16, n_beams=4).compress_reconstruct(W)).mean()
    assert s2 > s1


def test_autoencoder_improves_with_payload(data):
    W_train, W_test, _ = data
    small = LinearAutoencoder(8, 2).fit(W_train)
    large = LinearAutoencoder(96, 6).fit(W_train)
    assert large.payload_bits() > small.payload_bits()
    s_small = sgcs(W_test, small.compress_reconstruct(W_test)).mean()
    s_large = sgcs(W_test, large.compress_reconstruct(W_test)).mean()
    assert s_large > s_small


def test_generalization_gap_on_unseen_site(data):
    W_train, W_test, W_unseen = data
    model = LinearAutoencoder(64, 4).fit(W_train)
    same = sgcs(W_test, model.compress_reconstruct(W_test)).mean()
    unseen = sgcs(W_unseen, model.compress_reconstruct(W_unseen)).mean()
    assert same > unseen


def test_payload_bits():
    assert TypeICodebook(32, 4).payload_bits(13) == 13 * 7
