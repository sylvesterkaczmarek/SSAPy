import numpy as np
import pytest

from ssapy.particles import Particles


class _Probability:
    epoch = 0.0

    @staticmethod
    def lnprior(orbit):
        return 0.0

    @staticmethod
    def lnlike(orbit):
        return 0.0


def test_3d_particle_chain_flattens_matching_log_weights():
    particles = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    lnpriors = np.zeros((2, 3))
    ln_weights = np.arange(6, dtype=float).reshape(2, 3)

    result = Particles(
        particles,
        _Probability(),
        lnpriors=lnpriors,
        ln_weights=ln_weights,
    )

    assert result.particles.shape == (6, 6)
    assert result.ln_wts.shape == (6,)
    np.testing.assert_array_equal(result.ln_wts, np.arange(6, dtype=float))


def test_3d_particle_chain_rejects_mismatched_log_weights():
    particles = np.zeros((2, 3, 6))

    with pytest.raises(ValueError, match="ln_weights"):
        Particles(
            particles,
            _Probability(),
            lnpriors=np.zeros((2, 3)),
            ln_weights=np.zeros((2, 2)),
        )
