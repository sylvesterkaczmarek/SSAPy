import numpy as np

from ssapy.orbit import (
    _hyperbolicTrueToEccentricAnomaly,
    _hyperbolicTrueToEccentricAnomalyMany,
)


def test_hyperbolic_true_to_eccentric_many_broadcasts_by_orbit():
    eccentricity = np.array([1.2, 2.0])
    true_anomaly = np.array([
        [-0.4, 0.0, 0.4],
        [-0.6, 0.1, 0.5],
    ])

    result = _hyperbolicTrueToEccentricAnomalyMany(
        true_anomaly, eccentricity
    )
    expected = np.vstack([
        _hyperbolicTrueToEccentricAnomaly(true_anomaly[i], eccentricity[i])
        for i in range(len(eccentricity))
    ])

    assert result.shape == true_anomaly.shape
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
