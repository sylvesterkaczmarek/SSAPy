import numpy as np
import pytest

from ssapy.compute import radec, radecRate
from ssapy.constants import EARTH_MU
from ssapy.orbit import Orbit


def test_radecrate_matches_radec_rate_outputs():
    radius = 7_000_000.0
    orbit = Orbit(
        r=np.array([radius, 0.0, 0.0]),
        v=np.array([0.0, np.sqrt(EARTH_MU / radius), 0.0]),
        t=0.0,
    )
    time = np.array([0.0, 10.0])
    obs_pos = np.zeros((2, 3))
    obs_vel = np.zeros((2, 3))

    expected = radec(
        orbit,
        time,
        obsPos=obs_pos,
        obsVel=obs_vel,
        rate=True,
    )[3:]

    with pytest.warns(UserWarning, match="deprecated"):
        actual = radecRate(
            orbit,
            time,
            obsPos=obs_pos,
            obsVel=obs_vel,
        )

    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference)
