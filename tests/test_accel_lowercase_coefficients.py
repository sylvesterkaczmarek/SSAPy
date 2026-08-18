import numpy as np

import ssapy.accel as accel_module
from ssapy.accel import AccelDrag, AccelEarthRad, AccelSolRad
from ssapy.constants import EARTH_RADIUS


def test_radiation_models_accept_lowercase_cr(monkeypatch):
    monkeypatch.setattr(
        accel_module,
        "sunPos",
        lambda t: np.array([1.5e11, 2.0e10, -1.0e10]),
    )
    r = np.array([7.0e6, 1.0e6, 2.0e6])
    v = np.zeros(3)
    common = dict(area=2.0, mass=100.0)

    for model in (AccelSolRad(), AccelEarthRad()):
        upper = model(r, v, 0.0, CR=1.7, **common)
        lower = model(r, v, 0.0, cr=1.7, **common)
        np.testing.assert_allclose(lower, upper)


def test_drag_accepts_lowercase_cd(monkeypatch):
    monkeypatch.setattr(
        accel_module,
        "sunPos",
        lambda t: np.array([1.5e11, 0.0, 0.0]),
    )
    model = AccelDrag()
    model.atm = type(
        "Atmosphere",
        (),
        {"density": staticmethod(lambda *args: 1.0e-12)},
    )()

    r = np.array([EARTH_RADIUS + 400e3, 0.0, 0.0])
    v = np.array([0.0, 7_700.0, 0.0])
    common = dict(area=2.0, mass=100.0, _T=np.eye(3))

    upper = model(r, v, 0.0, CD=2.2, **common)
    lower = model(r, v, 0.0, cd=2.2, **common)
    np.testing.assert_allclose(lower, upper)


def test_uppercase_coefficient_takes_precedence(monkeypatch):
    monkeypatch.setattr(
        accel_module,
        "sunPos",
        lambda t: np.array([1.5e11, 0.0, 0.0]),
    )
    model = AccelSolRad()
    r = np.array([7.0e6, 0.0, 0.0])

    both = model(r, np.zeros(3), 0.0, area=1.0, mass=1.0, CR=1.5, cr=9.0)
    upper = model(r, np.zeros(3), 0.0, area=1.0, mass=1.0, CR=1.5)
    np.testing.assert_allclose(both, upper)
