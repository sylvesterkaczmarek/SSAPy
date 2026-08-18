from types import SimpleNamespace

import numpy as np

import ssapy.orbit_solver as orbit_solver
from ssapy.constants import MOON_MU


def test_three_angle_solver_propagates_custom_mu_to_shefer(monkeypatch):
    seen_mu = []

    class FakeSheferSolver:
        def __init__(self, r1, r2, t1, t2, mu=None, **kwargs):
            seen_mu.append(mu)
            self.mu = mu

        def _getP(self):
            return 1.0

        def _getEta(self, p):
            return 1.0

        def solve(self):
            return SimpleNamespace(mu=self.mu)

    monkeypatch.setattr(
        orbit_solver, "SheferTwoPosOrbitSolver", FakeSheferSolver
    )

    solver = orbit_solver.ThreeAngleOrbitSolver(
        e1=np.array([1.0, 0.0, 0.0]),
        e2=np.array([0.0, 1.0, 0.0]),
        e3=np.array([0.0, 0.0, 1.0]),
        R1=np.zeros(3),
        R2=np.zeros(3),
        R3=np.zeros(3),
        t1=0.0,
        t2=1.0,
        t3=2.0,
        mu=MOON_MU,
        maxiter=1,
    )

    result = solver.solve()

    assert seen_mu == [MOON_MU, MOON_MU, MOON_MU, MOON_MU]
    assert result.mu == MOON_MU
