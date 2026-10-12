"""Hookstep trust radii read relative to the starting norm.

The v0.1 radii are absolute (delta0 = 1 in [1e-10, 10]): right for a state of norm O(1), off the
scale for a state of norm 1e-3 or 1e3. ``radii="relative"`` reads the same numbers as fractions of
‖z0‖; the default stays absolute (v0.1 results unchanged) and warns when ‖z0‖ is outside the
declared range. The scalar problem z³ = a³ scaled by s has its solution at s·a: the same relative
radii solve it at every scale, and the result says which radii were used."""
import warnings

import numpy as np
import pytest

from conatus.ecs.newton import (HOOKSTEP_NORM_MAX, HOOKSTEP_NORM_MIN, NewtonKrylovHookstep,
                                TrustRadiusWarning)
from conatus.errors import OptionError


def _cubic(s):
    a = 1.7 * s
    return (lambda z: z ** 3 - a ** 3), (lambda z, v: 3 * z ** 2 * v), a


@pytest.mark.parametrize("s", [1e-3, 1.0, 1e3])
def test_relative_radii_solve_every_scale(s):
    R, J, a = _cubic(s)
    nk = NewtonKrylovHookstep(tol=1e-12, radii="relative", delta0=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TrustRadiusWarning)
        res = nk.solve(R, J, np.array([3.0 * s]), scale=a ** 3)
    assert abs(res.z[0] - a) < 1e-10 * a
    assert "relative to ‖z0‖ = " in res.trust and "relative to ‖z0‖" in res.explain()
    assert "(× ‖z0‖)" in nk.explain()


def test_default_absolute_radii_warn_off_scale_and_say_so():
    R, J, a = _cubic(1e3)
    nk = NewtonKrylovHookstep(tol=1e-12)
    assert nk.radii == "absolute" and HOOKSTEP_NORM_MIN < 1 < HOOKSTEP_NORM_MAX
    # the critic's case: Δmax = 10 against a distance of 1.3e3 to the root, the budget runs out;
    # the warning comes first and the failure names the radii it used
    from conatus.errors import NewtonNotConverged
    with pytest.warns(TrustRadiusWarning, match="radii='relative'"):
        with pytest.raises(NewtonNotConverged, match="TrustRadiusWarning"):
            nk.solve(R, J, np.array([3e3]), scale=a ** 3)
    # on the scale of the defaults: silent, and the absolute reading is stated
    R1, J1, a1 = _cubic(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TrustRadiusWarning)
        r1 = nk.solve(R1, J1, np.array([3.0]), scale=a1 ** 3)
        # explicit radii are the user's declared choice: no warning either
        NewtonKrylovHookstep(tol=1e-12, delta0=1e3, delta_max=1e4).solve(R, J, np.array([3e3]),
                                                                          scale=a ** 3)
    assert r1.trust.startswith("absolute")


def test_radii_values_checked():
    with pytest.raises(OptionError, match="radii"):
        NewtonKrylovHookstep(tol=1e-12, radii="scaled")
    R, J, _ = _cubic(1.0)
    with pytest.raises(OptionError, match="zero"):
        NewtonKrylovHookstep(tol=1e-12, radii="relative").solve(R, J, np.zeros(1), scale=1.0)


def test_trust_radius_warning_points_at_the_users_line():
    # a fixed stacklevel=3 pointed into the finder that called solve()
    R, J, a = _cubic(1e3)
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=1)
    from conatus.errors import NewtonNotConverged
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(NewtonNotConverged):
            nk.solve(R, J, np.array([3e3]), scale=a ** 3)
    w = [x for x in rec if issubclass(x.category, TrustRadiusWarning)]
    assert len(w) == 1 and w[0].filename == __file__
