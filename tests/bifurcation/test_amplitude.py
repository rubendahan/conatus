"""Landau and Ginzburg–Landau coefficients from the IR against published closed forms.

Swift–Hohenberg: A² = 4r/3 (Cross & Hohenberg 1993, RMP 65, §IV), τ0 = 1, ξ0² = 4, g0 = 3.
Stress-free Rayleigh–Bénard: Ra_c = 27π⁴/4 (Rayleigh 1916) and Nu − 1 = 2ε (Malkus & Veronis 1958),
independent of Pr, with spectral convergence in Nz.
"""
import numpy as np
import pytest


def test_swift_hohenberg_landau_and_ginzburg_landau(example):
    wn = example("bifurcation/weakly_nonlinear")
    le, gl = wn.sh_coefficients()
    assert abs(le.mu - 1) < 1e-13 and abs(le.g3 + 0.75) < 1e-13
    assert abs(le.amplitude(0.1) ** 2 - 4 * 0.1 / 3) < 1e-13
    assert abs(gl.tau0 - 1) < 1e-13 and abs(gl.xi0_sq - 4) < 1e-10 and abs(gl.g0 - 3) < 1e-13
    assert abs(gl.v_g) < 1e-12
    assert "Cross–Hohenberg form" in gl.explain()


@pytest.mark.parametrize("Pr", [0.1, 7.0])
def test_stress_free_convection_nusselt_slope(example, Pr):
    wn = example("bifurcation/weakly_nonlinear")
    slope, nf = wn.nusselt_slope(8, 20, Pr)
    assert abs(nf.params["Ra"] - 27 * np.pi ** 4 / 4) < 1e-9 and nf.checks["|λ_0|"] < 1e-10
    assert nf.kind == "pitchfork" and nf.a3 < 0          # supercritical rolls
    assert abs(slope - 2) < 1e-10


def test_transcritical_refused_by_landau():
    from conatus.bifurcation import amplitude_equation, ir_system, ode_problem
    from conatus.errors import OptionError
    s = ir_system(ode_problem(["x"], {"p": 0.0}, lambda X, P: [P["p"] * X["x"] - X["x"] * X["x"]]), ["p"])
    with pytest.raises(OptionError, match="transcritical"):
        amplitude_equation(s, np.zeros(1), "p", kind="stationary", observable=lambda v: v[0])
