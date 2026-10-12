"""Two-mode SO(2)-equivariant model of Budanur, Borrero-Echeverry & Cvitanović, *Periodic orbit
analysis of a system with continuous symmetry — a tutorial*, arXiv:1411.3303 (read; journal
reference Chaos 25, 073112 (2015) to be re-checked) — validation of the slice and of relative
periodic orbits in the slice.

Complex form (eq. 20): ż₁ = (μ₁ − ie₁)z₁ + a₁z₁|z₁|² + b₁z₁|z₂|² + c₁ z̄₁z₂,
ż₂ = (μ₂ − ie₂)z₂ + a₂z₂|z₁|² + b₂z₂|z₂|² + c₂z₁²; real form (eq. 21) with z_j = x_j + i y_j;
SO(2) acts by (z₁, z₂) → (e^{iθ}z₁, e^{2iθ}z₂) (generator eq. 24, charges (1, 2)).
Parameters of eq. (22) (read in the arXiv PDF): μ₁ = −2.8, μ₂ = 1, e₁ = 0, e₂ = 1, a₁ = −1,
a₂ = −2.66, b₁ = 0, b₂ = 0, c₁ = −7.75, c₂ = 1.
"""
from __future__ import annotations

__all__ = ["TWO_MODE_PARAMETERS", "two_mode_problem", "two_mode_callables"]

TWO_MODE_PARAMETERS = {"mu1": -2.8, "mu2": 1.0, "e1": 0.0, "e2": 1.0, "a1": -1.0, "a2": -2.66,
                       "b1": 0.0, "b2": 0.0, "c1": -7.75, "c2": 1.0}


def two_mode_problem(parameters=None):
    """IR IVP (0-D, ``conatus.bifurcation.ode_problem``) of the real form (eq. 21), variables
    (x1, y1, x2, y2); ``parameters`` overrides TWO_MODE_PARAMETERS."""
    from ..bifurcation.system import ode_problem
    par = {**TWO_MODE_PARAMETERS, **(parameters or {})}

    def rhs(X, P):
        x1, y1, x2, y2 = X["x1"], X["y1"], X["x2"], X["y2"]
        r1, r2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        g1 = P["mu1"] + P["a1"] * r1 + P["b1"] * r2
        g2 = P["mu2"] + P["a2"] * r1 + P["b2"] * r2
        return [(g1 + P["c1"] * x2) * x1 + P["c1"] * y1 * y2 + P["e1"] * y1,
                (g1 - P["c1"] * x2) * y1 + P["c1"] * x1 * y2 - P["e1"] * x1,
                g2 * x2 + P["c2"] * (x1 * x1 - y1 * y1) + P["e2"] * y2,
                g2 * y2 + 2 * P["c2"] * x1 * y1 - P["e2"] * x2]
    return ode_problem(["x1", "y1", "x2", "y2"], par, rhs,
                       "two-mode (Budanur, Borrero-Echeverry & Cvitanovic)")


def two_mode_callables(parameters=None):
    """Plain-numpy (f, jac) of the same real form (eq. 21), for long integrations where the
    per-call cost of the IR evaluator dominates; ``tests/slices`` checks f = the IR right-hand side."""
    import numpy as np
    p = {**TWO_MODE_PARAMETERS, **(parameters or {})}

    def f(x, par=None):
        x1, y1, x2, y2 = x
        r1, r2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        g1 = p["mu1"] + p["a1"] * r1 + p["b1"] * r2
        g2 = p["mu2"] + p["a2"] * r1 + p["b2"] * r2
        return np.array([(g1 + p["c1"] * x2) * x1 + p["c1"] * y1 * y2 + p["e1"] * y1,
                         (g1 - p["c1"] * x2) * y1 + p["c1"] * x1 * y2 - p["e1"] * x1,
                         g2 * x2 + p["c2"] * (x1 * x1 - y1 * y1) + p["e2"] * y2,
                         g2 * y2 + 2 * p["c2"] * x1 * y1 - p["e2"] * x2])

    def jac(x, par=None):
        x1, y1, x2, y2 = x
        r1, r2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        g1 = p["mu1"] + p["a1"] * r1 + p["b1"] * r2
        g2 = p["mu2"] + p["a2"] * r1 + p["b2"] * r2
        a1, a2, b1, b2, c1, c2 = p["a1"], p["a2"], p["b1"], p["b2"], p["c1"], p["c2"]
        e1, e2 = p["e1"], p["e2"]
        return np.array([
            [g1 + c1 * x2 + 2 * a1 * x1 * x1, 2 * a1 * x1 * y1 + c1 * y2 + e1,
             2 * b1 * x2 * x1 + c1 * x1, 2 * b1 * y2 * x1 + c1 * y1],
            [2 * a1 * x1 * y1 + c1 * y2 - e1, g1 - c1 * x2 + 2 * a1 * y1 * y1,
             2 * b1 * x2 * y1 - c1 * y1, 2 * b1 * y2 * y1 + c1 * x1],
            [2 * a2 * x1 * x2 + 2 * c2 * x1, 2 * a2 * y1 * x2 - 2 * c2 * y1,
             g2 + 2 * b2 * x2 * x2, 2 * b2 * y2 * x2 + e2],
            [2 * a2 * x1 * y2 + 2 * c2 * y1, 2 * a2 * y1 * y2 + 2 * c2 * x1,
             2 * b2 * x2 * y2 - e2, g2 + 2 * b2 * y2 * y2]])
    return f, jac
