"""Model systems with closed-form eigenpairs and wavemakers, for the sensitivity toolkit.

Linear complex Ginzburg–Landau equation with a spatially varying growth rate (Hunt & Crighton
1991; Chomaz 2005, §4; Bagheri, Henningson, Hœpffner & Schmid 2009, the "global" case):

    ∂t q = −ν ∂x q + γ ∂x² q + μ(x) q,     μ(x) = μ_c + μ₂ x²/2   (μ₂ < 0),

on x ∈ ℝ with q → 0 at ±∞. With χ⁴ = −μ₂/(2γ) (the root with Re χ² > 0) and h = 2γχ²:

    λ_n = μ_c − ν²/(4γ) − (n + ½) h,
    φ_n(x) = exp(ν x/(2γ) − χ² x²/2) H_n(χ x)                (direct, physicists' Hermite H_n)
    ψ_n(x) = exp(−ν̄ x/(2γ̄) − χ̄² x²/2) H_n(χ̄ x)              (adjoint for ∫ ā b dx)

and, since |e^{νx/(2γ)}| |e^{−ν̄x/(2γ̄)}| = 1, the wavemaker (Giannetti & Luchini 2007) and the
eigenvalue gradient with respect to μ(x) are closed-form:

    S_n(x) = |ψ_n| |φ_n| / |⟨ψ_n, φ_n⟩| = |e^{−χ²x²}| |H_n(χx)|² |χ| / (√π 2ⁿ n!),
    δλ_n/δμ(x) = ψ̄_n φ_n / ⟨ψ_n, φ_n⟩ = e^{−χ²x²} H_n(χx)² χ / (√π 2ⁿ n!),

using ∫ e^{−χ²x²} H_n(χx)² dx = √π 2ⁿ n!/χ (contour rotation, Re χ² > 0). The whole-line problem
is computed on a *declared* finite interval with homogeneous Dirichlet conditions; the truncation
error is exp(−Re χ² L²/2 ± Re(ν/2γ) L) and is reported by ``GinzburgLandauExact.truncation``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy
from numpy.polynomial.hermite import hermval

from ..errors import OptionError
from ..nonmodal.system import ModalSystem

__all__ = ["linear_ginzburg_landau", "GinzburgLandauExact", "ginzburg_landau_exact"]


def linear_ginzburg_landau(*, nu: complex, gamma: complex, mu, bounds: tuple[float, float],
                           name: str = "linear Ginzburg-Landau") -> ModalSystem:
    """∂t q = −ν ∂x q + γ ∂x² q + μ(x) q on ``bounds`` with q = 0 at both ends (declared).

    ``mu`` is a sympy expression in ``conatus.nonmodal.Y`` (the coordinate, called x above) or a
    number. Energy ½ ∫ |q|² dx (weight 1). Every argument is mandatory (no default model).
    """
    if not complex(gamma).real > 0:
        raise OptionError(f"gamma = {gamma}: Re γ must be > 0 (diffusion; otherwise ill-posed)")
    a, b = (float(v) for v in bounds)
    if not b > a:
        raise OptionError(f"bounds = {bounds}: expected (a, b) with b > a")
    nu_, g_ = sympy.nsimplify(complex(nu)), sympy.nsimplify(complex(gamma))
    op = {2: g_, 1: -nu_, 0: sympy.sympify(mu)}
    op = {k: c for k, c in op.items() if c != 0}
    bcs = (("q", 0, "left"), ("q", 0, "right"))
    return ModalSystem(f"{name}: nu = {complex(nu)}, gamma = {complex(gamma)}, mu = {mu}",
                       (a, b), ("q",), {"q": 1.0}, (), {("q", "q"): op}, {}, {}, bcs,
                       ["scalar complex field q, Dirichlet q = 0 at both ends (truncation of "
                        "the whole line, declared)"])


@dataclass
class GinzburgLandauExact:
    """Closed-form eigenpair n of the whole-line problem (see module doc)."""

    nu: complex
    gamma: complex
    mu_c: float
    mu2: float
    n: int
    chi: complex                  # χ, with Re χ² > 0
    eigenvalue: complex

    def direct(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        c = np.zeros(self.n + 1)
        c[-1] = 1.0
        return (np.exp(self.nu * x / (2 * self.gamma) - self.chi ** 2 * x ** 2 / 2)
                * hermval(self.chi * x, c))

    def adjoint(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        c = np.zeros(self.n + 1)
        c[-1] = 1.0
        cb, gb, nb = np.conj(self.chi), np.conj(self.gamma), np.conj(self.nu)
        return np.exp(-nb * x / (2 * gb) - cb ** 2 * x ** 2 / 2) * hermval(cb * x, c)

    def _norm(self) -> complex:
        return math.sqrt(math.pi) * 2.0 ** self.n * math.factorial(self.n) / self.chi

    def gradient_mu(self, x) -> np.ndarray:
        """δλ/δμ(x) = ψ̄ φ / ⟨ψ, φ⟩ (complex density; δλ = ∫ (δλ/δμ) δμ dx)."""
        x = np.asarray(x, float)
        c = np.zeros(self.n + 1)
        c[-1] = 1.0
        return np.exp(-self.chi ** 2 * x ** 2) * hermval(self.chi * x, c) ** 2 / self._norm()

    def wavemaker(self, x) -> np.ndarray:
        """S(x) = |ψ(x)| |φ(x)| / |⟨ψ, φ⟩|."""
        return np.abs(self.gradient_mu(x))

    def d_eigenvalue(self) -> dict[str, complex]:
        """dλ_n/dν, dλ_n/dμ_c, dλ_n/dμ₂ (closed form: −ν/(2γ), 1, (n + ½)/(2χ²))."""
        return {"nu": -self.nu / (2 * self.gamma), "mu_c": 1.0 + 0j,
                "mu2": (self.n + 0.5) / (2 * self.chi ** 2)}

    def truncation(self, bounds: tuple[float, float]) -> float:
        """max over the two ends of |φ|·|ψ| relative to its maximum: size of the neglected tail."""
        x = np.linspace(bounds[0], bounds[1], 4001)
        s = self.wavemaker(x)
        e = np.array([abs(self.direct(np.array([v]))[0]) * abs(self.adjoint(np.array([v]))[0])
                      for v in bounds]) / abs(self._norm())
        return float(max(e.max(), max(s[0], s[-1])) / s.max())

    def explain(self) -> str:
        return (f"Ginzburg-Landau (whole line) mode n = {self.n}: λ = {self.eigenvalue:.15g}, "
                f"χ = {self.chi:.12g} (Re χ² > 0); ν = {self.nu}, γ = {self.gamma}, μ_c = "
                f"{self.mu_c}, μ₂ = {self.mu2}. Closed-form direct/adjoint modes, wavemaker, "
                "gradient w.r.t. μ(x) and dλ/d(ν, μ_c, μ₂) (Chomaz 2005; Bagheri et al. 2009).")


def ginzburg_landau_exact(*, nu: complex, gamma: complex, mu_c: float, mu2: float,
                          n: int) -> GinzburgLandauExact:
    """Closed-form eigenpair n of ∂t q = −ν q' + γ q'' + (μ_c + μ₂ x²/2) q on the whole line."""
    if not mu2 < 0:
        raise OptionError(f"mu2 = {mu2}: the closed form needs μ₂ < 0 (confining potential)")
    from ..errors import check_count       # audit inputs: n = 2.5 was stored as 2, lambda at 2.5
    n = check_count(n, "n", "ginzburg_landau_exact()", minimum=0)
    nu, gamma = complex(nu), complex(gamma)
    chi2 = np.sqrt(-mu2 / (2 * gamma) + 0j)           # principal root: Re χ² > 0 for Re γ > 0
    if not chi2.real > 0:
        raise OptionError(f"Re χ² = {chi2.real:.3g} <= 0: no decaying eigenfunctions")
    chi = np.sqrt(chi2)
    h = 2 * gamma * chi2
    lam = mu_c - nu ** 2 / (4 * gamma) - (n + 0.5) * h
    return GinzburgLandauExact(nu, gamma, float(mu_c), float(mu2), int(n), complex(chi), complex(lam))
