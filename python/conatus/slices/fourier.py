"""First Fourier mode slice: reduction of a continuous translation symmetry.
Notes: Budanur, Cvitanović, Davidchack & Siminos, PRL 114,
084102 (2015), arXiv:1405.1096.

Group: translations τ_ℓ f(x) = f(x + ℓ) along one periodic RealFourier direction of period L,
acting on the solver vector through :class:`conatus.ecs.FourierOps` (1D) or
:class:`conatus.ecs.fourier_ops.AxisOps` (any dimension). Generator T = ∂x.

Template p: a state containing **only the first wavenumber** k₁ = 2π/L along the symmetry
direction (checked: τ_ℓ p = cos(k₁ℓ) p + sin(k₁ℓ) T p / k₁). Then for any state X,

    ⟨τ_{−ℓ} X, p⟩ = a cos k₁ℓ + b sin k₁ℓ,   a = ⟨X, p⟩,  b = ⟨X, T p⟩ / k₁,

and the slice representative is X̂ = τ_{−ℓ(X)} X with k₁ ℓ(X) = atan2(b, a): it satisfies the slice
condition ⟨X̂, T p⟩ = 0 and lies in the half-hyperplane ⟨X̂, p⟩ = √(a² + b²) > 0. For the default
1D template (p = the cos₁ coefficient) a + ib = c₁ + i s₁ is the complex first Fourier coefficient
and k₁ℓ its phase: X̂ has a real positive first mode (the first Fourier mode slice).

Slice border (singularity): a² + b² = 0 (the first mode vanishes), where the phase is undefined.
It is **detected and reported**, never regularized: ``border_distance(X) = √(a²+b²)/‖X‖`` and
every reduced trajectory lists the samples below the declared ``border_tol`` and the largest
phase jump between samples (a large jump means the trajectory passed close to the border and the
unwrapped phase may be ambiguous).

Reduced dynamics (in-slice velocity) and reconstruction equation: with v the full vector field
(equivariant, v(τX) = τ v(X)),

    dX̂/dt = v(X̂) − ℓ̇ T X̂,      ℓ̇ = ⟨v(X̂), T p⟩ / ⟨T X̂, T p⟩,

and X(t) = τ_{ℓ(t)} X̂(t), ℓ(t) = ℓ(0) + ∫ ℓ̇ dt (the phase is reconstructed by quadrature and
compared with the unwrapped phase of the projected trajectory).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError, SymmetryError
from .._repr import compact_repr

__all__ = ["FirstFourierModeSlice", "SliceTrajectory", "RotationOps"]


class RotationOps:
    """SO(2) acting on consecutive coordinate pairs (x_j, y_j) by rotations of angle m_j θ
    (integer ``charges`` m_j), for ODE models such as the two-mode system of Budanur,
    Borrero-Echeverry & Cvitanović (arXiv:1411.3303, eq. 24: charges (1, 2)). Same interface as FourierOps:
    ``translate(X, θ)`` (counter-clockwise rotation of pair j by m_j θ), ``ddx`` (generator T,
    T_j = [[0, −m_j], [m_j, 0]]) and ``k1`` = 1 (the group angle plays the role of k₁ℓ)."""

    def __init__(self, charges) -> None:
        self.charges = np.asarray(charges, float)
        self.k1 = 1.0
        self.n = 2 * len(self.charges)

    def translate(self, X, theta: float) -> np.ndarray:
        X = np.asarray(X, float)
        x, y = X[0::2], X[1::2]
        c, s = np.cos(self.charges * theta), np.sin(self.charges * theta)
        out = np.empty_like(X)
        out[0::2], out[1::2] = c * x - s * y, s * x + c * y
        return out

    def ddx(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        out = np.empty_like(X)
        out[0::2], out[1::2] = -self.charges * X[1::2], self.charges * X[0::2]
        return out

    def explain(self) -> str:
        return f"RotationOps: SO(2) on {len(self.charges)} pairs with charges {self.charges.tolist()}"


@dataclass
class SliceTrajectory:
    """A trajectory in the slice: reduced states, unwrapped shift ℓ(t), border evidence."""
    __repr__ = compact_repr   # one line, arrays by shape
    t: np.ndarray
    X_hat: np.ndarray
    ell: np.ndarray                    # unwrapped shift (length units): X = τ_ℓ X̂
    border_distance: np.ndarray        # √(a² + b²)/‖X‖ at every sample
    border_hits: np.ndarray            # indices with border_distance < border_tol
    max_phase_jump: float              # max |Δ(k₁ℓ)| between consecutive samples (radians)
    border_tol: float
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"SliceTrajectory: {len(self.t)} samples, net shift ℓ(T) − ℓ(0) = "
                 f"{self.ell[-1] - self.ell[0]:.10g}; min border distance "
                 f"{self.border_distance.min():.3e} (border_tol {self.border_tol:g}, "
                 f"{len(self.border_hits)} samples below); max phase jump between samples "
                 f"{self.max_phase_jump:.3f} rad"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


class FirstFourierModeSlice:
    """Slice of the translation group through a first-Fourier-mode template (see module doc).

    ``ops``: object with ``translate(X, ℓ)``, ``ddx(X)`` and ``k`` (wavenumbers of the pairs),
    e.g. ``FourierOps(solver)``; ``template``: solver vector holding only the first wavenumber
    (default: the cos₁ coefficient of the first variable, 1D FourierOps only); ``border_tol``:
    declared threshold on √(a² + b²)/‖X‖ below which a state is reported on the slice border."""

    def __init__(self, ops, *, border_tol: float, template: np.ndarray | None = None) -> None:
        if not border_tol > 0:
            raise OptionError("FirstFourierModeSlice: border_tol must be > 0 (declared threshold)")
        self.ops, self.border_tol = ops, float(border_tol)
        self.k1 = float(getattr(ops, "k1", None) or ops.k[1])
        self.L = 2 * np.pi / self.k1
        if template is None:
            if isinstance(ops, RotationOps):
                template = np.zeros(ops.n)
                template[0] = 1.0              # (1, 0, 0, ...): Budanur et al. eq. (11)
                self.template_desc = "(1, 0, ..., 0) (first pair, default)"
            elif not hasattr(ops, "mean_index"):
                raise OptionError("FirstFourierModeSlice: give a template for AxisOps (no default)")
            else:
                c = [np.zeros(ops.basis.size) for _ in ops.system.variables]
                c[0][2] = 1.0                   # (cos_0, msin_0, cos_1, ...): index 2 = cos_1
                template = ops.system.gather_state(c, check_subspace=False)
                self.template_desc = "cos₁ coefficient of the first variable (default)"
        else:
            self.template_desc = "user template"
        p = np.asarray(template, float)
        p = p / np.linalg.norm(p)
        Tp = ops.ddx(p)
        ell = 0.3719 * self.L                   # generic shift for the single-mode check
        err = np.linalg.norm(ops.translate(p, ell) - (np.cos(self.k1 * ell) * p
                                                      + np.sin(self.k1 * ell) * Tp / self.k1))
        if err > 1e-10:
            raise SymmetryError(f"FirstFourierModeSlice: the template is not a pure first-mode "
                                f"state (single-mode identity violated by {err:.1e})",
                                fix="use a template holding only the wavenumber 2π/L")
        self.p, self.Tp = p, Tp
        self.tTp = Tp / self.k1              # unit-normalised slice tangent (‖T p‖ = k₁ ‖p‖)

    # ---------------------------------------------------------------- projections
    def _ab(self, X):
        X = np.asarray(X, float)
        return float(X @ self.p), float(X @ self.tTp)

    def angle(self, X) -> float:
        """Group angle k₁ ℓ(X) ∈ (−π, π] of the state (undefined on the border)."""
        a, b = self._ab(X)
        return float(np.arctan2(b, a))

    def shift(self, X) -> float:
        """ℓ(X) ∈ (−L/2, L/2]: X = τ_ℓ X̂."""
        return self.angle(X) / self.k1

    def border_distance(self, X) -> float:
        a, b = self._ab(X)
        return float(np.hypot(a, b) / max(np.linalg.norm(X), 1e-300))

    def on_border(self, X) -> bool:
        return self.border_distance(X) < self.border_tol

    def reduce(self, X) -> tuple[np.ndarray, float]:
        """(X̂, ℓ) with X = τ_ℓ X̂; refuses a state on the border (phase undefined)."""
        d = self.border_distance(X)
        if d < self.border_tol:
            raise SymmetryError(f"state on the slice border: √(a²+b²)/‖X‖ = {d:.2e} < border_tol "
                                f"= {self.border_tol:g} (first Fourier mode ~ 0, phase undefined)",
                                fix="use another template (e.g. a higher mode) or treat the state "
                                    "in its invariant subspace")
        ell = self.shift(X)
        return self.ops.translate(np.asarray(X, float), -ell), ell

    def reconstruct(self, X_hat, ell: float) -> np.ndarray:
        """Full-space state τ_ℓ X̂."""
        return self.ops.translate(np.asarray(X_hat, float), ell)

    def slice_condition(self, X_hat) -> float:
        """⟨X̂, T p⟩/(k₁‖X̂‖): zero on the slice (to rounding)."""
        return float(np.asarray(X_hat) @ self.tTp / max(np.linalg.norm(X_hat), 1e-300))

    def velocity(self, X_hat, v) -> tuple[np.ndarray, float]:
        """In-slice velocity v̂ = v(X̂) − ℓ̇ T X̂ and phase velocity ℓ̇ (``v``: full vector field)."""
        X_hat = np.asarray(X_hat, float)
        vX = v(X_hat)
        TX = self.ops.ddx(X_hat)
        den = float(TX @ self.Tp)
        if abs(den) < self.border_tol * self.k1 ** 2 * np.linalg.norm(X_hat):
            raise SymmetryError("in-slice velocity on the slice border: ⟨T X̂, T p⟩ ~ 0 (ℓ̇ undefined)")
        elld = float(vX @ self.Tp) / den
        return vX - elld * TX, elld

    # ---------------------------------------------------------------- trajectories
    def reduce_trajectory(self, t, X, *, v=None) -> SliceTrajectory:
        """Project samples X (m, n) at times t; the phase is unwrapped between samples. If the
        vector field ``v`` is given, ℓ is also reconstructed by trapezoidal quadrature of ℓ̇ and
        the largest difference with the unwrapped projection is reported (notes)."""
        t = np.asarray(t, float)
        X = np.asarray(X, float)
        ang = np.array([self.angle(x) for x in X])
        dist = np.array([self.border_distance(x) for x in X])
        un = np.unwrap(ang)
        jumps = np.abs(np.diff(ang))
        jumps = np.minimum(jumps, 2 * np.pi - jumps)
        ell = un / self.k1
        Xh = np.array([self.ops.translate(x, -e) for x, e in zip(X, ell, strict=True)])
        st = SliceTrajectory(t, Xh, ell, dist, np.flatnonzero(dist < self.border_tol),
                             float(jumps.max(initial=0.0)), self.border_tol)
        st.notes.append(f"template: {self.template_desc}; phase unwrapped between samples "
                        "(ambiguous if the max phase jump approaches π)")
        if v is not None and len(t) > 1:
            if len(st.border_hits):
                st.notes.append("phase quadrature skipped: the trajectory touches the border")
            else:
                ld = np.array([self.velocity(x, v)[1] for x in Xh])
                quad = ell[0] + np.concatenate([[0.0], np.cumsum(0.5 * (ld[1:] + ld[:-1]) * np.diff(t))])
                st.notes.append(f"reconstruction ℓ(0) + ∫ℓ̇ dt (trapezoid) vs unwrapped projection: "
                                f"max difference {np.max(np.abs(quad - ell)):.3e}")
                st.ell_dot = ld
        return st

    def explain(self) -> str:
        return (f"FirstFourierModeSlice (Budanur et al. 2015): translations of period L = {self.L:.10g}, "
                f"k₁ = {self.k1:.10g}; template: {self.template_desc}; slice ⟨X̂, T p⟩ = 0, "
                f"⟨X̂, p⟩ > 0; border (first mode ~ 0) reported below border_tol = {self.border_tol:g}")
