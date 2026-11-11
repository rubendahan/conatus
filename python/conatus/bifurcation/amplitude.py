"""Amplitude equations derived from the IR: Landau (Stuart–Landau) and Ginzburg–Landau.

**Landau equation** (``amplitude_equation``): the centre-manifold normal form of
``normal_form`` rewritten for a declared *observable* amplitude A = ℓ(x − x0) (a linear
functional given by the user: e.g. the coefficient of cos kx of one field), so that the numbers do
not depend on the normalization of q:

    stationary (pitchfork):  dA/dt = μ Δp A + g3 A³,         g3 = a3 / ℓ(q)²,
    oscillatory (Hopf):      dZ/dt = λ' Δp Z + c1 |Z|² Z / |ℓ(q)|²,  x ≈ x0 + 2 Re(Z q e^{iωt}) / ℓ(q) ...

and predictions: A² = −μ Δp / g3 (steady amplitude), and any quadratic observable of the
bifurcated state from the second-order correction h2 (e.g. the Nusselt number of convection rolls:
Nu − 1 = −(a²/2) ∂zθ̄_{h2}(0)).

**Ginzburg–Landau** (``ginzburg_landau``, multiple scales; Newell & Whitehead 1969, Segel 1969;
Cross & Hohenberg 1993, §IV): for a pattern-forming instability at wavenumber k_c the envelope A of
u = A e^{i k_c x} q + c.c. obeys

    ∂_t A = μ Δp A − i σ_k ∂_X A... (group velocity) + D ∂_X² A + g |A|² A,
    D = −½ ∂²σ/∂k² (k_c),   v_g = −Im ∂σ/∂k (k_c),

with σ(k) the critical eigenvalue of the linearization on the domain of period 2π/k (computed on
systems rebuilt at k_c ± h, k_c ± 2h by the user's ``system_at``: fourth-order central differences,
declared step h, the two orders compared and reported), and g = 4 g3 for real stationary modes
(u = A e^{ikx} + c.c. = 2|A| cos kx). In the Cross–Hohenberg form τ0 ∂_t A = ε A + ξ0² ∂_X² A − g0
|A|² A with ε = Δp / p_c: τ0 = 1/(μ p_c), ξ0² = D τ0, g0 = −g τ0.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from ..errors import ConvergenceError, OptionError
from .equilibria import pencil_spectrum
from .normal_form import HopfNormalForm, ZeroEigenvalueNormalForm, normal_form
from .system import DynamicalSystem

__all__ = ["LandauEquation", "amplitude_equation", "GinzburgLandau", "ginzburg_landau",
           "cos_mode_observable"]


def cos_mode_observable(system: DynamicalSystem, name: str) -> Callable[[np.ndarray], complex]:
    """Complex-linear functional: coefficient of cos(2π x / L_x) of field ``name`` along the first
    (Fourier) axis, from its grid values (mean over the other axes' first grid index). A complex
    vector (a Hopf eigenvector) gives ℓ(Re v) + i ℓ(Im v). IR systems only."""
    names = [v.name for v in system.solver.system.variables]
    if name not in names:
        raise OptionError(f"cos_mode_observable: no variable {name!r}", fix=f"variables: {names}")
    k = names.index(name)

    def real_part(v):
        g = system.solver.plan.to_grid(name, system.solver.system.scatter(v)[k])
        xs = 2 * np.pi * np.arange(g.shape[0]) / g.shape[0]
        c = np.cos(xs).reshape((-1,) + (1,) * (g.ndim - 1))
        return float(2 * np.mean(g * c, axis=0).ravel()[0])

    def ell(v):
        if np.iscomplexobj(v):
            return complex(real_part(np.real(v)), real_part(np.imag(v)))
        return real_part(v)

    def wave(v):
        # (|content of e^{+i 2π x/L}|, |content of e^{-i 2π x/L}|) of a complex vector: tells the
        # eigenvalue of the e^{+ikx} mode from its conjugate (hunt 5D: v_g had the wrong sign)
        def grid(w):
            return system.solver.plan.to_grid(name, system.solver.system.scatter(w)[k])
        g = grid(np.real(v)) + 1j * grid(np.imag(v))
        xs = 2 * np.pi * np.arange(g.shape[0]) / g.shape[0]
        e = np.exp(1j * xs).reshape((-1,) + (1,) * (g.ndim - 1))
        return (abs(np.mean(g * e.conj(), axis=0).ravel()[0]),
                abs(np.mean(g * e, axis=0).ravel()[0]))
    ell.wave = wave
    return ell


#: An observable sees the critical mode q when |ℓ(q)| > OBSERVABLE_TOL x max_i |ℓ(e_i)| (its size on
#: unit vectors): below, ℓ(q) is rounding and g3 = c / ℓ(q)² meaningless (P2: division by 0).
OBSERVABLE_TOL = 1e-10


def _check_sees(observable, q, value: float, what: str) -> None:
    n = np.size(q)
    size = max(abs(complex(observable(np.eye(n)[i]))) for i in range(n))
    if not abs(value) > OBSERVABLE_TOL * size:
        raise OptionError(f"amplitude_equation: the observable does not see the critical mode q "
                          f"({what} = {abs(value):.3g}, observable size on unit vectors "
                          f"{size:.3g}, OBSERVABLE_TOL = {OBSERVABLE_TOL:g})",
                          fix="an observable with a nonzero component on the critical mode")


@dataclass
class LandauEquation:
    """Landau (Stuart–Landau) equation of an observable amplitude, from a normal form."""

    kind: str                       # "stationary" | "oscillatory"
    parameter: str
    p_c: float
    mu: complex                     # linear coefficient per unit Δp (λ' for Hopf)
    g3: complex                     # cubic coefficient on the observable amplitude
    observable_of_q: complex
    normal_form: object
    notes: list = dc_field(default_factory=list)

    def amplitude(self, dp: float) -> float:
        """Steady (or limit-cycle modulus) amplitude of the observable at p_c + dp."""
        if np.real(self.g3) == 0 or not np.isfinite(np.real(self.g3)):
            raise ConvergenceError(f"LandauEquation.amplitude: Re g3 = {np.real(self.g3):.3g}, so the "
                                   "cubic law A² = −Re μ Δp / Re g3 does not hold (a degenerate "
                                   "point: higher-order terms set the amplitude)",
                                   fix="a normal form of higher order; this one is degenerate")
        a2 = -np.real(self.mu) * dp / np.real(self.g3)
        return float(np.sqrt(a2)) if a2 >= 0 else float("nan")

    def explain(self) -> str:
        """The equation, its coefficients and the predicted amplitude law."""
        eq = ("dA/dt = μ Δ{p} A + g3 A³" if self.kind == "stationary"
              else "dZ/dt = λ' Δ{p} Z + g3 |Z|² Z").format(p=self.parameter)
        lines = [f"Landau equation ({self.kind}) at {self.parameter}_c = {self.p_c:.12g}: {eq}",
                 f"  μ = {complex(self.mu):.10g}, g3 = {complex(self.g3):.10g} "
                 f"(observable of q: {complex(self.observable_of_q):.6g})",
                 (f"  A² = −Re μ Δp / Re g3 = {-np.real(self.mu) / np.real(self.g3):.10g} "
                  f"Δ{self.parameter}" if np.real(self.g3) != 0 else
                  "  Re g3 = 0: no cubic amplitude law (degenerate)")]
        lines += ["  " + s for s in self.notes]
        return "\n".join(lines)


def amplitude_equation(system: DynamicalSystem, x0: np.ndarray, parameter: str, *,
                       kind: str, observable: Callable[[np.ndarray], complex],
                       par: Mapping[str, float] | None = None, **nf_kw) -> LandauEquation:
    """Landau equation for the observable amplitude A = observable(x − x0) (linear functional).

    ``kind``: "stationary" (zero eigenvalue; the quadratic coefficient must vanish, e.g. by a
    declared symmetry, else it is refused: a transcritical point has no cubic Landau law) or
    "oscillatory" (Hopf). Extra keywords go to :func:`normal_form` (symmetry, weight, omega...).
    """
    if kind not in ("stationary", "oscillatory"):
        raise OptionError("amplitude_equation: kind must be 'stationary' or 'oscillatory'")
    q_par = system._full(par)
    if kind == "stationary":
        nf: ZeroEigenvalueNormalForm = normal_form(system, x0, kind="zero", par=q_par,
                                                   parameter=parameter, **nf_kw)
        if nf.kind != "pitchfork":
            raise OptionError(f"amplitude_equation: the point is a {nf.kind}, not a pitchfork "
                              "(a2 ≠ 0 or β ≠ 0): no cubic Landau equation",
                              fix="declare the symmetry that makes a2 vanish, or use normal_form")
        if nf.degenerate or nf.a3 == 0 or nf.mu == 0:
            # P2: u' = p u − u⁵ (a3 = 0) built g3 = 0 and amplitude() divided by it
            raise ConvergenceError(f"amplitude_equation: the pitchfork is degenerate (μ = {nf.mu:.3g}, "
                                   f"a3 = {nf.a3:.3g}): the cubic Landau equation does not set the "
                                   "amplitude", fix="normal_form(...) reports the degeneracy; a "
                                   "higher-order expansion is needed")
        lq = complex(observable(nf.q))
        if lq.real == 0:
            # g3 = a3 / Re ℓ(q)²: an observable seeing q only through an imaginary part used to give
            # g3 = inf without a word
            raise OptionError("amplitude_equation: the observable vanishes on the critical mode q "
                              f"(ℓ(q) = {lq:.3g}: its real part is the amplitude of a real mode)",
                              fix="choose a real functional that sees the mode (e.g. its cos kx "
                                  "coefficient)")
        _check_sees(observable, nf.q, lq.real, "its real part")
        g3 = nf.a3 / np.real(lq) ** 2
        le = LandauEquation("stationary", parameter, q_par[parameter], nf.mu, g3, lq, nf)
        le.notes.append("from the pitchfork normal form: a3 / ℓ(q)², A = a ℓ(q)")
        return le
    nf: HopfNormalForm = normal_form(system, x0, kind="hopf", par=q_par, parameter=parameter, **nf_kw)
    lq = complex(observable(nf.q))
    _check_sees(observable, nf.q, abs(lq), "|ℓ(q)|")
    if not np.isfinite(nf.c1) or nf.dlambda == 0:
        raise ConvergenceError(f"amplitude_equation: degenerate Hopf point (c1 = {nf.c1:.3g}, "
                               f"λ' = {nf.dlambda:.3g}): no cubic Landau equation",
                               fix="normal_form(...) reports the degeneracy")
    g3 = nf.c1 / abs(lq) ** 2
    le = LandauEquation("oscillatory", parameter, q_par[parameter], nf.dlambda, g3, lq, nf)
    le.notes.append("from the Hopf normal form: Z ℓ(q) is the complex amplitude of the observable "
                    "(observable(x − x0) ≈ 2 Re(Z ℓ(q) e^{iωt}))")
    return le


@dataclass
class GinzburgLandau:
    """Ginzburg–Landau envelope coefficients at k_c (and their Cross–Hohenberg form)."""

    k_c: float
    p_c: float
    parameter: str
    mu: complex
    D: complex
    v_g: float
    g: complex
    h: float
    fd_gap: float                   # |D(order 4) − D(order 2)|
    landau: LandauEquation
    notes: list = dc_field(default_factory=list)

    @property
    def tau0(self) -> float:
        """Cross–Hohenberg time scale τ0 = 1/(μ p_c) (1/μ when p_c = 0, ε = Δp)."""
        if np.real(self.mu) == 0:
            raise ConvergenceError("GinzburgLandau.tau0: Re μ = 0 (degenerate unfolding): no time scale")
        return float(1.0 / (np.real(self.mu) * self.p_c)) if self.p_c else float(1.0 / np.real(self.mu))

    @property
    def xi0_sq(self) -> complex:
        """Cross–Hohenberg coherence length squared ξ0² = D τ0."""
        return self.D * self.tau0

    @property
    def g0(self) -> complex:
        """Cross–Hohenberg cubic coefficient g0 = −g τ0."""
        return -self.g * self.tau0

    def explain(self) -> str:
        """Coefficients, Cross–Hohenberg form and the finite-difference check of σ_kk."""
        return "\n".join([
            f"Ginzburg–Landau at k_c = {self.k_c:.12g}, {self.parameter}_c = {self.p_c:.12g}:",
            f"  ∂t A = μ Δ{self.parameter} A + v_g-term + D ∂X² A + g |A|² A, u = A e^(i k_c x) q + c.c.",
            f"  μ = {complex(self.mu):.10g}, D = −σ_kk/2 = {complex(self.D):.10g}, v_g = {self.v_g:.3g}, "
            f"g = {complex(self.g):.10g}",
            f"  Cross–Hohenberg form (ε = Δ{self.parameter}"
            + (f"/{self.parameter}_c" if self.p_c else "") + f"): τ0 = {self.tau0:.10g}, "
            f"ξ0² = {complex(self.xi0_sq):.10g}, g0 = {complex(self.g0):.10g}",
            f"  σ_kk: 4th-order central differences, h = {self.h:g} (declared); |4th − 2nd order| = "
            f"{self.fd_gap:.1e}"] + ["  " + s for s in self.notes])


def ginzburg_landau(system_at: Callable[[float], tuple[DynamicalSystem, np.ndarray]], k_c: float,
                    parameter: str, *, kind: str, observable: Callable[[np.ndarray], complex],
                    h: float, par: Mapping[str, float] | None = None, **nf_kw) -> GinzburgLandau:
    """Ginzburg–Landau coefficients of the instability at wavenumber k_c.

    ``system_at(k) -> (system, x0)``: the system discretized on the period 2π/k (same
    parameter values), and its base state. ``observable``: linear functional on the k_c system
    giving the amplitude of the e^{ik_c x} content (for a cos kx mode: its coefficient, then
    u = A e^{ikx} + c.c. means observable = 2|A|). ``h``: declared FD step in k.
    """
    sys_c, x0 = system_at(k_c)
    le = amplitude_equation(sys_c, x0, parameter, kind=kind, observable=observable, par=par, **nf_kw)
    lam_c, v_c = _critical_eig(sys_c, x0, par, None, wave=getattr(observable, "wave", None))
    wave_note = ""
    if kind == "oscillatory" and getattr(observable, "wave", None) is None:
        wave_note = ("the observable has no wave direction (cos_mode_observable has one): the "
                     "branch e^{+ik_c x} cannot be told from its conjugate, so v_g is left NaN")

    def sig(k):
        s, x = system_at(k)
        lam, _ = _critical_eig(s, x, par, (lam_c, v_c), k=k)
        return lam

    s = {j: sig(k_c + j * h) for j in (-2, -1, 1, 2)}
    s[0] = lam_c
    d2_4 = (-s[2] + 16 * s[1] - 30 * s[0] + 16 * s[-1] - s[-2]) / (12 * h * h)
    d2_2 = (s[1] - 2 * s[0] + s[-1]) / (h * h)
    d1_4 = (-s[2] + 8 * s[1] - 8 * s[-1] + s[-2]) / (12 * h)
    g = 4 * le.g3 if kind == "stationary" else le.g3
    v_g = float("nan") if wave_note else float(-np.imag(d1_4))
    gl = GinzburgLandau(k_c, le.p_c, parameter, le.mu, -0.5 * d2_4, v_g, g, h,
                        float(abs(d2_4 - d2_2) / 2), le)
    if wave_note:
        gl.notes.append(wave_note)
    gl.notes.append("g = 4 g3 (u = A e^{ikx} + c.c. = 2|A| cos kx)" if kind == "stationary"
                    else "g = c1 / |ℓ(q)|² (complex amplitude of the observable)")
    return gl


#: The critical mode at k_c ± j h is the eigenvector of largest overlap |v_c^H v| with the critical
#: eigenvector at k_c; below this overlap (or when another mode overlaps as much) the identity of
#: the mode is lost and ginzburg_landau refuses (G10: the nearest eigenvalue jumped to a slow
#: decoupled mode and gave D = 0.405 for 4).
GL_MODE_OVERLAP = 0.5


def _critical_eig(system, x0, par, ref, k=None, wave=None):
    """(λ, v) of the critical mode: the eigenvalue of smallest |Re λ| when ``ref`` is None (of a
    conjugate pair, the one whose eigenvector is a wave e^{+ikx} by ``wave(v)``), else the mode
    whose eigenvector overlaps most with ``ref`` = (λ_c, v_c)."""
    J = system.J(x0, system._full(par))
    spec = pencil_spectrum(J, system.M, vectors=True)
    ev, V = spec.values, spec.right
    if ev.size == 0:
        raise ConvergenceError("ginzburg_landau: no finite eigenvalue at the critical wavenumber"
                               if k is None else f"ginzburg_landau: no finite eigenvalue at k = {k:g}")
    V = V / np.maximum(np.linalg.norm(V, axis=0), np.finfo(float).tiny)
    if ref is None:
        i = int(np.argmin(np.abs(ev.real)))
        if wave is not None and ev[i].imag != 0:
            j = int(np.argmin(np.abs(ev - np.conj(ev[i]))))
            if j != i:
                ai, aj = wave(V[:, i]), wave(V[:, j])
                if aj[0] - aj[1] > ai[0] - ai[1]:
                    i = j
        return complex(ev[i]), V[:, i]
    lam_c, v_c = ref
    if V.shape[0] != v_c.size:
        raise OptionError(f"ginzburg_landau: the system at k = {k:g} has {V.shape[0]} unknowns, "
                          f"{v_c.size} at k_c: the critical mode cannot be followed by its "
                          "eigenvector", fix="system_at(k) with the same unknowns for every k")
    ov = np.abs(v_c.conj() @ V)
    order = np.argsort(-ov)
    i = int(order[0])
    second = float(ov[order[1]]) if ov.size > 1 else 0.0
    if ov[i] < GL_MODE_OVERLAP or second >= ov[i] and abs(ev[order[1]] - ev[i]) > 0:
        raise ConvergenceError(
            f"ginzburg_landau: the critical mode is not identified at k = {k:g} (overlap "
            f"{ov[i]:.3f} with the critical eigenvector, next {second:.3f}; GL_MODE_OVERLAP = "
            f"{GL_MODE_OVERLAP:g})", fix="a smaller FD step h")
    return complex(ev[i]), V[:, i]
