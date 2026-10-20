"""Transient growth and Floquet exponents of any linear problem about a time-periodic base flow.

The user writes one linear IR problem per wave (kx, kz) in which the base flow appears through
non-constant-coefficient (NCC) fields, and gives the base flow by its time harmonics,

    U(y, t) = Σ_m U_m(y) e^{2πimt/P}       ,

exactly as for :func:`conatus.floquet.hill_from_plan`. The plan is compiled with U := 0, U_m and
2 U_m (linearity and mass independence checked, :func:`conatus.floquet.harmonic_matrices`), which
gives the descriptor system

    M X' + L0 X = -B(t) X,      B(t) = Σ_m e^{2πimt/P} B_m,      B_m = L(U := U_m) - L(U := 0).

Two analyses share this operator (:class:`PeriodicOperator`):

* optimal energy gain G(t0, T) = max ‖u(t0 + T)‖² / ‖u(t0)‖² over admissible u(t0): the
  propagator of the IMEX Runge–Kutta scheme (L0 implicit, every base-flow term explicit at the
  exact stage times, like an IVPSolver with the base flow on the right-hand side), the exact
  energy Gram matrix of the declared fields, the SVD on the admissible inputs;
* Floquet exponents: the Hill operator of the same matrices (continuous in time), or the
  eigenvalues of the discrete monodromy matrix Φ(t0, P) on the admissible states.

"Admissible" = the states for which the algebraic rows of the system (rows without mass:
continuity, boundary conditions) hold for some value of the non-state unknowns (pressure, taus,
gauges): the discrete Leray projection with the boundary conditions of this discretization, as
:class:`conatus.adjoint.nlop.AdmissibleProjector` for the IVP. Mathematics:

"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import OptionError
from ..floquet.hill_general import HillBlocks, harmonic_matrices, hill_from_blocks
from ..tolerances import NCC_TAIL_TOL, REQUIRED, require
from .propagator import GainResult, gain, propagator
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["PeriodicOperator", "periodic_operator", "periodic_operator_from_plan", "OptimalGain",
           "optimal_gain", "operator_gain", "FloquetExponents", "floquet_exponents",
           "operator_floquet"]

_DOC = "docs/site/guide/dynamics/nonmodal.md"

LEAK_MAX = 1e-8
"""Default bound on the admissible leak of the monodromy matrix, ||Phi Z - Z (Z* Phi Z)|| / ||Phi Z||
(method='monodromy'): the same bound as the residual filter of the Hill method (residual_max)."""

#: Folding of Im mu into (-pi/P, pi/P]: an exponent within FOLD_ROUNDING x (2 pi / P) of the lower
#: end is on it to the rounding of the mod (a few eps of w), and moves up by one period; nothing
#: else is moved.
FOLD_ROUNDING = 1e3 * 2.220446049250313e-16


# ---------------------------------------------------------------------------- the operator
@dataclass
class PeriodicOperator:
    """Discrete operator M X' + L0 X = -Σ_m e^{2πimt/P} B_m X of one wave, and its state layout.

    ``state_cols``: columns of the prognostic unknowns (those with a time derivative), in plan
    order; ``other_cols``: pressure, taus, gauges. ``W``: exact energy Gram matrix on the state
    columns (weights of the declared fields, zero for the other prognostic fields, which is
    refused when it matters, see :meth:`energy_factor`)."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    M: np.ndarray
    L0: np.ndarray
    harmonics: dict[int, np.ndarray]
    cols: list
    state_cols: np.ndarray
    other_cols: np.ndarray
    W: np.ndarray
    period: float
    fields: dict[str, float]
    label: str
    rank_tol: float
    notes: list[str] = field(default_factory=list)
    _Z: np.ndarray | None = field(default=None, repr=False)
    _rank: str = field(default="", repr=False)

    @property
    def n(self) -> int:
        return self.M.shape[0]

    @property
    def state_names(self) -> list[str]:
        out: list[str] = []
        for j in self.state_cols:
            nm = self.cols[j][0]
            if nm not in out:
                out.append(nm)
        return out

    def B(self, t: float) -> np.ndarray:
        """B(t) = Σ_m e^{2πimt/P} B_m."""
        out = np.zeros_like(self.L0)
        for m, Bm in self.harmonics.items():
            out = out + np.exp(2j * np.pi * m * t / self.period) * Bm
        return out

    def admissible_basis(self) -> np.ndarray:
        """Z (state coordinates, orthonormal columns) spanning the admissible states.

        Algebraic rows C = [C_s, C_o] (state / non-state columns); s is admissible iff
        C_s s ∈ range(C_o), i.e. (I - Q_o Q_oᴴ) C_s s = 0 with Q_o an orthonormal basis of
        range(C_o). Both ranks are decided with the declared relative tolerance ``rank_tol``
        (reported by :meth:`explain`)."""
        if self._Z is not None:
            return self._Z
        alg = np.nonzero(~np.any(self.M != 0, axis=1))[0]
        Cs = self.L0[np.ix_(alg, self.state_cols)]
        Co = self.L0[np.ix_(alg, self.other_cols)]
        if Co.size:
            U, s, _ = np.linalg.svd(Co, full_matrices=False)
            r = int(np.sum(s > self.rank_tol * s[0])) if s.size and s[0] > 0 else 0
            Qo = U[:, :r]
            A = Cs - Qo @ (Qo.conj().T @ Cs)
        else:
            A = Cs
        _, s, Vh = np.linalg.svd(A, full_matrices=True)
        smax = s[0] if s.size else 0.0
        r = int(np.sum(s > self.rank_tol * smax)) if smax > 0 else 0
        Z = Vh[r:].conj().T
        kept = s[r - 1] / smax if r else float("nan")
        dropped = s[r] / smax if r < s.size else 0.0
        self._rank = (f"admissible states: {Z.shape[1]} of {len(self.state_cols)} state "
                      f"coefficients ({len(alg)} algebraic rows, constraint rank {r}; smallest kept "
                      f"singular value {kept:.1e}, largest dropped {dropped:.1e}, relative; "
                      f"rank_tol = {self.rank_tol:g})")
        self._Z = Z
        return Z

    def energy_factor(self) -> np.ndarray:
        """F with ‖F x‖² = xᴴ W x: the Hermitian square root W^½ = V Λ^½ Vᴴ of the exact Gram W.

        Being a function of W, it is block-diagonal whenever W is (row j of F belongs to the
        component of state column j). Until this returned Λ^½ Vᴴ (rows ordered by
        eigenvalue, mixing components when blocks of W are equal): same norms and gains (a
        unitary factor on the left), but rows that could not be read per component."""
        lam, V = np.linalg.eigh(self.W)
        if lam.min() < -1e-12 * max(lam.max(), 1e-300):
            raise RuntimeError("energy Gram matrix is not positive semi-definite")
        return (V * np.sqrt(np.clip(lam, 0.0, None))[None, :]) @ V.conj().T

    def hill(self, Nh: int) -> HillBlocks:
        """Hill operator of the same matrices: (L0 + B_0 + (μ + i n ω) M) x_n + Σ_{m≠0} B_m x_{n-m}."""
        return hill_from_blocks(self.M, self.L0, self.harmonics, Nh=Nh,
                                omega=2 * math.pi / self.period, label=self.label,
                                note="Blocks of the periodic operator (growth.periodic): the same "
                                     "M, L0 and base-flow harmonics B_m as the optimal gain.")

    def explain(self) -> str:
        if not self._rank:
            self.admissible_basis()
        ms = sorted(self.harmonics)
        w = ", ".join(f"{k}: {v:g}" for k, v in self.fields.items())
        return (f"Periodic linear operator ({self.label}): system size {self.n}, state "
                f"{'+'.join(self.state_names)} ({len(self.state_cols)} coefficients), base-flow "
                f"harmonics m in {ms}, period P = {self.period:g}. M X' + L0 X = -B(t) X, "
                f"B(t) = sum_m e^(2 pi i m t/P) B_m, B_m = L(U := U_m) - L(U := 0) (same plan, "
                f"linearity in U and mass independence checked). Energy E = 1/2 sum_f w_f "
                f"int |f|^2 (exact Gram matrix; weights {{{w}}}). {self._rank}. "
                + " ".join(self.notes))


def _zero_like(f) -> Callable:
    return lambda *X: 0.0 * X[0]


def periodic_operator_from_plan(make_plan: Callable[[Mapping[Any, Any]], Any],
                                base: Mapping[Any, Mapping[int, Any]], *,
                                fields: Mapping[str, float] | tuple[str, ...], period: float,
                                group: Any = None, label: str = "periodic base flow",
                                linearity_tol: float = 1e-12,
                                rank_tol: float = 1e-10,
                                steady: bool = False) -> PeriodicOperator:
    """:class:`PeriodicOperator` of the plan ``make_plan(ncc)`` (ncc = base-flow data).

    ``base[field][m]``: harmonic U_m of each base field (callable on the grid, like any NCC).
    ``fields``: energy fields, a tuple of unknown names (weight 1) or a mapping name -> weight
    (E = 1/2 Σ w_f ∫ |f|²); every prognostic unknown that is not listed gets weight 0, which
    :meth:`PeriodicOperator.energy_factor` users see refused if it leaves admissible states
    without energy. ``group``: the group of the plan (Fourier wavenumbers), None for a 1D plan.
    """
    from ..linearize.inner_product import InnerProduct
    if not (isinstance(period, (int, float)) and period > 0 and math.isfinite(period)):
        raise OptionError(f"period = {period!r}", why="the base flow is periodic in time",
                          fix="pass period > 0", doc=_DOC)
    if isinstance(fields, str):
        fields = (fields,)
    weights = dict(fields) if isinstance(fields, Mapping) else {f: 1.0 for f in fields}
    if not weights or any(not (w > 0) for w in weights.values()):
        raise OptionError(f"fields = {fields!r}", why="the energy needs at least one field, "
                          "each with a weight > 0", fix='fields=("u", "v", "w")', doc=_DOC)
    if steady and base:
        raise OptionError("steady=True with base-flow harmonics", doc=_DOC)
    if not base and not steady:
        raise OptionError("no base-flow harmonics", why="the base flow U(y, t) defines the "
                          "problem", fix="base_harmonics={'U': {1: U_p, -1: U_m}}", doc=_DOC)
    M, L0, B, cols = harmonic_matrices(make_plan, base, group=group, linearity_tol=linearity_tol)
    plan = make_plan({f: _zero_like(f) for f in base})
    names = [c[0] for c in cols]
    var_names = list(dict.fromkeys(names))
    unknown = sorted(set(weights) - set(var_names))
    if unknown:
        raise OptionError(f"energy fields {unknown} are not unknowns of the problem",
                          fix="unknowns: " + ", ".join(var_names), doc=_DOC)
    massive = np.any(M != 0, axis=0)
    prog = [v for v in var_names
            if any(massive[j] for j, c in enumerate(cols) if c[0] == v and c[2] != "tau")]
    not_prog = sorted(set(weights) - set(prog))
    if not_prog:
        raise OptionError(f"energy fields {not_prog} have no time derivative",
                          why="the energy is a norm of the state, which is the set of unknowns "
                              "with dt()", fix="list velocity components (and other evolved "
                              "fields), never the pressure", doc=_DOC)
    state = np.array([j for j, c in enumerate(cols) if c[0] in prog and c[2] != "tau"], dtype=int)
    other = np.setdiff1d(np.arange(len(cols)), state)
    for m, Bm in B.items():
        if np.abs(Bm[:, other]).max(initial=0.0) > 0:
            bad = sorted({cols[j][0] for j in other if np.any(Bm[:, j] != 0)})
            raise OptionError(f"the base flow multiplies {bad} (harmonic m = {m})",
                              why="the propagator acts on the evolved state; pressure, taus and "
                                  "gauges at t0 are not part of it",
                              fix="write the base-flow terms on the evolved fields only", doc=_DOC)
        alg = ~np.any(M != 0, axis=1)
        if np.abs(Bm[alg]).max(initial=0.0) > 0:
            raise OptionError(f"the base flow enters an algebraic row (continuity or a boundary "
                              f"condition, harmonic m = {m})",
                              why="time-dependent constraints change the admissible set in time",
                              fix="put the base flow in the evolution equations only", doc=_DOC)
    taus = {t.name for t in plan.taus}
    allw = {v.name: weights.get(v.name, 0.0) for v in plan.variables if v.name not in taus}
    Wfull = InnerProduct(plan, allw).gram(group).toarray()
    W = Wfull[np.ix_(state, state)]
    notes = []
    zero_w = [v for v in prog if v not in weights]
    if zero_w:
        notes.append(f"Evolved fields outside the energy (weight 0): {', '.join(zero_w)}.")
    if steady:
        notes.append("Steady base: it is part of L0, implicit in the propagator (B = 0).")
    else:
        notes.append("Base-flow terms (every harmonic, m = 0 included) are explicit in the "
                     "propagator; L0 = L(U := 0) is implicit.")
    return PeriodicOperator(M, L0, B, list(cols), state, other, W, float(period), weights, label,
                            float(rank_tol), notes)


def _base_by_name(problem, base_harmonics: Mapping[Any, Mapping[int, Any]]) -> dict:
    """{Field: harmonics} from names (or fields) of the base-flow NCC of ``problem``."""
    free = {f.name: f for eq in problem.equations for side in (eq.lhs, eq.rhs)
            for f in side.free_fields()}
    out = {}
    for k, h in base_harmonics.items():
        nm = getattr(k, "name", k)
        if nm not in free:
            raise OptionError(f"base field {nm!r} is not used by the problem",
                              fix="known (NCC) fields of the problem: "
                                  + (", ".join(sorted(free)) or "none"), doc=_DOC)
        if not isinstance(h, Mapping) or not h:
            raise OptionError(f"base_harmonics[{nm!r}] = {h!r}",
                              why="each base field is given by its harmonics {m: U_m}",
                              fix=f"base_harmonics={{{nm!r}: {{1: U_p, -1: U_m}}}}", doc=_DOC)
        bad = [m for m in h if not (isinstance(m, (int, np.integer)) and not isinstance(m, bool))]
        if bad:
            raise OptionError(f"harmonic indices {bad} of {nm!r} are not integers", doc=_DOC)
        out[free[nm]] = {int(m): v for m, v in h.items()}
    return out


def periodic_operator(build: Callable[[Mapping[str, float]], Any],
                      base_harmonics: Mapping[Any, Mapping[int, Any]], kx: float, kz: float, *,
                      fields, period: float = 1.0, linearity_tol: float = 1e-12,
                      rank_tol: float = 1e-10, ncc_cutoff: float | None = None,
                      ncc_tail_tol: float = NCC_TAIL_TOL) -> PeriodicOperator:
    """:class:`PeriodicOperator` of ``build({"kx": kx, "kz": kz})``, a linear IR problem of one wave
    (IVP with dt(), or EVP in the convention (L + σM) x = 0), or a ``(problem, ncc)`` pair whose
    ``ncc`` holds the other (time-independent) coefficient data. ``base_harmonics`` maps each
    base-flow NCC field (name or Field) to its harmonics {m: U_m(y)} of e^{2πimt/period}.
    ``ncc_cutoff`` / ``ncc_tail_tol``: acceptance of the non-constant coefficients, as in
    ``discretize`` (an under-resolved one is refused)."""
    from ..discretize import discretize
    if not callable(build):
        raise OptionError("build must be a callable params -> problem",
                          fix='def build(p): ... uses p["kx"], p["kz"] ...; return problem', doc=_DOC)
    for nm, k in (("kx", kx), ("kz", kz)):
        if not (isinstance(k, (int, float)) and math.isfinite(k)):
            raise OptionError(f"{nm} = {k!r} is not a finite real wavenumber", doc=_DOC)
    out = build({"kx": float(kx), "kz": float(kz)})
    problem, static = (out if isinstance(out, tuple) else (out, {}))
    if not hasattr(problem, "equations"):
        raise OptionError(f"build returned {type(problem).__name__}, not an IR problem",
                          fix="return the IVP/EVP (or (problem, ncc))", doc=_DOC)
    base = _base_by_name(problem, base_harmonics)
    clash = sorted(f.name for f in static if f in base)
    if clash:
        raise OptionError(f"{clash} given both as static NCC data and as base-flow harmonics",
                          doc=_DOC)

    def make_plan(ncc):
        return discretize(problem, ncc={**static, **ncc}, ncc_cutoff=ncc_cutoff,
                          ncc_tail_tol=ncc_tail_tol)
    op = periodic_operator_from_plan(make_plan, base, fields=fields, period=period,
                                     label=f"{problem.name}, kx = {kx:g}, kz = {kz:g}",
                                     linearity_tol=linearity_tol, rank_tol=rank_tol)
    return op


# ---------------------------------------------------------------------------- optimal gain
@dataclass
class OptimalGain:
    """Optimal energy gain over [t0, t0 + T] and its structures (state coordinates)."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    G: float
    gains: np.ndarray            # σ_i² in decreasing order (the k largest kept)
    seed: np.ndarray             # optimal initial state (state coefficients, unit energy)
    response: np.ndarray         # its image at t0 + T
    t0: float
    T: float
    nsteps: int
    scheme: str
    backend: str
    operator: PeriodicOperator
    seconds: float

    def field(self, name: str, which: str = "seed") -> np.ndarray:
        """Coefficients of one unknown in the seed (``which="seed"``) or the response."""
        x = self.seed if which == "seed" else self.response
        sel = [i for i, j in enumerate(self.operator.state_cols)
               if self.operator.cols[j][0] == name]
        if not sel:
            raise OptionError(f"{name!r} is not an evolved unknown",
                              fix="unknowns: " + ", ".join(self.operator.state_names), doc=_DOC)
        return x[sel]

    def explain(self) -> str:
        op = self.operator
        return (f"Optimal energy gain G = {self.G:.10g} over t in [{self.t0:g}, {self.t0 + self.T:g}] "
                f"({self.scheme}, {self.nsteps} steps, dt = T/steps = {self.T / self.nsteps:.6g} "
                f"exactly, backend {self.backend}, {self.seconds:.1f} s); next gains "
                + ", ".join(f"{g:.6g}" for g in self.gains[1:4])
                + ". G = largest squared singular value of F Phi Z R^-1 (F: energy factor, Z: "
                "admissible inputs, R: energy-orthonormalisation), i.e. max E(t0 + T)/E(t0) over "
                "admissible initial states. The discrete gain converges in dt at the order of the "
                "scheme on the algebraic part (observed 2-3 for RK443): report a convergence "
                "study.\n" + op.explain())


def check_gain_args(t0: Any, T: Any, steps: Any, backend: Any) -> None:
    """Times, step count and backend of an optimal gain, checked before the operator is built
    (review of the inputs audit: ``isinstance(T, (int, float))`` took ``T=True`` as
    a horizon of 1 and refused ``np.float32`` and ``Fraction`` times)."""
    from ..errors import check_real
    for nm, v in (("t0", t0), ("T", T)):
        if isinstance(v, (bool, np.bool_)):
            raise OptionError(f"{nm} = {v!r} is not a finite time", doc=_DOC)
        try:
            check_real(v, nm, "optimal_gain")
        except OptionError:
            raise OptionError(f"{nm} = {v!r} is not a finite time", doc=_DOC) from None
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise OptionError(f"steps = {steps!r}", why="dt = T / steps exactly: an integer >= 1",
                          fix="a whole number of time steps over the horizon, e.g. steps=400",
                          doc=_DOC)
    if not float(T) > 0:
        raise OptionError(f"T = {T!r}", why="the horizon must be > 0", doc=_DOC)
    if backend not in ("cpu", "cuda"):
        from ..timestep.integrator import backend_hint
        raise OptionError(f"backend = {backend!r}", fix="'cpu' or 'cuda'." + backend_hint(backend),
                          doc=_DOC)


@renames("conatus.growth.periodic:operator_gain")
def operator_gain(op: PeriodicOperator, t0: float, T: float, steps: int, scheme: str,
                  backend: str = "cpu", *, n_modes: int = 4) -> OptimalGain:
    """Optimal gain of a :class:`PeriodicOperator` (see :func:`optimal_gain`); ``n_modes``: the
    number of leading gains kept in ``.gains`` (``k=``: deprecated former name)."""
    check_gain_args(t0, T, steps, backend)
    t = time.perf_counter()
    try:
        Phi = propagator(op, float(t0), float(T), int(steps), scheme, backend)
    except ValueError as e:
        raise OptionError(str(e), doc=_DOC) from None
    Z = op.admissible_basis()
    F = op.energy_factor()
    FZ = F @ Z
    sv = np.linalg.svd(FZ, compute_uv=False)
    if sv.size == 0 or sv[-1] <= 1e-12 * sv[0]:
        raise OptionError("some admissible states carry no energy",
                          why="an evolved field is outside the energy (weight 0) and not "
                              "constrained by the others, so the gain is unbounded",
                          fix="give every evolved field a weight in fields=, or remove it from "
                              "the problem", doc=_DOC)
    r: GainResult = gain(Phi, F, Z)
    seed = r.v_opt / np.linalg.norm(F @ r.v_opt)
    resp = Phi @ seed
    return OptimalGain(r.G, (r.sigmas[:n_modes] ** 2), seed, resp, float(t0), float(T), int(steps),
                       scheme, backend, op, time.perf_counter() - t)


@renames("conatus.growth.periodic:optimal_gain")
def optimal_gain(build: Callable[[Mapping[str, float]], Any],
                 base_harmonics: Mapping[Any, Mapping[int, Any]], kx: float, kz: float,
                 t0: float, T: float, steps: int, scheme: str, backend: str = "cpu", *,
                 fields, period: float = 1.0, rank_tol: float = 1e-10,
                 ncc_cutoff: float | None = None, ncc_tail_tol: float = NCC_TAIL_TOL) -> OptimalGain:
    """Optimal energy gain G(t0, T) of one wave (kx, kz) about a time-periodic base flow.

    ``build(params)`` returns the linear IR problem of the wave (``params["kx"]``,
    ``params["kz"]``): the base flow enters through NCC fields, on the left-hand side, linearly;
    ``base_harmonics`` maps each of them (by name) to {m: U_m(y)} with U = Σ_m U_m e^{2πimt/period}.
    ``fields``: the energy, e.g. ``("u", "v", "w")`` for E = ½∫|u|² dy (exact Gram matrix).
    ``scheme``: IMEX Runge–Kutta (RK111, RK222, RK443); dt = T / steps exactly. Every argument
    that changes the number is explicit; ``explain()`` restates them with the admissible-set
    rank decision.
    """
    op = periodic_operator(build, base_harmonics, kx, kz, fields=fields, period=period,
                           rank_tol=rank_tol, ncc_cutoff=ncc_cutoff, ncc_tail_tol=ncc_tail_tol)
    return operator_gain(op, t0, T, steps, scheme, backend)


# ---------------------------------------------------------------------------- Floquet
@dataclass
class FloquetExponents:
    """Floquet exponents μ (solutions ∝ e^{μt}, Im μ folded into (-ω/2, ω/2], ω = 2π/P), sorted by
    decreasing real part, with their relative residuals."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    mu: np.ndarray
    residuals: np.ndarray
    method: str
    period: float
    details: dict
    # Hill truncation of each returned exponent: relative
    # weight of the outermost harmonics of its eigenvector (None: monodromy), and the warning
    # sentence when the leading one exceeds conatus.tolerances.HILL_TAIL_TOL ("" otherwise)
    harmonic_tails: np.ndarray | None = None
    truncation_note: str = ""

    @property
    def multipliers(self) -> np.ndarray:
        return np.exp(self.mu * self.period)

    @property
    def leading(self) -> complex:
        return complex(self.mu[0])

    def explain(self) -> str:
        top = ", ".join(f"{m.real:+.10f}{m.imag:+.10f}i (res {r:.1e})"
                        for m, r in zip(self.mu[:4], self.residuals[:4], strict=True))
        d = "; ".join(f"{k} = {v}" for k, v in self.details.items())
        out = (f"Floquet exponents ({self.method}), solutions proportional to e^(mu t), period "
               f"{self.period:g}, Im mu folded into (-pi/P, pi/P]: {top}. {d}")
        if self.harmonic_tails is not None and self.harmonic_tails.size:
            from ..tolerances import HILL_TAIL_TOL
            out += ("\n  harmonic tail (weight of the outermost harmonics) of each exponent above: "
                    + ", ".join(f"{t:.1e}" for t in self.harmonic_tails[:4])
                    + f" (truncated in Nh above HILL_TAIL_TOL = {HILL_TAIL_TOL:g})")
        if self.truncation_note:
            out += "\n  WARNING: " + self.truncation_note
        return out


def _fold(mu: np.ndarray, period: float) -> np.ndarray:
    w = 2 * np.pi / period
    im = np.mod(mu.imag + w / 2, w) - w / 2
    # the interval is (-w/2, w/2]: an exponent on its lower end (to the rounding of the mod)
    # moves by one period, keeping its offset (np.isclose's default
    # rtol = 1e-5 replaced every Im mu within 1e-5 w/2 of -w/2 by w/2 exactly)
    im = np.where(np.abs(im + w / 2) <= FOLD_ROUNDING * w, im + w, im)
    return mu.real + 1j * im


def _hill_residual(H, mu: complex, x: np.ndarray) -> float:
    """Relative residual of a Hill eigenpair of the mu-linear pencil M(mu) = M(0) + mu B (B = the
    mass block on every harmonic): ||M(mu) x|| / (||M(0) x|| + omega ||B x||).

    the former denominator ||M(0) x|| vanishes with mu, since
    M(0) x = -mu B x for an eigenpair, so the ratio grew like 1/|mu| near a neutral point
    and the leading exponent was dropped
    by residual_max = 1e-8 exactly where the neutral curve is decided. The frequency omega of the
    harmonics (the scale of the exponents: the diagonal blocks carry (mu + i n omega) B) keeps the
    denominator away from 0; for |mu| >~ omega it is the former ratio within a factor 2."""
    num = np.linalg.norm(H.matvec(mu, x))
    den = np.linalg.norm(H.matvec(0.0, x)) + abs(H.omega) * np.linalg.norm(x @ H.mass.T)
    return float(num / (den + 1e-300))


@renames("conatus.growth.periodic:operator_floquet")
def operator_floquet(op: PeriodicOperator, *, method: str, Nh: int | None = None,
               targets: tuple | list | None = None, n_modes: int = 6, tol: float = REQUIRED,
               steps: int | None = None, scheme: str | None = None, t0: float = 0.0,
               backend: str = "cpu", residual_max: float = 1e-8,
               leak_max: float = LEAK_MAX) -> FloquetExponents:
    """Floquet exponents of a :class:`PeriodicOperator` (see :func:`floquet`). ``leak_max``
    (monodromy): largest accepted ||Phi Z - Z (Z* Phi Z)|| / ||Phi Z||, the part of the monodromy
    that leaves the admissible states (refused above, :data:`LEAK_MAX`). ``tol`` (hill): the
    Arnoldi stopping tolerance, no default."""
    if method == "hill":
        if Nh is None or targets is None:
            raise OptionError("method='hill' needs Nh (harmonics -Nh..Nh) and targets (shifts)",
                              fix="floquet_exponents(..., method='hill', Nh=200, targets=[0.0, 2.2j], tol=1e-12)",
                              doc=_DOC)
        require(tol, "tol", where="floquet(method='hill')", suggest="tol=1e-12",
                why="tol is the Arnoldi (ARPACK) stopping tolerance of the shift-invert Hill "
                    "eigenproblem: it sets the accuracy of the returned exponents (each one is "
                    f"then checked against residual_max = {residual_max:g}); 1e-12 converges "
                    "them to near machine precision, 1e-8 is a faster survey")
        from scipy.sparse.linalg import LinearOperator, eigs

        from ..timestep.integrator import check_steps
        # an integer, never truncated (Nh=200.7 ran with 200)
        H = op.hill(check_steps(Nh, "floquet(method='hill')", "Nh"))
        K, m = len(H.harmonics), H.size
        harm = np.asarray(H.harmonics, dtype=int)
        found = []
        for ti, tgt in enumerate(targets):
            Fct = H.factor(complex(tgt))

            def mv(x, Fct=Fct):
                return Fct.solve(x.reshape(K, m) @ H.mass.T).ravel()
            th, vecs = eigs(LinearOperator((K * m, K * m), matvec=mv, dtype=complex), k=n_modes,
                            which="LM", tol=tol)
            for j, tj in enumerate(th):
                mu = complex(tgt) - 1 / tj
                x = vecs[:, j].reshape(K, m)
                res = _hill_residual(H, mu, x)
                wts = np.linalg.norm(x, axis=1)
                edge = float((wts[:2].sum() + wts[-2:].sum()) / max(wts.sum(), 1e-300))
                found.append((mu, res, edge, ti, x, int(harm[int(np.argmax(wts))])))
        mus = _fold(np.array([f[0] for f in found]), op.period)
        res = np.array([f[1] for f in found])
        edge = np.array([f[2] for f in found])
        keep = res < residual_max
        if not keep.any():
            raise OptionError(f"no Floquet exponent with a residual below {residual_max:g} "
                              f"(smallest {res.min():.1e})",
                              why="the eigenpairs near the targets are not converged",
                              fix="raise n_modes or tol, change the targets, or check Nh", doc=_DOC)
        # the same exponent is found from several targets, and as mu + i n omega (one Hill
        # family) from one target; after folding they coincide (mu[0] and mu[1]
        # were the same exponent). Multiplicity of a folded value = largest number of equal
        # *unfolded* values from one target (a true degeneracy, e.g. the waves +-kz of one
        # real-Fourier group, is kept), the best residuals are kept.
        tid = np.array([f[3] for f in found])
        raw = np.array([f[0] for f in found])
        centre = np.array([f[5] for f in found])
        close = lambda a, b: abs(a - b) <= 1e-8 * max(1.0, abs(b))  # noqa: E731

        def shifted_copy(i: int, j: int) -> bool:
            # in a truncated Hill matrix the copy mu + i n omega
            # of an exponent is an eigenpair whose harmonics are those of mu shifted by n; the
            # window -Nh..Nh then cuts it asymmetrically, so the two folded values differ by the
            # truncation error in Nh and were both
            # returned, the shifted copy first. Copy = same exponent within 1e-3 and the
            # eigenvectors equal up to the harmonic shift (|cos| > 0.99).
            d = int(centre[j] - centre[i])
            if d == 0 or abs(mus[i] - mus[j]) > 1e-3 * max(1.0, abs(mus[i])):
                return False
            xi, xj = found[i][4], found[j][4]
            a, b = (xi[:K - d], xj[d:]) if d > 0 else (xi[-d:], xj[:K + d])
            c = abs(np.vdot(a, b)) / max(np.linalg.norm(xi) * np.linalg.norm(xj), 1e-300)
            return c > 0.99
        # representatives: most centred in the harmonic window first (best resolved), then residual
        clusters: list[list[int]] = []
        for i in sorted((int(j) for j in np.flatnonzero(keep)), key=lambda j: (abs(centre[j]), res[j])):
            for cl in clusters:
                if close(mus[i], mus[cl[0]]) or shifted_copy(cl[0], i):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        picked, spread = [], 0.0
        for cl in clusters:
            mult = max(sum(1 for j in cl if tid[j] == t and close(raw[j], raw[i]))
                       for t in set(tid[cl]) for i in cl if tid[i] == t)
            picked += cl[:mult]                            # most centred, then best residual
            spread = max([spread] + [float(abs(mus[j] - mus[cl[0]])) for j in cl])
        merged = int(keep.sum()) - len(picked)
        pk = np.array(picked, dtype=int)
        order = pk[np.argsort(-mus[pk].real, kind="stable")]
        # Y10: the harmonic tail of the returned modes is judged, not only
        # reported: at Nh = 2 30 % of the leading mode sat in the outer
        # harmonics and its exponent was right to 6e-6 by luck (Nh = 20: 3e-5; the error is not
        # monotone in Nh until the tail is small)
        from ..tolerances import HILL_TAIL_TOL
        tails = edge[order]
        note = ""
        if tails.size and tails[0] > HILL_TAIL_TOL:
            note = (f"leading exponent mu = {complex(mus[order][0]):.10g} truncated in Nh: "
                    f"{100 * tails[0]:.2g} % of its eigenvector is in the outermost harmonics "
                    f"(Nh = {Nh}; above HILL_TAIL_TOL = {HILL_TAIL_TOL:g}), so its digits are not "
                    "converged; raise Nh until this weight falls below the tolerance (doubling Nh "
                    "and comparing is not enough at small Nh)")
            from ..errors import ResolutionWarning, _warn_user
            _warn_user(note, ResolutionWarning)
        return FloquetExponents(mus[order], res[order], "hill", op.period,
                                {"Nh": Nh, "targets": list(targets), "n_modes per target": n_modes,
                                 f"dropped (residual >= {residual_max:.0e})": int((~keep).sum()),
                                 "copies merged (same exponent from several targets or harmonic shifts)": merged,
                                 "largest gap between merged copies (truncation in Nh)": spread,
                                 "max weight of the 4 outer harmonics": float(edge[keep].max())
                                 if keep.any() else float("nan"),
                                 "Hill": H.explain()}, tails, note)
    if method == "monodromy":
        if steps is None or scheme is None:
            raise OptionError("method='monodromy' needs steps (time steps per period) and scheme "
                              "(the time stepper of the monodromy matrix Phi(t0, P))",
                              fix="operator_floquet(op, method='monodromy', steps=400, "
                                  "scheme='RK443')", doc=_DOC)
        from ..timestep.integrator import check_steps
        steps = check_steps(steps, "floquet(method='monodromy')", "steps")   # never truncated
        Phi = propagator(op, float(t0), op.period, steps, scheme, backend)
        Z = op.admissible_basis()
        PZ = Phi @ Z
        A = Z.conj().T @ PZ
        leak = float(np.linalg.norm(PZ - Z @ A) / max(np.linalg.norm(PZ), 1e-300))
        if not (leak <= leak_max):           # NaN refused too
            raise OptionError(
                f"the monodromy matrix leaves the admissible states: leak {leak:.1e} > "
                f"leak_max = {leak_max:g}",
                why="every step imposes the algebraic rows (continuity, boundary conditions), so "
                    "Phi(t0, P) maps admissible states onto admissible states up to rounding "
                    "(measured ~1e-15); a larger leak means the eigenvalues of Z* Phi Z are not "
                    "those of the propagator (admissible basis wrong for this operator, e.g. "
                    "rank_tol, or constraints that change in time)",
                fix="check op.admissible_basis() (op.explain() gives the constraint rank and the "
                    "smallest kept singular value; adjust rank_tol), or use method='hill'; pass "
                    "leak_max explicitly only after that check", doc=_DOC)
        lam, V = np.linalg.eig(A)
        res = np.linalg.norm(A @ V - V * lam, axis=0) / max(np.linalg.norm(A, 2), 1e-300)
        nz = np.abs(lam) > 0
        mus = _fold(np.log(lam[nz].astype(complex)) / op.period, op.period)
        order = np.argsort(-mus.real)[:n_modes]
        return FloquetExponents(mus[order], res[nz][order], "monodromy", op.period,
                                {"steps": steps, "scheme": scheme, "t0": t0,
                                 "admissible leak of Phi": f"{leak:.1e} (refused above {leak_max:g})",
                                 "residual": "relative to ||A||_2 of the monodromy on Z"})
    raise OptionError(f"method = {method!r}", fix="method='hill' (continuous in time, Nh "
                      "harmonics) or method='monodromy' (eigenvalues of the discrete Phi(t0, P))",
                      doc=_DOC)


def floquet_exponents(build: Callable[[Mapping[str, float]], Any],
            base_harmonics: Mapping[Any, Mapping[int, Any]], kx: float, kz: float, *,
            method: str, fields, period: float = 1.0, **kw) -> FloquetExponents:
    """Floquet exponents of the wave (kx, kz) with the same ``build`` and ``base_harmonics`` as
    :func:`optimal_gain`. ``method="hill"`` (Nh, targets: shift-invert Arnoldi on the μ-linear Hill
    pencil, exponents nearest each target, residual-filtered at ``residual_max``) or
    ``"monodromy"`` (steps, scheme: eigenvalues of the discrete monodromy matrix on the
    admissible states). ``fields`` only defines the state (energy is not used)."""
    nccd = {k: kw.pop(k) for k in ("ncc_cutoff", "ncc_tail_tol") if k in kw}
    op = periodic_operator(build, base_harmonics, kx, kz, fields=fields, period=period, **nccd)
    return operator_floquet(op, method=method, **kw)
