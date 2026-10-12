"""Parametric pencils: assemble once, evaluate L(p), M(p) for any parameter values.

When a Parameter enters the equations as a Laurent polynomial (p, p**2, 1/p, 1/(a*Re), ...),
the discrete matrices are Laurent polynomials in it:

    L(p) = sum_e (p/p0)^e C_e ,   M(p) = sum_e (p/p0)^e D_e .

The exponent set of each parameter is read off the IR (``laurent_exponents``), never guessed.
The coefficient matrices are recovered exactly from plans built at the roots of unity
p = p0 exp(2 i pi k/m) (a discrete Fourier transform, perfectly conditioned), then checked
against a directly assembled plan at an independent real point: the reconstruction error is
reported and a mismatch is refused. A parameter entering non-polynomially is refused with a
message (rebuild the plan per value instead).

``ParametricEVP`` then evaluates spectra, tracks one eigenpair by bordered Newton, and gives
exact matrix derivatives d/dp for sensitivities. ``critical`` finds a neutral parameter value
(growth = 0) and optionally minimizes it over other parameters (Re_c, alpha_c).
"""
from __future__ import annotations

import itertools
import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import scipy.linalg as sl
import scipy.optimize as so

from .._deprecation import renames
from ..discretize.layout import GroupItem
from ..discretize.param_keys import problem_parameter_names
from ..discretize.plan import discretize
from ..errors import ConvergenceError, OptionError, ParameterError, did_you_mean
from ..ir import expr as E
from ..ir.problem import EVP, ProblemError
from ..tolerances import NCC_TAIL_TOL, REQUIRED, require
from .evp import EVPResult, solve_dense
from .plan_evp import pencil_rank

__all__ = ["laurent_exponents", "ParametricEVP", "TrackedMode", "CriticalPoint", "critical"]

RECON_TOL = 1e-10
# Declared defaults (listed in conatus.tolerances.DECLARED, shown by explain()):
TRACK_TOL = 1e-13               # bordered-Newton stop |d sigma| <= TRACK_TOL max(|sigma|, 1): at the
                                # attainable floor, the residual of the tracked pair is returned
TRACK_MAX_ITERATIONS = 30       # iteration cap of track(): reaching it raises ConvergenceError
MULTIPLICITY_TOL = 1e-8         # singular values <= this x scale: multiple eigenvalue (declared note)
MODE_SWITCH_TOL = 1e-8          # critical(): another mode more unstable by more than this: refused
CRITICAL_MAX_ITERATIONS = 50    # critical(): Newton cap on the neutral parameter (raises when hit)
BRACKET_SCAN_POINTS = 8         # critical(): interior points scanned to name sub-brackets of a refused bracket


def laurent_exponents(x: E.Expr, p: E.Parameter) -> frozenset[int] | None:
    """Exponents of ``p`` in ``x`` seen as a Laurent polynomial in p; None if it is not one."""
    if isinstance(x, E.Parameter):
        return frozenset({1}) if x == p else frozenset({0})
    if not x.args:
        return frozenset({0})
    kids = [laurent_exponents(a, p) for a in x.args]
    if any(k is None for k in kids):
        return None
    if isinstance(x, E.Add):
        return frozenset().union(*kids)
    if isinstance(x, E.Power):
        (b,) = kids
        if b == frozenset({0}):
            return b
        e = x.exponent
        if not (isinstance(e, numbers.Integral) or (isinstance(e, float) and e.is_integer())):
            return None
        e = int(e)
        if e < 0:
            if len(b) != 1:
                return None          # 1/(p + 1) is not a Laurent polynomial in p
            return frozenset({next(iter(b)) * e})
        out = frozenset({0})
        for _ in range(e):
            out = frozenset(i + j for i in out for j in b)
        return out
    if isinstance(x, E.UnaryFunction):          # exp(p), sin(p)...: not a Laurent polynomial
        return kids[0] if kids[0] == frozenset({0}) else None
    if isinstance(x, E.Conj) and kids[0] != frozenset({0}) and np.iscomplexobj(np.zeros(1, p.dtype)):
        return None
    # products and multilinear operators: Minkowski sum; unary linear operators: the child's set
    out = frozenset({0})
    for k in kids:
        out = frozenset(i + j for i in out for j in k)
    return out


@dataclass
class TrackedMode:
    """One eigenpair (L + sigma M) x = 0 followed by bordered Newton; y is the left vector."""

    sigma: complex
    x: np.ndarray
    y: np.ndarray | None
    residual: float
    iterations: int
    values: dict
    note: str = ""
    # resolution of the mode: its spectral tail (None: not
    # measured: drift test set, no tau, spurious='none') and the warning sentence above
    # conatus.tolerances.UNRESOLVED_TAIL_TOL ("" otherwise)
    tail: float | None = None
    unresolved_note: str = ""

    def __repr__(self) -> str:
        vals = ", ".join(f"{k}={float(np.real(v)):.10g}" for k, v in self.values.items())
        return (f"TrackedMode(sigma={self.sigma:.10g} at {vals}, residual {self.residual:.1e}, "
                f"{self.x.size} coefficients"
                + (", UNDER-RESOLVED" if self.unresolved_note else "") + ")")

    def explain(self) -> str:
        return (f"TrackedMode sigma = {self.sigma:.12e} at {self.values}: residual {self.residual:.1e} "
                f"after {self.iterations} bordered-Newton iterations"
                + ("" if self.y is None else "; left vector available (sensitivities)")
                + ("" if self.tail is None else f"; spectral tail {self.tail:.1e}")
                + (f"\n  WARNING: {self.unresolved_note}" if self.unresolved_note else "")
                + (f"\n  {self.note}" if self.note else ""))


GROWTH = {"real": lambda s: s.real, "imag": lambda s: s.imag}

#: Growth of a mode read off the name of its eigenvalue when the model does not declare it
#:: a growth rate exp(s t) grows with Re s; a phase speed c of
#: exp(i alpha (x - c t)) grows with Im c (alpha > 0). Any other name must be declared
#: (``m.eigenvalue("omega", growth="imag")`` or ``critical(..., growth=...)``): never guessed.
GROWTH_BY_NAME = {"s": "real", "sigma": "real", "lam": "real", "lmbda": "real",
                  "lambda_": "real", "c": "imag"}
_GROWTH_WHY = {"real": "growth rate, mode exp({e} t): growth = Re {e}",
               "imag": "phase speed, mode exp(i alpha (x - {e} t)): growth = Im {e}"}


def _growth(g) -> Callable[[complex], float]:
    if isinstance(g, str):
        if g not in GROWTH:
            raise OptionError(f"growth={g!r}", fix="growth='real' (exp(s t)), 'imag' (phase "
                              "speed, exp(i alpha (x - c t))) or a callable of sigma"
                              + did_you_mean(g, GROWTH))
        return GROWTH[g]
    if not callable(g):
        raise OptionError(f"growth={g!r}", fix="growth='real', 'imag' or a callable of sigma")
    return g


def growth_convention(name: str | None, declared: Any = None,
                      why: str | None = None) -> tuple[Any, str]:
    """(growth, sentence) for an eigenvalue ``name``: the declared one first (``why``: the
    sentence when the construction fixes it, e.g. a linearization), else ``GROWTH_BY_NAME``
    (a v0.1 guess: ``m.eigenvalue`` warns); (None, reason) when neither decides."""
    if declared is not None and why:
        return declared, why
    if declared is not None:
        what = ({"real": f"Re {name}", "imag": f"Im {name}"}.get(declared, declared)
                if isinstance(declared, str) else f"growth({name}), a callable")
        return declared, f"growth = {what}, declared with the eigenvalue {name}"
    g = GROWTH_BY_NAME.get(name or "")
    if g is None:
        return None, f"eigenvalue {name!r}: its growth convention is not declared"
    return g, (_GROWTH_WHY[g].format(e=name) + f" (inferred from the name {name!r}, "
               "conatus.solvers.parametric.GROWTH_BY_NAME)")


class ParametricEVP:
    """Pencil L(p) + sigma M(p) of one group, assembled once for all values of ``params``.

    ``derived``: parameter-only expressions that are not Laurent polynomials (e.g. ``sqrt(Ta)``)
    declared as independent pencil coordinates g = f(params): each is replaced by an internal
    parameter, L and M are Laurent polynomials in it, and its value (and d g / d p, by exact
    symbolic differentiation) is computed from the user's parameters. Undeclared non-Laurent
    dependence is refused, as before.

    ``wavenumbers``: {coordinate name: Parameter} makes the wavenumber of a Fourier direction a
    continuous pencil parameter (no rediscretization for a sweep in k). The group matrices are
    polynomials in k of degree = the differential order along that direction; they are built at
    synthetic wavenumbers with the index structure of the chosen group (default: the first group
    with k > 0 along these directions), so the domain length along them plays no role (declared).

    ``spurious``: ``"tail"`` (default): :meth:`least_stable` and :func:`critical`
    select only modes that pass the spectral-tail test (``conatus.solvers.spurious``; spurious tau
    modes and unresolved modes fail it); ``"none"`` switches it off.
    """

    def __init__(self, problem: EVP, params: Sequence[E.Parameter], *,
                 ncc: Mapping[E.Field, Any] | None = None, group=None, reduced: bool | None = None,
                 fixed: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
                 ncc_tail_tol: float = NCC_TAIL_TOL,
                 derived: Sequence[E.Expr] = (), wavenumbers: Mapping[str, E.Parameter] | None = None,
                 spurious: str = "tail") -> None:
        if not isinstance(problem, EVP):
            raise TypeError("ParametricEVP needs an EVP")
        problem.check()
        self.source = problem
        self.wavenumbers = dict(wavenumbers or {})
        self.params = tuple(params) + tuple(p for p in self.wavenumbers.values() if p not in params)
        self.ncc, self.fixed, self.ncc_cutoff = ncc, dict(fixed or {}), ncc_cutoff
        # growth convention of the eigenvalue (declared on the problem by the native layer,
        # else read off its name); declared values of the parameters (native layer only:
        # Model.parametric fills it with the model's values); optional drift pencil (set_drift)
        self.growth, self.growth_note = growth_convention(
            problem.eigenvalue.name if problem.eigenvalue is not None else None,
            getattr(problem, "growth", None), getattr(problem, "growth_why", None))
        self.declared: dict[str, Any] = {}
        self.drift: tuple | None = None
        # spectral-tail spurious-mode flag of least_stable / critical: "tail"
        # (default) or "none"; set by m.parametric(spurious=) / lin.parametric(spurious=)
        from .spurious import check_spurious
        self.spurious = check_spurious(spurious, "ParametricEVP")
        self.ncc_tail_tol = ncc_tail_tol                  # as discretize(ncc_tail_tol=)
        self._group, self.reduced = group, reduced
        for p in self.params:
            if p == problem.eigenvalue:
                raise ProblemError("the eigenvalue is not a pencil parameter")
            if p.value is None or p.value == 0:
                raise ProblemError(f"parameter {p.name} needs a nonzero nominal value (it sets the scale)")
        # declared functions of parameters -> internal pencil coordinates g_i
        self.derived: dict[str, E.Expr] = {}
        subst = {}
        for d in derived:
            d = E.as_expr(d)
            if d.free_fields() or not d.free_parameters() <= set(self.params):
                raise ProblemError(f"derived '{d}' must depend on the pencil parameters only")
            g = E.Parameter(f"[{d}]", _eval_param_expr(d, {q.name: q.value for q in self.params}))
            if g.value == 0:
                raise ProblemError(f"derived '{d}' vanishes at the nominal values (it sets the scale)")
            self.derived[g.name] = d
            subst[d] = g
        self.problem = _substituted(problem, subst) if subst else problem
        # parameters still present after the substitution (a pencil parameter used only through a
        # derived coordinate, e.g. Ta in sqrt(Ta), is not a key discretize accepts)
        self._pnames = problem_parameter_names(self.problem)
        self._g = tuple(subst.values())
        kparams = set(self.wavenumbers.values())
        self._pen = tuple(p for p in self.params if p not in kparams) + self._g
        self.p0 = {p.name: p.value for p in self.params + self._g}
        self.exponents: dict[str, tuple[int, ...]] = {}
        for p in self._pen:
            ex = frozenset()
            for eq in self.problem.equations:
                e = laurent_exponents(eq.lhs, p)
                if e is None:
                    raise ProblemError(
                        f"parameter {p.name} does not enter equation '{eq}' as a Laurent polynomial: "
                        "no parametric pencil (declare the offending function of parameters with "
                        "derived=[...], or rebuild the plan per value)")
                ex |= e
            self.exponents[p.name] = tuple(range(min(ex), max(ex) + 1))
        if self.wavenumbers:
            plan = discretize(self.problem, ncc=self.ncc, params={**self.fixed, **{
                q.name: q.value for q in self._pen if q.name in self._pnames}},
                ncc_cutoff=self.ncc_cutoff, ncc_tail_tol=self.ncc_tail_tol)
            fnames = [d.coord.name for d in plan.layout.fdirs]
            unknown = [n for n in self.wavenumbers if n not in fnames]
            if unknown:
                raise ProblemError(f"wavenumbers: {unknown} are not Fourier directions ({fnames})")
            self._kaxis = {n: fnames.index(n) for n in self.wavenumbers}
            degs = plan._fourier_degrees()
            for n, q in self.wavenumbers.items():
                self.exponents[q.name] = tuple(range(0, degs[self._kaxis[n]] + 1))
        self._pen = self._pen + tuple(self.wavenumbers.values())
        self._build()

    # ------------------------------------------------------------------ assembly
    def _plan_mats(self, values: Mapping[str, Any]):
        kn = {q.name for q in self.wavenumbers.values()}
        plan = discretize(self.problem, ncc=self.ncc,
                          params={**self.fixed, **{k: v for k, v in values.items()
                                                   if k not in kn and k in self._pnames}},
                          ncc_cutoff=self.ncc_cutoff, ncc_tail_tol=self.ncc_tail_tol)
        group = self._group
        if group is None:
            if self.wavenumbers:     # first group with k > 0 along every wavenumber direction
                axes = list(self._kaxis.values())
                group = next(g for g in plan.groups if all(g[a].k > 0 for a in axes))
            else:
                group = plan.groups[0]
        elif callable(group):
            group = next(g for g in plan.groups if group(g))
        if self.wavenumbers:
            # synthetic group: same index structure, wavenumber = the parameter value (complex at
            # the roots of unity; plan._synthetic casts to float and would drop the imaginary part)
            group = list(group)
            for n, q in self.wavenumbers.items():
                it = group[self._kaxis[n]]
                group[self._kaxis[n]] = GroupItem(it.kind, it.index, values[q.name], it.indices)
            group = tuple(group)
        self._group_used = group
        gm = plan.matrices(group, self.reduced)
        if not hasattr(self, "_layout_of"):
            # layout of the pencil's vectors for the spectral-tail test
            self._layout_of = (plan, group, bool(getattr(gm, "reduced", False)))
        return gm

    def _build(self) -> None:
        names = [p.name for p in self._pen]
        ms = [len(self.exponents[n]) for n in names]
        grids = [np.exp(2j * np.pi * np.arange(m) / m) for m in ms]
        Ls = Ms = None
        self.n_plans = 0
        for idx in itertools.product(*(range(m) for m in ms)):
            vals = {n: self.p0[n] * grids[i][k] for i, (n, k) in enumerate(zip(names, idx, strict=True))}
            gm = self._plan_mats(vals)
            self.n_plans += 1
            if Ls is None:
                self.gm0 = gm
                n = gm.shape
                Ls = np.zeros(tuple(ms) + n, complex)
                Ms = np.zeros(tuple(ms) + n, complex)
            Ls[idx] = gm.L.toarray()
            Ms[idx] = gm.M.toarray()
        # DFT along each parameter axis: C_e = (1/m) sum_k L(t_k) t_k^{-e}
        for ax, (n, m) in enumerate(zip(names, ms, strict=True)):
            e = np.array(self.exponents[n])
            F = np.exp(-2j * np.pi * np.outer(e, np.arange(m)) / m) / m
            Ls = np.moveaxis(np.tensordot(F, np.moveaxis(Ls, ax, 0), axes=(1, 0)), 0, ax)
            Ms = np.moveaxis(np.tensordot(F, np.moveaxis(Ms, ax, 0), axes=(1, 0)), 0, ax)
        self._coef = {}
        scale = max(np.abs(Ls).max(), np.abs(Ms).max(), 1e-300)
        dropped = 0
        for idx in itertools.product(*(range(m) for m in ms)):
            ex = tuple(self.exponents[n][i] for n, i in zip(names, idx, strict=True))
            C, D = Ls[idx], Ms[idx]
            if max(np.abs(C).max(), np.abs(D).max()) <= 1e-13 * scale:
                dropped += 1
                continue
            self._coef[ex] = (C, D)
        # real pencil: real matrices at real parameter values (wavenumbers make RealFourier real)
        self.real = not (np.iscomplexobj(self.gm0.L.data) or np.iscomplexobj(self.gm0.M.data))
        # independent real check point, never a sampling point
        check = {p.name: p.value * (1.0 + 0.137 * (i + 1)) for i, p in enumerate(self.params)}
        gm = self._plan_mats(self._internal(check))
        L, M = self.matrices(**check)
        err = max(np.abs(L - gm.L.toarray()).max(), np.abs(M - gm.M.toarray()).max()) / scale
        if err > RECON_TOL:
            raise ProblemError(f"parametric pencil reconstruction error {err:.1e} > {RECON_TOL:.0e}")
        self.recon_error, self.n_terms, self.n_dropped = err, len(self._coef), dropped
        self.rank_check = pencil_rank(gm)

    # ------------------------------------------------------------------ evaluation
    def _internal(self, values: Mapping[str, Any]) -> dict:
        """User values -> values of every pencil coordinate (derived ones computed)."""
        missing = [p.name for p in self.params if p.name not in values]
        if missing:
            raise ParameterError(
                f"values missing for parameters {missing} (no default is assumed)",
                where="ParametricEVP call with " + (", ".join(f"{k}=" for k in values) or "no values"),
                why="the pencil parameters are declared symbols; a nominal value is never "
                    "substituted in silence",
                fix="pass every pencil parameter by name, e.g. "
                    + ", ".join(f"{p.name}=..." for p in self.params)
                    + did_you_mean(next(iter(set(values) - {p.name for p in self.params}), ""),
                                   [p.name for p in self.params]))
        out = {p.name: values[p.name] for p in self.params}
        for g, d in self.derived.items():
            out[g] = _eval_param_expr(d, out)
        return out

    def _t(self, values: Mapping[str, Any]) -> dict:
        v = self._internal(values)
        return {p.name: v[p.name] / self.p0[p.name] for p in self._pen}

    def matrices(self, **values) -> tuple[np.ndarray, np.ndarray]:
        """Dense L, M at the given parameter values (all parameters must be given)."""
        t = self._t(values)
        names = [p.name for p in self._pen]
        L = np.zeros(self.gm0.shape, complex)
        M = np.zeros(self.gm0.shape, complex)
        for ex, (C, D) in self._coef.items():
            w = np.prod([t[n] ** e for n, e in zip(names, ex, strict=True)])
            L += w * C
            M += w * D
        if self.real and all(np.isrealobj(v) for v in t.values()):
            return L.real, M.real
        return L, M

    def _partial(self, j: int, t: dict) -> tuple[np.ndarray, np.ndarray]:
        """Exact d/dq of L, M for the j-th pencil coordinate q (term by term)."""
        names = [p.name for p in self._pen]
        dL = np.zeros(self.gm0.shape, complex)
        dM = np.zeros(self.gm0.shape, complex)
        for ex, (C, D) in self._coef.items():
            if ex[j] == 0:
                continue
            # d/dp (p/p0)^e = e (p/p0)^(e-1) / p0: finite at p = 0 for e >= 1 (the former
            # (p/p0)^e * e / p gave 0/0 = NaN there; gallery)
            w = np.prod([t[n] ** (e - 1 if k == j else e) for k, (n, e) in enumerate(zip(names, ex, strict=True))])
            w = w * ex[j] / self.p0[names[j]]
            dL += w * C
            dM += w * D
        return dL, dM

    def derivative(self, name: str, **values) -> tuple[np.ndarray, np.ndarray]:
        """Exact dL/dp, dM/dp (term by term; chain rule through the derived coordinates, whose
        derivatives are symbolic), no finite differences."""
        if name not in {p.name for p in self.params}:
            raise ParameterError(f"unknown parameter {name!r}",
                                 why="not a parameter of this pencil: "
                                     + ", ".join(p.name for p in self.params),
                                 fix="use one of the pencil parameters."
                                     + did_you_mean(name, [p.name for p in self.params]))
        t = self._t(values)
        pen = [p.name for p in self._pen]
        dL = np.zeros(self.gm0.shape, complex)
        dM = np.zeros(self.gm0.shape, complex)
        if name in pen:
            a, b = self._partial(pen.index(name), t)
            dL, dM = dL + a, dM + b
        if self.derived:
            v = self._internal(values)
            p = next(q for q in self.params if q.name == name)
            for g, d in self.derived.items():
                dg = _eval_param_expr(_dparam(d, p), v)
                if dg != 0:
                    a, b = self._partial(pen.index(g), t)
                    dL, dM = dL + dg * a, dM + dg * b
        return dL, dM

    def solve(self, *, left: bool = False, **values) -> EVPResult:
        """All eigenvalues (dense QZ) at the given values."""
        L, M = self.matrices(**values)
        r = solve_dense(L, M, left=left, label=f"{self.gm0.label} at {values}")
        r.log.insert(0, self.rank_check)
        return r

    @renames("conatus.solvers.parametric:ParametricEVP.track")
    def track(self, sigma: complex, x: np.ndarray, *, left: bool = True, tol: float = TRACK_TOL,
              max_iterations: int = TRACK_MAX_ITERATIONS, **values) -> TrackedMode:
        """Follow the eigenpair near (sigma, x) at new parameter values by bordered Newton.

        A multiple (semisimple) eigenvalue, e.g. the (cos, sin) pair of a RealFourier group of a
        reflection-symmetric problem, makes the bordered matrix singular (it used to raise, or
        with rcond just above zero only warn). This is detected (rcond < eps), the eigenspace is
        read off the SVD of L + sigma M, and the returned vector is a *declared* member of it: the
        one continuing the given x (w^H x = 1 with w = x, and orthogonal to the rest of the
        eigenspace), obtained by Gauss-Newton with one border row per extra dimension. The
        multiplicity and this choice are stated in ``TrackedMode.note``.

        ``tol`` / ``max_iterations``: declared defaults :data:`TRACK_TOL` (stop at the attainable
        floor, residual returned) and :data:`TRACK_MAX_ITERATIONS` (cap: raises when reached).
        ``maxit`` is the deprecated former name of ``max_iterations`` (DeprecationWarning)."""
        L, M = self.matrices(**values)
        L, M = L.astype(complex), M.astype(complex)
        n = L.shape[0]
        w = np.asarray(x, complex) / np.vdot(x, x)
        x = np.asarray(x, complex) / np.vdot(w, x)
        s = complex(sigma)
        extra = np.zeros((0, n), complex)          # extra border rows (multiple eigenvalue)
        note = ""
        for it in range(1, max_iterations + 1):  # noqa: B007 -- read after the loop
            A = L + s * M
            J = np.zeros((n + 1 + extra.shape[0], n + 1), complex)
            J[:n, :n] = A
            J[:n, n] = M @ x
            J[n, :n] = w.conj()
            J[n + 1:, :n] = extra.conj()
            rhs = np.concatenate([-(A @ x), [1.0 - np.vdot(w, x)], -(extra.conj() @ x)])
            if extra.shape[0]:
                d = np.linalg.lstsq(J, rhs, rcond=None)[0]
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", sl.LinAlgWarning)
                        d = sl.solve(J, rhs)
                except (sl.LinAlgWarning, np.linalg.LinAlgError):
                    scale = np.linalg.norm(L, 1) + max(abs(s), 1.0) * np.linalg.norm(M, 1)
                    extra, note = self._eigenspace_rows(A, x, scale)
                    continue
            x = x + d[:n]
            s = s + d[n]
            if abs(d[n]) <= tol * max(abs(s), 1.0) and np.linalg.norm(d[:n]) <= 1e3 * tol * np.linalg.norm(x):
                break
            if abs(d[n]) <= tol * max(abs(s), 1.0) and not extra.shape[0]:
                break
        else:
            raise ConvergenceError(
                f"eigenpair tracking did not converge in {max_iterations} iterations at {values}",
                why="the starting eigenpair is too far from the tracked mode at these values "
                    "(large parameter step, or a mode crossing)",
                fix="track in smaller parameter steps from the last converged mode, or restart "
                    "from least_stable(...) at the new values")
        A = L + s * M
        nrm = np.linalg.norm(L, 1) + abs(s) * np.linalg.norm(M, 1)
        res = float(np.linalg.norm(A @ x) / (nrm * np.linalg.norm(x)))
        y = None
        if left and not extra.shape[0]:
            # A^H y = 0 with (M x)^H y = 1; the border column x spans null(A) = range(A^H)^perp
            K = np.zeros((n + 1, n + 1), complex)
            K[:n, :n] = A.conj().T
            K[:n, n] = x / np.linalg.norm(x)
            K[n, :n] = (M @ x).conj()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", sl.LinAlgWarning)
                    y = sl.solve(K, np.concatenate([np.zeros(n), [1.0]]), check_finite=False)[:n]
            except (sl.LinAlgWarning, np.linalg.LinAlgError):
                # Newton converged in one linear step without meeting the singular border: the
                # multiplicity shows up here (same detection and declared choice as above)
                scale = np.linalg.norm(L, 1) + max(abs(s), 1.0) * np.linalg.norm(M, 1)
                extra, note = self._eigenspace_rows(A, x, scale)
        if left and extra.shape[0]:
            # left eigenspace Y (last left singular vectors); y in it with y^H M [x, X2] = e1^T
            m = extra.shape[0] + 1
            U, sv, Vh = np.linalg.svd(A)
            Y = U[:, -m:]
            X = np.column_stack([x] + [v for v in extra])
            G = Y.conj().T @ (M @ X)
            y = Y @ np.linalg.solve(G.conj().T, np.eye(m)[:, 0])
        return TrackedMode(s, x, y, res, it, dict(values), note)

    @staticmethod
    def _eigenspace_rows(A: np.ndarray, x: np.ndarray, scale: float) -> tuple[np.ndarray, str]:
        """Rows fixing the member of a multiple eigenvalue's eigenspace (declared choice)."""
        sv, V = np.linalg.svd(A)[1:]
        V = V.conj().T
        small = sv <= MULTIPLICITY_TOL * scale          # relative to ||L|| + max(|sigma|, 1) ||M||
        m = int(small.sum())
        if m < 2:
            raise np.linalg.LinAlgError(
                "bordered Newton singular but L + sigma M has a simple null space: the pencil is "
                "singular or the eigenvalue is defective (not tracked)")
        N = V[:, -m:]                                   # eigenspace basis at the current sigma
        xn = x / np.linalg.norm(x)
        others = N - np.outer(xn, xn.conj() @ N)        # the eigenspace minus the tracked direction
        Q, R = np.linalg.qr(others)
        keep = np.abs(np.diag(R)) > MULTIPLICITY_TOL
        rows = Q[:, keep][:, :m - 1].T
        note = (f"eigenvalue of multiplicity {m} (singular values <= MULTIPLICITY_TOL = "
                f"{MULTIPLICITY_TOL:g} x scale; semisimple, singular values "
                f"{', '.join(f'{v:.1e}' for v in sv[-m:])}): the tracked vector is the eigenspace "
                "member continuing the given x (orthogonal to the rest of the eigenspace), a "
                "declared choice; sensitivities use the dual left vector")
        return rows, note

    def set_drift(self, fine: ParametricEVP, *, drift_tol: float = REQUIRED) -> ParametricEVP:
        """Spurious-mode rejection by resolution drift: a mode is admissible
        only if the pencil ``fine`` (same problem and parameters, finer resolution) has an
        eigenvalue within ``drift_tol`` x max(|sigma|, 1) at the same parameter values, as
        ``eigenvalues(drift=)``. Used by :meth:`least_stable` and :func:`critical` in place of a
        hand-written ``select`` bound. Returns self."""
        from .evp import check_drift_tol
        drift_tol = check_drift_tol(drift_tol, "ParametricEVP.set_drift()")   # given and in range
        if not isinstance(fine, ParametricEVP):
            raise OptionError("set_drift(fine): a ParametricEVP of the same problem at a finer "
                              "resolution", fix="m.parametric(N, over=..., drift=N2, "
                              "drift_tol=1e-6)")
        if {p.name for p in fine.params} != {p.name for p in self.params}:
            raise OptionError("set_drift(fine): the two pencils have different parameters "
                              f"({sorted(p.name for p in fine.params)} vs "
                              f"{sorted(p.name for p in self.params)})")
        if fine.gm0.shape[0] <= self.gm0.shape[0]:
            from ..errors import ResolutionError
            raise ResolutionError(
                f"set_drift(fine): the drift pencil is not finer ({fine.gm0.shape[0]} unknowns "
                f"against {self.gm0.shape[0]})",
                why="the drift test keeps the modes found again at a FINER resolution",
                fix="m.parametric(N, over, drift=N2, drift_tol=1e-6) with N2 > N, e.g. 3N/2")
        self.drift = (fine, float(drift_tol))
        return self

    def resolve_growth(self, growth: Any = None) -> tuple[Callable[[complex], float], str]:
        """(growth function, sentence) for ``growth`` (None: the declared or inferred
        convention of the eigenvalue, refused if neither decides)."""
        if growth is not None:
            return _growth(growth), (f"growth {growth!r} given" if isinstance(growth, str)
                                     else "growth given as a callable")
        if self.growth is None:
            name = self.problem.eigenvalue.name
            raise OptionError(
                f"the growth of the eigenvalue {name!r} is not declared",
                why="which part of the eigenvalue measures growth depends on how the mode is "
                    "written (exp(s t): Re s; exp(i alpha (x - c t)): Im c; exp(-i omega t): "
                    "Im omega); it decides which physics is solved, so it is never guessed",
                fix=f"growth='real' or growth='imag' in this call, or declare it once: "
                    f"m.eigenvalue({name!r}, growth='imag')")
        return _growth(self.growth), self.growth_note

    def admissible(self, r: EVPResult, values: Mapping[str, Any],
                   select: Callable | None = None,
                   growth: Callable | None = None) -> tuple[np.ndarray, str]:
        """(mask of admissible modes of ``r``, sentence): ``select`` if given, the spectral-tail
        test (``spurious='tail'``) and the drift test of :meth:`set_drift` if set. With
        ``growth`` (a callable of sigma), the sentence also says when the spectral-tail test set
        aside a mode of larger growth than every mode it kept (a selection that could otherwise
        jump to another branch without a word)."""
        keep = np.ones(r.values.size, bool) if select is None else np.asarray(select(r.values), bool)
        notes = [] if select is None else [f"select() kept {int(keep.sum())} of {keep.size}"]
        if self.spurious == "tail":
            layout = self._tail_layout()
            if layout is not None and r.vectors is not None and r.values.size:
                from ..tolerances import SPURIOUS_TAIL_TOL
                from .spurious import spectral_tails
                tails = spectral_tails(layout, r.vectors)
                ok = tails <= SPURIOUS_TAIL_TOL
                notes.append(f"spectral-tail test passed {int((keep & ok).sum())} of {keep.size} "
                             f"(SPURIOUS_TAIL_TOL = {SPURIOUS_TAIL_TOL:g})")
                if growth is not None and (keep & ~ok).any():
                    gv = np.array([growth(v) for v in r.values])
                    best = np.max(np.where(keep & ok, gv, -np.inf))
                    above = np.nonzero(keep & ~ok & (gv > best))[0]
                    if above.size:
                        j = int(above[np.argmax(gv[above])])
                        notes.append(
                            f"the spectral-tail test set aside {above.size} mode(s) of larger "
                            f"growth than every mode it kept (largest: sigma = {r.values[j]:.10g}, "
                            f"growth {gv[j]:.3e}, tail {tails[j]:.1e}): spurious tau or "
                            "unresolved modes; raise N, or spurious='none' to select them")
                keep = keep & ok
            elif self._tail_why:
                notes.append(self._tail_why)
        if self.drift is not None:
            fine, tol = self.drift
            hi = fine.solve(**values).values
            if hi.size:
                d = np.array([np.min(np.abs(hi - s)) / max(abs(s), 1.0) for s in r.values])
                ok = d <= tol
            else:
                ok = np.zeros(r.values.size, bool)
            notes.append(f"drift test kept {int((keep & ok).sum())} of {keep.size} (resolution "
                         f"drift <= {tol:.0e} against the finer pencil "
                         f"[{fine.gm0.label}] size {fine.gm0.shape})")
            keep = keep & ok
        return keep, "; ".join(notes) + (f" finite modes ({r.n_infinite} infinite)"
                                         if notes else "")

    def _tail_layout(self):
        """Layout of the pencil's vectors for the spectral-tail test (None: not applicable, the
        reason in ``_tail_why``, said by :meth:`admissible`)."""
        if not hasattr(self, "_tail"):
            from .spurious import tail_layout
            try:
                plan, group, reduced = self._layout_of
                self._tail, self._tail_why = tail_layout(plan, group, reduced)
            except Exception as exc:  # noqa: BLE001 -- detection only: no layout, no flag, said by admissible
                self._tail = None
                self._tail_why = f"spectral-tail test not run ({type(exc).__name__}: {exc})"
        return self._tail

    def _values(self, values: Mapping[str, Any]) -> tuple[dict, list[str]]:
        """``values`` completed by the model's declared values (native layer), with the names
        taken from the model (logged)."""
        out = dict(values)
        used = []
        for p in self.params:
            if p.name not in out and p.name in self.declared:
                out[p.name] = self.declared[p.name]
                used.append(f"{p.name} = {self.declared[p.name]:g}")
        return out, used

    def least_stable(self, *, growth: Any = None, select=None, **values) -> TrackedMode:
        """Full QZ, admissible modes (``select`` and the drift test of :meth:`set_drift`,
        logged), the one of largest growth, tracked once more to get its left vector.
        ``growth``: 'real', 'imag' or a callable of sigma; None (default): the convention
        declared with the eigenvalue or read off its name (``GROWTH_BY_NAME``), else refused."""
        g, gnote = self.resolve_growth(growth)
        values, used = self._values(values)
        r = self.solve(**values)
        keep, knote = self.admissible(r, values, select, growth=g)
        if not keep.any():
            raise ConvergenceError(f"no admissible mode at {values} ({knote})",
                                   fix="widen select=, check the drift resolution, or raise N (every mode failing "
                                       "the spectral-tail test is unresolved; spurious='none' "
                                       "turns that test off)")
        gv = np.array([g(v) for v in r.values])
        i = int(np.argmax(np.where(keep, gv, -np.inf)))
        m = self.track(r.values[i], r.vectors[:, i], **values)
        m.note = "; ".join(t for t in (m.note, knote, gnote, (
            "declared values of the model: " + ", ".join(used)) if used else "") if t)
        self._judge_resolution(m, "the least stable mode")
        return m

    def _mode_tail(self, x: np.ndarray) -> float | None:
        """Spectral tail of the vector ``x`` of this pencil (None when the tail test does not
        apply: spurious='none', no tau, no wall-normal axis)."""
        if self.spurious != "tail":
            return None
        layout = self._tail_layout()
        if layout is None:
            return None
        from .spurious import spectral_tails
        return float(spectral_tails(layout, np.asarray(x).reshape(-1, 1))[0])

    def _judge_resolution(self, m: TrackedMode, what: str) -> None:
        """Y1: without a drift pencil, the tail of the selected mode ``m`` against
        ``UNRESOLVED_TAIL_TOL``; above it ``m.unresolved_note`` and one ResolutionWarning."""
        if self.drift is not None:
            return
        m.tail = self._mode_tail(m.x)
        from ..tolerances import UNRESOLVED_TAIL_TOL
        if m.tail is not None and m.tail > UNRESOLVED_TAIL_TOL:
            from ..errors import ResolutionWarning, _warn_user
            from .spurious import unresolved_sentence
            vals = ", ".join(f"{k} = {float(np.real(v)):.10g}" for k, v in m.values.items())
            m.unresolved_note = unresolved_sentence(
                f"{what} ({self.problem.eigenvalue.name} = {m.sigma:.10g} at {vals})", m.tail)
            _warn_user(m.unresolved_note, ResolutionWarning)

    def sensitivity(self, mode: TrackedMode, name: str) -> complex:
        """d sigma / d p = - y^H (dL + sigma dM) x / (y^H M x), with exact dL, dM."""
        if mode.y is None:
            raise ValueError("mode tracked without its left vector (track(..., left=True))")
        dL, dM = self.derivative(name, **mode.values)
        _, M = self.matrices(**mode.values)
        x, y = mode.x, mode.y
        return complex(-(y.conj() @ ((dL + mode.sigma * dM) @ x)) / (y.conj() @ (M @ x)))

    def explain(self) -> str:
        ex = ", ".join(f"{n}: exponents {e[0]}..{e[-1]} (nominal {self.p0[n]})"
                       for n, e in self.exponents.items())
        if self.derived:
            ex += "; declared functions of parameters (independent pencil coordinates, exact " \
                  "symbolic derivatives): " + ", ".join(f"{g} = {d}" for g, d in self.derived.items())
        if self.wavenumbers:
            ex += ("; wavenumber parameters " + ", ".join(
                f"k_{n} = {q.name}" for n, q in self.wavenumbers.items())
                + f" (matrices polynomial in k, index structure of group "
                  f"[{self.gm0.label.split(' [')[0]}], synthetic wavenumbers; the domain length "
                  "along these directions is not used)")
        return (f"ParametricEVP [{self.gm0.label}] size {self.gm0.shape}: {ex}; "
                f"{self.n_terms} coefficient matrices from {self.n_plans} plans at roots of unity "
                f"({self.n_dropped} identically zero dropped); reconstruction error at an "
                f"independent real point {self.recon_error:.1e} (refused above {RECON_TOL:.0e}); "
                f"{self.rank_check}; mode tracking: bordered Newton, declared defaults "
                f"TRACK_TOL = {TRACK_TOL:g}, TRACK_MAX_ITERATIONS = {TRACK_MAX_ITERATIONS}")


@dataclass
class CriticalPoint:
    """Neutral value of ``parameter`` (growth = 0), minimized over ``over`` if given."""

    parameter: str
    value: float
    over: dict
    mode: TrackedMode
    evaluations: int
    log: list[str] = dc_field(default_factory=list)
    growth: float = float("nan")          # growth(sigma) of the tracked mode at the critical point
    verified: bool = False                # full QZ confirmed the tracked mode is the least stable
    # resolution of the critical mode: "" or the warning
    # sentence (its tail exceeds UNRESOLVED_TAIL_TOL and no drift pencil checked it)
    unresolved_note: str = ""

    def __repr__(self) -> str:
        # one line (the dataclass repr printed every eigenvector)
        o = "".join(f", {k}={v:.10g}" for k, v in self.over.items())
        return (f"CriticalPoint({self.parameter}_c={self.value:.10g}{o}, growth "
                f"{self.growth:.1e}, residual {self.mode.residual:.1e}, verified={self.verified}"
                + (", CRITICAL MODE UNDER-RESOLVED" if self.unresolved_note else "")
                + "; .explain() for the log)")

    def explain(self) -> str:
        o = ", ".join(f"{k}={v:.10g}" for k, v in self.over.items())
        warn = [f"WARNING: {self.unresolved_note}"] if self.unresolved_note else []
        return (f"CriticalPoint {self.parameter}_c = {self.value:.10g}" + (f" at {o}" if o else "")
                + f" ({self.evaluations} neutral-curve evaluations)\n  "
                + "\n  ".join(warn + self.log + [self.mode.explain()]))


def critical(pevp: ParametricEVP, parameter: str, *, bracket: tuple[float, float],
             growth: Callable[[complex], float] | str | None = None,
             select: Callable[[np.ndarray], np.ndarray] | None = None,
             over: Mapping[str, tuple[float, float]] | None = None,
             values: Mapping[str, float] | None = None, xtol: float = REQUIRED,
             start: float | None = None, over_start: float | None = None,
             atol: float | None = None) -> CriticalPoint:
    """Neutral ``parameter`` (growth(sigma) = 0 for the least stable selected mode).

    ``growth`` (None by default): the convention declared with the eigenvalue
    (``m.eigenvalue("c", growth="imag")``) or read off its name (``GROWTH_BY_NAME``: ``s``,
    ``sigma`` -> Re, phase speed ``c`` -> Im), logged; an undeclared name is refused. Parameters
    of the pencil missing from ``values`` take the model's declared values (native layer, logged).

    The least stable mode is chosen once by a full QZ among the admissible modes (the drift test
    of ``m.parametric(N, over, drift=N2, drift_tol=)`` / ``P.set_drift``, which removes spurious
    tau modes, and/or ``select``; the counts are logged), then
    followed by bordered Newton; the neutral value is found by Newton on growth with the exact
    sensitivity, safeguarded by the bracket. With ``over`` the neutral value is minimized over
    one other parameter (bounded Brent). The final mode is re-checked against a full QZ: if
    another admissible mode is less stable there, the result is refused.

    ``start`` is where the mode is first selected and Newton starts (default: the bracket
    midpoint, logged as such); ``over_start`` likewise for the ``over`` parameter (default: the
    midpoint of its bounds). Convergence: |dp| <= xtol |p| + atol, with atol = 1e-3 xtol (hi - lo)
    unless given (declared in the log): a pure relative test never ends at p = 0. ``xtol`` has no
    default: it sets how many digits of the critical value are returned.
    """
    xtol = require(xtol, "xtol", where="critical()", suggest="xtol=1e-9",
                   why="xtol is the relative precision of the returned critical value (Newton and "
                       "Brent stop at |dp| <= xtol |p| + atol): how many digits you get is your "
                       "choice; 1e-9 is ample for a neutral curve, 1e-6 for a quick survey")
    growth, gnote = pevp.resolve_growth(growth)
    (oname, (olo, ohi)), = (over or {"_": (None, None)}).items()
    if over is None:
        oname = None
    skip = {parameter} | ({oname} if oname else set())
    values, used = pevp._values({**{p.name: None for p in pevp.params if p.name in skip},
                                 **dict(values or {})})
    values = {k: v for k, v in values.items() if k not in skip}
    lo, hi = bracket
    if not lo < hi:
        raise OptionError(f"critical(): bracket={bracket} must satisfy lo < hi",
                          fix=f"bracket=({min(lo, hi)!r}, {max(lo, hi)!r})")
    p_start = 0.5 * (lo + hi) if start is None else float(start)
    if not lo <= p_start <= hi:
        raise OptionError(f"critical(): start {p_start} lies outside the bracket {bracket}",
                          fix="a start inside the bracket, or no start (bracket midpoint)")
    atol = 1e-3 * xtol * (hi - lo) if atol is None else float(atol)
    log: list[str] = [
        f"start: {parameter} = {p_start:.10g} ({'bracket midpoint (default)' if start is None else 'given'})"
        + (f", {oname} = {over_start:.10g} (given)" if oname is not None and over_start is not None else ""),
        f"tolerance: |d{parameter}| <= {xtol:g} |{parameter}| + {atol:.3g}",
        gnote] + (["declared values of the model: " + ", ".join(used)] if used else [])
    state: dict = {}
    count = [0]

    def least_stable(vals) -> TrackedMode:
        r = pevp.solve(**vals)
        keep, knote = pevp.admissible(r, vals, select, growth=growth)
        if not keep.any():
            raise ConvergenceError(f"critical(): no admissible mode at {vals} ({knote})",
                                   fix="widen select=, check the drift resolution, or raise N (every mode failing "
                                       "the spectral-tail test is unresolved; spurious='none' "
                                       "turns that test off)")
        g = np.array([growth(s) for s in r.values])
        i = int(np.argmax(np.where(keep, g, -np.inf)))
        if "filter" not in state and knote:
            state["filter"] = knote
        return pevp.track(r.values[i], r.vectors[:, i], **vals)

    def growth_at(q: float, mode: TrackedMode, vals: dict) -> float:
        return float(growth(pevp.track(mode.sigma, mode.x, left=False,
                                       **{**vals, parameter: q}).sigma))

    def max_growth(vals) -> float:
        """Growth of the least stable admissible mode at ``vals`` (full QZ; -inf if none)."""
        r = pevp.solve(**vals)
        keep, _ = pevp.admissible(r, vals, select, growth=growth)
        g = np.array([growth(s) for s in r.values])
        return float(np.max(np.where(keep, g, -np.inf))) if keep.any() else -np.inf

    def check_ends(vals) -> None:
        """Both ends of the bracket, whatever the path (with
        values= Newton started at the midpoint and returned the upper-branch neutral point
        Re = 31956 of Orr-Sommerfeld at alpha = 1 as verified, with the growth negative at both
        ends of (5000, 40000)). No sign change: refused, after a coarse scan of the bracket
        (BRACKET_SCAN_POINTS) that names the sub-brackets holding neutral points."""
        g_lo = max_growth({**vals, parameter: lo})
        g_hi = max_growth({**vals, parameter: hi})
        state["ends"] = (f"bracket ends (least stable admissible mode, full QZ): growth "
                         f"{g_lo:.3e} at {parameter} = {lo:.10g}, {g_hi:.3e} at {parameter} = "
                         f"{hi:.10g}: a sign change, so a neutral point lies inside")
        if np.sign(g_lo) * np.sign(g_hi) <= 0:
            return
        geo = lo > 0 and hi >= 4 * lo           # wide positive brackets (Re, Ra): geometric scan
        qs = (np.geomspace if geo else np.linspace)(lo, hi, BRACKET_SCAN_POINTS + 2)
        gs = [g_lo] + [max_growth({**vals, parameter: float(q)}) for q in qs[1:-1]] + [g_hi]
        cross = [(float(qs[i]), float(qs[i + 1])) for i in range(len(qs) - 1)
                 if np.sign(gs[i]) * np.sign(gs[i + 1]) <= 0]
        raise _bracket_error(parameter, lo, hi, g_lo, g_hi, gnote, vals,
                             select is None and pevp.drift is None,
                             what="the least stable admissible mode", crossings=cross,
                             scan=len(qs))

    def neutral(ov=None) -> tuple[float, TrackedMode]:
        count[0] += 1
        vals = dict(values)
        if oname is not None:
            vals[oname] = ov
        check_ends(vals)
        p = state.get("p", p_start)
        vals[parameter] = p
        mode = state.get("mode")
        if mode is None:                 # first selection, at the explicit start (over_start)
            first = dict(vals)
            if oname is not None and over_start is not None:
                first[oname] = over_start
            mode = least_stable(first)
        mode = pevp.track(mode.sigma, mode.x, **vals)
        for _it in range(CRITICAL_MAX_ITERATIONS):
            g = growth(mode.sigma)
            ds = pevp.sensitivity(mode, parameter)
            dg = growth(mode.sigma + ds) - growth(mode.sigma)      # growth is R-linear in sigma
            step = -g / dg if dg != 0 else np.inf
            p_new = p + step
            if not (lo <= p_new <= hi) or not np.isfinite(p_new):
                # safeguarded fallback: Brent on the bracket, tracking from the current mode
                def gfun(q, mode=mode, vals=vals):
                    return growth_at(q, mode, vals)
                g_lo, g_hi = gfun(lo), gfun(hi)
                if not (np.sign(g_lo) * np.sign(g_hi) <= 0):
                    raise _bracket_error(parameter, lo, hi, g_lo, g_hi, gnote, vals,
                                         select is None and pevp.drift is None)
                # absolute tolerance never 0 (p = 0 gave brentq(xtol = 0) -> ValueError)
                p_new = so.brentq(gfun, lo, hi, xtol=xtol * abs(p) + atol)
                vals[parameter] = p_new
                mode = pevp.track(mode.sigma, mode.x, **vals)
                p = p_new
                break
            vals[parameter] = p_new
            mode = pevp.track(mode.sigma, mode.x, **vals)
            converged = abs(p_new - p) <= xtol * abs(p_new) + atol
            p = p_new
            if converged:
                break
        else:
            raise ConvergenceError(
                f"critical(): Newton on the neutral {parameter} did not converge in "
                f"{CRITICAL_MAX_ITERATIONS} iterations (CRITICAL_MAX_ITERATIONS) at {vals} (last "
                f"step above {xtol:g}|p| + {atol:.3g})",
                why="the growth rate is not smooth or not monotone in the bracket (mode switch, "
                    "or a start far from the neutral point)",
                fix="narrow the bracket around the sign change, give start= close to it, or "
                    "check the growth curve with least_stable(...) at a few values")
        state["p"], state["mode"] = p, mode
        return p, mode

    if oname is None:
        pc, mode = neutral()
        ov_best = {}
    else:
        opt = so.minimize_scalar(lambda a: neutral(a)[0], bounds=(olo, ohi), method="bounded",
                                 options={"xatol": xtol * max(abs(olo), abs(ohi))})
        pc, mode = neutral(opt.x)
        ov_best = {oname: float(opt.x)}
    # verification: full QZ at the critical point, tracked mode must be the least stable one
    r = pevp.solve(**mode.values)
    keep, cnote = pevp.admissible(r, mode.values, select, growth=growth)
    g = np.array([growth(s) for s in r.values])
    gmax = float(np.max(np.where(keep, g, -np.inf)))
    if gmax > growth(mode.sigma) + MODE_SWITCH_TOL * max(1.0, abs(mode.sigma)):
        raise RuntimeError(f"mode switch: another admissible mode has growth {gmax:.3e} > 0 at the "
                           f"critical point {mode.values}")
    if select is None and pevp.drift is None:
        if pevp.spurious != "tail":
            what = "no spurious-mode rejection (spurious='none')"
        elif pevp._tail_layout() is None:
            what = ("no spurious-mode rejection: " + (pevp._tail_why or "the spectral-tail test "
                                                      "does not apply (no tau)"))
        else:
            what = "spurious modes: spectral-tail test only"
        log.append(what
                   + " (no drift pencil, no select): m.parametric(N, over, drift=N2, drift_tol=) "
                   "or P.set_drift(...) is the stronger filter")
    log += [state.get("ends", ""), state.get("filter", ""),
            ("at the critical point: " + cnote) if "set aside" in cnote else "", pevp.explain(),
            f"verification: full QZ at the critical point, tracked mode is the least stable "
            f"admissible one (max growth {gmax:.1e}), residual {mode.residual:.1e}"]
    # Y1 (N = 24 OS gave Re_c = 5037.6 for 5772.2 as a normal
    # result): without a drift pencil, the tail of the critical mode is printed and judged
    pevp._judge_resolution(mode, f"the critical mode at {parameter}_c = {pc:.10g}")
    if mode.tail is not None:
        from ..tolerances import UNRESOLVED_TAIL_TOL
        log.append(f"spectral tail of the critical mode {mode.tail:.1e} (under-resolved above "
                   f"UNRESOLVED_TAIL_TOL = {UNRESOLVED_TAIL_TOL:g}; the residual does not show it)")
    return CriticalPoint(parameter, float(pc), ov_best, mode, count[0], [s for s in log if s],
                         float(growth(mode.sigma)), True, mode.unresolved_note)


def _bracket_error(parameter: str, lo: float, hi: float, g_lo: float, g_hi: float,
                   gnote: str, vals: Mapping[str, Any], unfiltered: bool, *,
                   what: str = "the tracked mode", crossings: list | None = None,
                   scan: int = 0) -> ConvergenceError:
    """The refusal of a bracket without a sign change of the growth (it used to be
    SciPy's raw "f(a) and f(b) must have different signs"), with a suggested bracket: the
    sub-brackets of a coarse scan that hold a sign change (``crossings``), else
    the secant through the two ends."""
    def state(g: float) -> str:
        if g == -np.inf:        # max_growth() of an end where every mode was set aside
            return "no admissible mode there (every mode set aside by the filter: raise N or the drift resolution)"
        return f"{g:.3e}, " + ("growing" if g > 0 else ("decaying" if g < 0 else "neutral"))
    sugg = ""
    if crossings:
        subs = ", ".join(f"({a:.6g}, {b:.6g})" for a, b in crossings)
        sugg = (f"; a scan of {scan} points finds {len(crossings)} sign change"
                f"{'s' if len(crossings) > 1 else ''} inside, for {parameter} in {subs}"
                + (" (several neutral points: choose the branch)" if len(crossings) > 1 else "")
                + f": try bracket=({crossings[0][0]:.6g}, {crossings[0][1]:.6g})")
    elif crossings is not None and scan:
        sugg = f"; a scan of {scan} points finds no sign change inside either"
    if not crossings and np.isfinite(g_lo) and np.isfinite(g_hi) and g_hi != g_lo:
        # the zero of the secant through the two ends: where the sign change probably lies
        pz = lo - g_lo * (hi - lo) / (g_hi - g_lo)
        w = hi - lo
        sugg += (f"; the secant through the two ends crosses zero near {parameter} = {pz:.6g}: "
                f"try bracket=({pz - 0.5 * w:.6g}, {pz + 0.5 * w:.6g})")
    others = ", ".join(f"{k} = {v:g}" for k, v in vals.items() if k != parameter)
    return ConvergenceError(
        f"critical(): the growth of {what} has the same sign at both ends of the "
        f"bracket ({parameter} = {lo:g}: {state(g_lo)}; {parameter} = {hi:g}: "
        f"{state(g_hi)})" + (f" at {others}" if others else ""),
        why=f"a neutral value needs a sign change of the growth inside the bracket ({gnote})",
        fix="move or widen the bracket" + sugg + "; if the growth sign looks wrong, check the "
            "growth convention (growth='real' for exp(s t), 'imag' for a phase speed c)"
            + ("; no spurious-mode rejection is active, so the tracked mode may be a tau mode: "
               "m.parametric(N, over, drift=N2, drift_tol=1e-6)" if unfiltered else ""))


# ---------------------------------------------------------------------- parameter expressions
def _eval_param_expr(d: E.Expr, values: Mapping[str, Any]) -> complex | float:
    """Numeric value of a parameter-only expression (constant folding by the IR builders)."""
    e = d.substitute({p: E.Constant(values[p.name]) for p in d.free_parameters()})
    if not isinstance(e, E.Constant):
        raise ProblemError(f"cannot evaluate '{d}' from the parameter values")
    v = e.value
    return v.real if isinstance(v, complex) and v.imag == 0 and not np.iscomplexobj(
        [values[p.name] for p in d.free_parameters()]) else v


def _dparam(d: E.Expr, p: E.Parameter) -> E.Expr:
    """Exact symbolic d/dp of a parameter-only expression (sum, product, power, pointwise f)."""
    if d == p:
        return E.Constant(1)
    if p not in d.free_parameters():
        return E.Constant(0)
    if isinstance(d, E.Add):
        return E.add(*(_dparam(a, p) for a in d.args))
    if isinstance(d, E.Mul):
        return E.add(*(E.mul(*(d.args[:i] + (_dparam(a, p),) + d.args[i + 1:]))
                       for i, a in enumerate(d.args)))
    if isinstance(d, E.Power):
        b = d.args[0]
        return E.mul(d.exponent, E.power(b, d.exponent - 1), _dparam(b, p))
    if isinstance(d, E.UnaryFunction):
        u = d.args[0]
        der = E._FUNCS[d.func][3]
        if der is None:
            raise ProblemError(f"'{d}': {d.func} is not differentiable (derived parameter)")
        return E.mul(der(u), _dparam(u, p))
    raise ProblemError(f"derived parameter '{d}': cannot differentiate {type(d).__name__}")


def _substituted(problem: EVP, subst: Mapping[E.Expr, E.Expr]) -> EVP:
    """Copy of an EVP with sub-expressions replaced (declared derived parameters)."""
    out = EVP(problem.variables, problem.eigenvalue,
              name=f"{problem.name} [derived: {', '.join(str(k) for k in subst)}]")
    for eq in problem.equations:
        out.add_equation((eq.lhs.substitute(subst), eq.rhs))
    out._tau_log = list(problem._tau_log)
    return out
