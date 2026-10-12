"""Discrete dynamical system M dx/dt = G(x; p) seen by the bifurcation core.

Two constructors, never mixed:

* :func:`ir_system` (the normal path): an IR :class:`IVP` is discretized once; G = F(x; p) - L(p) x
  where F is the right-hand side evaluated by the spectral evaluator and L, M the IMEX matrices.
  The multilinear derivatives of F are **symbolic Fréchet derivatives of the IR**
  (``Expr.linearize`` applied once, twice, three times): J = F' - L, B(u, v) = F''[u, v],
  C(u, v, w) = F'''[u, v, w]; nothing is finite-differenced in x. The parameter dependence is read
  off the IR: when a parameter enters an LHS (hence L or M) or the RHS as a Laurent polynomial, the
  coefficient matrices / vectors are recovered exactly from samples (Vandermonde in t = p / p0,
  checked at an independent point, refused above ``RECON_TOL``); ∂G/∂p is then exact. A parameter
  entering non-polynomially needs an explicit FD step (``dp_fd``), declared by ``explain()``.
  For 0-D systems (every basis holds one mode: ODEs written in the IR) the IR expressions are
  compiled to vectorised numpy (``pointwise=True``), which evaluates many states at once (the
  collocation points of a periodic orbit); the spectral-evaluator path is the oracle.
* :meth:`DynamicalSystem.from_callables`: user callables (constructed test systems whose right-hand
  side is not expressible in the IR, e.g. a half-turn transverse rotation). Missing derivatives
  are finite-differenced with **declared** steps, shown by ``explain()``; the origin is always
  printed ("callables, not IR").
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..errors import OptionError, ParameterError, UnsupportedError, did_you_mean
from ..ir import expr as E

__all__ = ["DynamicalSystem", "ir_system", "ode_problem", "RECON_TOL"]

RECON_TOL = 1e-10      # relative error of the Laurent reconstruction at the check point


# ---------------------------------------------------------------------------- pointwise compiler
_POINTWISE_TYPES = (E.Field, E.Parameter, E.Constant, E.Add, E.Mul, E.Power, E.UnaryFunction)


def _is_pointwise(e: E.Expr) -> bool:
    return all(isinstance(n, _POINTWISE_TYPES) for n in e.walk())


def _compile(e: E.Expr) -> Callable[[Mapping[E.Field, np.ndarray], Mapping[str, float]], np.ndarray]:
    """Vectorised numpy function of a pointwise IR expression (fields -> arrays, params -> floats)."""
    if isinstance(e, E.Field):
        return lambda d, p: d[e]
    if isinstance(e, E.Parameter):
        return lambda d, p: p[e.name] if e.name in p else e.value
    if isinstance(e, E.Constant):
        v = e.value
        return lambda d, p: v
    kids = [_compile(a) for a in e.args]
    if isinstance(e, E.Add):
        def f_add(d, p):
            out = kids[0](d, p)
            for k in kids[1:]:
                out = out + k(d, p)
            return out
        return f_add
    if isinstance(e, E.Mul):
        def f_mul(d, p):
            out = kids[0](d, p)
            for k in kids[1:]:
                out = out * k(d, p)
            return out
        return f_mul
    if isinstance(e, E.Power):
        ex = e.exponent
        return lambda d, p: kids[0](d, p) ** ex
    if isinstance(e, E.UnaryFunction):
        fn = e.numpy_func
        return lambda d, p: fn(kids[0](d, p))
    raise UnsupportedError(f"pointwise compiler: {type(e).__name__} is not pointwise")


# ---------------------------------------------------------------------------- Laurent helpers
def _sample_points(m: int) -> np.ndarray:
    """m distinct real sample points t = p / p0 (Chebyshev nodes in [0.5, 1.5])."""
    if m == 1:
        return np.array([1.0])
    return 1.0 + 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)


def _laurent_fit(values: Sequence[np.ndarray], t: np.ndarray, exps: Sequence[int]) -> list:
    """Coefficients C_e with values[k] = sum_e t_k^e C_e (square Vandermonde solve)."""
    V = np.array([[tk ** e for e in exps] for tk in t])
    Vi = np.linalg.inv(V)
    return [sum(Vi[i, k] * values[k] for k in range(len(t))) for i in range(len(exps))]


class DynamicalSystem:
    """M dx/dt = G(x; p): residual, Jacobian, parameter derivatives and multilinear forms.

    Parameters are held in ``self.params`` (name -> value); every method takes ``par`` (a dict of
    values overriding them). Vectors may be complex: B and C are extended C-multilinearly from the
    real forms (real system), so B(q, conj(q)) etc. are exact.
    """

    def __init__(self, n: int, *, params: Mapping[str, float], origin: str,
                 G: Callable, J: Callable, Gp: Callable, JF_dir: Callable | None,
                 B: Callable, C: Callable, M: Any = None, G_many: Callable | None = None,
                 J_many: Callable | None = None, Jp: Callable | None = None,
                 notes: Sequence[str] = ()) -> None:
        self.n = int(n)
        self.params = dict(params)
        self.origin = origin
        self._G, self._J, self._Gp, self._B, self._C = G, J, Gp, B, C
        self._Jv = JF_dir
        self._G_many, self._J_many = G_many, J_many
        self._Jp = Jp
        self._Gp_many = None          # vectorised ∂G/∂p (set by ir_system for 0-D systems)
        self.M = sp.identity(n, format="csr") if M is None else M
        self.identity_mass = M is None or _is_identity(self.M)
        self.notes = list(notes)
        self.n_G = 0

    # ------------------------------------------------------------------ parameters
    def par(self, **values) -> dict:
        """Parameter values with ``values`` overriding ``self.params`` (unknown names refused)."""
        unknown = [k for k in values if k not in self.params]
        if unknown:
            raise ParameterError(f"unknown parameter(s) {unknown}", where="DynamicalSystem.par",
                                 fix=f"known: {sorted(self.params)}")
        return {**self.params, **values}

    def _check_parameter(self, name: str, where: str) -> None:
        """Refuse a parameter name the system does not hold (a typo used to surface as a bare
        ``KeyError`` deep inside the continuation)."""
        if name not in self.params:
            raise ParameterError(f"{where}: unknown parameter {name!r}",
                                 fix=f"known parameters: {sorted(self.params)}."
                                     + did_you_mean(str(name), self.params))

    def _full(self, par: Mapping[str, float] | None) -> dict:
        return dict(self.params) if par is None else {**self.params, **par}

    # ------------------------------------------------------------------ evaluations
    def G(self, x: np.ndarray, par: Mapping[str, float] | None = None) -> np.ndarray:
        """Right-hand side G(x; p) of M dx/dt = G (real state)."""
        self.n_G += 1
        return np.asarray(self._G(np.asarray(x, float), self._full(par)))

    def J(self, x: np.ndarray, par: Mapping[str, float] | None = None) -> np.ndarray:
        """Dense Jacobian dG/dx (the systems handled here are small or moderate)."""
        return np.asarray(self._J(np.asarray(x, float), self._full(par)))

    def Gp(self, x: np.ndarray, name: str, par: Mapping[str, float] | None = None,
           order: int = 1) -> np.ndarray:
        """∂G/∂p (order 1) or ∂²G/∂p² (order 2) for parameter ``name`` (exact Laurent
        reconstruction for IR systems, or declared FD)."""
        if name not in self.params:
            raise ParameterError(f"unknown parameter {name!r}", fix=f"known: {sorted(self.params)}")
        if order not in (1, 2):
            raise OptionError("Gp: order must be 1 or 2")
        return np.asarray(self._Gp(np.asarray(x, float), self._full(par), name, order))

    def Jp(self, x: np.ndarray, v: np.ndarray, name: str, par=None) -> np.ndarray:
        """∂/∂p (J(x; p) v) (C-linear in v)."""
        par = self._full(par)
        x = np.asarray(x, float)
        if np.iscomplexobj(v):
            return self.Jp(x, np.real(v), name, par) + 1j * self.Jp(x, np.imag(v), name, par)
        return np.asarray(self._Jp(x, np.asarray(v, float), par, name))

    def G_many(self, X: np.ndarray, par=None) -> np.ndarray:
        """G at many states (rows of X); vectorised for 0-D IR systems, else a loop."""
        par = self._full(par)
        if self._G_many is not None:
            return self._G_many(X, par)
        return np.array([self.G(x, par) for x in X])

    def J_many(self, X: np.ndarray, par=None) -> np.ndarray:
        """Jacobians at many states (rows of X), shape (m, n, n); vectorised for 0-D IR systems."""
        par = self._full(par)
        if self._J_many is not None:
            return self._J_many(X, par)
        return np.array([self.J(x, par) for x in X])

    def Gp_many(self, X: np.ndarray, name: str, par=None) -> np.ndarray:
        """∂G/∂p at many states (rows of X); vectorised for 0-D IR systems."""
        if self._Gp_many is not None:
            return self._Gp_many(np.atleast_2d(X), self._full(par), name)
        return np.array([self.Gp(x, name, par) for x in X])

    def B(self, x, u, v, par=None) -> np.ndarray:
        """Second derivative F''(x)[u, v] (complex arguments: C-bilinear extension)."""
        par = self._full(par)
        x = np.asarray(x, float)
        if np.iscomplexobj(u) or np.iscomplexobj(v):
            u, v = np.asarray(u, complex), np.asarray(v, complex)
            b = lambda a, c: np.asarray(self._B(x, a, c, par))       # noqa: E731
            return (b(u.real, v.real) - b(u.imag, v.imag)) + 1j * (b(u.real, v.imag) + b(u.imag, v.real))
        return np.asarray(self._B(x, np.asarray(u, float), np.asarray(v, float), par))

    def C(self, x, u, v, w, par=None) -> np.ndarray:
        """Third derivative F'''(x)[u, v, w] (complex arguments: C-trilinear extension)."""
        par = self._full(par)
        x = np.asarray(x, float)
        if not (np.iscomplexobj(u) or np.iscomplexobj(v) or np.iscomplexobj(w)):
            return np.asarray(self._C(x, np.asarray(u, float), np.asarray(v, float),
                                      np.asarray(w, float), par))
        parts = [(np.real(a), np.imag(a)) for a in (u, v, w)]
        out = np.zeros(self.n, complex)
        for idx in itertools.product((0, 1), repeat=3):
            coef = 1j ** sum(idx)
            args = [parts[k][i] for k, i in enumerate(idx)]
            if any(not np.any(a) for a in args):
                continue
            out += coef * np.asarray(self._C(x, *args, par))
        return out

    def Jv(self, x, v, par=None) -> np.ndarray:
        """J(x) v without forming J (C-linear extension for complex v)."""
        par = self._full(par)
        if self._Jv is None:
            return self.J(x, par) @ v
        if np.iscomplexobj(v):
            return self._Jv(x, np.real(v), par) + 1j * self._Jv(x, np.imag(v), par)
        return self._Jv(x, np.asarray(v, float), par)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_callables(cls, f: Callable, n: int, params: Mapping[str, float], *,
                       jac: Callable | None = None, dfdp: Callable | None = None,
                       fd_x: float | None = None, fd_p: float | None = None,
                       name: str = "callables") -> DynamicalSystem:
        """dx/dt = f(x, par) from Python callables (M = I). ``jac(x, par)`` and
        ``dfdp(x, par, name)`` are optional; without them, **declared** central FD steps
        ``fd_x`` / ``fd_p`` are required. B and C are always central FD of J (step ``fd_x``,
        required): exact only up to O(fd_x²), stated by ``explain()``."""
        if fd_x is None:
            raise OptionError("from_callables: fd_x (central FD step for B, C and a missing "
                              "Jacobian) is required", fix="e.g. fd_x=1e-4 (declared)")
        if dfdp is None and fd_p is None:
            raise OptionError("from_callables: no dfdp given, fd_p (central FD step) is required")
        notes = [f"origin: {name} (callables, not IR)",
                 f"Jacobian: {'analytic' if jac else f'central FD, h = {fd_x:g}'}",
                 f"B, C: central FD of the Jacobian, h = {fd_x:g} (O(h^2) error)",
                 f"dG/dp: {'analytic' if dfdp else f'central FD, h = {fd_p:g}'}; d2G/dp2 and "
                 f"dJ/dp: central FD, h = {fd_p if fd_p is not None else fd_x:g}"]

        def J(x, par):
            if jac is not None:
                return np.asarray(jac(x, par), float)
            out = np.empty((n, n))
            for i in range(n):
                e = np.zeros(n)
                e[i] = fd_x
                out[:, i] = (np.asarray(f(x + e, par)) - np.asarray(f(x - e, par))) / (2 * fd_x)
            return out

        hp = fd_p if fd_p is not None else fd_x

        def Gp(x, par, nm, order=1):
            if order == 1 and dfdp is not None:
                return np.asarray(dfdp(x, par, nm), float)
            hi, lo = dict(par), dict(par)
            hi[nm] += hp
            lo[nm] -= hp
            if order == 1:
                return (np.asarray(f(x, hi)) - np.asarray(f(x, lo))) / (2 * hp)
            return (np.asarray(f(x, hi)) - 2 * np.asarray(f(x, par)) + np.asarray(f(x, lo))) / hp ** 2

        def Jp(x, v, par, nm):
            hi, lo = dict(par), dict(par)
            hi[nm] += hp
            lo[nm] -= hp
            return (J(x, hi) @ v - J(x, lo) @ v) / (2 * hp)

        def B(x, u, v, par):
            return (J(x + fd_x * u, par) @ v - J(x - fd_x * u, par) @ v) / (2 * fd_x)

        def C(x, u, v, w, par):
            return (B(x + fd_x * w, u, v, par) - B(x - fd_x * w, u, v, par)) / (2 * fd_x)

        return cls(n, params=params, origin=f"{name} (callables, not IR)",
                   G=lambda x, par: np.asarray(f(x, par), float), J=J, Gp=Gp, JF_dir=None,
                   B=B, C=C, Jp=Jp, notes=notes)

    # ------------------------------------------------------------------ report
    def explain(self) -> str:
        """Size, origin, parameter values, mass matrix and how each derivative is computed."""
        lines = [f"DynamicalSystem: M dx/dt = G(x; p), n = {self.n}, origin: {self.origin}",
                 "  parameters: " + ", ".join(f"{k} = {v:.12g}" for k, v in self.params.items()),
                 f"  mass matrix: {'identity' if self.identity_mass else 'general (DAE rows allowed)'}"]
        lines += ["  " + s for s in self.notes]
        return "\n".join(lines)


def _is_identity(M) -> bool:
    M = sp.csr_matrix(M)
    if M.shape[0] != M.shape[1]:
        return False
    return abs(M - sp.identity(M.shape[0])).max() == 0 if M.nnz else False


# ---------------------------------------------------------------------------- ODEs in the IR
def ode_problem(variables: Sequence[str], parameters: Mapping[str, float], rhs: Callable,
                name: str = "ODE"):
    """IR :class:`IVP` of an ODE system dx_i/dt = f_i(x; p) (a "0-D" problem).

    ``rhs(x, p)`` receives dicts name -> Field / Parameter and returns the list of right-hand
    sides (IR expressions). The domain is one RealFourier direction holding a single mode (the
    constant): every field is a number, products are exact (declared in ``explain()``).
    """
    from ..ir import IVP, CartesianCoordinates, Domain, Field, Parameter, RealFourier, dt
    cs = CartesianCoordinates("ode_dummy")
    D = Domain(cs, [RealFourier(cs["ode_dummy"], 2, (0.0, 1.0), dealias=1)])
    X = {v: Field(v, D) for v in variables}
    P = {k: Parameter(k, float(v)) for k, v in parameters.items()}
    f = list(rhs(X, P))
    if len(f) != len(variables):
        raise OptionError(f"ode_problem: {len(f)} right-hand sides for {len(variables)} variables")
    pb = IVP(list(X.values()), name=f"{name} (0-D: one constant Fourier mode)")
    for v, fi in zip(variables, f, strict=True):
        pb.add_equation((dt(X[v]), fi))
    return pb


# ---------------------------------------------------------------------------- IR adapter
def ir_system(problem, parameters: Sequence[str | E.Parameter], *,
              values: Mapping[str, float] | None = None, dp_fd: float | None = None,
              pointwise: bool | None = None, ncc: Mapping | None = None) -> DynamicalSystem:
    """:class:`DynamicalSystem` of an IR :class:`IVP` with free parameters ``parameters``.

    ``values`` fixes the starting values (default: the values declared on the Parameters; a
    Parameter without value must be given here). The IVP is discretized once (plus one plan per
    Laurent sample if a free parameter enters an LHS). ``pointwise`` (0-D systems only): compile
    the IR expressions to numpy; ``None`` = automatic (True when every basis holds one mode),
    declared by ``explain()``. ``dp_fd``: central FD step in p, required only if a free parameter
    enters non-polynomially.
    """
    from ..discretize import discretize
    from ..solvers.parametric import laurent_exponents
    from ..timestep import IVPSolver

    names = [p if isinstance(p, str) else p.name for p in parameters]
    declared = {}
    for eq in problem.equations:
        for side in (eq.lhs, eq.rhs):
            for q in side.free_parameters():
                declared[q.name] = q
    missing = [n for n in names if n not in declared]
    if missing:
        raise ParameterError(f"ir_system: parameter(s) {missing} do not appear in the problem",
                             fix=f"parameters of the problem: {sorted(declared)}")
    vals = {n: (values or {}).get(n, declared[n].value) for n in names}
    for n, v in vals.items():
        if v is None:
            raise ParameterError(f"ir_system: parameter {n!r} has no value", fix=f"values={{'{n}': ...}}")
    base_params = {**{k: q.value for k, q in declared.items() if q.value is not None},
                   **(values or {})}

    def build(pv: Mapping[str, float]):
        plan = discretize(problem, ncc=ncc, params={**base_params, **pv})
        return IVPSolver(plan, "RK443", 1.0)       # evaluator + matrices only: no time step taken

    solver = build(vals)
    sysm = solver.system
    n = sysm.n
    if np.iscomplexobj(np.zeros(1, sysm.dtype)):
        raise UnsupportedError("ir_system: complex unknowns are not supported yet (real systems only)")
    notes = ["origin: IR (discretized IVP; F by the spectral evaluator, derivatives symbolic)",
             "IVPSolver used as evaluator/matrix container only: no time step is taken"]

    # --- parameter dependence of L, M (LHS) and of F (RHS): Laurent exponents from the IR
    lhs_exp, rhs_exp = {}, {}
    for nm in names:
        p = declared[nm]
        el, er = set(), set()
        for eq in problem.equations:
            for side, acc in ((eq.lhs, el), (eq.rhs, er)):
                if p in side.free_parameters():
                    ex = laurent_exponents(side, p)
                    if ex is None:
                        acc.add(None)
                    else:
                        acc.update(ex)
        lhs_exp[nm], rhs_exp[nm] = el, er
        if (None in el or None in er) and dp_fd is None:
            raise UnsupportedError(f"ir_system: parameter {nm!r} enters non-polynomially",
                                   fix="give dp_fd (declared central FD step in p)")
    lhs_params = [nm for nm in names if lhs_exp[nm]]
    if any(None in lhs_exp[nm] for nm in lhs_params):
        raise UnsupportedError("ir_system: a free parameter enters an LHS non-polynomially "
                               "(L(p) cannot be reconstructed exactly)",
                               fix="move that term to the RHS, or rebuild the system per value")
    L0, M0 = sysm.L.tocsr(), sysm.M.tocsr()
    # tensor Laurent reconstruction of L(p), M(p) over the LHS parameters
    lhs_coef = {}
    if lhs_params:
        exps = [sorted(lhs_exp[nm] | {0}) for nm in lhs_params]
        p0 = {nm: (vals[nm] if vals[nm] != 0 else 1.0) for nm in lhs_params}
        grids = [_sample_points(len(e)) for e in exps]
        samples = {}
        for idx in itertools.product(*(range(len(e)) for e in exps)):
            pv = dict(vals)
            pv.update({nm: p0[nm] * grids[i][k] for i, (nm, k) in enumerate(zip(lhs_params, idx, strict=True))})
            s = build(pv).system
            samples[idx] = (s.L.toarray(), s.M.toarray())
        arr_L = np.array([[samples[idx][0]] for idx in samples]).reshape(
            tuple(len(e) for e in exps) + (n, n))
        arr_M = np.array([[samples[idx][1]] for idx in samples]).reshape(
            tuple(len(e) for e in exps) + (n, n))
        for ax, (e, g) in enumerate(zip(exps, grids, strict=True)):
            Vi = np.linalg.inv(np.array([[tk ** ee for ee in e] for tk in g]))
            arr_L = np.moveaxis(np.tensordot(Vi, np.moveaxis(arr_L, ax, 0), axes=(1, 0)), 0, ax)
            arr_M = np.moveaxis(np.tensordot(Vi, np.moveaxis(arr_M, ax, 0), axes=(1, 0)), 0, ax)
        for idx in itertools.product(*(range(len(e)) for e in exps)):
            ex = tuple(exps[i][k] for i, k in enumerate(idx))
            lhs_coef[ex] = (sp.csr_matrix(arr_L[idx]), sp.csr_matrix(arr_M[idx]))
        # independent check point
        chk = dict(vals)
        chk.update({nm: p0[nm] * (1.0 + 0.137 * (i + 1)) for i, nm in enumerate(lhs_params)})
        s = build(chk).system
        Lc, Mc = _lm(lhs_coef, lhs_params, p0, chk)
        scale = max(abs(L0).max(), abs(M0).max(), 1e-300)
        err = max(abs(Lc - s.L).max(), abs(Mc - s.M).max()) / scale
        if err > RECON_TOL:
            raise UnsupportedError(f"ir_system: L(p), M(p) reconstruction error {err:.1e} > {RECON_TOL:.0e}")
        notes.append(f"LHS parameters {lhs_params}: L(p), M(p) exact Laurent reconstruction "
                     f"(exponents {exps}, {len(samples)} plans, check error {err:.1e})")
    else:
        p0 = {}
    mass_depends = False
    if lhs_coef:
        # relative threshold: the reconstructed M coefficients of nonzero exponents are round-off
        mref = max(abs(M0).max(), 1e-300)
        mass_depends = any(abs(Mm).max() > RECON_TOL * mref for ex, (_, Mm) in lhs_coef.items() if any(ex))
        if not mass_depends:          # drop the round-off (M independent of the parameters)
            lhs_coef = {ex: (Lm, Mm if not any(ex) else 0 * Mm) for ex, (Lm, Mm) in lhs_coef.items()}

    def LM(par):
        if not lhs_coef:
            return L0, M0
        return _lm(lhs_coef, lhs_params, p0, par)

    # --- symbolic derivatives of the RHS
    V = list(sysm.variables)
    d1 = [E.Field("_b1" + v.name, v.domain, v.tensorsig, v.dtype) for v in V]
    d2 = [E.Field("_b2" + v.name, v.domain, v.tensorsig, v.dtype) for v in V]
    d3 = [E.Field("_b3" + v.name, v.domain, v.tensorsig, v.dtype) for v in V]
    rhs = list(solver._rhs_exprs)
    lin1, lin2, lin3 = [], [], []
    for e in rhs:
        if e is None or not (set(e.free_fields()) & set(V)):
            lin1.append(None), lin2.append(None), lin3.append(None)
            continue
        a = e.linearize(None, dict(zip(V, d1, strict=True)))
        b = a.linearize(None, dict(zip(V, d2, strict=True))) if not a.is_zero() else a
        c = b.linearize(None, dict(zip(V, d3, strict=True))) if not b.is_zero() else b
        lin1.append(None if a.is_zero() else a)
        lin2.append(None if b.is_zero() else b)
        lin3.append(None if c.is_zero() else c)
    eqs = solver.plan.problem.equations
    ev = solver.evaluator

    # 0-D: every (scalar) variable holds exactly one coefficient in the discrete state. (Before,
    # "every basis has size <= 2" was tested: a Chebyshev basis of size 2 holds T0 and T1, yet it
    # was compiled pointwise as one number per variable, so G ignored the T1 coefficients without
    # a word; tests/bifurcation/test_bifurcation_quality_lot.py.)
    zero_d = all(v.rank == 0 for v in V) and n == len(V)
    if pointwise is None:
        pointwise = zero_d and all(e is None or _is_pointwise(e) for e in rhs)
    if pointwise and not zero_d:
        raise OptionError("pointwise=True needs a 0-D system (every basis holds one mode)")
    if pointwise:
        # state layout: variable k is the single coefficient at index perm[k]
        perm = []
        for k in range(len(V)):
            e = np.zeros(len(V))
            e[k] = 1.0
            X = sysm.gather_state([np.full(s, e[i]) for i, s in enumerate(sysm.var_shapes)])
            perm.append(int(np.flatnonzero(X)[0]))
        rowperm = []
        for k in range(len(eqs)):
            out = [None] * len(eqs)
            out[k] = np.ones(sysm.eq_shapes[k])
            rowperm.append(int(np.flatnonzero(sysm.gather_rhs(out))[0]))
        comp = [[None if e is None else _compile(e) for e in lst] for lst in (rhs, lin1, lin2, lin3)]
        notes.append("pointwise=True: 0-D system, IR expressions compiled to vectorised numpy "
                     "(same expressions as the spectral evaluator)")
    else:
        notes.append("F, F', F'', F''' evaluated by the spectral evaluator "
                     "(Jacobian: n directional derivatives)")

    def set_params(par):
        solver.plan.params.update(par)

    def _data(X, dirs=()):
        data = dict(solver._ncc)
        data.update(zip(V, sysm.scatter(X), strict=True))
        for fields, vec in dirs:
            data.update(zip(fields, sysm.scatter(vec), strict=True))
        return data

    def _eval_list(exprs, data, par):
        out = []
        for le, shape, eq in zip(exprs, sysm.eq_shapes, eqs, strict=True):
            if le is None:
                out.append(None)
                continue
            c = ev.evaluate(le, data, par)
            out.append(ev._embed(c, eq.lhs.rank, shape))
        return sysm.gather_rhs(out)

    def _pw(which, Xs, dirs, par):
        """Vectorised pointwise evaluation; Xs (m, n), dirs: list of (fields, (m, n) arrays)."""
        Xs = np.atleast_2d(Xs)
        m = Xs.shape[0]
        data = {v: Xs[:, perm[k]] for k, v in enumerate(V)}
        for fields, arr in dirs:
            arr = np.atleast_2d(arr)
            data.update({f: arr[:, perm[k]] for k, f in enumerate(fields)})
        out = np.zeros((m, n))
        for k, fn in enumerate(comp[which]):
            if fn is not None:
                out[:, rowperm[k]] = np.broadcast_to(fn(data, par), (m,))
        return out

    def Fx(X, par):
        if pointwise:
            return _pw(0, X, (), par)[0]
        set_params(par)
        return _eval_list(rhs, _data(X), par)

    def G(X, par):
        L, _ = LM(par)
        return Fx(X, par) - L @ X

    def JF_dir(X, v, par):
        if pointwise:
            return _pw(1, X, [(d1, v)], par)[0]
        set_params(par)
        return _eval_list(lin1, _data(X, [(d1, v)]), par)

    def Jv(X, v, par):
        L, _ = LM(par)
        return JF_dir(X, v, par) - L @ v

    def J(X, par):
        L, _ = LM(par)
        if pointwise:
            Jm = J_many(X[None, :], par)[0]
            return Jm
        cols = np.empty((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            cols[:, i] = JF_dir(X, e, par)
        return cols - L.toarray()

    def G_many(Xs, par):
        L, _ = LM(par)
        if pointwise:
            return _pw(0, Xs, (), par) - Xs @ L.T.toarray()
        return np.array([G(x, par) for x in Xs])

    def J_many(Xs, par):
        L, _ = LM(par)
        Xs = np.atleast_2d(Xs)
        m = Xs.shape[0]
        if not pointwise:
            return np.array([J(x, par) for x in Xs])
        out = np.empty((m, n, n))
        for i in range(n):
            e = np.zeros((m, n))
            e[:, i] = 1.0
            out[:, :, i] = _pw(1, Xs, [(d1, e)], par)
        return out - L.toarray()[None]

    def B(X, u, v, par):
        if pointwise:
            return _pw(2, X, [(d1, u), (d2, v)], par)[0]
        set_params(par)
        return _eval_list(lin2, _data(X, [(d1, u), (d2, v)]), par)

    def Cf(X, u, v, w, par):
        if pointwise:
            return _pw(3, X, [(d1, u), (d2, v), (d3, w)], par)[0]
        set_params(par)
        return _eval_list(lin3, _data(X, [(d1, u), (d2, v), (d3, w)]), par)

    # --- ∂G/∂p, ∂²G/∂p², ∂(Jv)/∂p: exact Laurent in the RHS (samples of F in t) + exact ∂L/∂p
    def _dp_rhs(fun, par, nm, order):
        """d^order/dp^order of the RHS part fun(par) (vector), exact Laurent or declared FD."""
        er = rhs_exp.get(nm, set())
        if not er:
            return 0.0
        if None in er:
            h = dp_fd
            q = [dict(par) for _ in range(3)]
            q[0][nm] -= h
            q[2][nm] += h
            f = [fun(qq) for qq in q]
            return (f[2] - f[0]) / (2 * h) if order == 1 else (f[2] - 2 * f[1] + f[0]) / h ** 2
        exps = sorted(er | {0})
        pc = par[nm] if par[nm] != 0 else 1.0
        t = _sample_points(len(exps))
        vals_ = []
        for tk in t:
            q = dict(par)
            q[nm] = pc * tk
            vals_.append(fun(q))
        coef = _laurent_fit(vals_, t, exps)
        tc = par[nm] / pc
        out = 0.0
        for e, c in zip(exps, coef, strict=True):
            k = e * (e - 1) if order == 2 else e
            if k:
                out = out + k * tc ** (e - order) / pc ** order * c
        return out

    def _dL(par, nm, order):
        if nm not in lhs_params:
            return None
        i = lhs_params.index(nm)
        out = sp.csr_matrix((n, n))
        for ex, (Lm, _) in lhs_coef.items():
            e = ex[i]
            k = e * (e - 1) if order == 2 else e
            if k == 0:
                continue
            w = k * (par[nm] / p0[nm]) ** (e - order) / p0[nm] ** order
            for j, other in enumerate(lhs_params):
                if j != i:
                    w *= (par[other] / p0[other]) ** ex[j]
            out = out + w * Lm
        return out

    def Gp(X, par, nm, order=1):
        out = np.zeros(n) + _dp_rhs(lambda q: Fx(X, q), par, nm, order)
        dl = _dL(par, nm, order)
        return out if dl is None else out - dl @ X

    def Jp(X, v, par, nm):
        """∂/∂p (J(x; p) v), exact (Laurent) or declared FD."""
        out = np.zeros(n) + _dp_rhs(lambda q: JF_dir(X, v, q), par, nm, 1)
        dl = _dL(par, nm, 1)
        return out if dl is None else out - dl @ v

    for nm in names:
        er = rhs_exp[nm]
        if er:
            notes.append(f"dG/d{nm} (RHS): " + (f"central FD, h = {dp_fd:g} (non-polynomial)" if None in er
                                                  else f"exact Laurent reconstruction, exponents {sorted(er)}"))
    ds = DynamicalSystem(n, params=vals, origin=f"IR: {problem.name}", G=G, J=J, Gp=Gp,
                         JF_dir=Jv, B=B, C=Cf, Jp=Jp, M=None if (_is_identity(M0) and not mass_depends) else M0,
                         G_many=G_many if pointwise else None, J_many=J_many if pointwise else None,
                         notes=notes)
    if mass_depends:
        raise UnsupportedError("ir_system: the mass matrix depends on a free parameter",
                               fix="rescale time so that the dt() terms have constant coefficients")
    if pointwise:
        def Gp_many(Xs, par, nm):
            out = np.zeros(Xs.shape) + _dp_rhs(lambda q: _pw(0, Xs, (), q), par, nm, 1)
            dl = _dL(par, nm, 1)
            return out if dl is None else out - Xs @ dl.T.toarray()
        ds._Gp_many = Gp_many
    ds.solver, ds.problem, ds.pointwise = solver, problem, bool(pointwise)
    ds.lm = LM
    return ds


def _lm(coef, names, p0, par):
    L = M = None
    for ex, (Lm, Mm) in coef.items():
        w = np.prod([(par[nm] / p0[nm]) ** e for nm, e in zip(names, ex, strict=True)])
        L = w * Lm if L is None else L + w * Lm
        M = w * Mm if M is None else M + w * Mm
    return L.tocsr(), M.tocsr()
