"""Normal modes of a native model about a base state (:class:`Linearization`), split out of
:mod:`conatus.model.run` (which re-exports it): a declared wrapper over linearize and eigenvalues;
the expert objects stay reachable (``lin.discretize(N)``)."""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError, ResolutionError
from ..ir import Domain, Field, Parameter, VectorField, Zero
from ..tolerances import NCC_TAIL_TOL, REQUIRED
from .compiler import Compiled, _coord_args, _on_all_grids
from .._deprecation import renames


_GROWTH_DOC = "docs/site/guide/dynamics/nonmodal.md"
# normal modes, wavenumbers, groups (they pointed to the time steppers page)
_LIN_DOC = "docs/site/guide/dynamics/eigenvalues.md"


class Linearization:
    """Normal modes of a model about a base state (``Model.linearize(about=...)``).

    Base values: 0 (the zero field), a number, or a callable of coordinate names; a callable is a
    non-constant coefficient on the coordinates it depends on (a Fourier coordinate along which
    it is exactly constant on the grid is dropped, declared).

    Time-periodic base (``period=P``): a callable may also take ``t``; the base is then
    U = Σ_m U_m e^{2πimt/P}, its harmonics computed by a discrete Fourier transform over
    ``_TIME_SAMPLES`` instants per period (the harmonics kept and the size of the dropped ones
    are reported by ``explain()``; an under-sampled base is refused). Such a base has no normal
    modes: use :meth:`floquet` (Floquet exponents) and :meth:`optimal_gain` (transient growth)."""

    _TIME_SAMPLES = 64          # instants per period for the harmonics of a periodic base
    _TIME_TOL = 1e-13           # harmonics below this (relative) are dropped, declared

    def __init__(self, model, about: Mapping[Any, Any], eigenvalue: str = "s",
                 period: float | None = None, base_residual: str = "check") -> None:
        if base_residual not in ("check", "accept"):
            from ..errors import did_you_mean
            raise OptionError(f"linearize(base_residual={base_residual!r})",
                              why="'check' (default) computes the steady residual of the base state "
                                  "and says when it is not a steady solution; 'accept' declares a "
                                  "base that is knowingly not one (frozen or forced profile)",
                              fix="base_residual='check' or base_residual='accept'"
                                  + did_you_mean(str(base_residual), ("check", "accept")),
                              doc=_LIN_DOC)
        self.base_residual_mode = base_residual
        self._base_reports: dict = {}
        self._base_warned = False
        self.model = model
        names = {u.name for u in model.unknowns}
        self.about = {}
        for k, v in about.items():
            n = getattr(k, "name", k)
            if n not in names:
                # list the unknowns and a dict to write (the message named
                # neither)
                from ..errors import did_you_mean
                order = [u.name for u in model.unknowns]
                raise InputTypeError(
                    f"about: {n!r} is not an unknown of the model '{model.name}'"
                    + did_you_mean(str(n), order),
                    why=f"the base state gives a value to unknowns; this model's are "
                        f"{', '.join(order)}",
                    fix="about={" + ", ".join(f"{u.name!r}: ..." for u in model.unknowns
                                              if u.role != "pressure") + "} (each value a "
                        "number or a function of the coordinates)",
                    doc=_LIN_DOC)
            self.about[n] = v
        if period is not None and (isinstance(period, bool) or not (    # True: period 1
                isinstance(period, numbers.Real) and period > 0 and math.isfinite(period))):
            raise InputTypeError(f"period = {period!r}", why="the period of the base flow is a "
                                 "positive time", fix="period=1.0", doc=_LIN_DOC)
        self.period = None if period is None else float(period)
        self.eigenvalue = eigenvalue
        self.notes: list[str] = []
        self._size_note: str | None = None
        self._harmonics: dict = {}

    def _base_args(self, n: str, v: Callable, names: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
        """(coordinate arguments, depends on t) of a base callable; ``t`` needs ``period=``."""
        import inspect
        try:
            params = [p.name for p in inspect.signature(v).parameters.values()]
        except (TypeError, ValueError):
            params = []
        if "t" in params and "t" not in names:
            if self.period is None:
                raise InputTypeError(f"base state {n} depends on t but no period is given",
                                     why="a time-dependent base flow is analysed as a periodic "
                                         "one (Floquet, transient growth)",
                                     fix=f"m.linearize(about={{{n}: ...}}, period=1.0)", doc=_LIN_DOC)
            args = tuple(p for p in params if p != "t")
            bad = [a for a in args if a not in names]
            if bad:
                raise InputTypeError(f"base state {n}: argument(s) {bad} are neither coordinates "
                                     f"{names} nor t", fix=f"lambda {', '.join(names)}, t: ...",
                                     doc=_LIN_DOC)
            return args, True
        return _coord_args(v, names, f"base state {n}"), False

    def _time_harmonics(self, comp, n: str, v: Callable, args: tuple[str, ...],
                        names: tuple[str, ...]) -> tuple[dict[int, Callable], str]:
        """Harmonics {m: U_m(coords)} of a t-dependent base, U = Σ_m U_m e^{2πimt/P}, from
        ``_TIME_SAMPLES`` instants; the kept set is decided on the reference grid."""
        from ..reference.transforms import grid as _grid
        Q, P = self._TIME_SAMPLES, self.period
        ts = P * np.arange(Q) / Q

        def at(grids, t):
            return np.asarray(v(**{a: grids[names.index(a)] for a in args}, t=t))

        def coeffs(grids):
            vals = np.stack([np.asarray(at(grids, t), dtype=complex) for t in ts])
            return np.fft.fft(vals, axis=0) / Q            # index m mod Q
        grids = []
        for i, c in enumerate(names):
            g = np.asarray(_grid(comp.bases[c], dealias=False))
            sh = [1] * len(names)
            sh[i] = g.size
            grids.append(g.reshape(sh))
        C = coeffs(grids)
        # every harmonic of the Q samples, the Nyquist one (m = -Q/2) included: it used to be
        # left out of both the kept and the dropped sets, so a base with content there was
        # truncated without a word; now it is refused as aliased
        ms = np.arange(-Q // 2, Q // 2)
        amp = np.array([np.abs(C[m % Q]).max() for m in ms])
        big = amp.max() if amp.size else 0.0
        if not big > 0:
            raise InputTypeError(f"base state {n} is zero at every sampled time",
                                 fix=f"about={{{n}: 0}}", doc=_LIN_DOC)
        keep = ms[amp > self._TIME_TOL * big]
        if np.abs(keep).max() > Q // 4:
            raise InputTypeError(f"base state {n} has harmonics up to |m| = {np.abs(keep).max()} "
                                 f"at {Q} samples per period",
                                 why="its time harmonics are not resolved (aliasing): the base "
                                     "flow must be smooth in t",
                                 fix="give a smooth periodic base flow (a few harmonics)", doc=_LIN_DOC)
        drop = float(amp[amp <= self._TIME_TOL * big].max(initial=0.0) / big)
        out = {}
        for m in keep:
            def data(*g, m=int(m)):
                return coeffs(g)[m % Q]
            out[int(m)] = data
        note = (f"{n} = sum_m {n}_m e^(2 pi i m t/P), P = {P:g}: harmonics m = "
                f"{', '.join(str(int(m)) for m in keep)} from {Q} instants per period (largest "
                f"dropped harmonic {drop:.1e} relative)")
        return out, note

    def compile(self, N: Any, dealias: Any = None):
        """(EVP, ncc, compiled IVP) at resolution ``N``; the harmonics of a time-periodic base are
        in ``self.harmonics`` ({base field: {m: data}}, empty for a steady base)."""
        from ..linearize import linearize
        comp = self.model.compile(N, dealias)
        if comp.kind != "IVP":
            raise OptionError("linearize(): the model has no dt(): it is already steady/linear",
                              doc=_LIN_DOC)
        geo = self.model.geometry
        names = geo.coord_names
        around, ncc = {}, dict(comp.ncc)
        self.notes = []
        self._harmonics = {}
        for n, v in self.about.items():
            f = comp.fields[n]
            if isinstance(v, numbers.Number) and v == 0:
                around[f] = Zero(f.tensorsig)
                self.notes.append(f"{n} = 0 about the zero field")
                continue
            periodic = False
            if isinstance(v, numbers.Number):
                v = (lambda c0: (lambda *g: np.asarray(c0)))(v)
                args = ()
            else:
                args, periodic = self._base_args(n, v, names)
            if periodic:
                harm, note = self._time_harmonics(comp, n, v, args, names)
                # coordinate dependence read off the sum of the harmonic amplitudes sum_m |U_m|
                data = (lambda hs: (lambda *g: sum(np.abs(h(*g)) for h in hs.values())))(harm)
            else:
                data = _on_all_grids(v, args, names)
            keep = self._dependence(comp, data, args)
            dom = Domain(comp.coordsys, [comp.bases[c] for c in names if c in keep])
            bsym = None
            if comp.symmetry is not None:      # sine/cosine axis: the base state's checked parity
                if periodic:
                    raise OptionError("a time-periodic base on a sine/cosine axis",
                                      why="the parity of each time harmonic is not checked",
                                      fix="use a periodic or walls axis", doc=_LIN_DOC)
                from .sincos import data_parity, reflection
                R = reflection(geo, comp.coordsys)
                bsym = {R: data_parity(comp, data, f.rank, f"base state {n}", R=R)}
            b = (VectorField(f"{n}0", dom, symmetry=bsym) if f.rank == 1
                 else Field(f"{n}0", dom, symmetry=bsym))
            around[f] = b
            dropped = [c for c in args if c not in keep]
            if periodic:
                self._harmonics[b] = harm
                self.notes.append(note + (f"; exactly constant along {', '.join(dropped)}"
                                          if dropped else ""))
                continue
            ncc[b] = data
            self.notes.append(f"{n} = {n}0({', '.join(keep)}), a non-constant coefficient"
                              + (f"; exactly constant along {', '.join(dropped)} on the grid, so it "
                                 "is taken independent of it" if dropped else ""))
        s = Parameter(self.eigenvalue, dtype=complex)
        sym = None
        if comp.symmetry is not None:
            # sine/cosine axis: the perturbations live in the same parity subspace (it is the
            # discretization, not a restriction); names follow linearize's prefix "d"
            from ..ir import SymmetrySubspace
            sym = SymmetrySubspace(*comp.symmetry.generators,
                                   **{"d" + k: v for k, v in comp.symmetry.parities.items()})
            self.notes.append("perturbations in the sine/cosine parity subspace of the model "
                              "(same parities as the unknowns)")
        evp = linearize(comp.problem, around, s, symmetry=sym)
        self._check_base(comp, N, dealias, around, ncc)
        # res.mode(i) shows the perturbation variables ("du") under the unknowns' names ("u")
        evp.mode_names = {"d" + u.name: u.name for u in self.model.unknowns}
        # growth convention (normal modes exp(s t): growth = Re s, whatever the name of s): set
        # by conatus.linearize itself, one sentence for both layers
        if comp.time is not None:
            # (native2): a coefficient written with t in the equations (a base
            # flow in the right-hand side, cos(t)*Uc*dx(u)) used to be frozen silently at the
            # Parameter's value t = 0 here: the "normal modes" were those of the frozen profile.
            hit = [i for i, eq in enumerate(evp.equations)
                   if comp.time in eq.lhs.free_parameters() | eq.rhs.free_parameters()]
            if hit:
                raise OptionError(
                    f"linearize(): the linearized equations {hit} depend on the time t",
                    why="the model writes a time-dependent coefficient of the unknowns in its "
                        "equations (e.g. a base flow cos(t)*U(y) in the right-hand side); "
                        "linearizing would freeze it at t = 0",
                    fix="write the model for the full field without the base flow and pass the "
                        "base as about={u: lambda y, t: ...}, period=P (Floquet, transient "
                        "growth), e.g. cn.problems.native.this case_floquet", doc=_GROWTH_DOC)
        # a given field that only enters a source term (the
        # forcing of Kolmogorov flow) leaves with the linearization; passing its data on made
        # discretize refuse "ncc given for field(s) not used by the problem"
        used = set()
        for eq in evp.equations:
            used |= eq.lhs.free_fields() | eq.rhs.free_fields()
        gone = [f for f in ncc if f not in used]
        for f in gone:
            del ncc[f]
        if gone:
            self.notes.append("given field(s) " + ", ".join(sorted(f.name for f in gone))
                              + " only in source terms: absent from the linearized equations")
        return evp, ncc, comp

    def _check_base(self, comp: Compiled, N: Any, dealias: Any, around: Mapping,
                    ncc: Mapping) -> None:
        """Y11: the steady residual of the base state, in the
        notes (explain()); above BASE_RESIDUAL_TOL and not declared (``base_residual='accept'``),
        a BaseStateWarning once (0.1 and 0.2; refused from 0.3). Not a DeprecationWarning: Python hides
        those outside ``__main__``."""
        from ..tolerances import BASE_RESIDUAL_TOL
        # the residual depends on everything the problem is compiled from, not only (N, dealias):
        # keyed on all of it (earlier versions: after m.params["F"] = 0 the cached residual of
        # the forced flow still said "a steady solution")
        key = self._base_key(comp, N, dealias, ncc)
        if key not in self._base_reports:
            self._base_reports[key] = self._base_residual(comp, around, ncc)
        worst, where, how = self._base_reports[key]
        if worst is None:
            self.notes.append("base state: steady residual not computed (" + how + ")")
            return
        line = (f"base state: steady residual {worst:.1e} (largest relative residual of the "
                f"equations and boundary conditions at the base, {how}; worst: {where})")
        if worst <= BASE_RESIDUAL_TOL:
            self.notes.append(line + f": a steady solution to BASE_RESIDUAL_TOL = "
                              f"{BASE_RESIDUAL_TOL:g}")
            return
        if self.base_residual_mode == "accept":
            self.notes.append(line + ": not a steady solution, declared (base_residual='accept')")
            return
        msg = (line + f": NOT a steady solution (above BASE_RESIDUAL_TOL = {BASE_RESIDUAL_TOL:g}),"
               " so these are not the normal modes of a flow that exists; check the base (sign, "
               "boundary values, forcing), or declare a frozen or forced base with "
               "linearize(..., base_residual='accept'). Accepted in 0.1 and 0.2, refused from 0.3")
        self.notes.append("WARNING: " + msg)
        if not self._base_warned:
            self._base_warned = True
            from ..errors import BaseStateWarning, _warn_user
            _warn_user(msg, BaseStateWarning)

    def _base_key(self, comp: Compiled, N: Any, dealias: Any, ncc: Mapping) -> tuple:
        """Cache key of the base residual: the resolution, the parameter values, the equation
        and condition texts and values, and the grid values of every non-constant coefficient
        (base state and given fields, evaluated now: a callable whose data changed in place
        gives a new key). A coefficient without a reference grid is keyed by identity (a fresh
        callable per compilation then recomputes the residual: slower, never stale)."""
        import hashlib

        from ..reference.transforms import grid as _grid

        def digest(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return (v.shape, str(v.dtype), hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest())
            if isinstance(v, (list, tuple)):
                return tuple(digest(x) for x in v)
            if callable(v):
                return ("callable", id(v))
            return repr(v)

        names = comp.model.geometry.coord_names
        try:
            grids = []
            for i, c in enumerate(names):
                g = np.asarray(_grid(comp.bases[c], dealias=False))
                sh = [1] * len(names)
                sh[i] = g.size
                grids.append(g.reshape(sh))
        except (NotImplementedError, KeyError):
            grids = None
        coeffs = []
        for f, v in ncc.items():
            d = digest(v)
            if callable(v) and grids is not None:
                try:
                    d = digest(np.asarray(v(*grids)))
                except Exception:  # noqa: BLE001 -- another grid convention: keyed by identity
                    pass
            coeffs.append((getattr(f, "name", repr(f)), d))
        m = self.model
        conditions = tuple((c.boundary, c.text, c.unknown, digest(c.value), c.into)
                           for c in m.conditions)
        return (repr(N), repr(dealias), repr(sorted(dict(m.params or {}).items())),
                tuple(m.eqs), conditions, tuple(sorted(coeffs, key=lambda t: t[0])))

    def _base_residual(self, comp: Compiled, around: Mapping, ncc: Mapping
                       ) -> tuple[float | None, str, str]:
        """(largest relative residual, where, how) of the steady equations at the base state:
        dt() terms dropped, every equation and boundary condition evaluated at the base on the
        discretization of ``comp``; the unknowns without a base value (pressure, taus, unknowns
        that enter linearly) are fitted by least squares. Per equation, the residual is relative
        to the largest of its terms at the base. (None, '', why) when it does not apply."""
        if self.period is not None or self._harmonics:
            return None, "", "time-periodic base"
        if comp.symmetry is not None:
            return None, "", "sine/cosine axis"
        if comp.time is not None and any(
                comp.time in eq.lhs.free_parameters() | eq.rhs.free_parameters()
                for eq in comp.problem.equations):
            return None, "", "the equations depend on t"
        try:
            return self._steady_residual(comp, around, ncc)
        except Exception as exc:  # noqa: BLE001 -- a detection: say why it did not run, never fail the solve
            return None, "", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

    def _steady_residual(self, comp: Compiled, around: Mapping, ncc: Mapping
                         ) -> tuple[float, str, str]:
        """Residual F(U) of the steady equations at the base U, from the problem that the
        linearization already discretizes: J d = -F(U) with J = F'(U) (same base fields, same
        non-constant coefficients as the eigenproblem) and F(U) evaluated with every unknown
        without a base value set to 0; those unknowns (pressure, taus) then enter F linearly, so
        min over them of |F(U) + J_free d_free| is a least-squares problem per group."""
        import scipy.sparse as sp

        from ..discretize import discretize
        from ..ir import expr as E
        from ..ir.problem import LBVP, _terms
        pb = comp.problem
        V = frozenset(pb.variables)
        pert = {v: E.Field("d" + v.name, v.domain, v.tensorsig, v.dtype) for v in pb.variables}
        # unknowns at 0 (no base value, or a zero base): every term holding one vanishes at the
        # base (the terms are polynomial in the unknowns) and is left out; the others are
        # evaluated with the base fields (integ(0) has no domain to integrate over)
        zero = frozenset(v for v in pb.variables if v not in around or around[v].is_zero())
        at_base = {v: b for v, b in around.items() if v not in zero}

        def value_at_base(terms):
            kept = [t for t in terms if not (t.free_fields() & zero)]
            return E.add(*kept).substitute(at_base) if kept else None

        def nodt(e):
            m = {n: Zero(n.tensorsig) for n in e.walk() if isinstance(n, E.TimeDerivative)}
            return e.substitute(m) if m else e
        total = LBVP(list(pert.values()), name=f"steady residual of {pb.name}")
        linear = LBVP(list(pert.values()), name=f"linear part, {pb.name}")
        texts = []
        for eq in pb.equations:
            F = nodt(E.add(eq.lhs, E.neg(eq.rhs)))
            J = F.linearize(dict(around), pert)
            lin = [t for t in _terms(F) if t.degrees(V) == frozenset({1})]
            ft, fl = value_at_base(_terms(F)), value_at_base(lin)
            total.add_equation((J, 0 if ft is None else E.neg(ft)))
            linear.add_equation((J, 0 if fl is None else E.neg(fl)))
            texts.append(_without_taus(str(eq)))

        def rhs_of(lb):
            used = set()
            for eq in lb.equations:
                used |= eq.lhs.free_fields() | eq.rhs.free_fields()
            data = {f: d for f, d in ncc.items() if f in used}
            plan = discretize(lb, ncc=data or None)
            gdata = {f: plan._as_grid(f, d) for f, d in data.items()}
            # evaluated on the full domain: the right-hand side of a boundary condition holds the
            # trace of a base field here (T0(z=-0.5)), which the evaluator of the boundary
            # domain alone cannot sample
            from ..reference.evaluate import ReferenceEvaluator
            ev = ReferenceEvaluator(plan.domain)
            out = []
            for eq in lb.equations:
                v = ev.evaluate(eq.rhs, gdata, plan.params)
                g_ = np.broadcast_to(v, plan._grid_shape(eq.domain, eq.rhs.tensorsig)).copy()
                out.append(plan._to_coeffs(g_, eq.domain, eq.rhs.rank))
            return plan, out
        plan, rt = rhs_of(total)
        plan_l, rl = rhs_of(linear)
        given = {"d" + n for n in self.about}
        free_names = [v.name for v in plan.variables if v.name not in given]
        worst, where = 0.0, "-"
        for g in plan.groups:
            gm = plan._assemble(g)
            if gm.L.shape[0] == 0:
                continue
            b = plan._group_rhs(g, rt, gm)                       # -F(U), free unknowns at 0
            bl = plan_l._group_rhs(g, rl, plan_l._assemble(g))   # -(linear terms)(U)
            A = sp.csr_matrix(gm.L)
            free = [np.arange(vl.offset, vl.offset + vl.size)
                    for vl in plan.var_layouts(g) if vl.name in free_names]
            Jd = np.zeros_like(b)
            if free:
                Af = A[:, np.concatenate(free)].toarray()
                d = np.linalg.lstsq(Af, b, rcond=None)[0]
                Jd = Af @ d
            r = Jd - b                                          # F(U) + J_free d_free
            # rounding floor: an equation whose terms all vanish at the base (div u with u = 0,
            # the pressure gauge) holds the rounding of the least-squares fit; a residual below
            # 1e3 eps x the largest term of the group is that rounding, not a defect
            floor = 1e3 * np.finfo(float).eps * max(float(np.abs(bl).max(initial=0.0)),
                                                    float(np.abs(b - bl).max(initial=0.0)),
                                                    float(np.abs(Jd).max(initial=0.0)))
            for i, el in enumerate(plan.eq_layouts(g)):
                if el.size == 0:
                    continue
                rows = slice(el.offset, el.offset + el.size)
                scale = max(float(np.abs(bl[rows]).max()), float(np.abs((b - bl)[rows]).max()),
                            float(np.abs(Jd[rows]).max()), floor)
                res = float(np.abs(r[rows]).max())
                rel = 0.0 if res <= floor else res / scale
                if rel > worst:
                    worst, where = rel, texts[i][:80]
        # the user's unknowns by name; the automatic boundary (tau) variables are not shown
        names_u = {u.name for u in self.model.unknowns}
        free_shown = [n[1:] for n in free_names if n[1:] in names_u]
        taus = len(free_shown) < len(free_names)
        how = ("fitted by least squares: " + ", ".join(free_shown + (
            ["the automatic boundary terms"] if taus else []))) if free_names else \
            "every unknown given"
        return worst, where, how

    @property
    def harmonics(self) -> dict:
        """{base field: {m: data}} of the last compiled time-periodic base (empty if steady)."""
        return self._harmonics

    @staticmethod
    def _dependence(comp: Compiled, data: Callable, args: tuple[str, ...]) -> tuple[str, ...]:
        """Coordinates the base state depends on: its arguments, minus Fourier coordinates along
        which its grid values are exactly constant."""
        from ..reference.transforms import grid as _grid
        names = comp.model.geometry.coord_names
        grids = []
        try:
            for i, c in enumerate(names):
                g = np.asarray(_grid(comp.bases[c], dealias=False))
                sh = [1] * len(names)
                sh[i] = g.size
                grids.append(g.reshape(sh))
        except NotImplementedError:        # no reference grid for this basis: keep every argument
            return tuple(args)
        # The user's base state itself is evaluated outside the try: an error in it is the
        # user's error and is raised here (a blind except used to turn it into "depends on all").
        val = np.asarray(data(*grids))
        keep = []
        for c in args:
            i = names.index(c)
            ax = val.ndim - len(names) + i
            if comp.bases[c].periodic and val.shape[ax] > 1 and \
                    np.array_equal(val, np.take(val, [0], axis=ax).repeat(val.shape[ax], axis=ax)):
                continue
            if comp.bases[c].periodic and val.shape[ax] == 1:
                continue
            keep.append(c)
        return tuple(keep)

    def _N(self, N: Any, k: Any) -> Any:
        """Int N with Fourier coordinates: N applies to the wall coordinate, and each Fourier
        coordinate gets the smallest size holding wavenumber k (declared: a group's matrices do
        not depend on the other wavenumbers)."""
        geo = self.model.geometry
        names = geo.coord_names
        if getattr(geo, "terrain", None) is not None and k is not None:
            raise OptionError(f"k={k!r} on a mapped domain",
                              why="the metric couples all horizontal wavenumbers: the spectrum is "
                                  "computed for all of them at once, there is no group per k",
                              fix="drop k= and give one size per coordinate, N=(N_x, N_z)",
                              doc="docs/site/guide/mapped.md")
        if not isinstance(N, numbers.Integral) or len(names) == 1:
            return N
        wall = geo.bounded_coord               # walls, half line or whole line
        fourier = [c for c in names if c != wall]
        if k is None or wall is None:
            raise ResolutionError(f"N={N} for coordinates {names}: give one size per coordinate, "
                                  "or k= to fix the Fourier sizes", doc=_LIN_DOC)
        ks = self._k_map(k, fourier)
        out = {wall: int(N)}
        for c in fourier:
            L = geo._axes[c].bounds[1] - geo._axes[c].bounds[0]
            n = ks[c] * L / (2 * math.pi)
            if abs(n - round(n)) > 1e-9:
                raise ResolutionError(f"k_{c} = {ks[c]:g} is not a wavenumber of the periodic "
                                      f"length {L:g} (2 pi n / L)", doc=_LIN_DOC)
            out[c] = 2 * int(round(abs(n))) + 2
        self._size_note = ("Fourier sizes chosen to hold k: " + ", ".join(
            f"N_{c} = {out[c]}" for c in fourier) + " (the matrices of one wavenumber do not "
            "depend on the others)")
        return out

    @staticmethod
    def _k_map(k: Any, fourier: list[str]) -> dict:
        # the one wavenumber check of the native layer (audit inputs: here a dict
        # key that named no periodic coordinate was ignored, a complex value was a bare TypeError
        # and True was read as 1)
        from .model import _k_map
        return _k_map(k, fourier)

    def discretize(self, N: Any, dealias: Any = None, k: Any = None, **kw: Any):
        """The Plan of the EVP (expert: its groups and matrices)."""
        from ..discretize import discretize
        self._size_note = None
        evp, ncc, _ = self.compile(self._N(N, k), dealias)
        if self._size_note:
            self.notes.append(self._size_note)
        if self._harmonics:
            raise OptionError("a time-periodic base flow has no normal modes exp(s t)",
                              why="the operator depends on t; its stability is a Floquet problem",
                              fix="lin.floquet(N, k=..., method='hill', Nh=..., targets=[...], tol=1e-12) or "
                                  "lin.optimal_gain(N, k=..., t0=..., T=..., steps=..., "
                                  "scheme='RK443')", doc=_GROWTH_DOC)
        return discretize(evp, ncc=ncc or None, **kw)

    def group(self, plan, k: Any = None, **which: Any):
        """The group of wavenumber ``k`` (Cartesian) or ``m=``/``kz=``/``ell=`` (curvilinear)."""
        if which:
            return plan.group(**which)
        if k is None:
            if len(plan.groups) > 1:
                # Y3: with one size per coordinate and no k, the first group was
                # solved without a word
                from .model import _select_group
                return _select_group(plan, self._wavenumber_coords(), None, {})
            return plan.groups[0]
        geo = self.model.geometry
        fourier = [c for c in geo.coord_names if c != geo.bounded_coord]
        ks = self._k_map(k, fourier)
        want = [ks[c] for c in fourier]
        for g in plan.groups:
            if all(abs(item.k - kk) < 1e-12 * max(1.0, abs(kk)) for item, kk in zip(g, want, strict=True)):
                return g
        raise ResolutionError(f"no group with k = {k!r} at this resolution", doc=_LIN_DOC)

    def eigenvalues(self, N: Any, *, k: Any = None, dealias: Any = None, drift: Any = None,
                    n: int | None = None, **kw: Any):
        """Eigenvalues s of the normal modes exp(s t) of wavenumber k (conatus.eigenvalues).
        On a sine/cosine axis every vertical wavenumber n pi / L (n < N) is its own group: the
        result gathers all of them (or only ``n=``)."""
        from ..solvers.plan_evp import eigenvalues
        which = {n_: kw.pop(n_) for n_ in ("m", "kz", "ell") if n_ in kw}
        from .model import _EIG_OPTIONS, _check_drift, check_call_options, ncc_options
        given = [n_ for n_ in kw if n_ in (self.model.params or {})
                 and n_ not in _EIG_OPTIONS + ("n_modes",)]
        if given:
            # a declared parameter was reported as "unknown keyword ... declared
            # parameters: none"; a linearization takes its values from the model
            raise OptionError(f"lin.eigenvalues(..., {given[0]}=...): a parameter value is not a "
                              "keyword of a linearization",
                              why="the base state and the operator are compiled with the model's "
                                  "parameter values",
                              fix=f"m.params[{given[0]!r}] = ... before the call, or "
                                  f"lin.parametric(N, over=({given[0]!r},)) to sweep it",
                              doc=_LIN_DOC)
        check_call_options(None, kw, _EIG_OPTIONS + ("n_modes",), "lin.eigenvalues")
        from ..solvers.plan_evp import check_eig_method   # before the base state
        check_eig_method(kw.get("method"), kw.get("target"), kw.get("region"), "lin.eigenvalues")
        if which and k is not None:
            # a curvilinear group is chosen by m=/kz=/ell=: k used to be ignored without a
            # word (the shell docstring passed k=4 meaning four shift-invert modes)
            if isinstance(k, numbers.Integral) and not isinstance(k, bool) and \
                    "n_modes" not in kw and kw.get("method") == "shift-invert":
                from ..errors import _warn_user
                _warn_user(f"lin.eigenvalues(..., {', '.join(f'{a}={b!r}' for a, b in which.items())}"
                           f", k={k}): the group is chosen by {', '.join(which)}; k={k} is read as "
                           f"the number of shift-invert modes, write n_modes={k}",
                           DeprecationWarning)
                kw["n_modes"], k = int(k), None
            else:
                raise OptionError(f"lin.eigenvalues(..., k={k!r}) with "
                                  + ", ".join(f"{a}={b!r}" for a, b in which.items()),
                                  why="on a curvilinear geometry the group is chosen by m= (and "
                                      "kz= or ell=); k would be ignored",
                                  fix="drop k= (kz= is the axial wavenumber of a pipe or a "
                                      "cylinder); n_modes= is the number of shift-invert modes",
                                  doc=_LIN_DOC)
        dkw = ncc_options(kw)
        geo = self.model.geometry
        if geo.sincos_coord is not None:
            if drift is not None:
                raise OptionError("drift= on a sine/cosine axis",
                                  why="drift= filters spurious tau modes; a sine/cosine axis has "
                                      "no tau (each group is exact)", fix="drop drift=", doc=_LIN_DOC)
            plan = self.discretize(N, dealias, k=k, **dkw)
            fourier = [c for c in geo.coord_names if c != geo.sincos_coord]
            want = self._k_map(k, fourier) if fourier else {}
            return sincos_eigenvalues(plan, geo, want, n, **kw)
        if n is not None:
            raise OptionError("n= selects a sine/cosine wavenumber: no sine/cosine axis here",
                              doc=_LIN_DOC)
        plan = self.discretize(N, dealias, k=k, **dkw)
        g = self.group(plan, k, **which)
        _check_drift(drift, kw, N)
        if drift is not None:
            kw["drift_plan"] = self.discretize(drift, dealias, k=k, **dkw)
        return eigenvalues(plan, g, **kw)

    def _wavenumber_coords(self) -> list[str]:
        """The periodic coordinates whose wavenumber ``k=`` selects a group."""
        geo = self.model.geometry
        return [c for c in geo.coord_names if c != geo.bounded_coord]

    def wavenumber_names(self) -> dict[str, str]:
        """{name usable in ``parametric(over=...)``: Fourier coordinate}: ``"k"`` with one
        periodic axis, ``"k<coord>"`` (``"kx"``, ``"ky"``) with several."""
        geo = self.model.geometry
        fourier = [c for c in geo.coord_names if c != geo.bounded_coord]
        if len(fourier) == 1:
            return {"k": fourier[0]}
        return {f"k{c}": c for c in fourier}

    def parametric(self, N: Any, over: Any, *, k: Any = None, dealias: Any = None,
                   drift: Any = None, drift_tol: Any = None, **kw: Any):
        """``ParametricEVP`` of the normal modes in ``over`` (same contract as
        ``Model.parametric``): declared parameters of the model (``"Ra"``) and the
        wavenumber (``"k"``; ``"kx"``, ``"ky"`` with several periodic axes, see
        :meth:`wavenumber_names`), which becomes a continuous pencil parameter, so that
        ``X.critical(lin.parametric(N, over=("Ra", "k"), drift=N2, drift_tol=1e-6), "Ra",
        bracket=(a, b), over={"k": (k1, k2)}, xtol=1e-9)`` gives the critical point of a ready
        linearization. Values left out of a call take the declared ones (logged): the model's
        parameters, and for a wavenumber ``k=`` or, if not given, the fundamental 2 pi / L of the
        box. A periodic axis not in ``over`` is fixed at ``k=`` (the group of that wavenumber, as
        ``eigenvalues(N, k=)``). ``drift=N2`` with its required ``drift_tol`` keeps only the
        modes found again at N2 (the others are not converged by this test), as ``Model.parametric``."""
        from ..solvers.parametric import ParametricEVP
        from .model import _check_drift
        from ..errors import did_you_mean
        if isinstance(over, str):
            over = (over,)
        over = tuple(over)
        if self.period is not None:
            raise OptionError("parametric(): a time-periodic base flow has no normal modes "
                              "exp(s t)", fix="lin.floquet(N, k=..., method='hill', ...)",
                              doc=_GROWTH_DOC)
        geo = self.model.geometry
        if getattr(geo, "terrain", None) is not None or geo.sincos_coord is not None:
            raise OptionError("parametric() on a mapped domain or a sine/cosine axis",
                              why="the pencil is built for one group of Fourier wavenumbers on "
                                  "Chebyshev walls", fix="lin.eigenvalues(N, ...) per value",
                              doc=_LIN_DOC)
        wn = self.wavenumber_names() if geo.bounded_coord is not None else {}
        params = self.model.params
        clash = [n for n in wn if n in params and n in over]
        if clash:
            raise OptionError(f"parametric(over={over!r}): {clash[0]!r} is both a parameter of the "
                              "model and the name of the wavenumber", fix="rename the model "
                              "parameter", doc=_LIN_DOC)
        unknown = [n for n in over if n not in params and n not in wn]
        if unknown:
            raise OptionError(f"parametric(over={over!r}): {unknown[0]!r} is neither a declared "
                              "parameter nor a wavenumber." + did_you_mean(
                                  unknown[0], list(params) + list(wn)),
                              fix="declared parameters: " + (", ".join(params) or "none")
                                  + "; wavenumbers: " + (", ".join(wn) or "none (no periodic axis)"),
                              doc=_LIN_DOC)
        if drift is not None:
            _check_drift(drift, {"drift_tol": REQUIRED if drift_tol is None else drift_tol}, N)
        elif drift_tol is not None:
            raise OptionError("parametric(drift_tol=...) without drift=N2", fix="add drift=N2 "
                              "(a finer resolution) or drop drift_tol", doc=_LIN_DOC)
        fourier = list(wn.values())
        free = {wn_name: c for wn_name, c in wn.items() if wn_name in over}
        fixed = [c for c in fourier if c not in free.values()]
        given = {}
        if k is not None:
            given = self._k_map(k, fourier) if fixed or isinstance(k, Mapping) or len(fourier) > 1 \
                else {fourier[0]: float(k)}
        elif fixed:
            raise ResolutionError(f"parametric(): the wavenumber along {fixed} is not in over: "
                                  "give it with k=", fix="k=<wavenumber>, or add it to over",
                                  doc=_LIN_DOC)
        # sizes: a fixed axis holds its wavenumber; a free one is synthetic (domain length unused)
        size_k = {c: (given[c] if c in fixed else 2 * math.pi / (geo._axes[c].bounds[1]
                                                                 - geo._axes[c].bounds[0]))
                  for c in fourier}
        nominal = {n_: given.get(c, size_k[c]) for n_, c in free.items()}
        for n_, v in nominal.items():
            if not v:
                raise OptionError(f"parametric(): k = 0 for {n_}", why="the wavenumber is a "
                                  "pencil coordinate and sets its scale (nonzero)",
                                  fix="k=<nonzero wavenumber>", doc=_LIN_DOC)

        def group_of(g) -> bool:
            for c, item in zip(fourier, g, strict=True):
                if c in fixed and abs(item.k - size_k[c]) > 1e-12 * max(1.0, abs(size_k[c])):
                    return False
                if c not in fixed and not item.k > 0:
                    return False
            return True

        def pencil(n: Any):
            self._size_note = None
            evp, ncc, comp = self.compile(self._N(n, size_k or None), dealias)
            if self._harmonics:
                raise OptionError("parametric(): a time-periodic base flow", doc=_GROWTH_DOC)
            P = ParametricEVP(evp, [comp.params[n_] for n_ in over if n_ in params],
                              ncc=ncc or None, group=group_of if fourier else None,
                              wavenumbers={c: Parameter(n_, nominal[n_]) for n_, c in free.items()},
                              **kw)
            P.declared = {**{n_: params[n_] for n_ in over if n_ in params}, **nominal}
            return P
        P = pencil(N)
        if drift is not None:
            P.set_drift(pencil(drift), drift_tol=drift_tol)
        return P

    def periodic_operator(self, N: Any, *, k: Any = None, dealias: Any = None,
                          fields: Any = None, rank_tol: float = 1e-10,
                          ncc_cutoff: float | None = None, ncc_tail_tol: float = NCC_TAIL_TOL,
                          **which: Any):
        """The :class:`conatus.growth.PeriodicOperator` of wavenumber ``k``: the linearized
        equations M X' + L0 X = -Σ_m e^{2πimt/P} B_m X with the base flow's harmonics (a steady
        base is in L0, implicit, and B = 0), the exact energy of ``fields`` (default: the velocity unknowns)
        and the admissible states (continuity and boundary conditions)."""
        from ..discretize import discretize
        from ..growth.periodic import periodic_operator_from_plan
        self._size_note = None
        evp, ncc, comp = self.compile(self._N(N, k), dealias)
        if self._size_note:
            self.notes.append(self._size_note)
        if fields is None:
            fields = tuple(u.name for u in self.model.unknowns if u.role == "velocity")
            others = [u.name for u in self.model.unknowns
                      if u.role not in ("velocity", "pressure")]
            if not fields:
                raise OptionError("no velocity unknown: the energy is not defined by default",
                                  fix="fields=('T',) (unknowns whose energy 1/2 int |f|^2 is "
                                      "measured)", doc=_GROWTH_DOC)
            if others:
                raise OptionError(f"the model also evolves {others}: which energy?",
                                  why="the kinetic energy ignores them, which can make the gain "
                                      "unbounded or meaningless",
                                  fix=f"fields={fields + tuple(others)!r} or fields={{...: weight}}",
                                  doc=_GROWTH_DOC)
        if isinstance(fields, str):
            fields = (fields,)
        wts = (dict(fields) if isinstance(fields, Mapping) else {f: 1.0 for f in fields})
        bad = [f for f in wts if f not in {u.name for u in self.model.unknowns}]
        if bad:
            raise OptionError(f"fields {bad} are not unknowns of the model", doc=_GROWTH_DOC)
        wts = {"d" + f: w for f, w in wts.items()}          # perturbation names (prefix d)
        static = dict(ncc)

        def make_plan(h):
            return discretize(evp, ncc={**static, **h} or None, ncc_cutoff=ncc_cutoff,
                              ncc_tail_tol=ncc_tail_tol)
        plan0 = make_plan({b: (lambda *g: 0.0 * g[0]) for b in self._harmonics})
        g = self.group(plan0, k, **which)
        op = periodic_operator_from_plan(make_plan, self._harmonics, fields=wts,
                                         period=self.period or 1.0, group=g,
                                         label=f"{self.model.name} linearized, k = {k!r}",
                                         rank_tol=rank_tol, steady=not self._harmonics)
        geo = self.model.geometry
        real_axes = [c for c in geo.coord_names if c != geo.bounded_coord
                     and not getattr(geo._axes.get(c), "complex", True)]
        if real_axes and k is not None:
            op.notes.append(f"Real Fourier axes {real_axes}: the group holds the cos/sin pairs, i.e. "
                            "the complex waves (kx, +-kz) together; the gain is the larger of "
                            "the two (cn.periodic(L, complex=True) separates them).")
        op.notes += list(self.notes)
        return op

    @renames("conatus.model.run:Linearization.optimal_gain")
    def optimal_gain(self, N: Any, *, k: Any = None, t0: float, T: float, steps: int,
                     scheme: str, backend: str = "cpu", dealias: Any = None, fields: Any = None,
                     **which: Any):
        """Optimal energy gain max E(t0 + T)/E(t0) over admissible perturbations of wavenumber
        ``k`` (conatus.growth.OptimalGain: ``.G``, ``.seed``, ``.explain()``). The time stepper
        (IMEX Runge-Kutta: RK111, RK222, RK443) and dt = T/steps are yours, as in simulate()."""
        from ..growth.periodic import check_gain_args, operator_gain
        check_gain_args(t0, T, steps, backend)          # before the base (review, inputs)
        op = self.periodic_operator(N, k=k, dealias=dealias, fields=fields, **which)
        return operator_gain(op, t0, T, steps, scheme, backend)

    @renames("conatus.model.run:Linearization.floquet")
    def floquet(self, N: Any, *, k: Any = None, method: str, dealias: Any = None,
                fields: Any = None, steps: int | None = None, **kw: Any):
        """Floquet exponents of the perturbations of wavenumber ``k`` about the time-periodic base
        (conatus.growth.FloquetExponents): ``method="hill"`` with ``Nh`` and ``targets``, or
        ``method="monodromy"`` with ``steps`` (time steps per period) and ``scheme``."""
        import inspect

        from .._deprecation import aliases_of
        from ..growth.periodic import operator_floquet
        from .model import check_call_options
        # every keyword is checked here, with the close names (earlier versions: nmodes=2
        # reached operator_floquet as a bare TypeError naming neither lin.floquet nor n_modes)
        accepted = tuple(n for n in inspect.signature(operator_floquet).parameters
                         if n not in ("op", "method", "steps")) + tuple(aliases_of(operator_floquet)) \
            + ("m", "kz", "ell", "ncc_cutoff", "ncc_tail_tol")
        check_call_options(None, kw, accepted, "lin.floquet")
        if method == "monodromy" and (steps is None or kw.get("scheme") is None):
            raise OptionError("lin.floquet(method='monodromy') needs steps= (time steps per period) "
                              "and scheme= (the time stepper of the monodromy matrix)",
                              fix="lin.floquet(N, k=..., method='monodromy', steps=400, "
                                  "scheme='RK443')", doc=_GROWTH_DOC)
        if steps is not None:
            from ..timestep.integrator import check_steps
            kw["steps"] = check_steps(steps, "lin.floquet(method='monodromy')", "steps")
        if self.period is None:
            raise OptionError("floquet(): the base flow is steady",
                              fix="eigenvalues(...) for a steady base; linearize(about=..., "
                                  "period=P) for a periodic one", doc=_GROWTH_DOC)
        which = {n_: kw.pop(n_) for n_ in ("m", "kz", "ell", "ncc_cutoff", "ncc_tail_tol")
                 if n_ in kw}
        op = self.periodic_operator(N, k=k, dealias=dealias, fields=fields, **which)
        return operator_floquet(op, method=method, **kw)




    def explain(self, N: Any = None, *, k: Any = None, full: bool = False) -> str:
        """Base state and eigenvalue convention (a few lines); with ``N``, how the base enters.
        ``full=True`` with ``N``: also every decision of the compiled eigenproblem (operator
        terms, taus and their modes, gauge, bases). ``N`` as in
        :meth:`eigenvalues`: one int is the wall-normal size, the Fourier sizes then hold ``k``
        (default: the fundamental wavenumber 2 pi / L of each periodic axis; the decisions do not
        depend on it; ``lin.explain(24)`` takes the same N as
        ``lin.eigenvalues(24, k=...)``)."""
        lines = [f"Linearization of '{self.model.name}' about a base state, eigenvalue "
                 f"{self.eigenvalue} (normal modes exp({self.eigenvalue} t)):"]
        compiled = None
        if N is not None:
            geo = self.model.geometry
            if (k is None and isinstance(N, numbers.Integral) and len(geo.coord_names) > 1
                    and geo.bounded_coord is not None and getattr(geo, "terrain", None) is None):
                k = {c: 2 * math.pi / (geo._axes[c].bounds[1] - geo._axes[c].bounds[0])
                     for c in geo.coord_names if c != geo.bounded_coord}
            compiled = self.compile(self._N(N, k))
        lines += ["  " + n for n in self.notes] or ["  base: " + ", ".join(self.about)]
        if compiled is not None and full:
            evp, _ncc, comp = compiled
            wall = [d for d in comp.decisions() if d.startswith("  ") and " lifted into " in d]
            lines += ["  boundary conditions -> tau terms (as in the model):"] + ["  " + d for d in wall]
            lines += ["  " + g for g in comp.gauge]
            lines.append("  eigenproblem (expert layer):")
            lines += ["    " + ln for ln in evp.explain().splitlines()]
        elif compiled is not None:
            lines.append("  (tau placement and the eigenproblem: explain(N, full=True))")
        return "\n".join(lines)


def _without_taus(text: str) -> str:
    """An equation as the user wrote it: the automatic tau terms left out (for messages)."""
    import re
    t = re.sub(r"\s*\+\s*dot\(\w+, lift\(tau_\w+, \w+, -?\d+\)\)", "", text)
    t = re.sub(r"\s*\+\s*lift\(tau_\w+, \w+, -?\d+\)", "", t)
    t = re.sub(r"lift\(tau_\w+, \w+, -?\d+\)\s*\+\s*", "", t)
    t = re.sub(r"\s*\+\s*tau_\w+\b", "", t)
    return t


def sincos_eigenvalues(plan, geo, want: Mapping, n: int | None = None, **kw: Any):
    """Eigenvalues of every group of ``plan`` whose Fourier wavenumbers are ``want`` (name -> k),
    across the sine/cosine wavenumbers n pi / L (or only ``n``), gathered in one EVPResult
    (eigenvectors dropped: their sizes differ per group; use the IR plan for them)."""
    from ..solvers.evp import EVPResult
    from ..solvers.plan_evp import eigenvalues
    if kw.get("region") is not None:
        # the gathered result has no single count; refused, not approximated
        raise OptionError("eigenvalues(region=...) on a sine/cosine axis",
                          why="the result gathers one group per vertical wavenumber; a certified "
                              "region count is defined per group",
                          fix="n=<one vertical wavenumber> with the expert call "
                              "conatus.eigenvalues(plan, group, region=...), or no region=",
                          doc="docs/site/guide/dynamics/contour.md")
    names = [d.coord.name for d in plan.layout.fdirs]
    w = geo.sincos_coord
    a, b = geo._axes[w].bounds
    kz0 = math.pi / (b - a)
    groups = []
    for g in plan.groups:
        ks = dict(zip(names, (item.k for item in g), strict=True))
        if any(abs(ks[c] - kk) > 1e-12 * max(1.0, abs(kk)) for c, kk in want.items()):
            continue
        nz = int(round(ks[w] / kz0))
        if n is None or nz == n:
            groups.append((nz, g))
    if not groups:
        raise ResolutionError(f"no group with k = {dict(want)!r}" + (f", n = {n}" if n is not None
                                                                     else "") + " at this resolution",
                              doc=_LIN_DOC)
    if kw.get("left"):
        # left=True was overwritten silently (eigenvectors are dropped here)
        raise OptionError("left=True on a sine/cosine axis: the gathered result has no "
                          "eigenvectors (their sizes differ per group)",
                          why="every argument given must be used",
                          fix="pick one group with n=... and use the IR plan "
                              "(conatus.eigenvalues(plan, group, left=True))", doc=_LIN_DOC)
    kw = dict(kw, left=False)
    res = [(nz, eigenvalues(plan, g, **kw)) for nz, g in groups]
    if len(res) == 1:
        return res[0][1]
    vals = np.concatenate([r.values for _, r in res])
    resid = np.concatenate([r.residuals for _, r in res])
    log = [f"sine/cosine axis {w}: groups n = {res[0][0]}..{res[-1][0]} (n pi/L) gathered"]
    log += [f"n = {nz}: {r.values.size} finite, {r.n_infinite} infinite" for nz, r in res]
    label = ", ".join(f"k_{c} = {kk:g}" for c, kk in want.items()) or "all groups"
    return EVPResult(vals, None, resid, res[0][1].method, f"{label}, n < N_{w}",
                     sum(r.n_infinite for _, r in res), log=log)
