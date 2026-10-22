"""The native model: unknowns, parameters, equations as on paper, conditions by boundary name.

A :class:`Model` only *declares*. The IR problem (bases, fields, tau variables, lifts, gauges,
implicit/explicit sides) is built by :func:`conatus.model.compiler.compile_model` when a
resolution is given (``simulate``, ``eigenvalues``, ``discretize``, ``compile``), by fixed rules
that ``explain()`` lists. Declarations are checked as soon as they are made (the equations are
parsed at a probe resolution), so a typo fails at the line that contains it.
"""
from __future__ import annotations

import math
import numbers
import re
import warnings
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..errors import (BoundaryConditionError, EquationError, InputTypeError, OptionError,
                      ResolutionError, did_you_mean)
from .compiler import Compiled, compile_model
from .geometry import DEFAULT_DEALIAS, Geometry

__all__ = ["Model", "Unknown", "Given"]

_DOC = "docs/site/guide/equations.md"   # public guide page (never an internal design note)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LABEL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:(?!=)\s*(.*)$", re.S)
_MACRO = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)$", re.S)
_RESERVED = {"dt", "grad", "div", "lap", "curl", "dot", "cross", "outer", "conj", "exp", "log",
             "sin", "cos", "tanh", "sqrt", "abs", "absolute", "sign", "comp", "diff", "integ",
             "interp", "lift", "pi", "inf", "left", "right", "explicit", "implicit", "rvec"}
# instance attributes of Model (a parameter cannot take these names: m.<parameter> = value)
_MODEL_ATTRIBUTES = {"geometry", "name", "params", "unknowns", "givens", "macros", "functions",
                     "eqs", "conditions", "eig", "eig_growth"}


@dataclass(eq=False)
class Unknown:
    """Handle of an unknown field (``m.velocity("u")``); usable as a key (``about={u: 0}``)."""
    name: str
    rank: int
    role: str

    def __repr__(self) -> str:
        return f"<unknown {self.role} {self.name}>"

    def explain(self) -> str:
        """What this unknown is (every key object has one)."""
        kind = {0: "scalar", 1: "vector"}.get(self.rank, f"rank-{self.rank} tensor")
        role = "" if self.role == kind else f" ({self.role})"
        return (f"unknown {self.name}: a {kind} field{role}; its initial value "
                f"sim.initial({self.name}=...), its grid values sim.grid({self.name!r}), its "
                f"coefficients sim.coeffs({self.name!r})")

    def __str__(self) -> str:
        return self.name


@dataclass(eq=False)
class Given:
    """A known field (non-constant coefficient or data): a callable of coordinate names, or a
    number. ``lambda y: 1 - y**2`` depends on y only (its argument names declare it)."""
    name: str
    value: Any
    rank: int

    def __repr__(self) -> str:
        return f"<given {self.name}>"

    def explain(self) -> str:
        """What this given field is."""
        v = self.value
        what = (f"the function {getattr(v, '__name__', type(v).__name__)} of its argument names"
                if callable(v) else f"the constant {v!r}")
        return (f"given field {self.name} (rank {self.rank}): {what}; expanded on the grid at "
                "compile time, its coefficient tail checked (NCC_TAIL_TOL)")

    def __str__(self) -> str:
        return self.name


@dataclass
class _Condition:
    boundary: str
    text: str | None
    unknown: str | None
    value: Any
    into: Any


def _split_lines(text: str) -> list[str]:
    """Logical lines: '#' comments dropped; a line continues while parentheses are open or it
    ends with an operator, ',' or '\\'."""
    out, cur, depth = [], "", 0
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() and not cur:
            continue
        cont = line.endswith("\\")
        if cont:
            line = line[:-1]
        cur = (cur + " " + line.strip()) if cur else line.strip()
        depth = cur.count("(") - cur.count(")")
        if depth > 0 or cont or re.search(r"[-+*/@=,(·]\s*$", cur):
            continue
        out.append(cur)
        cur = ""
    if cur:
        out.append(cur)
    return out


def _check_drift(drift: Any, kw: dict, N: Any = None) -> None:
    """``drift=N2`` needs its tolerance, in (0, DRIFT_TOL_MAX], and a resolution finer than
    ``N``: refused before any solve, with a value to write."""
    if drift is not None:
        from ..solvers.evp import check_drift_tol
        from ..tolerances import REQUIRED
        check_drift_tol(kw.get("drift_tol", REQUIRED), "eigenvalues(N, drift=N2)")
        if N is not None:
            _check_drift_finer(N, drift)


def _size_total(n: Any) -> int | None:
    """Number of modes of one resolution entry (an int, or a tuple of piece sizes)."""
    if isinstance(n, numbers.Integral) and not isinstance(n, bool):
        return int(n)
    if isinstance(n, (tuple, list)) and all(isinstance(v, numbers.Integral) for v in n):
        return int(sum(n))
    return None


def _check_drift_finer(N: Any, drift: Any) -> None:
    """Refuse a drift resolution that is not finer than ``N``: no entry smaller and at least one
    larger (an int against an int: the wall-normal size). Shapes that do not compare (a dict
    against a tuple) are left to the expert check on the matrix sizes."""
    pairs: list[tuple[Any, Any]] = []
    if isinstance(N, Mapping) and isinstance(drift, Mapping) and set(N) == set(drift):
        pairs = [(N[c], drift[c]) for c in N]
    elif isinstance(N, (tuple, list)) and isinstance(drift, (tuple, list)) and len(N) == len(drift):
        pairs = list(zip(N, drift, strict=True))
    elif _size_total(N) is not None and _size_total(drift) is not None and \
            isinstance(N, numbers.Integral) == isinstance(drift, numbers.Integral):
        pairs = [(N, drift)]
    tot = [(_size_total(a), _size_total(b)) for a, b in pairs]
    if not tot or any(a is None or b is None for a, b in tot):
        return
    if any(b < a for a, b in tot) or not any(b > a for a, b in tot):
        raise ResolutionError(
            f"drift={drift!r} is not finer than N={N!r}",
            why="the drift test keeps the modes found again at a FINER resolution: drift = N "
                "keeps every mode (nothing filtered, while the result reads as filtered) and a "
                "coarser drift removes resolved modes too",
            fix="drift larger than N in the wall-normal direction, e.g. drift=3*N//2 for an int "
                "N", doc=_DOC)


def _check_value(name: str, value: Any, declared: Any = None) -> float | complex:
    """A parameter value: a finite real or complex number (not a bool), stored as float or
    complex. ``declared``: the value the parameter was declared with; a parameter declared real
    takes real values only (``m.kappa = 1j`` and ``m.kappa = nan``
    were accepted, the complex one built a simulation)."""
    if not isinstance(value, numbers.Number) or isinstance(value, bool):
        raise InputTypeError(f"parameter {name}={value!r} is not a number",
                             fix="parameters are numbers; a function of space is a given "
                                 f"field: m.given({name}=lambda ...: ...)", doc=_DOC)
    c = complex(value)
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        raise InputTypeError(f"parameter {name}={value!r} is not finite",
                             why="a NaN or infinite parameter makes every result NaN or "
                                 "meaningless, found only steps later",
                             fix=f"a finite number for {name}", doc=_DOC)
    # complex only for a complex number: a Decimal has .imag but is real (audit inputs,
    # Ra=Decimal("2") was stored as 2+0j and compiled a complex problem)
    is_complex = isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real)
    if is_complex and declared is not None and not isinstance(declared, complex):
        raise InputTypeError(f"parameter {name}={value!r}: {name} was declared real "
                             f"({name} = {declared:g})",
                             why="the equations were compiled for a real parameter; a complex "
                                 "value changes the problem (complex matrices, complex fields)",
                             fix=f"a real value for {name}, or declare it complex: "
                                 f"cn.Model(geometry, ..., {name}={declared:g}+0j)", doc=_DOC)
    return c if is_complex else float(value)



def _check_boundary_value(boundary: str, name: str, v: Any) -> None:
    """A boundary value of ``m.at(boundary, name=v)``: a wall condition (``cn.no_slip``), an
    expression as text, a callable of the coordinates (and t), a tuple of finite numbers (a
    Cartesian vector) or a finite number (audit inputs: ``u=nan``, ``u=inf``,
    ``u=True`` and ``u=None`` reached the right-hand side as they were: a NaN boundary row)."""
    from .conditions import WallCondition
    if isinstance(v, (WallCondition, str)) or callable(v):
        return
    items = list(v) if isinstance(v, (tuple, list)) else [v]
    for c in items:
        ok = isinstance(c, numbers.Number) and not isinstance(c, bool)
        if ok:
            z = complex(c)
            ok = math.isfinite(z.real) and math.isfinite(z.imag)
        if not ok:
            raise BoundaryConditionError(
                f"at({boundary!r}, {name}={v!r}): not a boundary value",
                why="a boundary value is a finite number, a tuple of them (a vector), an "
                    "expression as text, a function of the coordinates or a wall condition; "
                    "NaN, infinity or a flag would become a boundary row of the system",
                fix=f"at({boundary!r}, {name}=0.0), {name}='sin(t)', {name}=lambda x: ... or "
                    f"{name}=cn.no_slip", doc=_DOC)


#: Keywords of ``Model.discretize`` / ``Model.solve`` besides the declared parameters (audit
#: inputs: an unknown one was a bare TypeError of conatus.discretize).
_DISCRETIZE_OPTIONS = ("ncc_cutoff", "ncc_tail_tol", "assembly", "params")
#: Keywords of ``Model.parametric`` passed to ``ParametricEVP`` (the same audit).
_PARAMETRIC_OPTIONS = ("group", "reduced", "fixed", "ncc_cutoff", "ncc_tail_tol", "derived",
                       "wavenumbers", "spurious")

#: Keywords accepted by the native eigenvalue calls besides the declared parameters (U1:
#: an unknown keyword, ``eigenvalues(re=5000)``, was a bare TypeError of the expert function).
_EIG_OPTIONS = ("method", "target", "left", "reduced", "spurious", "drift_tol", "ncc_cutoff",
                "ncc_tail_tol", "region")


def check_call_options(model: Any, kw: dict, accepted: tuple, where: str) -> None:
    """Refuse a keyword that is neither an option of ``where`` nor a declared parameter of the
    model, with the close names (the same name in another case first: ``re`` -> ``Re``)."""
    params = list(getattr(model, "params", {}) or {})
    bad = [n for n in kw if n not in accepted and n not in params]
    if bad:
        choices = list(accepted) + params
        raise OptionError(f"{where}(..., {bad[0]}=...): unknown keyword." + did_you_mean(bad[0], choices),
                          why="a keyword is an option of the call or a declared parameter of the "
                              "model (names are case-sensitive)",
                          fix="declared parameters: " + (", ".join(params) or "none")
                              + "; options: " + ", ".join(accepted), doc=_DOC)


def _fourier_coords(geo: Any) -> list[str]:
    """Periodic coordinates of a Cartesian box whose wavenumber selects a group (none on a mapped
    domain, whose metric couples the wavenumbers, nor on a curvilinear geometry)."""
    from .geometry import Box, Periodic
    if not isinstance(geo, Box) or getattr(geo, "terrain", None) is not None:
        return []
    return [c for c, ax in geo._axes.items() if isinstance(ax, Periodic)]


def _k_map(k: Any, fourier: list[str]) -> dict[str, float]:
    """{periodic coordinate: wavenumber} from one number, a tuple in coordinate order or a dict."""
    if isinstance(k, Mapping):
        missing = [c for c in fourier if c not in k]
        extra = [c for c in k if c not in fourier]
        if missing or extra:
            raise ResolutionError(f"k={dict(k)!r} for the periodic coordinates {fourier}",
                                  fix="one wavenumber per periodic coordinate: k={"
                                      + ", ".join(f"'{c}': ..." for c in fourier) + "}", doc=_DOC)
        ks = tuple(k[c] for c in fourier)
    else:
        ks = (k,) if isinstance(k, numbers.Number) or not hasattr(k, "__iter__") else tuple(k)
    # finite real wavenumbers (audit inputs: a dict value 1+1j was a bare TypeError,
    # True was read as 1, NaN searched for a group)
    if len(ks) != len(fourier) or not all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                                          and math.isfinite(v) for v in ks):
        raise ResolutionError(f"k={k!r} for the periodic coordinates {fourier}",
                              fix=f"{len(fourier)} real wavenumber(s), in the order {fourier}",
                              doc=_DOC)
    return dict(zip(fourier, (float(v) for v in ks), strict=True))


def _k0_warning(want: Mapping[str, float]) -> None:
    # a GroupWarning, not a DeprecationWarning: the modes of every other wavenumber are absent,
    # which Python must show outside __main__ too
    from ..errors import GroupWarning, _warn_user
    ks = ", ".join(f"k_{c} = {v:g}" for c, v in want.items())
    _warn_user(f"eigenvalues() without k=: only the group {ks} is solved (v0.1 behaviour); each "
               "wavenumber of a periodic direction is its own set of modes, so give it: "
               "eigenvalues(N, k=<wavenumber>) (k=0 for this group). Required in v0.2 "
               "", GroupWarning)


def _wavenumber_or_count(fourier: list[str], k: Any, n_modes: Any, method: str) -> tuple[Any, Any]:
    """(wavenumber, shift-invert count) of a ``Model.eigenvalues`` call (Y3): ``k`` is a
    wavenumber; an integer ``k`` keeps its v0.1 meaning (number of shift-invert modes) with a
    DeprecationWarning where it cannot be a wavenumber (no periodic direction); on a periodic
    model an integer ``k`` with shift-invert and no ``n_modes`` reads both ways and is refused
   ; a non-integer ``k`` on a model without a periodic direction is refused (it was
    ignored)."""
    integral = isinstance(k, numbers.Integral) and not isinstance(k, bool)
    if fourier and integral and n_modes is None and method == "shift-invert":
        # on a periodic model this call reads both ways (v0.1: k shift-invert
        # modes of the k = 0 group; now: the group of wavenumber k); reading it as the v0.1
        # count solved the k = 0 group for a user who meant the wavenumber, with only a
        # DeprecationWarning (hidden outside __main__): refused, both spellings given
        raise OptionError(f"eigenvalues(k={k}, method='shift-invert') on a model with a periodic "
                          "direction: k is ambiguous",
                          why="k is the wavenumber of the periodic direction (it selects a group); "
                              "in v0.1 an integer k was the number of shift-invert modes and the "
                              "k = 0 group was solved",
                          fix=f"eigenvalues(N, k={k}, method='shift-invert', target=..., "
                              f"n_modes=6) for the wavenumber {k}, or eigenvalues(N, k=0, "
                              f"method='shift-invert', target=..., n_modes={k}) for {k} modes "
                              "of the k = 0 group", doc=_DOC)
    old_count = k is not None and integral and n_modes is None and not fourier
    if old_count:
        from ..errors import _warn_user
        _warn_user(f"eigenvalues(k={k}): k as the number of shift-invert modes is deprecated "
                      f"since v0.1, write n_modes={k} (same meaning; k is the "
                      "wavenumber of a periodic direction, as in lin.eigenvalues)"
                      + ("; with method='dense' it has no effect (every mode is returned)"
                         if method != "shift-invert" else ""),
                      DeprecationWarning)
        return None, int(k)
    if n_modes is not None:
        from ..errors import check_count
        check_count(n_modes, "n_modes", "eigenvalues()")
        if method != "shift-invert":
            raise OptionError(f"eigenvalues(n_modes={n_modes}) with method={method!r}",
                              why="n_modes is the number of modes of method='shift-invert' "
                                  "(nearest target); the dense method returns every mode",
                              fix="drop n_modes, or method='shift-invert', target=..., "
                                  f"n_modes={n_modes}", doc=_DOC)
    if k is not None and not fourier:
        raise OptionError(f"eigenvalues(k={k!r}): the model has no periodic direction",
                          why="k is the wavenumber of a periodic direction (it selects a group of "
                              "modes); here it would be ignored. A wavenumber of a model without "
                              "a periodic axis is one of its parameters (e.g. alpha)",
                          fix="drop k=, or pass the parameter (eigenvalues(N, alpha=...)); "
                              "n_modes= is the number of shift-invert modes", doc=_DOC)
    return k, n_modes


def _select_group(plan: Any, fourier: list[str], k: Any, which: Mapping[str, Any]) -> Any:
    """The group of wavenumber ``k`` (Cartesian periodic axes) or of ``m=``/``kz=``/``ell=``
    (curvilinear); none given with several groups: the first one, with a GroupWarning."""
    groups = plan.groups
    if which:
        return plan.group(**which)
    if k is not None:
        want = _k_map(k, fourier)
        names = [d.coord.name for d in plan.layout.fdirs]
        for g in groups:
            ks = dict(zip(names, (it.k for it in g), strict=True))
            if all(abs(ks[c] - v) <= 1e-12 * max(1.0, abs(v)) for c, v in want.items()):
                return g
        avail = sorted({round(float(it.k), 10) for g in groups for it in g})
        raise ResolutionError(f"no group with k = {k!r} at this resolution",
                              why="the wavenumbers of a periodic axis of length L are 2 pi n / L, "
                                  "|n| below half its size",
                              fix="a wavenumber among " + ", ".join(f"{v:g}" for v in avail[:12])
                                  + (" ..." if len(avail) > 12 else "")
                                  + ", or a larger size along the periodic axis", doc=_DOC)
    if len(groups) > 1:
        g = groups[0]
        if fourier and isinstance(g, tuple) and all(hasattr(it, "k") for it in g):
            names = [d.coord.name for d in plan.layout.fdirs]
            _k0_warning(dict(zip(names, (float(it.k) for it in g), strict=True)))
        else:
            from ..errors import GroupWarning, _warn_user
            _warn_user(f"eigenvalues() without a group: only the first group "
                       f"[{plan.matrices(g).label}] of {len(groups)} is solved (v0.1 behaviour); "
                       "choose it with m= (and kz= or ell=). Required in v0.2",
                       GroupWarning)
        return g
    return groups[0]


class ParameterValues(dict):
    """The parameters of a :class:`Model` (``m.params``): a dict whose writes are validated.

    ``m.params["Ra"] = 1e5`` (or ``m.Ra = 1e5``) changes a declared parameter, checked as in
    ``Model(...)`` (a number); a name that was not declared is refused with the declared names
    (``m.params`` used to be a plain dict, so a typo or a string value went
    through and failed later, or never). Parameters cannot be removed (the equations use them).
    The next compilation (``simulate``, ``eigenvalues``, ``lin.eigenvalues``, ...) uses the
    new value; an already-built ``Simulation`` keeps the values it was compiled with."""

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            raise OptionError(f"params[{key!r}]: {key!r} is not a declared parameter."
                              + did_you_mean(key, self),
                              why="the equations were parsed with the declared parameters only; "
                                  "a new name would never reach them",
                              fix=("declared parameters: " + ", ".join(self) if self else
                                   "this model declares no parameter")
                              + "; declare new ones in cn.Model(geometry, ..., name=value)",
                              doc=_DOC)
        old = dict.__getitem__(self, key)
        new = _check_value(key, value, old)
        super().__setitem__(key, new)
        if new != old:
            self._notify(key, old, new)

    def _notify(self, key: str, old: Any, new: Any) -> None:
        """A note when simulations built from the model exist: they keep the value they were
        compiled with (the change was silently ignored by them)."""
        owner = getattr(self, "_owner", None)
        model = owner() if owner is not None else None
        sims = [r() for r in getattr(model, "_sims", ())] if model is not None else []
        sims = [s for s in sims if s is not None]
        if sims:
            warnings.warn(f"{key} = {new:g} applies to the next simulate()/eigenvalues()/... of "
                          f"model '{model.name}'; the existing simulation(s) "
                          + ", ".join(repr(getattr(s, "id", "?")) for s in sims)
                          + f" keep {key} = {old:g} (sim.explain() lists the values in use; "
                          "rebuild with m.simulate(...) to use the new one)",
                          UserWarning, stacklevel=3)

    def update(self, *args: Any, **kw: Any) -> None:  # type: ignore[override]
        """Validated update (every key must be declared, every value a number); all values are
        checked before any is written (a failing update changes nothing)."""
        new = dict(*args, **kw)
        for k, v in new.items():
            if k not in self:
                self[k] = v                 # refused with the declared names
            _check_value(k, v, dict.__getitem__(self, k))
        for k, v in new.items():
            self[k] = v

    def __ior__(self, other: Any) -> ParameterValues:  # type: ignore[override,misc]
        """``m.params |= {...}``: the validated update (review finding 28: dict.__ior__ wrote
        NaN and undeclared names without any check). All values are checked before any is
        written."""
        new = dict(other)
        for k, v in new.items():
            if k not in self:
                self[k] = v                 # refused with the declared names
            _check_value(k, v, dict.__getitem__(self, k))
        self.update(new)
        return self

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """Declared parameters always have a value: returns it (an unknown name is refused)."""
        if key not in self:
            self[key] = default            # refused with the declared names
        return self[key]

    def _no_removal(self, *args: Any, **kw: Any) -> Any:
        raise OptionError("a declared parameter cannot be removed from m.params",
                          why="the equations of the model use it",
                          fix="change its value instead: m.params['Ra'] = 1e5 (or m.Ra = 1e5)",
                          doc=_DOC)

    __delitem__ = pop = popitem = clear = _no_removal   # type: ignore[assignment]

    def _declare(self, key: str, value: Any) -> None:
        """Declaration of a new parameter (``Model.__init__`` only)."""
        super().__setitem__(key, _check_value(key, value))

    def __reduce__(self) -> tuple:
        # copy/deepcopy/pickle rebuild a dict subclass through __setitem__ on an empty
        # instance, which refuses every (undeclared) name: rebuild by declaration instead
        # (copy.deepcopy(m) and pickle of a Model failed)
        return (_parameter_values, (dict(self),))


def _parameter_values(values: dict) -> ParameterValues:
    """``ParameterValues`` holding ``values`` (unpickling, copies)."""
    out = ParameterValues()
    for k, v in values.items():
        out._declare(k, v)
    return out


def split_overrides(model: Model, kw: dict) -> dict:
    """Pop from ``kw`` the keywords that name a declared parameter of ``model``: per-call values
    (``m.simulate(N, ..., Ra=3000)``), validated as numbers; the model itself is not changed."""
    return {k: _check_value(k, kw.pop(k), model.params[k]) for k in list(kw) if k in model.params}


def ncc_options(kw: dict) -> dict:
    """Pop the discretization options of non-constant coefficients from ``kw``: the
    declared truncation ``ncc_cutoff=`` and refusal threshold ``ncc_tail_tol=``, accepted by
    every native call that takes a resolution (``simulate``, ``solve``, ``eigenvalues``, ...)."""
    return {k: kw.pop(k) for k in ("ncc_cutoff", "ncc_tail_tol") if k in kw}


def _transactional(method: Any) -> Any:
    """A declaration of :class:`Model` run in ``Model._transaction``: on failure, the model is
    unchanged (signature and docstring kept for help() and the API surface)."""
    import functools

    @functools.wraps(method)
    def wrapper(self: Model, *args: Any, **kw: Any) -> Any:
        with self._transaction():
            return method(self, *args, **kw)
    return wrapper


class Model:
    """A PDE model on a geometry, declared as on paper.

    ``Model(geometry, name=..., **parameters)``: parameters by keyword (``Ra=2500, Pr=1``), each
    becoming a named Parameter of the equations. Unknowns, given fields, equations and boundary conditions
    are then declared on the model; resolution, scheme and time step are given later to
    ``simulate``, ``eigenvalues`` or ``compile``.
    """

    def __init__(self, geometry: Geometry, *, name: str = "model", **parameters: Any) -> None:
        if not isinstance(geometry, Geometry):
            raise InputTypeError(f"Model needs a geometry (cn.Box, cn.Channel, cn.Disk, ...), got "
                                 f"{type(geometry).__name__}", fix="cn.Model(cn.Box(x=cn.periodic(2),"
                                 " z=cn.walls(-0.5, 0.5)), Ra=2500)", doc=_DOC)
        if not isinstance(name, str):              # a label of ids, files and messages
            raise InputTypeError(f"Model(name={name!r}): the name is a string",
                                 fix="cn.Model(geometry, name='convection', ...)", doc=_DOC)
        self.geometry = geometry
        self.name = name
        self.params: ParameterValues = ParameterValues()
        for k, v in parameters.items():
            self._check_name(k, "parameter")
            self.params._declare(k, v)
        self.unknowns: list[Unknown] = []
        self.givens: list[Given] = []
        self.macros: list[tuple[str, tuple[str, ...], str, str]] = []
        self.functions: dict[str, Any] = {}        # name -> builder (conatus.ir.pointwise)
        self.eqs: list[tuple[str, str]] = []
        self.conditions: list[_Condition] = []
        self.eig: str | None = None
        self.eig_growth: Any = None                # growth convention of the eigenvalue
        self._degree: Any = None                   # highest nonlinear degree (set by _probe)
        self._unknown_degree: Any = None           # the same in the unknowns only (set by _probe)
        self._sealed = True                        # from here on, __setattr__ checks the names

    # --- parameters as attributes ------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """``m.Ra``: the value of a declared parameter (only reached when no attribute or method
        has that name)."""
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """``m.Ra = 1e5`` sets a declared parameter (validated as ``m.params["Ra"] = 1e5``).
        Any other new public name is refused: ``m.Ra = 1e5`` used to create an unrelated
        attribute while the equations kept the old Ra."""
        d = self.__dict__
        if not d.get("_sealed") or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in d["params"]:            # never an attribute name (refused by _check_name)
            d["params"][name] = value
            return
        if name in d or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return
        known = list(d["params"]) + [n for n in d if not n.startswith("_")]
        raise OptionError(f"m.{name} = ...: {name!r} is neither a declared parameter nor an "
                          "attribute of the model." + did_you_mean(name, known),
                          why="setting an undeclared name would create an unrelated attribute "
                              "that the equations never see",
                          fix=("declared parameters: " + ", ".join(d["params"])
                               if d["params"] else "this model declares no parameter")
                              + f"; declare a new one in cn.Model(geometry, ..., {name}=value)",
                          doc=_DOC)

    # --- declarations --------------------------------------------------------------------
    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Validate, then mutate (review findings 27, 61): a declaration that fails leaves the
        model exactly as it was. ``equations("dt(u) = typo")`` used to keep the bad line, so the
        corrected call failed too (a notebook could not recover without rebuilding the model)."""
        d = self.__dict__
        lists = ("unknowns", "givens", "macros", "eqs", "conditions")
        saved = {k: list(d[k]) for k in lists}
        functions = dict(d["functions"])
        scalars = {k: d.get(k) for k in ("eig", "eig_growth", "_degree", "_unknown_degree")}
        dyes = dict(d["_dyes"]) if "_dyes" in d else None
        try:
            yield
        except BaseException:
            for k in lists:
                d[k][:] = saved[k]
            d["functions"].clear()
            d["functions"].update(functions)
            d.update(scalars)
            if dyes is None:
                d.pop("_dyes", None)
            else:
                d["_dyes"] = dyes
            raise

    def _names(self) -> set[str]:
        return (set(self.params) | {u.name for u in self.unknowns} | {g.name for g in self.givens}
                | {m[0] for m in self.macros} | set(getattr(self, "functions", {}))
                | ({self.eig} if self.eig else set()))

    def _check_name(self, name: str, what: str) -> None:
        if not isinstance(name, str) or not _NAME.match(name):
            raise InputTypeError(f"{what} name {name!r} is not an identifier", doc=_DOC)
        if name in _RESERVED or name.startswith("tau_"):
            raise InputTypeError(f"{what} name {name!r} is reserved",
                                 why="operators, markers and the automatic tau variables (tau_*) "
                                     "use these names", fix=f"rename it ({name}_)", doc=_DOC)
        if name in getattr(self, "params", {}) or (hasattr(self, "unknowns") and name in self._names()):
            raise InputTypeError(f"{what} name {name!r} is already declared", doc=_DOC)
        if what == "parameter" and (name in _MODEL_ATTRIBUTES or hasattr(type(self), name)):
            # m.<parameter> reads and sets the parameter: it cannot also be an attribute or a
            # method of the model (m.at, m.solve, m.params, ...)
            raise InputTypeError(f"parameter name {name!r} is an attribute of cn.Model",
                                 why="m.<name> = value sets a parameter; this name already "
                                     "means something else on the model",
                                 fix=f"rename it ({name}_)", doc=_DOC)
        if name in self.geometry.coord_names:
            raise InputTypeError(f"{what} name {name!r} is a coordinate of the geometry", doc=_DOC)

    def _unknown(self, name: str, rank: int, role: str) -> Unknown:
        self._check_name(name, "unknown")
        if self.eqs:
            raise EquationError(f"unknown {name!r} declared after the equations",
                                fix="declare every unknown before m.equations(...)", doc=_DOC)
        h = Unknown(name, rank, role)
        self.unknowns.append(h)
        return h

    def scalar(self, name: str) -> Unknown:
        """A scalar unknown (temperature, concentration, streamfunction...)."""
        return self._unknown(name, 0, "scalar")

    def vector(self, name: str) -> Unknown:
        """A vector unknown."""
        return self._unknown(name, 1, "vector")

    def velocity(self, name: str = "u") -> Unknown:
        """A velocity (a vector unknown; the role is only printed)."""
        return self._unknown(name, 1, "velocity")

    def pressure(self, name: str = "p") -> Unknown:
        """A pressure: a scalar that must have the structure of one (only grad(p), no condition,
        a div constraint); checked at compilation, never assumed."""
        return self._unknown(name, 0, "pressure")

    @_transactional
    def dye(self, name: str, *, velocity: str, kappa: Any) -> Unknown:
        """A numerical dye: the passive scalar ``name`` advected by the vector
        unknown ``velocity``, with its own diffusivity ``kappa`` (a number >= 0 or the name of
        a declared parameter). Declares the unknown and its equation in conservative form
        ``dt(c) - kappa*lap(c) = -div(c*u)``, with no-flux walls ``dz(c) = 0`` on the wall axis
        when kappa > 0; ``kappa=0`` is pure advection (no diffusion, no filter, no wall
        condition). May follow ``m.equations`` (nothing else uses the dye). Cartesian boxes;
        printed by ``explain()``; written with its mass and variance by ``sim.dye(...)``
        (``conatus.model.dye``)."""
        from .dye import declare_dye
        return declare_dye(self, name, velocity=velocity, kappa=kappa)

    @_transactional
    def given(self, name: str | None = None, value: Any = None, *, rank: int = 0,
              **named: Any) -> Given | tuple[Given, ...]:
        """Known fields: ``m.given(U=lambda y: 1 - y**2)`` (argument names = coordinates it depends
        on); ``rank=1`` for a vector (the callable returns the components)."""
        items = ([(name, value)] if name is not None else []) + list(named.items())
        # rank 0 (scalar) or 1 (vector), never a float, a flag or another rank (audit inputs,
        # rank=1.5, True, -1 and 3 were stored and compiled as a scalar)
        if isinstance(rank, bool) or rank not in (0, 1) or not isinstance(rank, numbers.Integral):
            raise InputTypeError(f"given(..., rank={rank!r}): 0 (a scalar) or 1 (a vector)",
                                 fix="rank=0 (default) or rank=1", doc=_DOC)
        out = []
        for n, v in items:
            self._check_name(n, "given field")
            if not (callable(v) or (isinstance(v, numbers.Number) and not isinstance(v, bool))):
                raise InputTypeError(f"given {n}={v!r}: a callable of coordinates or a number",
                                     doc=_DOC)
            if not callable(v) and not (math.isfinite(complex(v).real)
                                        and math.isfinite(complex(v).imag)):
                raise InputTypeError(f"given {n}={v!r}: not a finite number",
                                     why="a NaN or infinite coefficient makes every result NaN",
                                     fix=f"a finite value for {n}", doc=_DOC)
            if callable(v):                   # its arguments name coordinates: checked now
                from .compiler import _coord_args
                _coord_args(v, tuple(self.geometry.coord_names), f"given {n}")
            g = Given(n, v, rank)
            self.givens.append(g)
            out.append(g)
        return out[0] if len(out) == 1 else tuple(out)

    @_transactional
    def function(self, name: str, f: Any, *, derivative: Any = None, parity: str = "none") -> Any:
        """A pointwise function of a scalar field usable in the equations by its name
        (``m.function("g", np.tanh, derivative=lambda u: 1 - np.tanh(u)**2)``, then
        ``dt(u) - lap(u) = -g(u)``): the role of Dedalus' ``GeneralFunction`` for pointwise
        functions (``conatus.ir.pointwise``: evaluated on the dealiased grid, not alias-free,
        declared; without ``derivative`` linearization and Newton are refused). ``f`` takes and
        returns arrays (CuPy arrays on the GPU backend)."""
        from ..ir import pointwise
        self._check_name(name, "function")
        try:
            g = pointwise(name, f, derivative=derivative, parity=parity)
        except (ValueError, TypeError) as err:
            raise InputTypeError(f"function({name!r}): {err}", doc=_DOC) from err
        self.functions[name] = g
        return g

    @_transactional
    def define(self, text: str) -> None:
        """A named operator used in the equations: ``m.define("L2(f) = dy(dy(f)) - alpha**2*f")``
        (the body is parsed at each use with its arguments bound: pure substitution)."""
        mt = _MACRO.match(text)
        if not mt:
            raise EquationError(f"define({text!r}): expected 'name(a, b) = expression'",
                                fix='m.define("L2(f) = dy(dy(f)) - alpha**2*f")', doc=_DOC)
        name, args, body = mt.group(1), tuple(a.strip() for a in mt.group(2).split(",") if a.strip()), mt.group(3)
        self._check_name(name, "operator")
        # arguments: distinct identifiers (review finding 31: "pick(a, a) = a" was accepted and
        # bound the last value silently)
        bad = [a for a in args if not _NAME.match(a)]
        if bad:
            raise EquationError(f"define({text!r}): argument(s) {bad} are not identifiers",
                                fix='m.define("L2(f) = dy(dy(f)) - alpha**2*f")', doc=_DOC)
        dup = sorted({a for a in args if args.count(a) > 1})
        if dup:
            raise EquationError(f"define({text!r}): argument(s) {dup} repeated",
                                why="each argument names one value of the call",
                                fix="give every argument its own name", doc=_DOC)
        self.macros.append((name, args, body, text))

    @_transactional
    def eigenvalue(self, name: str = "s", *, growth: Any = None) -> str:
        """Declare the model an eigenvalue problem in ``name`` (complex); returns the name.

        ``growth``: which part of the eigenvalue measures growth, used by ``critical`` and
        ``least_stable``: ``"real"`` for a mode exp(s t), ``"imag"`` for a phase speed c of
        exp(i alpha (x - c t)) (or exp(-i omega t)), or a callable of the eigenvalue. Printed by
        ``explain()``; it ranks the modes (``res.order()``) and selects the least stable one.

        Declare it: ``m.eigenvalue("c", growth="imag")``. Left out,
        v0.1 keeps the former guess from the name (``s``, ``sigma``, ``lam`` -> real, ``c`` ->
        imag; ``conatus.solvers.parametric.GROWTH_BY_NAME``; any other name: ``order()`` ranks
        by the real part and ``critical`` refuses) with a ``DefaultChoiceWarning`` naming the rule
        guessed: a ``c`` of a temporal problem (``c*T = ...``) was ranked by Im c. Required from
        v0.2."""
        if growth is not None and not callable(growth) and growth not in ("real", "imag"):
            raise OptionError(f"eigenvalue({name!r}, growth={growth!r})",
                              fix="growth='real' (exp(s t)) or 'imag' (exp(i alpha (x - c t))), "
                                  "or a callable of the eigenvalue", doc=_DOC)
        self._check_name(name, "eigenvalue")
        if growth is None:
            # the user's line whatever the call depth (a problem helper calling it, a wrapper)
            from ..errors import DefaultChoiceWarning, _warn_user
            from ..solvers.parametric import GROWTH_BY_NAME
            g = GROWTH_BY_NAME.get(name)
            rule = ({"real": f"growth = Re {name} (a growth rate of exp({name} t))",
                     "imag": f"growth = Im {name} (a phase speed of exp(i alpha (x - {name} t)))"}
                    [g] + " guessed from the name") if g else (
                f"modes ranked by Re {name} (v0.1 default; critical() refuses)")
            _warn_user(f"m.eigenvalue({name!r}) without growth=: {rule}. Declare it: "
                       f"m.eigenvalue({name!r}, growth='real') for exp({name} t), "
                       f"growth='imag' for a phase speed; required from v0.2",
                       DefaultChoiceWarning)
        self.eig = name
        self.eig_growth = growth
        return name

    @_transactional
    def equations(self, *texts: str, **named: str) -> Model:
        """Equations as on paper, one per line or one per argument (or ``label=...`` keywords, or
        ``label: ...`` prefixes). Terms may sit on either side: the split is automatic
        (explain())."""
        items: list[tuple[str | None, str]] = []
        bad = [t for t in list(texts) + list(named.values()) if not isinstance(t, str)]
        if bad:                                   # audit inputs: a bare AttributeError before
            raise EquationError(f"equations({bad[0]!r}): an equation is text",
                                fix='m.equations("dt(u) - lap(u) = 0")', doc=_DOC)
        for text in texts:
            for line in _split_lines(text):
                mt = _LABEL.match(line)
                items.append((mt.group(1), mt.group(2)) if mt else (None, line))
        items += list(named.items())
        if not items:
            raise EquationError("equations(): no equation given", doc=_DOC)
        for lab, eq in items:
            if "=" not in eq:
                raise EquationError(f"'{eq}' is not an equation (no '=')", fix=f"{eq} = 0", doc=_DOC)
            label = lab or f"eq{len(self.eqs)}"
            if label in [l for l, _ in self.eqs]:
                raise EquationError(f"two equations labelled {label!r}", doc=_DOC)
            self.eqs.append((label, eq.strip()))
        self._probe()
        return self

    @_transactional
    def at(self, boundary: str, *conditions: str, into: Any = None, **values: Any) -> Model:
        """Boundary conditions at a named boundary: ``m.at("bottom", u=0, T=0.5)``, or as text
        ``m.at("walls", "v = 0", "dy(v) = 0")`` (texts first, then keywords, in order).
        ``into=`` names the equation that receives the taus when the rule cannot decide."""
        self.geometry.boundary(boundary)          # unknown names fail here
        if not conditions and not values:
            raise BoundaryConditionError(f"at({boundary!r}): no condition", doc=_DOC)
        for c in conditions:
            if not isinstance(c, str) or "=" not in c:
                raise BoundaryConditionError(f"at({boundary!r}, {c!r}): a text condition is "
                                             "'lhs = rhs'", fix="at('bottom', 'dz(T) = 0')", doc=_DOC)
            self.conditions.append(_Condition(boundary, c, None, None, into))
        for k, v in values.items():
            if k not in {u.name for u in self.unknowns}:
                raise BoundaryConditionError(f"at({boundary!r}, {k}=...): {k} is not an unknown",
                                             fix="unknowns: " + ", ".join(u.name for u in self.unknowns),
                                             doc=_DOC)
            _check_boundary_value(boundary, k, v)
            self.conditions.append(_Condition(boundary, None, k, v, into))
        return self

    # --- compilation ---------------------------------------------------------------------
    def _probe_N(self) -> dict:
        geo = self.geometry
        pieces = len(geo.interfaces) + 1          # compound interval: one size per piece
        return {c: ((8,) * pieces if pieces > 1 and c == geo.wall_coord else 8)
                for c in geo.coord_names}

    def _probe(self) -> None:
        """Parse and split the equations at a probe resolution (errors at the declaring line);
        conditions are checked only at compilation (they may come later)."""
        saved = self.conditions
        self.conditions = []
        try:
            comp = compile_model(self, self._probe_N(), DEFAULT_DEALIAS, check_conditions=False)
            # highest polynomial degree of the equations: the default dealiasing factor
            self._degree = comp.degree
            self._unknown_degree = comp.unknown_degree     # weighted-axis floor (Hermite/Laguerre)
        finally:
            self.conditions = saved

    def compile(self, N: Any, dealias: Any = None, *, params: Mapping[str, Any] | None = None
                ) -> Compiled:
        """The IR problem at resolution ``N`` with the record of every automatic decision.
        ``params``: values of declared parameters for this compilation only (the model keeps its
        own; listed as per-call values by ``explain()``)."""
        if not self.eqs:
            raise EquationError("no equation declared", fix="m.equations('...')", doc=_DOC)
        over = {}
        for k, v in (params or {}).items():
            if k not in self.params:
                raise OptionError(f"compile(params={{{k!r}: ...}}): {k!r} is not a declared "
                                  "parameter." + did_you_mean(k, self.params),
                                  fix="declared parameters: " + (", ".join(self.params) or "none"),
                                  doc=_DOC)
            over[k] = _check_value(k, v, self.params[k])
        if getattr(self, "_dyes", None):
            from .dye import check_dye_parameters   # kappa given by name
            check_dye_parameters(self, over)
        return compile_model(self, N, dealias, params=over)

    def discretize(self, N: Any, dealias: Any = None, **kw: Any):
        """``conatus.discretize`` of the compiled problem (expert: the Plan, its matrices); a
        keyword naming a declared parameter (``Re=3000``) is its value for this call only."""
        over = split_overrides(self, kw)
        check_call_options(self, kw, _DISCRETIZE_OPTIONS, "discretize")   # audit inputs: bare TypeError
        return self.compile(N, dealias, params=over).discretize(**kw)

    def explain(self, N: Any = None, dealias: Any = None, *, full: bool = False) -> str:
        """The model in a few lines: geometry, unknowns, parameters, equations
        and conditions as written, the problem kind, which terms are explicit, the gauge, the
        dealiasing and every warning. ``full=True``: every automatic decision (term-by-term
        split, tau names and modes, symbolic unless ``N`` is given, bases)."""
        if not full:
            return self._summary(N, dealias)
        lines = [f"Model '{self.name}'", self.geometry.explain()]
        lines.append("Unknowns: " + ", ".join(f"{u.name} ({u.role})" for u in self.unknowns))
        if self.params:
            lines.append("Parameters: " + ", ".join(f"{k} = {v:g}" for k, v in self.params.items())
                         + " (change: m.<name> = value; for one call: simulate(..., name=value))")
        if self.givens:
            lines.append("Given fields: " + ", ".join(g.name for g in self.givens))
        if self.eig:
            from ..solvers.parametric import growth_convention
            _, gnote = growth_convention(self.eig, self.eig_growth)
            lines.append(f"Eigenvalue: {self.eig} (complex); {gnote}")
        for _name, _, _, text in self.macros:
            lines.append(f"Operator: {text}")
        for fname in getattr(self, "functions", {}):
            lines.append(f"Pointwise function: {fname}(u) (user function on the dealiased grid; "
                         "not alias-free)")
        lines.append("Equations as written:")
        lines += [f"  [{i}] {lab}: {eq}" for i, (lab, eq) in enumerate(self.eqs)]
        if self.conditions:
            lines.append("Conditions as written:")
            for c in self.conditions:
                what = c.text if c.text else f"{c.unknown} = {c.value!r}"
                lines.append(f"  at {c.boundary}: {what}" + (f" (into={c.into!r})" if c.into is not None else ""))
        if self.eqs:
            comp = (compile_model(self, self._probe_N(), dealias) if N is None
                    else self.compile(N, dealias))
            lines += comp.decisions(symbolic=N is None)
            if N is None:
                lines.append("Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))")
        return "\n".join(lines)

    def _summary(self, N: Any, dealias: Any) -> str:
        """The default ``explain()``: declarations as written, then ``Compiled.summary``."""
        geo = self.geometry.explain().splitlines()[0]
        lines = [f"Model '{self.name}': {geo[len('Geometry: '):] if geo.startswith('Geometry: ') else geo}"]
        lines.append("Unknowns: " + ", ".join(f"{u.name} ({u.role})" for u in self.unknowns)
                     + ("; parameters: " + ", ".join(f"{k} = {v:g}" for k, v in self.params.items())
                        if self.params else ""))
        if self.givens:
            lines.append("Given fields: " + ", ".join(g.name for g in self.givens))
        if self.eig:
            from ..solvers.parametric import growth_convention
            _, gnote = growth_convention(self.eig, self.eig_growth)
            lines.append(f"Eigenvalue: {self.eig}; {gnote}")
        for fname in getattr(self, "functions", {}):
            lines.append(f"Pointwise function: {fname}(u) (not alias-free)")
        lines.append("Equations as written:")
        lines += [f"  {lab}: {eq}" for lab, eq in self.eqs]
        if self.eqs:
            comp = (compile_model(self, self._probe_N(), dealias) if N is None
                    else self.compile(N, dealias))
            lines += comp.summary()
            if not comp.taus and self.conditions:
                lines.append("Conditions as written: " + "; ".join(
                    f"{c.boundary}: {c.text if c.text else f'{c.unknown} = {c.value!r}'}"
                    for c in self.conditions))
            if N is None:
                lines.append("Resolution: not chosen yet (simulate/eigenvalues/discretize(N=...))")
        lines.append("(compiler decisions, tau by tau and term by term: m.explain(full=True))")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Model('{self.name}' on {self.geometry!r}: "
                f"{', '.join(u.name for u in self.unknowns)}; {len(self.eqs)} equation(s))")

    # --- solving -------------------------------------------------------------------------
    def simulate(self, N: Any, *, scheme: str, dt: float, dealias: Any = None,
                 backend: str = "cpu", **solver_kw: Any):
        """Time integration: ``Simulation`` of the compiled IVP (scheme and dt explicit).

        A keyword naming a declared parameter (``m.simulate(N, ..., Ra=3000)``) is the value of
        that parameter for this run only (validated; the model keeps its own; recorded by
        ``sim.explain()`` and the provenance). The other keywords are options of the solver
        (``conatus.expert.IVPSolver``: ``check_finite``, ``f_backend``, ``reduced``, ...); an
        unknown one is refused with the accepted names."""
        from .run import Simulation, check_solver_options
        over = split_overrides(self, solver_kw)
        check_solver_options(solver_kw, "simulate")
        comp = self.compile(N, dealias, params=over)
        if comp.kind != "IVP":
            raise OptionError(f"simulate(): the model is an {comp.kind}, not an IVP",
                              why="only models with dt() are time-stepped",
                              fix="eigenvalues(...) for an EVP, discretize(...).solve() for an "
                                  "LBVP", doc=_DOC)
        if getattr(self.geometry, "terrain", None) is not None:
            # mapped domain: coupled implicit operator, its own stepper (CPU)
            from ..discretize.terrain_plan import TerrainSimulation
            return TerrainSimulation(comp, scheme=scheme, dt=dt, backend=backend, **solver_kw)
        # each simulation registers itself with its model (_register: id, parameter notes)
        return Simulation(comp, scheme=scheme, dt=dt, backend=backend, **solver_kw)

    def _register(self, sim: Any) -> Any:
        """Record ``sim`` (weakly) so that a later parameter change can name it;
        gives it its id ``'<model name>/sim<k>'``. Called by the simulation's constructor."""
        sims = [r for r in self.__dict__.get("_sims", []) if r() is not None]
        self._count = self.__dict__.get("_count", 0) + 1
        try:
            sim.id = f"{self.name}/sim{self._count}"
            sims.append(weakref.ref(sim))
        except (AttributeError, TypeError):  # an object without attributes or weak references
            pass
        self._sims = sims
        self.params._owner = weakref.ref(self)
        return sim

    def __getstate__(self) -> dict:
        """Pickling: the weak references to live simulations and the operators kept for reuse
        (``plan_cache``: a cache, not state) are not part of the model."""
        d = dict(self.__dict__)
        d.pop("_sims", None)
        d.pop("_plan_cache", None)
        return d

    def solve(self, N: Any, *, dealias: Any = None, **kw: Any):
        """Solution of a linear boundary-value model (LBVP: no dt(), linear in the unknowns) at
        resolution ``N``: the ``LBVPSolution`` of ``discretize(N).solve()``, given fields as data
        (``sol.grid("u")``, ``sol.explain()`` with the residual of each group)."""
        over = split_overrides(self, kw)              # a declared parameter: this call only
        check_call_options(self, kw, _DISCRETIZE_OPTIONS, "solve")   # audit inputs: bare TypeError
        comp = self.compile(N, dealias, params=over)
        if comp.kind != "LBVP":
            raise OptionError(f"solve(): the model is an {comp.kind}, not a linear boundary-value "
                              "problem", why="solve() is the direct solve of L X = F",
                              fix="simulate(...) for an IVP, eigenvalues(...) for an EVP; a "
                                  "nonlinear BVP goes through the IR (NewtonSolver on "
                                  "m.compile(N).problem)", doc=_DOC)
        return comp.discretize(**kw).solve()

    def linearize(self, about: Mapping[Any, Any], eigenvalue: str = "s", *,
                  period: float | None = None, base_residual: str = "check"):
        """Linear dynamics of the model about a base state (``{u: 0, T: lambda z: -z}``): normal
        modes (``eigenvalues``) of a steady base; with ``period=P`` the base may depend on t
        (``{u: lambda y, t: ...}``, periodic) and is analysed by ``floquet`` and
        ``optimal_gain`` (transient growth, also for a steady base).

        ``base_residual``: ``"check"`` (default) evaluates the steady equations and boundary
        conditions at a steady base when the problem is compiled and prints the residual in
        ``explain()``; above ``conatus.tolerances.BASE_RESIDUAL_TOL`` the base is not a steady
        solution (BaseStateWarning in v0.1, refused in v0.2). ``"accept"``
        declares a base that knowingly is not one (a frozen or forced profile)."""
        from .run import Linearization
        return Linearization(self, about, eigenvalue, period, base_residual)

    def eigenvalues(self, N: Any, *, dealias: Any = None, drift: Any = None, k: Any = None,
                    n_modes: int | None = None, **kw: Any):
        """Eigenvalues of an EVP model (``m.eigenvalue("c")``); ``drift=N2`` keeps only the modes
        found again at resolution N2 (conatus.eigenvalues(drift_plan=)); a mode that moves is not
        converged by this test (a tau mode, or a physical mode N does not resolve) and is set aside.

        ``k``: the wavenumber of the periodic direction(s) (one number, a tuple in coordinate
        order or a dict by name), as in ``lin.eigenvalues(N, k=)``: each wavenumber is its own
        group of modes. Required on a model with a periodic direction (the k = 0 group alone was solved without a word); left out in v0.1, the
        k = 0 group is solved with a GroupWarning that says so. ``n_modes``: the number of
        modes of ``method="shift-invert"`` (the former meaning of an integer ``k`` on a model
        without a periodic direction, accepted with a DeprecationWarning until v0.2).
        ``m=``, ``kz=``, ``ell=``: the group of a curvilinear geometry (disk, pipe, sphere)."""
        from ..solvers.plan_evp import eigenvalues
        if not self.eig:
            raise OptionError("eigenvalues(): no eigenvalue declared",
                              fix="m.eigenvalue('s') before the equations, or m.linearize(about=...)"
                                  ".eigenvalues(...) for an evolution model", doc=_DOC)
        # a declared parameter keeps its name: k, n_modes, m, kz, ell and group
        # select a group or a count only when the model declares no parameter of that name (a
        # normal-mode model written with 1j*k, a mass ratio m): k= is then that parameter's value
        # for this call, as in v0.1
        params = self.params
        which = {n_: kw.pop(n_) for n_ in ("m", "kz", "ell") if n_ in kw and n_ not in params}
        group = None if "group" in params else kw.pop("group", None)  # an expert group (v0.1)
        if k is not None and "k" in params:
            kw["k"], k = k, None
        if n_modes is not None and "n_modes" in params:
            kw["n_modes"], n_modes = n_modes, None
        check_call_options(self, kw, _EIG_OPTIONS + (("n",) if self.geometry.sincos_coord
                                                     else ()), "eigenvalues")
        from ..solvers.plan_evp import check_eig_method   # before compiling (review, inputs)
        check_eig_method(kw.get("method"), kw.get("target"), kw.get("region"))
        dkw = ncc_options(kw)
        dkw.update(split_overrides(self, kw))      # a declared parameter (Re=...): this call only
        fourier = _fourier_coords(self.geometry)
        k, n_modes = _wavenumber_or_count(fourier, k, n_modes, kw.get("method", "dense"))
        plan = self.discretize(N, dealias, **dkw)
        if self.geometry.sincos_coord is not None:
            # sine/cosine axis: gather the groups of every vertical wavenumber n pi / L at the
            # Fourier wavenumbers k (as without the axis); no tau, no drift
            from .run import sincos_eigenvalues
            if drift is not None:
                raise OptionError("drift= on a sine/cosine axis",
                                  why="drift= filters spurious tau modes; a sine/cosine axis has "
                                      "no tau", fix="drop drift=", doc=_DOC)
            w = self.geometry.sincos_coord
            names = [d.coord.name for d in plan.layout.fdirs]
            if k is None:
                want = {c: it.k for c, it in zip(names, plan.groups[0], strict=True) if c != w}
                if want:
                    _k0_warning(want)
            else:
                want = _k_map(k, fourier)
            if n_modes is not None and kw.get("method") == "shift-invert":
                kw["n_modes"] = n_modes
            return sincos_eigenvalues(plan, self.geometry, want, kw.pop("n", None), **kw)
        g = group if group is not None else _select_group(plan, fourier, k, which)
        _check_drift(drift, kw, N)
        plan.problem.growth = self.eig_growth      # ordering of the result
        if drift is not None:
            kw["drift_plan"] = self.discretize(drift, dealias, **dkw)
        if n_modes is not None and kw.get("method") == "shift-invert":
            kw["n_modes"] = n_modes                 # (a v0.1 count with 'dense': warned, unused)
        return eigenvalues(plan, g, **kw)

    def parametric(self, N: Any, over: Any, *, dealias: Any = None, drift: Any = None,
                   drift_tol: Any = None, **kw: Any):
        """``ParametricEVP`` of an EVP model in the parameters ``over`` (names). The model's
        declared values of these parameters are used when a call leaves one out (logged), and
        the growth convention of the eigenvalue goes with the pencil. ``drift=N2`` (with its
        required ``drift_tol``) builds the same pencil at resolution N2: ``least_stable`` and
        ``critical`` then keep only the modes found again there (spurious tau modes removed),
        as ``eigenvalues(drift=)``."""
        from ..solvers.parametric import ParametricEVP
        from ..tolerances import REQUIRED
        if isinstance(over, str):
            over = (over,)
        # names, at least one, each once (audit inputs: over=3 was a bare TypeError,
        # over=() and over=("Re", "Re") built a pencil)
        if not hasattr(over, "__iter__") or not over or any(not isinstance(n, str) for n in over) \
                or len(set(over)) != len(tuple(over)):
            raise OptionError(f"parametric(over={over!r}): one or more distinct parameter names",
                              fix="over='Re' or over=('Re', 'alpha'); declared parameters: "
                                  + (", ".join(self.params) or "none"), doc=_DOC)
        over = tuple(over)
        bad = [k for k in kw if k not in _PARAMETRIC_OPTIONS]
        if bad:
            raise OptionError(f"parametric(..., {bad[0]}=...): unknown keyword."
                              + did_you_mean(bad[0], _PARAMETRIC_OPTIONS),
                              why="the parameters of the pencil are named by over=; the other "
                                  "keywords are options of the pencil",
                              fix="options: " + ", ".join(_PARAMETRIC_OPTIONS), doc=_DOC)
        unknown = [n for n in over if n not in self.params]
        if unknown:
            raise OptionError(f"parametric(over={tuple(over)!r}): {unknown} not declared "
                              "parameters." + did_you_mean(unknown[0], self.params),
                              fix="declared parameters: " + (", ".join(self.params) or "none"),
                              doc=_DOC)
        if drift is not None:
            _check_drift(drift, {"drift_tol": REQUIRED if drift_tol is None else drift_tol}, N)
        elif drift_tol is not None:
            raise OptionError("parametric(drift_tol=...) without drift=N2", fix="add drift=N2 "
                              "(a finer resolution) or drop drift_tol", doc=_DOC)

        def pencil(n: Any):
            comp = self.compile(n, dealias)
            if comp.kind != "EVP":
                raise OptionError("parametric(): the model is not an EVP", doc=_DOC)
            comp.problem.growth = self.eig_growth
            P = ParametricEVP(comp.problem, [comp.params[n_] for n_ in over],
                              ncc=comp.ncc or None, **kw)
            P.declared = {n_: self.params[n_] for n_ in over}
            return P
        P = pencil(N)
        if drift is not None:
            P.set_drift(pencil(drift), drift_tol=drift_tol)
        return P
