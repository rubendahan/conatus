"""Ill-posedness check of a native initial-value problem at compile time.

``dt(T) + kappa*lap(T) = 0`` (a sign error for the heat equation) was built, run and explained
without a word: its implicit operator makes every Fourier mode grow like ``exp(kappa k^2 t)``,
faster the finer the grid, an ill-posed problem (backward diffusion). The check reads the
**principal symbol** of each evolution equation off the compiled terms:

* an equation ``a*dt(u) + sum of implicit terms = explicit terms``, with ``a`` a number, is
  probed with a plane wave ``u = exp(i k x_c)`` along each coordinate ``x_c`` in turn;
* each implicit term that is a number times derivatives of ``u`` alone (``dx``, ``dz``, ...,
  ``lap``: the operators whose symbol is a power of ``i k``) contributes ``c_p (i k)^p``; the
  growth rate of the mode is ``s(k) = -sum_p c_p (i k)^p / a``;
* the leading power ``P`` decides: if ``Re(-c_P i^P / a) > 0`` the growth rate increases without
  bound with ``|k|`` and the problem is ill-posed; an ``IllPosedWarning`` names the leading term(s)
  and the sign that would make it well-posed, and ``explain()`` repeats it.

Only the diagonal part (terms in the unknown that ``dt`` acts on) is read. A term whose symbol is
not read (a non-constant coefficient, ``grad``/``div``/``curl``, another unknown) and whose
derivative order is at least ``P`` makes the check say nothing for that equation: a detection
never guesses. Lower-order terms of the wrong sign are physics (the ``+lap(u)`` of
Kuramoto--Sivashinsky is dominated by ``+lap(lap(u))``) and are not reported. Nothing is changed.
"""
from __future__ import annotations

import warnings
from typing import Any

from ..ir import Add, Constant, Field, Mul, Parameter
from ..ir.expr_algebra import Power
from ..ir.expr_calculus import Curl, Differentiate, Divergence, Gradient, Laplacian, TimeDerivative


def _num(e: Any) -> complex | None:
    """Numeric value of an expression without fields (constants and parameters), else None."""
    if isinstance(e, Constant):
        return complex(e.value)
    if isinstance(e, Parameter):
        return None if e.value is None else complex(e.value)
    if isinstance(e, Mul):
        out = 1 + 0j
        for a in e.args:
            v = _num(a)
            if v is None:
                return None
            out *= v
        return out
    if isinstance(e, Add):
        vals = [_num(a) for a in e.args]
        return None if any(v is None for v in vals) else complex(sum(vals))
    if isinstance(e, Power):
        base = _num(e.args[0])
        if base is None:
            return None
        try:
            return complex(base ** complex(e.exponent))
        except (ZeroDivisionError, OverflowError):
            return None
    return None


def _symbol(e: Any, u: Field, coord: Any) -> dict[int, complex] | None:
    """Symbol {p: c_p} of a term linear in ``u`` along ``coord`` (powers of q = i k), or None
    when it is not read (module docstring)."""
    if e == u:
        return {0: 1 + 0j}
    if isinstance(e, Differentiate):
        s = _symbol(e.args[0], u, coord)
        if s is None:
            return None
        return {p + 1: c for p, c in s.items()} if e.coord == coord else {}
    if isinstance(e, Laplacian):
        s = _symbol(e.args[0], u, coord)
        return None if s is None else {p + 2: c for p, c in s.items()}
    if isinstance(e, Mul):
        num, inner = 1 + 0j, None
        for a in e.args:
            v = _num(a)
            if v is not None:
                num *= v
            elif inner is None:
                inner = a
            else:
                return None
        if inner is None:
            return None
        s = _symbol(inner, u, coord)
        return None if s is None else {p: num * c for p, c in s.items()}
    if isinstance(e, Add):
        out: dict[int, complex] = {}
        for a in e.args:
            s = _symbol(a, u, coord)
            if s is None:
                return None
            for p, c in s.items():
                out[p] = out.get(p, 0) + c
        return out
    return None


def _order(e: Any) -> int:
    """Highest number of spatial derivatives on any path of the tree (an upper bound)."""
    own = 2 if isinstance(e, Laplacian) else 1 if isinstance(
        e, (Differentiate, Gradient, Divergence, Curl)) else 0
    return own + max((_order(a) for a in e.args), default=0)


def _dt_term(t: Any) -> tuple[complex, Field] | None:
    """(a, u) for a term a*dt(u) with a a number, else None."""
    if isinstance(t, TimeDerivative) and isinstance(t.args[0], Field):
        return 1 + 0j, t.args[0]
    if isinstance(t, Mul):
        num, dt = 1 + 0j, None
        for a in t.args:
            v = _num(a)
            if v is not None:
                num *= v
            elif isinstance(a, TimeDerivative) and isinstance(a.args[0], Field) and dt is None:
                dt = a.args[0]
            else:
                return None
        return (num, dt) if dt is not None else None
    return None


def check_ill_posed(comp: Any, eqs: list, labels: list, terms: Any) -> list[str]:
    """Warn (IllPosedWarning) and return the explain() lines for every evolution equation whose
    implicit principal symbol grows without bound with the wavenumber (module docstring).
    ``eqs``: (implicit lhs, explicit rhs) per equation, ``terms``: the term splitter."""
    from ..errors import IllPosedWarning
    from ..ir.expr_algebra import neg
    out = []
    coords = list(comp.coordsys.coords)
    for (lhs, _rhs), label in zip(eqs, labels, strict=True):
        ts = list(terms(lhs))
        dts = [d for d in (_dt_term(t) for t in ts) if d is not None]
        if len(dts) != 1 or abs(dts[0][0]) == 0:
            continue
        a, u = dts[0]
        rest = [t for t in ts if _dt_term(t) is None and u in t.free_fields()]
        if not rest:
            continue
        for c in coords:
            total: dict[int, complex] = {}
            by_power: dict[int, list] = {}
            blind = 0
            for t in rest:
                s = _symbol(t, u, c)
                if s is None:
                    blind = max(blind, _order(t))
                    continue
                for p, v in s.items():
                    if v != 0:
                        total[p] = total.get(p, 0) + v
                        by_power.setdefault(p, []).append(t)
            powers = [p for p, v in total.items() if abs(v) > 0]
            if not powers:
                continue
            P = max(powers)
            if P < 1 or blind >= P:
                continue
            growth = (-total[P] * (1j ** P) / a).real
            if growth <= 0:
                continue
            names = ", ".join(str(t) for t in by_power[P])
            flip = " + ".join(str(neg(t)) for t in by_power[P])
            right = " + ".join(str(t) for t in by_power[P])
            text = (f"ill-posed: in [{label}] the implicit term {names} makes dt({u.name}) grow "
                    f"like +{growth:.3g} k^{P} along {c.name} (faster the larger the wavenumber "
                    f"k: backward diffusion, or a negative viscosity); did you mean {flip} on the "
                    f"left side, that is dt({u.name}) = {right} + ...?")
            out.append(text)
            warnings.warn(f"model '{comp.model.name}': {text} (the run would only grow the "
                          "finest modes; nothing is changed: correct the sign if it is a "
                          "mistake)", IllPosedWarning, stacklevel=4)
            break                                      # one direction is enough per equation
    return out
