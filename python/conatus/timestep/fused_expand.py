"""Symbolic expansion of explicit right-hand sides for the fused evaluator (the compile step of
:class:`~conatus.timestep.fused.FusedRHS`): the polynomial form (monomials of differentiated
field components) and the general form (post-operators applied to products of linear forms);
derivatives of products are kept on the projected product, never expanded by Leibniz. Split out
of :mod:`conatus.timestep.fused`, which re-exports every name."""
from __future__ import annotations

from itertools import product as iproduct
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np

from ..ir import expr as E
from .spectral import SpectralError


Factor = tuple  # (field, comp tuple, derivative counts per coordinate)
Mono = tuple    # (coef, tuple of Factor), factors sorted by key


def _comps(e: E.Expr) -> list[tuple[int, ...]]:
    return list(iproduct(*[range(cs.dim) for cs in e.tensorsig]))


def _fkey(f: Factor):
    return (f[0].name, f[1], f[2])


def _peel(a: tuple) -> list:
    """Derivative counts from 0 up to ``a``, one derivative at a time, peeling the last
    differentiated axis first (the order of the former recursive memo): [(prev, cur, axis)].
    Iterative on purpose: a recursive closure is a reference cycle that keeps its memo's arrays
    alive until the cyclic GC (+7 to 28 % peak memory)."""
    steps, cur = [], tuple(a)
    while any(cur):
        ax = max(i for i, n in enumerate(cur) if n)
        prev = tuple(n - (i == ax) for i, n in enumerate(cur))
        steps.append((prev, cur, ax))
        cur = prev
    return steps[::-1]


def _mul_monos(a: list, b: list) -> list:
    return [(ca * cb, tuple(sorted(fa + fb, key=_fkey))) for ca, fa in a for cb, fb in b]


def _cross_monos(x: E.Expr, ex, mul) -> dict:
    """Expansion of cross(a, b): out[(k,)] = sum sign A[(i,)] B[(j,)] with the frame of
    ``cartesian_cross_table``; ``mul`` multiplies two monomial lists of the expander."""
    from ..ir.expr_algebra import cartesian_cross_table
    a, b = x.args
    A, B = ex(a), ex(b)
    out: dict = {}
    for (i, j), (k, sg) in cartesian_cross_table(a.tensorsig[0]).items():
        if (i,) in A and (j,) in B:
            out.setdefault((k,), []).extend((m[0] * sg,) + tuple(m[1:]) for m in mul(A[(i,)], B[(j,)]))
    return out


def _add_into(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)


def expand_polynomial(e: E.Expr, coords: Sequence, params: Mapping[str, Any]) -> dict:
    """{component tuple: [monomials]} for ``e``; raises SpectralError if not polynomial in
    differentiated field components."""
    nd = len(coords)

    def const(v) -> dict:
        return {(): [(complex(v) if np.iscomplexobj(v) else float(v), ())]}

    def diff1(d: dict, coord) -> dict:
        i = list(coords).index(coord)
        out: dict = {}
        for comp, monos in d.items():
            res = []
            for c, fs in monos:
                if len(fs) == 0:
                    continue                      # derivative of a constant
                if len(fs) > 1:
                    raise SpectralError("fused RHS: derivative of a product is not supported "
                                        "(Leibniz on the dealiased grid would change the numbers)")
                f, fc, a = fs[0]
                a = tuple(n + (1 if j == i else 0) for j, n in enumerate(a))
                res.append((c, ((f, fc, a),)))
            out[comp] = res
        return out

    def ex(x: E.Expr) -> dict:
        if isinstance(x, E.Field):
            return {c: [(1.0, ((x, c, (0,) * nd),))] for c in _comps(x)}
        if isinstance(x, E.Parameter):
            if x.rank:
                raise SpectralError("fused RHS: tensor parameter")
            v = params.get(x.name, x.value)
            if v is None:
                raise SpectralError(f"fused RHS: no value for parameter {x.name}")
            return const(v)
        if isinstance(x, E.Constant):
            return {} if x.rank else const(x.value)
        if isinstance(x, E.UnitVector):
            return {(x.index,): [(1.0, ())]}
        if isinstance(x, E.Add):
            out: dict = {}
            for a in x.args:
                _add_into(out, ex(a))
            return out
        if isinstance(x, E.Mul):
            parts = [ex(a) for a in x.args]
            ranks = [a.rank for a in x.args]
            if sum(1 for r in ranks if r) > 1:
                raise SpectralError("fused RHS: product of two tensors (use dot/outer)")
            out = {(): [(1.0, ())]}
            for p in parts:
                new: dict = {}
                for ca, ma in out.items():
                    for cb, mb in p.items():
                        new.setdefault(ca + cb, []).extend(_mul_monos(ma, mb))
                out = new
            return out
        if isinstance(x, E.Dot):
            a, b = x.args
            A, B = ex(a), ex(b)
            out = {}
            for ca, ma in A.items():
                for cb, mb in B.items():
                    if ca[-1] == cb[0]:
                        out.setdefault(ca[:-1] + cb[1:], []).extend(_mul_monos(ma, mb))
            return out
        if isinstance(x, E.Cross):                 # frame of cartesian_cross_table
            return _cross_monos(x, ex, _mul_monos)
        if isinstance(x, E.Component):
            A = ex(x.args[0])
            return {c[1:]: m for c, m in A.items() if c[0] == x.index}
        if isinstance(x, E.Differentiate):
            return diff1(ex(x.args[0]), x.coord)
        if isinstance(x, E.Gradient):
            A = ex(x.args[0])
            out = {}
            for i, co in enumerate(x.coordsys.coords):
                for c, m in diff1(A, co).items():
                    out[(i,) + c] = m
            return out
        if isinstance(x, E.Divergence):
            A = ex(x.args[0])
            out = {}
            for i, co in enumerate(x.args[0].tensorsig[0].coords):
                sub = {c[1:]: m for c, m in A.items() if c[0] == i}
                _add_into(out, diff1(sub, co))
            return out
        if isinstance(x, E.Laplacian):
            A = ex(x.args[0])
            out = {}
            for co in x.coordsys.coords:
                _add_into(out, diff1(diff1(A, co), co))
            return out
        raise SpectralError(f"fused RHS: node {type(x).__name__} not supported")

    return ex(e)


# ------------------------------------------------------------------------------------------------
# General form: sums of *post-operators* applied to products of *linear forms*,
#
#     F_(eq, comp) = sum_t  c_t  P_t [ prod_j  Lf_tj ],
#
# where a linear form Lf = sum_a c_a D^a f_a (+ constant) is computed in coefficient space (one
# backward transform per distinct form, however many fields and derivatives it combines), the
# products are formed on the dealiased grid, forward transformed, and the post-operator P_t (a
# product of spectral derivatives, then integrals along Fourier directions) is applied to the
# *projected* product in coefficient space. This is exactly what the evaluator does for
# ``diff(u * v, y)`` (product on the grid, projection, spectral derivative): no Leibniz rewrite.
# Example: the divergence form -d_j(u_i u_j) of Navier–Stokes, with u, v, w each a linear form of
# the poloidal/toroidal potentials, gives 3 forms and 6 products.
#
# Atom: (field, comp, derivative counts) or None (the constant 1).
# Form: tuple of (coef, atom) sorted by atom key, like atoms merged.
# Term: (coef, post, forms), post = (derivative counts, integrated flags) per coordinate.

def _akey(atom):
    return ("",) if atom is None else (atom[0].name, atom[1], atom[2])


def _form(pairs) -> tuple:
    acc: dict = {}
    for c, atom in pairs:
        k = _akey(atom)
        acc[k] = (acc[k][0] + c, atom) if k in acc else (c, atom)
    return tuple(sorted(((c, a) for c, a in acc.values() if c != 0), key=lambda p: _akey(p[1])))


def _formkey(form) -> tuple:
    return tuple((_akey(a), complex(c).real, complex(c).imag) for c, a in form)


def expand_general(e: E.Expr, coords: Sequence, params: Mapping[str, Any],
                   fourier_axes: Sequence[int]) -> dict:
    """{component: [terms]} of ``e`` in the general form above. Refused (SpectralError): a
    product whose factor is itself a post-processed product (nested products), a derivative
    along y after an integral along y, non-polynomial nodes."""
    nd = len(coords)
    P0 = ((0,) * nd, (False,) * nd)

    def const(v) -> dict:
        v = complex(v) if np.iscomplexobj(v) else float(v)
        return {(): [(v, P0, ())]}

    def collapse(terms: list):
        """Terms of a product operand -> one affine linear form, if they all are (degree <= 1,
        no post-operator); None otherwise."""
        pairs = []
        for c, post, forms in terms:
            if post != P0 or len(forms) > 1:
                return None
            if not forms:
                pairs.append((c, None))
            else:
                pairs.extend((c * cf, a) for cf, a in forms[0])
        return _form(pairs)

    def whole(A: list) -> list:
        """A linear (affine) operand as one term with one form (or a constant term)."""
        f = collapse(A)
        if f is None:
            return A
        if not f:
            return []
        if len(f) == 1 and f[0][1] is None:
            return [(f[0][0], P0, ())]
        c0 = f[0][0]                         # normalized: first coefficient 1 (u and -u: one form)
        return [(c0, P0, (tuple((c / c0, a) for c, a in f),))]

    def mul_terms(A: list, B: list) -> list:
        out = []
        for c1, p1, f1 in whole(A):
            for c2, p2, f2 in whole(B):
                if not f1 and p1 == P0:
                    out.append((c1 * c2, p2, f2))          # constant x anything
                elif not f2 and p2 == P0:
                    out.append((c1 * c2, p1, f1))
                elif p1 == P0 and p2 == P0:
                    out.append((c1 * c2, P0, tuple(sorted(f1 + f2, key=_formkey))))
                else:
                    raise SpectralError("fused RHS: product of a derivative/integral of a "
                                        "product (nested products) is not supported")
        return out

    def diff1(d: dict, i: int) -> dict:
        out: dict = {}
        for comp, terms in d.items():
            res = []
            for c, (dc, ig), forms in terms:
                if not forms:
                    continue                                   # constant
                if len(forms) == 1 and (dc, ig) == P0:          # linear: derivative inside
                    f = _form([(cf, None if a is None else
                                (a[0], a[1], tuple(n + (j == i) for j, n in enumerate(a[2]))))
                               for cf, a in forms[0] if a is not None])
                    if f:
                        res.append((c, P0, (f,)))
                    continue
                if ig[i]:
                    continue                                   # constant along i
                res.append((c, (tuple(n + (j == i) for j, n in enumerate(dc)), ig), forms))
            out[comp] = res
        return out

    def integ1(d: dict, i: int) -> dict:
        """Mark axis i as integrated; the interval length is applied in post() (a `length` argument was passed here and never read)."""
        out: dict = {}
        for comp, terms in d.items():
            res = []
            for c, (dc, ig), forms in terms:
                if not forms:
                    raise SpectralError("fused RHS: integral of a constant (length ambiguous)")
                if dc[i]:
                    if i in fourier_axes:
                        continue                               # periodic derivative: zero mean
                    raise SpectralError("fused RHS: integral of a derivative along a "
                                        "non-periodic direction is not supported")
                if ig[i]:
                    raise SpectralError("fused RHS: repeated integral")
                res.append((c, (dc, tuple(g or j == i for j, g in enumerate(ig))), forms))
            out[comp] = res
        return out

    def ex(x: E.Expr) -> dict:
        if isinstance(x, E.Field):
            return {c: [(1.0, P0, (((1.0, (x, c, (0,) * nd)),),))] for c in _comps(x)}
        if isinstance(x, E.Parameter):
            if x.rank:
                raise SpectralError("fused RHS: tensor parameter")
            v = params.get(x.name, x.value)
            if v is None:
                raise SpectralError(f"fused RHS: no value for parameter {x.name}")
            return const(v)
        if isinstance(x, E.Constant):
            return {} if x.rank else const(x.value)
        if isinstance(x, E.UnitVector):
            return {(x.index,): [(1.0, P0, ())]}
        if isinstance(x, E.Add):
            out: dict = {}
            for a in x.args:
                _add_into(out, ex(a))
            return out
        if isinstance(x, E.Mul):
            if sum(1 for a in x.args if a.rank) > 1:
                raise SpectralError("fused RHS: product of two tensors (use dot/outer)")
            out = {(): [(1.0, P0, ())]}
            for a in x.args:
                p = ex(a)
                new: dict = {}
                for ca, ma in out.items():
                    for cb, mb in p.items():
                        new.setdefault(ca + cb, []).extend(mul_terms(ma, mb))
                out = new
            return out
        if isinstance(x, E.Dot):
            a, b = x.args
            A, B = ex(a), ex(b)
            out = {}
            for ca, ma in A.items():
                for cb, mb in B.items():
                    if ca[-1] == cb[0]:
                        out.setdefault(ca[:-1] + cb[1:], []).extend(mul_terms(ma, mb))
            return out
        if isinstance(x, E.Cross):
            return _cross_monos(x, ex, mul_terms)
        if isinstance(x, E.Component):
            A = ex(x.args[0])
            return {c[1:]: m for c, m in A.items() if c[0] == x.index}
        if isinstance(x, E.Differentiate):
            return diff1(ex(x.args[0]), list(coords).index(x.coord))
        if isinstance(x, E.Gradient):
            A = ex(x.args[0])
            out = {}
            for i, co in enumerate(x.coordsys.coords):
                for c, m in diff1(A, list(coords).index(co)).items():
                    out[(i,) + c] = m
            return out
        if isinstance(x, E.Divergence):
            A = ex(x.args[0])
            out = {}
            for i, co in enumerate(x.args[0].tensorsig[0].coords):
                sub = {c[1:]: m for c, m in A.items() if c[0] == i}
                _add_into(out, diff1(sub, list(coords).index(co)))
            return out
        if isinstance(x, E.Laplacian):
            A = ex(x.args[0])
            out = {}
            for co in x.coordsys.coords:
                i = list(coords).index(co)
                _add_into(out, diff1(diff1(A, i), i))
            return out
        if isinstance(x, E.Integrate):
            A = ex(x.args[0])
            for co in x.coords:
                A = integ1(A, list(coords).index(co))
            return A
        raise SpectralError(f"fused RHS: node {type(x).__name__} not supported")

    out = ex(e)
    # merge identical (post, forms) terms of each component
    merged: dict = {}
    for comp, terms in out.items():
        acc: dict = {}
        for c, post, forms in terms:
            k = (post, tuple(_formkey(f) for f in forms))
            acc[k] = (acc[k][0] + c, post, forms) if k in acc else (c, post, forms)
        merged[comp] = [t for t in acc.values() if t[0] != 0]
    return merged
