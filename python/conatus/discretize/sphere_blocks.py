"""Spherical plans, part 1: groups, block entries and the ball's regular operators.

``SphereGroup`` (one azimuthal wavenumber m, or one (m, l)), ``_Ent`` (the sparse blocks of one
compiled tensor component, per degree l, with its radial level and r-power), and the identification
of the ball's first-order operators in regularity components from the spin operators (``ball_map``).
Public face and mathematical summary: ``conatus.discretize.sphere_plan``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiler import DiscretizationError
from .sphere_ops import regularity_Q

_SQ2 = np.sqrt(2.0)


@dataclass(frozen=True)
class SphereGroup:
    """Azimuthal wavenumber m (``index``: position along the stored azimuth axis) and the degrees
    ``ells`` held together (one degree in separable 3D groups, all of them otherwise)."""

    m: int
    index: int
    ells: tuple
    single: bool = False

    @property
    def label(self) -> str:
        return f"m={self.m}" + (f", l={self.ells[0]}" if self.single else "")

    def __str__(self) -> str:
        return self.label


class _Ent:
    """Compiled form of one output component: rows laid out over the group's degrees (bulk: n_l
    radial rows per valid l; point: one row per valid l), a Jacobi/ultraspherical level and an
    r-power p (None for a point entry; a lifted tau is held at p = lam on the shell, 0 in the ball,
    sphere_compiler._lift), and blocks per (field, field component)."""

    __slots__ = ("bulk", "comp", "level", "p", "blocks")

    def __init__(self, bulk: bool, comp: tuple, level, p, blocks: dict) -> None:
        self.bulk, self.comp, self.level, self.p, self.blocks = bulk, comp, level, p, blocks

    def map(self, A, **kw) -> _Ent:
        e = _Ent(self.bulk, self.comp, self.level, self.p,
                 {key: (A @ b).tocsr() for key, b in self.blocks.items()})
        for a, v in kw.items():
            setattr(e, a, v)
        return e

    def scale(self, c) -> _Ent:
        return _Ent(self.bulk, self.comp, self.level, self.p,
                    {key: (c * b).tocsr() for key, b in self.blocks.items()})


def _sum_blocks(ents: list[_Ent]) -> dict:
    out: dict = {}
    for e in ents:
        for key, b in e.blocks.items():
            out[key] = (out[key] + b).tocsr() if key in out else b
    return out


# ----------------------------------------------------------------------------------------------
# ball: regularity-component operators identified from the spin operators
# ----------------------------------------------------------------------------------------------

def _spin_first_order(H, kind: str, m: int, ell: int) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, beta) of the first-order spin operators alpha d_r + beta / r on one (m, l) block
    (unit sphere angular factors): grad (3x1), div (1x3), curl (3x3); spherical.md."""
    g = lambda sigma, s: H.G(sigma, m, s)[ell]           # noqa: E731
    if kind == "grad":
        a = np.array([[0], [0], [1]], dtype=complex)
        b = np.array([[g(-1, 0)], [g(1, 0)], [0]], dtype=complex)
    elif kind == "div":
        a = np.array([[0, 0, 1]], dtype=complex)
        b = np.array([[g(1, -1), g(-1, 1), 2]], dtype=complex)
    elif kind == "curl":
        a = np.array([[-1j, 0, 0], [0, 1j, 0], [0, 0, 0]], dtype=complex)
        b = np.array([[-1j, 0, 1j * g(-1, 0)], [0, 1j, -1j * g(1, 0)],
                      [1j * g(1, -1), -1j * g(-1, 1), 0]], dtype=complex)
    else:
        raise ValueError(kind)
    return a, b


def ball_map(H, kind: str, m: int, ell: int) -> dict:
    """{(i_out, i_in): ('up' | 'down', coefficient)}: the operator ``kind`` between regularity
    components of degree l, obtained as Q_out^H (alpha d_r + beta/r) Q_in and identified entry by
    entry with c D_+(k_in) (k_out = k_in + 1, beta = -k_in alpha) or c D_-(k_in) (k_out = k_in - 1,
    beta = (k_in + 1) alpha). Anything else is refused (it would not be regular at the centre)."""
    a, b = _spin_first_order(H, kind, m, ell)
    Q = regularity_Q(H, m, ell)
    Qi = Q if a.shape[1] == 3 else np.ones((1, 1))
    Qo = Q if a.shape[0] == 3 else np.ones((1, 1))
    A = Qo.conj().T @ a @ Qi
    B = Qo.conj().T @ b @ Qi
    reg = (-1, 1, 0)
    out = {}
    scale = max(np.abs(A).max(), np.abs(B).max(), 1.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            al, be = A[i, j], B[i, j]
            if abs(al) <= 1e-12 * scale and abs(be) <= 1e-12 * scale:
                continue
            k_in = ell + (reg[j] if A.shape[1] == 3 else 0)
            k_out = ell + (reg[i] if A.shape[0] == 3 else 0)
            if k_out == k_in + 1 and abs(be + k_in * al) <= 1e-10 * scale:
                out[(i, j)] = ("up", al)
            elif k_out == k_in - 1 and abs(be - (k_in + 1) * al) <= 1e-10 * scale:
                out[(i, j)] = ("down", al)
            else:
                raise DiscretizationError(
                    f"internal: ball {kind} at (m={m}, l={ell}) entry ({i},{j}) = {al:.3g} d_r + "
                    f"{be:.3g}/r is not a regular operator (k {k_in} -> {k_out})")
    return out


def _relabel(e: _Ent, comp: tuple) -> _Ent:
    return _Ent(e.bulk, comp, e.level, e.p, dict(e.blocks))
