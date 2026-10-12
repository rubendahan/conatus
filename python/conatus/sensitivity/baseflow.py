"""Sensitivity of an eigenvalue to a base-flow modification and to a steady forcing.

Marquet, Sipp & Jacquin (2008): for a linear operator A(U) built on a base state U, the
first-order eigenvalue change under U → U + δU is

    δσ = ⟨u†, (∂A/∂U · δU) u⟩ / ⟨u†, u⟩ =: ∫ G(y) δU(y) dy + Σ_{side, r} B_{side, r} δU^{(r)}(side).

Here the operator is a symbolic table (``ModalSystem``) produced by a *declared* factory
``factory(U) -> ModalSystem`` (e.g. ``lambda U: parallel_shear_flow(U, kx=..., ...)``, or the
Ginzburg–Landau model with U = μ(x)). The variation ∂A/∂U·δU is obtained symbolically (sympy:
U → U + ε φ(y), d/dε at 0), so every entry is Σ_m e_m(y) φ^{(m)}(y) × (derivatives of u). The
m-th derivatives of φ are moved onto the mode by m integrations by parts; the boundary terms are
**kept and reported** (``boundary``), not assumed to vanish. Only the state block A may depend on
U (a base-flow dependence of the pressure/constraint blocks is refused).

The *discrete* counterpart ``discrete_shift(δU)`` = −yᴴ δL x / (yᴴ M x) with δL = L(U + δU) − L(U)
is the exact derivative of the discrete eigenvalue (the tau descriptor is affine in U); the
density G converges to it spectrally (tested).

Steady forcing (Marquet et al. 2008, §2.3): if the base state obeys a *declared* linear equation
for its variation, Σ_k b_k(y) ∂^k δU + δF = 0 with homogeneous boundary conditions, then
δσ = ∫ H δF dy with H = −(L_bᵀ)⁻¹ G (the base-flow adjoint). It is computed on a Chebyshev tau
discretization of the declared operator (size declared), as the covector δF ↦ δσ expressed in the
exact Chebyshev Gram. See
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy
from numpy.polynomial import Chebyshev

from ..errors import BoundaryConditionError, OptionError, UnsupportedError
from ..linearize.inner_product import chebyshev_gram
from ..nonmodal.system import ModalSystem, Y
from . import _cheb
from .modes import ModePair, mode_pair
from .._repr import compact_repr

__all__ = ["BaseFlowSensitivity", "base_flow_sensitivity", "ForcingSensitivity",
           "steady_forcing_sensitivity"]

MAX_PHI_ORDER = 6


def _variation(factory: Callable, base) -> dict:
    """{(i, j): {k: {m: e_{k,m}(y)}}}: ∂A_ij/∂U·φ = Σ_k Σ_m e_{k,m} φ^{(m)} ∂^k (sympy)."""
    phi = sympy.Function("delta_base")(Y)
    eps = sympy.Symbol("epsilon_variation")
    s1 = factory(sympy.sympify(base) + eps * phi)
    P = [sympy.Symbol(f"P{m}") for m in range(MAX_PHI_ORDER + 1)]
    rep = {sympy.Derivative(phi, (Y, m)): P[m] for m in range(MAX_PHI_ORDER, 0, -1)}
    for blk in ("G", "C"):
        for key, op in getattr(s1, blk).items():
            for c in op.values():
                if sympy.diff(c, eps) != 0:
                    raise UnsupportedError(f"block {blk}{key} depends on the base state: only the "
                                           "state block A may (pressure/constraint coupling not "
                                           "implemented)")
    out: dict = {}
    for key, op in s1.A.items():
        for k, c in op.items():
            dc = sympy.expand(sympy.diff(c, eps).subs(eps, 0))
            if dc == 0:
                continue
            dc = sympy.expand(dc.xreplace(rep).xreplace({phi: P[0]}))
            if dc.has(phi) or dc.has(sympy.Derivative):
                raise UnsupportedError(f"A{key}: variation not reducible to φ^(m) up to order "
                                       f"{MAX_PHI_ORDER}")
            poly = sympy.Poly(dc, *P)
            if poly.total_degree() > 1 or poly.coeff_monomial(1) != 0:
                raise UnsupportedError(f"A{key}: coefficient not linear in the base state")
            for m in range(MAX_PHI_ORDER + 1):
                e = poly.coeff_monomial(P[m])
                if e != 0:
                    out.setdefault(key, {}).setdefault(k, {})[m] = e
    return out


@dataclass
class BaseFlowSensitivity:
    """δσ = ∫ G δU dy + Σ B_{side,r} δU^{(r)}(side) (first order; see module doc)."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    factory: Callable
    base: object
    pair: ModePair
    density: Chebyshev                         # G(y), exact Chebyshev series
    boundary: dict                             # {(side, r): B}
    degrees: dict = field(default_factory=dict)  # adaptive degrees of the e_{k,m}(y)

    @property
    def sigma(self) -> complex:
        return self.pair.sigma

    def gradient(self, yg) -> np.ndarray:
        """G(y) on the points ``yg``."""
        return self.density(np.asarray(yg, float))

    def predict(self, dU) -> complex:
        """First-order δσ for the base change ``dU`` (sympy in Y, number or callable)."""
        s, _ = _cheb.adaptive(dU, self.pair.bounds, what="base-state change δU")
        v = _cheb.integral(self.density * s)
        a, b = self.pair.bounds
        for (side, r), B in self.boundary.items():
            v += B * complex(s.deriv(r)(a if side == "left" else b) if r else
                             s(a if side == "left" else b))
        return complex(v)

    def discrete_shift(self, dU) -> complex:
        """Exact derivative of the discrete eigenvalue along ``dU`` (sympy in Y)."""
        return self.pair.shift_towards(self.factory(sympy.sympify(self.base) + sympy.sympify(dU)))

    def explain(self) -> str:
        bt = ", ".join(f"{s} φ^({r}): {abs(B):.1e}" for (s, r), B in self.boundary.items())
        return ("Base-flow sensitivity (Marquet, Sipp & Jacquin 2008): δσ = ∫ G δU dy + boundary "
                f"terms [{bt or 'none'}]; G from the symbolic variation of the operator table, "
                f"integrated by parts exactly (Chebyshev series, degree {self.density.degree()}; "
                f"coefficient fits {self.degrees}). " + self.pair.explain())


def base_flow_sensitivity(factory: Callable, base, *, N: int, target: complex, adjoint: str,
                          infinite_tol: float, adjoint_bcs=None) -> BaseFlowSensitivity:
    """Gradient of the eigenvalue nearest ``target`` of ``factory(base)`` w.r.t. the base state."""
    system = factory(sympy.sympify(base))
    if not isinstance(system, ModalSystem):
        raise OptionError("factory(U) must return a ModalSystem")
    pair = mode_pair(system, N=N, target=target, adjoint=adjoint, infinite_tol=infinite_tol,
                     adjoint_bcs=adjoint_bcs)
    var = _variation(factory, base)
    a, u, w = pair.fields("adjoint"), pair.fields("direct"), system.weights
    zero = _cheb.series([0.0], pair.bounds)
    h: dict[int, Chebyshev] = {}
    degrees = {}
    for (i, j), ops in var.items():
        for k, terms in ops.items():
            dku = u[j].deriv(k) if k else u[j]
            base_term = w[i] * _cheb.conj(a[i]) * dku
            for m, e in terms.items():
                es, deg = _cheb.adaptive(e, pair.bounds, what=f"variation of A{(i, j)}, ∂^{k}")
                degrees[f"A{(i, j)} k={k} m={m}"] = deg
                h[m] = h.get(m, zero) + es * base_term
    G = zero
    boundary: dict = {}
    lo, hi = pair.bounds
    for m, hm in h.items():
        G = G + (-1) ** m * (hm.deriv(m) if m else hm)
        for r in range(m):                     # ∫ h φ^(m) = Σ_r (−1)^r [h^(r) φ^(m−1−r)] + ...
            q = m - 1 - r
            hr = hm.deriv(r) if r else hm
            boundary[("right", q)] = boundary.get(("right", q), 0) + (-1) ** r * complex(hr(hi))
            boundary[("left", q)] = boundary.get(("left", q), 0) - (-1) ** r * complex(hr(lo))
    return BaseFlowSensitivity(factory, base, pair, G, boundary, degrees)


# ---------------------------------------------------------------------- steady forcing
@dataclass
class ForcingSensitivity:
    """δσ = ∫ H δF dy for a steady forcing δF of the declared base-state equation."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    base_flow: BaseFlowSensitivity
    density: Chebyshev                         # H(y)
    operator: dict
    bcs: tuple
    N: int
    solve: Callable                            # δF series -> δU series (declared operator)

    def gradient(self, yg) -> np.ndarray:
        return self.density(np.asarray(yg, float))

    def predict(self, dF) -> complex:
        """First-order δσ for the forcing ``dF`` (through H; degree < N exactly)."""
        s, _ = _cheb.adaptive(dF, self.base_flow.pair.bounds, what="forcing δF")
        return _cheb.integral(self.density * s)

    def base_change(self, dF) -> Chebyshev:
        """δU solving Σ b_k ∂^k δU + δF = 0 with the declared homogeneous conditions."""
        s, _ = _cheb.adaptive(dF, self.base_flow.pair.bounds, what="forcing δF")
        return self.solve(s)

    def explain(self) -> str:
        op = " + ".join(f"({c})·∂^{k}" for k, c in sorted(self.operator.items(), reverse=True))
        return (f"Steady-forcing sensitivity: base equation [{op}] δU + δF = 0, homogeneous BCs "
                f"{self.bcs}, Chebyshev tau N = {self.N}; H = −(L_bᵀ)⁻¹ G in the exact Chebyshev "
                "Gram (bilinear pairing ∫ H δF, real δF). " + self.base_flow.explain())


def _tau_operator(operator: dict, bcs, N: int, bounds) -> tuple[np.ndarray, int]:
    """Coefficient-space matrix of Σ b_k ∂^k (degree-N−1 truncation) with the BC rows last."""
    Lb = np.zeros((N, N), complex)
    for n in range(N):
        e = np.zeros(N)
        e[n] = 1.0
        Tn = Chebyshev(e, domain=list(bounds))
        acc = Chebyshev([0.0], domain=list(bounds))
        for k, c in operator.items():
            cs, _ = _cheb.adaptive(c, bounds, what=f"base operator coefficient b_{k}")
            acc = acc + cs * (Tn.deriv(k) if k else Tn)
        col = np.zeros(N, complex)
        cc = acc.coef[:N]
        col[:cc.size] = cc
        Lb[:, n] = col
    nb = len(bcs)
    for r, (order, side) in enumerate(bcs):
        yv = bounds[0] if side == "left" else bounds[1]
        row = []
        for n in range(N):
            e = np.zeros(N)
            e[n] = 1.0
            Tn = Chebyshev(e, domain=list(bounds))
            row.append((Tn.deriv(order) if order else Tn)(yv))
        Lb[N - nb + r] = row
    return Lb, nb


def steady_forcing_sensitivity(bfs: BaseFlowSensitivity, *, operator: dict, bcs, N: int
                               ) -> ForcingSensitivity:
    """Sensitivity to a steady forcing, for a base state obeying Σ b_k(y) ∂^k δU + δF = 0.

    ``operator``: {k: b_k} (sympy in Y or numbers), e.g. {2: 1/Re} for a parallel channel flow
    (ν U'' + F = 0); ``bcs``: homogeneous conditions [(order, 'left'|'right'), ...], e.g. no-slip
    [(0, 'left'), (0, 'right')]; ``N``: Chebyshev size of the base-state solve (declared).
    """
    for (order, side) in bcs:
        if side not in ("left", "right") or order < 0:
            raise BoundaryConditionError(f"bc {(order, side)}: expected (order >= 0, 'left'|'right')")
    if len(bcs) != max(operator):
        raise BoundaryConditionError(f"{len(bcs)} conditions for an operator of order "
                                     f"{max(operator)}: expected as many conditions as the order")
    bounds = bfs.pair.bounds
    Lb, nb = _tau_operator(operator, bcs, N, bounds)
    lu = np.linalg.inv(Lb)

    def solve(fs: Chebyshev) -> Chebyshev:
        f = np.zeros(N, complex)
        c = fs.coef[:N]
        f[:c.size] = c
        f[N - nb:] = 0.0                      # tau: the BC rows carry no forcing
        return _cheb.series(-(lu @ f), bounds)

    # covector ℓ on δU coefficients: ℓ_n = ∫ G T_n + Σ B T_n^{(r)}(side)
    ell = np.zeros(N, complex)
    for n in range(N):
        e = np.zeros(N)
        e[n] = 1.0
        Tn = Chebyshev(e, domain=list(bounds))
        v = _cheb.integral(bfs.density * Tn)
        for (side, r), B in bfs.boundary.items():
            yv = bounds[0] if side == "left" else bounds[1]
            v += B * complex((Tn.deriv(r) if r else Tn)(yv))
        ell[n] = v
    # δσ = ℓ · δU = −ℓ Lb⁻¹ P f  =: h · f ;  H with ∫ H δF = h · f  ->  Gram H = h
    P = np.eye(N)
    P[N - nb:] = 0.0
    hvec = -(ell @ lu @ P)
    Gm = chebyshev_gram(N, *bounds)
    Hc = np.linalg.solve(Gm, hvec)
    return ForcingSensitivity(bfs, _cheb.series(Hc, bounds), dict(operator), tuple(bcs), N, solve)
