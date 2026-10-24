"""Linear modal systems in one wall-normal coordinate, and their *continuous* adjoint.

A :class:`ModalSystem` is the linear operator of one Fourier mode (kx, kz), written as a table of
scalar differential operators in y with symbolic (sympy) coefficients:

    dt s_i = Σ_j A_ij s_j + Σ_m G_im p_m        (state rows, unit mass)
         0 = Σ_j C_mj s_j                       (constraint rows, e.g. continuity)
    ∂^k s_i (y_side) = 0                        (homogeneous boundary conditions, declared)

where each entry is Σ_k c_k(y) ∂_y^k. The state s carries the declared energy

    E = ½ Σ_i w_i ∫ |s_i|² dy                   (w_i > 0, constant, mandatory)

and the multipliers p (pressure) are not part of the state.

The *continuous* adjoint ("optimize, then discretize") is derived from this table by formal
integration by parts (``formal_adjoint``): for the block operator K = [[A, G], [C, 0]] and the
inner product ⟨(a,q),(u,p)⟩ = Σ w_i ∫ ā_i u_i + Σ ∫ q̄_m p_m,

    K† = [[W⁻¹ A* W, W⁻¹ C*], [G* W, 0]],   (c ∂^k)* f = (−1)^k ∂^k (c̄ f)  (Leibniz-expanded),

and the adjoint boundary conditions are *declared by the user* and *checked*: the bilinear
concomitant of Green's identity ⟨a, K u⟩ − ⟨K† a, u⟩ = [P(u, a)] must vanish identically once the
direct and adjoint conditions are imposed (``show_green_identity``). Nothing is guessed.

Discretization goes through the IR (tau + lifts, discretization plan): ``discretize(N)`` returns the
descriptor pencil M x' + L x = 0 with the exact Chebyshev Gram of the energy. Its *discrete*
adjoint W⁻¹ Lᴴ W ("discretize, then optimize") differs at finite N; ``tests/nonmodal`` checks that
both converge to the same quantities. See
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb

import numpy as np
import sympy

from ..errors import BoundaryConditionError, EquationError, OptionError
from ..linearize.inner_product import chebyshev_gram
from .._repr import compact_repr

__all__ = ["ModalSystem", "Descriptor", "parallel_shear_flow", "Y"]

# The wall-normal coordinate of every table (real, so that conj(y) = y).
Y = sympy.Symbol("y", real=True)

Op = dict      # {derivative order k: sympy coefficient c_k(y)}  meaning  Σ_k c_k(y) ∂_y^k


def _clean(op: Op) -> Op:
    """Drop zero coefficients (after expansion)."""
    out = {}
    for k, c in op.items():
        c = sympy.expand(c)
        if c != 0:
            out[k] = c
    return out


def _add(a: Op, b: Op) -> Op:
    out = dict(a)
    for k, c in b.items():
        out[k] = out.get(k, 0) + c
    return _clean(out)


def _scale(a: Op, s) -> Op:
    return _clean({k: s * c for k, c in a.items()})


def formal_adjoint_op(op: Op) -> Op:
    """(Σ_k c_k ∂^k)* = Σ_k (−1)^k ∂^k (c̄_k ·) = Σ_k (−1)^k Σ_m C(k,m) (∂^{k−m} c̄_k) ∂^m."""
    out: Op = {}
    for k, c in op.items():
        cb = sympy.conjugate(c)
        for m in range(k + 1):
            out[m] = out.get(m, 0) + (-1) ** k * comb(k, m) * sympy.diff(cb, Y, k - m)
    return _clean(out)


def _op_str(op: Op, f: str) -> str:
    terms = []
    for k in sorted(op, reverse=True):
        d = f if k == 0 else (f"∂y({f})" if k == 1 else f"∂y^{k}({f})")
        terms.append(f"({sympy.simplify(op[k])})·{d}")
    return " + ".join(terms) if terms else "0"


@dataclass
class ModalSystem:
    """dt s = A s + G p, 0 = C s, homogeneous BCs; energy ½ Σ w_i ∫|s_i|² (see module doc)."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    name: str
    bounds: tuple[float, float]
    state: tuple[str, ...]
    weights: dict[str, float]
    multipliers: tuple[str, ...]
    A: dict[tuple[str, str], Op]
    G: dict[tuple[str, str], Op]
    C: dict[tuple[str, str], Op]
    bcs: tuple[tuple[str, int, str], ...]          # (state field, derivative order, "left"|"right")
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        for n in self.state:
            w = self.weights.get(n)
            if w is None or not w > 0:
                raise OptionError(f"state field '{n}' needs an explicit energy weight > 0",
                                  fix="weights={'u': 1.0, ...}: every state field is in the norm")
        for (i, j) in self.A:
            if i not in self.state or j not in self.state:
                raise EquationError(f"A[{i},{j}]: both indices must be state fields")
        for (i, m) in self.G:
            if i not in self.state or m not in self.multipliers:
                raise EquationError(f"G[{i},{m}]: (state field, multiplier) expected")
        for (m, j) in self.C:
            if m not in self.multipliers or j not in self.state:
                raise EquationError(f"C[{m},{j}]: (multiplier, state field) expected")
        for (f, k, side) in self.bcs:
            if f not in self.state or side not in ("left", "right") or k < 0:
                raise BoundaryConditionError(f"boundary condition {(f, k, side)}: expected "
                                             "(state field, order >= 0, 'left'|'right')")

    # ------------------------------------------------------------------ continuous adjoint
    def formal_adjoint(self, bcs, *, suffix: str = "†") -> ModalSystem:
        """Continuous adjoint in the declared energy product, with **declared** adjoint BCs.

        ``bcs`` lists the adjoint conditions as (direct state name, order, side); the adjoint
        fields are named ``name + suffix``. The Green concomitant is checked to vanish; otherwise
        BoundaryConditionError (the declared conditions are not the adjoint ones).
        """
        w = self.weights
        a = {n: n + suffix for n in self.state}
        q = {m: m + suffix for m in self.multipliers}
        A2: dict = {}
        for (i, j), op in self.A.items():          # (A†)_{ji} = (w_i / w_j) (A_ij)*
            A2[(a[j], a[i])] = _add(A2.get((a[j], a[i]), {}),
                                    _scale(formal_adjoint_op(op), sympy.nsimplify(w[i] / w[j])))
        G2 = {(a[j], q[m]): _scale(formal_adjoint_op(op), sympy.nsimplify(1 / w[j]))
              for (m, j), op in self.C.items()}    # W⁻¹ C*
        C2 = {(q[m], a[j]): _scale(formal_adjoint_op(op), sympy.nsimplify(w[j]))
              for (j, m), op in self.G.items()}    # G* W
        adj = ModalSystem(f"adjoint of {self.name}", self.bounds, tuple(a[n] for n in self.state),
                          {a[n]: w[n] for n in self.state}, tuple(q[m] for m in self.multipliers),
                          {k: v for k, v in A2.items() if v}, G2, C2,
                          tuple((a[f], k, s) for (f, k, s) in bcs),
                          [f"continuous adjoint (formal, energy weights {w}) of '{self.name}'"])
        adj._direct = self
        res = self.green_concomitant(adj)
        if any(r != 0 for r in res.values()):
            raise BoundaryConditionError(
                "declared adjoint boundary conditions leave a nonzero Green concomitant",
                where=", ".join(f"{s}: {r}" for s, r in res.items()),
                why="⟨a, K u⟩ − ⟨K† a, u⟩ must vanish for all admissible u and a",
                fix="declare the adjoint BCs that cancel the terms listed (show_green_identity)")
        return adj

    def green_concomitant(self, adj: ModalSystem, *, impose: bool = True) -> dict[str, object]:
        """Boundary terms P at each side of ⟨a, K u⟩ − ⟨K† a, u⟩ = P(right) − P(left).

        Boundary values are symbols ``u^(k)(side)`` (direct) and ``ā^(k)(side)`` (adjoint,
        conjugated). With ``impose=True`` the direct and adjoint BCs set their symbols to 0.
        """
        fdir = {n: sympy.Function(n)(Y) for n in self.state + self.multipliers}
        fadj = {n: sympy.Function(f"conj({n})")(Y) for n in adj.state + adj.multipliers}
        amap = dict(zip(self.state + self.multipliers, adj.state + adj.multipliers, strict=True))
        wt = {**self.weights, **{m: 1.0 for m in self.multipliers}}
        P = 0
        blocks = list(self.A.items()) + list(self.G.items()) + list(self.C.items())
        for (row, col), op in blocks:              # row weight × ∫ ā_row · op(col)
            g0 = sympy.nsimplify(wt[row]) * fadj[amap[row]]
            for k, c in op.items():                # ∫ g ∂^k u = [Σ (−1)^m ∂^m g ∂^{k−1−m} u] + ...
                g = g0 * c
                for m in range(k):
                    P += (-1) ** m * sympy.diff(g, Y, m) * sympy.diff(fdir[col], Y, k - 1 - m)
        out = {}
        bc_zero = {(f, k, s) for (f, k, s) in self.bcs}
        bc_zero |= {(f, k, s) for (f, k, s) in adj.bcs}
        for side, yv in (("left", self.bounds[0]), ("right", self.bounds[1])):
            rep = {}
            for name, fn in list(fdir.items()) + list(fadj.items()):
                label = name if name in fdir else str(fn.func)
                for k in range(8, 0, -1):
                    rep[sympy.Derivative(fn, (Y, k))] = sympy.Symbol(f"{label}^({k})({side})")
                rep[fn] = sympy.Symbol(f"{label}({side})")
            e = sympy.expand(P).xreplace(rep).subs(Y, yv)
            if impose:
                zero = {}
                for (f, k, s) in bc_zero:
                    if s != side:
                        continue
                    lab = f if f in fdir else f"conj({f})"
                    zero[sympy.Symbol(f"{lab}({side})" if k == 0 else f"{lab}^({k})({side})")] = 0
                e = e.subs(zero)
            out[side] = sympy.simplify(sympy.expand(e))
        return out

    # ------------------------------------------------------------------ display
    def show_equations(self) -> str:
        lines = [f"{self.name}:"]
        for i in self.state:
            terms = [_op_str(op, j) for (r, j), op in self.A.items() if r == i]
            terms += [_op_str(op, m) for (r, m), op in self.G.items() if r == i]
            lines.append(f"  dt {i} = " + " + ".join(terms))
        for m in self.multipliers:
            terms = [_op_str(op, j) for (r, j), op in self.C.items() if r == m]
            lines.append("  0 = " + " + ".join(terms) + f"      (multiplier {m})")
        return "\n".join(lines)

    def show_boundary_conditions(self) -> str:
        a, b = self.bounds
        def lhs(f, k):
            return f if k == 0 else (f"∂y {f}" if k == 1 else f"∂y^{k} {f}")
        return "\n".join(f"  {lhs(f, k)}(y = {a if s == 'left' else b:g}) = 0"
                          for (f, k, s) in self.bcs)

    def show_green_identity(self, adj: ModalSystem) -> str:
        raw = self.green_concomitant(adj, impose=False)
        imp = self.green_concomitant(adj, impose=True)
        lines = ["⟨a, K u⟩ − ⟨K† a, u⟩ = P(right) − P(left), energy product "
                 f"Σ w_i ∫ ā_i u_i (w = {self.weights}) + ∫ q̄ p:"]
        for side in ("left", "right"):
            lines.append(f"  P({side}) = {raw[side]}")
            lines.append(f"    with the direct and adjoint BCs: {imp[side]}")
        return "\n".join(lines)

    def explain(self) -> str:
        return (f"ModalSystem '{self.name}' on y in {list(self.bounds)}: state {self.state} "
                f"(energy ½ Σ w_i ∫|s_i|², w = {self.weights}), multipliers {self.multipliers}, "
                f"{len(self.bcs)} homogeneous BCs. " + " ".join(self.notes) + "\n"
                + self.show_equations() + "\n" + self.show_boundary_conditions())

    # ------------------------------------------------------------------ discretization
    def discretize(self, N: int) -> Descriptor:
        """Tau/lift discretization through the IR: M x' + L x = 0, dense, exact energy Gram."""
        from ..discretize import discretize as _disc
        from ..ir import IVP, CartesianCoordinates, Chebyshev, Domain, Field, diff, dt, interp, lift
        nbc = {n: sum(1 for (f, _, _) in self.bcs if f == n) for n in self.state}
        if N < max(nbc.values()) + 2:
            from ..errors import ResolutionError
            raise ResolutionError(f"N = {N} is too small for {max(nbc.values())} BCs per field")
        cs = CartesianCoordinates("y")
        y = cs["y"]
        b = Chebyshev(y, N, tuple(float(v) for v in self.bounds), dealias=1)
        D = Domain(cs, [b])
        D0 = D.without(y)
        F = {n: Field(n.replace("†", "_adj"), D) for n in self.state + self.multipliers}
        taus = {n: [Field(f"tau_{n.replace('†', '_adj')}{k}", D0) for k in range(1, nbc[n] + 1)]
                for n in self.state}
        ncc: dict = {}
        counter = itertools.count()

        def coeff(c):
            c = sympy.nsimplify(c) if not c.free_symbols else c
            if Y not in c.free_symbols:
                return complex(c)
            fld = Field(f"ncc{next(counter)}", D)
            fn = sympy.lambdify(Y, c, "numpy")
            ncc[fld] = lambda Yg, fn=fn: np.asarray(fn(Yg), dtype=complex) + 0 * Yg
            return fld

        def apply(op, f):
            out = 0
            for k, c in sorted(op.items()):
                d = f
                for _ in range(k):
                    d = diff(d, y)
                out = out + coeff(c) * d
            return out

        pb = IVP([F[n] for n in self.state + self.multipliers]
                 + [t for n in self.state for t in taus[n]], name=self.name)
        for i in self.state:
            rhs = 0
            for (r, j), op in self.A.items():
                if r == i:
                    rhs = rhs + apply(op, F[j])
            for (r, m), op in self.G.items():
                if r == i:
                    rhs = rhs + apply(op, F[m])
            lifts = sum(lift(t, b, -(k + 1)) for k, t in enumerate(taus[i]))
            pb.add_equation((dt(F[i]) - rhs + lifts, 0))
        for m in self.multipliers:
            row = 0
            for (r, j), op in self.C.items():
                if r == m:
                    row = row + apply(op, F[j])
            if isinstance(row, int):
                raise EquationError(f"constraint of multiplier '{m}' is empty")
            pb.add_equation((row, 0))
        for (f, k, side) in self.bcs:
            d = F[f]
            for _ in range(k):
                d = diff(d, y)
            pb.add_equation((interp(d, y, self.bounds[0] if side == "left" else self.bounds[1]), 0))
        plan = _disc(pb, ncc=ncc)
        g = plan.matrices()
        names = {F[n].name: n for n in F}
        state_cols = {n: np.array([c for c, col in enumerate(g.cols)
                                   if col[2] != "tau" and names.get(col[0]) == n], dtype=int)
                      for n in self.state}
        cols = np.concatenate([state_cols[n] for n in self.state])
        a, bb = self.bounds
        with np.errstate(divide="ignore"):         # np.where evaluates the odd branch too
            Gz = chebyshev_gram(N, a, bb)
        W = np.zeros((len(cols), len(cols)))
        for k, n in enumerate(self.state):         # E = ½ Σ w_i ∫|s_i|²
            W[k * N:(k + 1) * N, k * N:(k + 1) * N] = 0.5 * self.weights[n] * Gz
        return Descriptor(self, N, g.M.toarray().astype(complex), g.L.toarray().astype(complex),
                          cols, W, b, g)


@dataclass
class Descriptor:
    """M x' + L x = 0 (dense), state columns ``cols`` (state fields in order, N coefficients each)
    and the exact energy Gram ``W`` on them (½ w_i × ∫ T_m T_n dy)."""
    __repr__ = compact_repr   # one line, arrays by shape

    system: ModalSystem
    N: int
    M: np.ndarray
    L: np.ndarray
    cols: np.ndarray
    W: np.ndarray
    basis: object
    matrices: object

    @property
    def n(self) -> int:
        return self.M.shape[0]

    def energy_factor(self) -> np.ndarray:
        """F with ‖F x_s‖₂² = E(x_s): Cholesky factor of the exact Gram (upper, W = Fᴴ F)."""
        return np.linalg.cholesky(self.W).conj().T

    def evaluate(self, xs: np.ndarray, yg: np.ndarray) -> dict[str, np.ndarray]:
        """State coefficients -> values of each state field on points ``yg``."""
        a, b = self.system.bounds
        t = (2 * np.asarray(yg, dtype=float) - (a + b)) / (b - a)      # affine map to [-1, 1]
        return {n: np.polynomial.chebyshev.chebval(t, xs[k * self.N:(k + 1) * self.N])
                for k, n in enumerate(self.system.state)}

    def explain(self) -> str:
        return (f"Descriptor of '{self.system.name}': Chebyshev N = {self.N}, system {self.n}, "
                f"state {len(self.cols)} coefficients {self.system.state}; exact Chebyshev Gram "
                "(closed-form ∫T_mT_n), energy ½ Σ w_i ∫|s_i|².")


def parallel_shear_flow(U, *, kx: float, kz: float, Re: float,
                        walls: str, bounds: tuple[float, float] = (-1.0, 1.0)) -> ModalSystem:
    """Linearized Navier–Stokes around U(y) e_x, mode e^{i(kx x + kz z)}, primitive variables.

        dt u = (Δ/Re − i kx U) u − U' v − i kx p
        dt v = (Δ/Re − i kx U) v − ∂y p
        dt w = (Δ/Re − i kx U) w − i kz p,        0 = i kx u + ∂y v + i kz w,

    Δ = ∂y² − kx² − kz². ``U`` is a sympy expression in ``conatus.nonmodal.Y`` (real). ``walls`` is
    mandatory (no default); only "no-slip" (u = v = w = 0 at both walls) is implemented. Energy:
    ½ ∫ (|u|² + |v|² + |w|²) dy (weights 1). kz = 0 keeps w (it decouples: Squire); the
    reduction is not applied silently.
    """
    if walls != "no-slip":
        raise OptionError(f"walls='{walls}': only 'no-slip' is implemented")
    if not Re > 0:
        raise OptionError(f"Re must be > 0 (got {Re})")
    U = sympy.sympify(U)
    kx_, kz_, Re_ = sympy.Float(kx), sympy.Float(kz), sympy.Float(Re)
    lap = {2: 1 / Re_, 0: -(kx_ ** 2 + kz_ ** 2) / Re_}
    adv = _add(lap, {0: -sympy.I * kx_ * U})
    A = {("u", "u"): adv, ("v", "v"): adv, ("w", "w"): adv}
    Uy = sympy.diff(U, Y)
    if Uy != 0:
        A[("u", "v")] = {0: -Uy}
    G = {("u", "p"): {0: -sympy.I * kx_}, ("v", "p"): {1: sympy.Integer(-1)},
         ("w", "p"): {0: -sympy.I * kz_}}
    C = {("p", "u"): {0: sympy.I * kx_}, ("p", "v"): {1: sympy.Integer(1)},
         ("p", "w"): {0: sympy.I * kz_}}
    bcs = tuple((f, 0, s) for f in ("u", "v", "w") for s in ("left", "right"))
    return ModalSystem(f"LNS, U = {U}, kx = {kx:g}, kz = {kz:g}, Re = {Re:g}",
                       tuple(float(v) for v in bounds), ("u", "v", "w"),
                       {"u": 1.0, "v": 1.0, "w": 1.0}, ("p",), A, G, C, bcs,
                       ["primitive variables (u, v, w, p), no-slip walls"])
