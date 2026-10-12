"""Extended DMD: Koopman eigenvalues, eigenfunctions and modes on a declared dictionary.

Williams, Kevrekidis & Rowley (2015). Snapshot pairs (x_j, y_j = T(x_j)), j = 1 … m, and a
*declared* dictionary ψ = (ψ_1 … ψ_K): Ψ_X (m × K), Ψ_Y (m × K) its values on the pairs. With the
empirical Gram matrices (uniform sample weights 1/m, declared)

    G = Ψ_Xᴴ Ψ_X / m,   A = Ψ_Xᴴ Ψ_Y / m,   K = G⁺ A   (least squares, ``rcond`` declared),

K ξ_k = μ_k ξ_k gives Koopman eigenvalues μ_k and eigenfunctions φ_k(x) = ψ(x) ξ_k. The Koopman
modes v_k (state = Σ_k φ_k(x) v_k) are the least-squares fit of the full-state observable on the
eigenfunctions. Residual of each pair (ResDMD, Colbrook & Townsend 2023):

    res_k² = ξᴴ (L − μ̄ A − μ Aᴴ + |μ|² G) ξ / ξᴴ G ξ,   L = Ψ_Yᴴ Ψ_Y / m,

i.e. ‖φ_k∘T − μ_k φ_k‖/‖φ_k‖ on the data (≈ rounding when span ψ is Koopman-invariant). It is
evaluated in that second, cancellation-free form ‖(Ψ_Y − μ_k Ψ_X) ξ_k‖ / ‖Ψ_X ξ_k‖ (the quadratic
form loses half the digits: its floor is √eps).

Dictionaries (``dictionary=``): a callable x (m × d) -> (m × K) with ``names`` for ``explain``,
or the declared built-ins ``monomials(d, degree)`` (all monomials of total degree ≤ degree,
constant included) and ``rbf(centers, epsilon)`` (Gaussian RBFs + the state + a constant).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["Dictionary", "monomials", "rbf", "EDMD", "edmd"]


@dataclass
class Dictionary:
    """Declared observables ψ(x): x (m × d) -> (m × K)."""
    __repr__ = compact_repr   # one line, arrays by shape

    evaluate: Callable[[np.ndarray], np.ndarray]
    names: list[str]
    state_columns: list[int] | None = None   # columns holding the state x_i itself, if any

    def explain(self) -> str:
        """The declared observables."""
        shown = ", ".join(self.names[:8]) + (", ..." if len(self.names) > 8 else "")
        return (f"dictionary of {len(self.names)} observables psi(x): {shown}"
                + ("" if self.state_columns is None else
                   f"; the state itself in columns {list(self.state_columns)}"))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        P = np.asarray(self.evaluate(np.asarray(x)))
        if P.ndim != 2 or P.shape[1] != len(self.names):
            raise OptionError(f"dictionary returned shape {P.shape}, expected (m, {len(self.names)})")
        return P


def monomials(d: int, degree: int) -> Dictionary:
    """All monomials x^α, |α| ≤ degree, in d variables (constant first, then by degree)."""
    exps = [a for k in range(degree + 1) for a in itertools.product(range(k + 1), repeat=d)
            if sum(a) == k]
    exps = sorted(set(exps), key=lambda a: (sum(a), tuple(-v for v in a)))

    def ev(x):
        return np.stack([np.prod(x ** np.array(a)[None, :], axis=1) for a in exps], axis=1)

    names = ["1" if sum(a) == 0 else "·".join(f"x{i}^{p}" if p > 1 else f"x{i}"
                                              for i, p in enumerate(a) if p) for a in exps]
    state = [exps.index(tuple(int(i == j) for i in range(d))) for j in range(d)] if degree >= 1 else None
    return Dictionary(ev, names, state)


def rbf(centers: np.ndarray, epsilon: float) -> Dictionary:
    """[1, x_1 … x_d, exp(−ε² ‖x − c_k‖²) for each center] (declared centers and shape ε)."""
    C = np.asarray(centers, float)
    d = C.shape[1]

    def ev(x):
        r2 = ((x[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return np.concatenate([np.ones((x.shape[0], 1)), x, np.exp(-epsilon ** 2 * r2)], axis=1)

    names = ["1"] + [f"x{i}" for i in range(d)] + [f"rbf{k}" for k in range(C.shape[0])]
    return Dictionary(ev, names, list(range(1, d + 1)))


@dataclass
class EDMD:
    __repr__ = compact_repr   # one line, arrays by shape
    eigenvalues: np.ndarray        # μ_k (discrete time)
    omega: np.ndarray | None       # log μ_k / Δt if dt was declared
    eigenvectors: np.ndarray       # ξ_k (K × K columns)
    residuals: np.ndarray          # ResDMD residual of each pair
    modes: np.ndarray | None       # Koopman modes (K × d rows), if the state is reconstructible
    dictionary: Dictionary
    rcond: float
    m: int
    notes: list[str] = field(default_factory=list)

    def eigenfunctions(self, x: np.ndarray) -> np.ndarray:
        """φ_k(x) = ψ(x) ξ_k, shape (m, K)."""
        return self.dictionary(x) @ self.eigenvectors

    def explain(self) -> str:
        o = np.argsort(self.residuals)[:8]
        rows = "; ".join(f"μ = {self.eigenvalues[k]:.10g} (res {self.residuals[k]:.1e})" for k in o)
        return (f"EDMD (Williams et al. 2015): dictionary of {len(self.dictionary.names)} "
                f"observables [{', '.join(self.dictionary.names[:8])}"
                f"{' …' if len(self.dictionary.names) > 8 else ''}], m = {self.m} pairs "
                f"(uniform weights), K = G⁺A with rcond = {self.rcond:g}; smallest ResDMD "
                f"residuals: {rows}. " + " ".join(self.notes))


def edmd(X: np.ndarray, Y: np.ndarray, *, dictionary: Dictionary, rcond: float,
         dt: float | None) -> EDMD:
    """EDMD of the pairs (X[j], Y[j]) (arrays m × d) on the declared ``dictionary``."""
    X, Y = np.asarray(X), np.asarray(Y)
    if X.shape != Y.shape or X.ndim != 2:
        raise OptionError("X and Y must be (m × d) arrays of the same shape")
    if not rcond > 0:
        raise OptionError("rcond must be > 0 (declared pseudo-inverse cutoff)")
    m = X.shape[0]
    PX, PY = dictionary(X), dictionary(Y)
    G = PX.conj().T @ PX / m
    A = PX.conj().T @ PY / m
    K = np.linalg.pinv(G, rcond=rcond, hermitian=True) @ A
    notes = []
    sv = np.linalg.svd(G, compute_uv=False)
    if sv[-1] < rcond * sv[0]:
        notes.append(f"G is rank-deficient at rcond (σ_min/σ_max = {sv[-1] / sv[0]:.1e}): the "
                     "dictionary is redundant on the data; K is the minimum-norm solution")
    mu, Xi = np.linalg.eig(K)
    res = (np.linalg.norm(PY @ Xi - (PX @ Xi) * mu[None, :], axis=0)
           / np.linalg.norm(PX @ Xi, axis=0))
    modes = None
    if dictionary.state_columns is not None:
        Phi = PX @ Xi
        modes = np.linalg.lstsq(Phi, X, rcond=None)[0]
    omega = None if dt is None else np.log(mu.astype(complex)) / dt
    return EDMD(mu, omega, Xi, res, modes, dictionary, float(rcond), m, notes)
