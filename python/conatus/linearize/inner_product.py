"""Inner products as objects (spec §30): Gram matrices on the coefficient vector of a group.

``InnerProduct(plan, weights={"u": 1, "T": 1})`` is the weighted L² product
sum_v w_v ∫ conj(v1) v2 dV over the field unknowns of one group. Gram
entries of the non-periodic basis are exact (Chebyshev: ∫ T_m T_n dz in closed form; Legendre,
Jacobi, Gegenbauer: Gauss–Legendre quadrature exact for their degree; Hermite and Laguerre
functions: orthonormal; compound intervals: one Chebyshev block per piece); Fourier
modes carry their exact norm (L for complex e^{ikx}, L/2 for cos/sin, L for the
mean). Tau and constant unknowns get weight 0: they are not part of the state.

The adjoint of an operator A for this product is A† = W⁻¹ Aᴴ W on the field
subspace (spec §4: transpose ≠ adjoint). It is only defined where W is
invertible: a singular W (tau or zero-weight columns) is refused by ``adjoint()``,
which points to the left eigenvectors (``solve_dense(..., left=True)``) or a tau-free
formulation. No column is ever dropped for you.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

from ..ir.maps import is_mapped
from ..errors import OptionError

__all__ = ["InnerProduct", "chebyshev_gram", "basis_gram"]


def chebyshev_gram(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """G[m, k] = ∫_a^b T_m T_k dz (affine map to [-1, 1]), exact."""
    def I(k):  # noqa: E743 -- ∫_{-1}^{1} T_k (mathematical notation)
        # 2/(1 - k²) for even k, 0 for odd k; the division runs on even k only (np.where would
        # evaluate it at k = 1 too and emit a spurious divide-by-zero warning)
        even = k % 2 == 0
        return np.divide(2.0, 1.0 - k.astype(float) ** 2, out=np.zeros(k.shape), where=even)
    m = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    return 0.5 * (b - a) * 0.5 * (I(m + k) + I(np.abs(m - k)))


def basis_gram(basis, n: int) -> np.ndarray:
    """G[m, k] = ∫ φ_m φ_k dx over the basis interval, exact, for the coefficients of ``basis``
    as declared by its ``coeff_space()`` (the Chebyshev Gram used to be applied to
    every non-periodic basis, a wrong norm and adjoint for Legendre, Jacobi, compound intervals,
    and an infinite one on Hermite/Laguerre axes). Refused for a basis without an exact Gram."""
    from scipy.special import eval_gegenbauer, eval_jacobi

    from ..discretize.jacobi_family import gauss_legendre
    from ..errors import UnsupportedError
    from ..ir.bases import (Chebyshev, CompoundChebyshev, Hermite, Jacobi, Laguerre, Legendre,
                            Ultraspherical)
    if is_mapped(basis):
        # The closed-form Gram is for the affine map only; a mapped axis needs the map's metric
        # dy = m'(xi) dxi (audit C5: non-finite or wrong entries).
        raise UnsupportedError(f"InnerProduct on {basis!r}: the Gram with the map metric is not "
                               "implemented (MappedChebyshev)")
    if n != basis.size:
        raise OptionError(f"basis_gram: {n} coefficients for a basis of size {basis.size}",
                          fix="give as many coefficients as the basis size")
    if isinstance(basis, Chebyshev):
        return chebyshev_gram(n, *basis.bounds)
    if isinstance(basis, Jacobi):
        if isinstance(basis, Ultraspherical) and basis.lam == 0:
            raise UnsupportedError("InnerProduct on Ultraspherical(lam=0): C_n^(0) vanishes for "
                                   "n >= 1 in the standard normalization",
                                   fix="Chebyshev(...) for the Chebyshev T basis")
        t, w = gauss_legendre(n + 1)                 # exact for degree 2n + 1 >= 2(n - 1)
        k = np.arange(n)
        if isinstance(basis, Ultraspherical):
            V = eval_gegenbauer(k[None, :], basis.lam, t[:, None])
        elif isinstance(basis, Legendre):
            V = eval_jacobi(k[None, :], 0.0, 0.0, t[:, None])
        else:
            V = eval_jacobi(k[None, :], basis.a, basis.b, t[:, None])
        a, b = basis.bounds
        return 0.5 * (b - a) * (V.T * w) @ V
    if isinstance(basis, (Hermite, Laguerre)):
        # orthonormal functions of y = (x - x0)/s (psi_n; L_n e^{-y/2}): dx = s dy
        return basis.scale * np.eye(n)
    if isinstance(basis, CompoundChebyshev):
        G = np.zeros((n, n))
        for o, m, (a, b) in zip(basis.offsets, basis.sizes, basis.pieces, strict=True):
            G[o:o + m, o:o + m] = chebyshev_gram(m, a, b)
        return G
    raise UnsupportedError(f"InnerProduct on {basis!r}: no exact Gram matrix for this basis")


class InnerProduct:
    """Weighted L² inner product on the field unknowns of a plan group."""

    def __init__(self, plan, weights: Mapping[str, float] | None = None) -> None:
        self.plan = plan
        names = [v.name for v in plan.variables]
        taus = {t.name for t in plan.taus}
        default = {n: (0.0 if n in taus else 1.0) for n in names}
        if weights is not None:
            unknown = set(weights) - set(names)
            if unknown:
                raise OptionError(f"weights for unknown variables: {sorted(unknown)}",
                                  fix="weights only for the unknowns of the problem")
            # Every state unknown needs an explicit weight (0 allowed): an unlisted one used to get
            # 0 silently, changing the norm and the adjoint (audit C3).
            missing = [n for n in names if n not in taus and n not in weights]
            if missing:
                raise OptionError(f"no weight for {', '.join(repr(n) for n in missing)}: give every "
                                  "non-tau unknown a weight (0 to exclude it from the state)",
                                  fix="give every non-tau unknown a weight (0 to exclude it from the state)")
            default = {n: float(weights.get(n, 0.0)) for n in names}
        negative = sorted(n for n, w in default.items() if not w >= 0)
        if negative:                    # W must be positive semi-definite (audit C4)
            raise OptionError(f"negative (or NaN) weights for {negative}: not an inner product",
                              fix="non-negative weights")
        self.weights = default

    def explain(self) -> str:
        """The product and its weights."""
        w = ", ".join(f"{n}: {v:g}" for n, v in self.weights.items())
        return ("Inner product: weighted L² over the domain, exact Gram of the non-periodic basis, exact Fourier "
                f"mode norms; weights {{{w}}} (0 = not part of the state)")

    def gram(self, group=None, reduced: bool | None = None) -> sp.csr_matrix:
        """Hermitian positive semi-definite W on the group's column space."""
        plan = self.plan
        group = plan.groups[0] if group is None else group
        L = plan.layout
        blocks = []
        for v, vl in zip(plan.variables, plan.var_layouts(group), strict=True):
            w = self.weights[v.name]
            fnorm = np.ones(vl.nf)
            for i, d in enumerate(L.fdirs):
                if not vl.fdep[i]:
                    continue
                Lx = d.basis.length
                idx = L.fourier_indices(vl.fdep, group)
                fnorm = fnorm * np.array([
                    Lx if (d.complex or fi[i] == 0) else Lx / 2 for fi in idx])
            if vl.zdep:
                Gz = basis_gram(L.zbasis, vl.nz)
            else:
                Gz = np.ones((1, 1))
            for _c in range(vl.ncomp):
                blocks.append(sp.kron(sp.diags(w * fnorm), sp.csr_matrix(Gz)))
        W = sp.block_diag(blocks, format="csr")
        if reduced is None:
            reduced = plan.problem.symmetry is not None
        if reduced:
            ci = plan.matrices(group, True).col_index
            W = W[ci][:, ci].tocsr()
        return W

    def inner(self, x: np.ndarray, y: np.ndarray, group=None, reduced=None) -> complex:
        """⟨x, y⟩ = xᴴ W y on the group's column space."""
        W = self.gram(group, reduced)
        return complex(np.conj(x) @ (W @ y))

    def norm(self, x: np.ndarray, group=None, reduced=None) -> float:
        """sqrt⟨x, x⟩ (round-off negative values clipped to 0)."""
        return float(np.sqrt(max(self.inner(x, x, group, reduced).real, 0.0)))

    def adjoint(self, A, group=None, reduced=None) -> np.ndarray:
        """A† = W⁻¹ Aᴴ W, refused when W is singular (tau / zero-weight columns)."""
        W = self.gram(group, reduced).toarray()
        s = np.linalg.svd(W, compute_uv=False)
        if s.size and s[-1] <= 1e-14 * s[0]:
            raise OptionError("Gram matrix is singular (tau or zero-weight unknowns): the adjoint for this "
                              "inner product is not defined on the full column space. Use left eigenvectors "
                              "(solve_dense(..., left=True)) for eigenvalue sensitivities, or a tau-free "
                              "(Galerkin) formulation.",
                              fix="left eigenvectors (solve_dense(..., left=True)) for sensitivities, or a "
                                  "tau-free (Galerkin) formulation")
        Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
        return np.linalg.solve(W, Ad.conj().T @ W)
