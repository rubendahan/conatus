"""Unstable manifolds of equilibria by the parametrization method.
Notes: Cabré, Fontich & de la Llave (2003, 2005); Haro,
Canadell, Figueras, Luque & Mondelo, *The Parameterization Method for Invariant Manifolds*
(Springer 2016), ch. 2.

For M ẋ = G(x), an equilibrium x₀ and d unstable eigenvalues λ₁..λ_d (right eigenvectors v_j,
complex allowed), we look for P : C^d → R^n with P(0) = x₀, ∂_j P(0) = v_j and the invariance
equation

    G(P(s)) = M DP(s) Λ s,       Λ = diag(λ₁..λ_d),

so the dynamics on the manifold is conjugate to ṡ = Λ s (exactly linear). With P(s) = Σ_α a_α s^α
(multi-indices |α| ≤ K, the **declared order**), the coefficient of s^α (|α| ≥ 2) gives the
homological equation

    (J − (α·λ) M) a_α = −[N(P)]_α,   N(P) = ½ B(P̃, P̃) + ⅙ C(P̃, P̃, P̃),   P̃ = P − x₀,

which is exact when G is a polynomial of degree ≤ 3 (``degree`` declared and **verified** on a
random line: G(x₀ + h u) must be a polynomial of that degree in h; the measured defect is
reported). A resonance α·λ = λ_k(J, M) makes the left side singular: refused (UnsupportedError)
when σ_min / σ_max < ``resonance_tol``. Complex-conjugate pairs are handled in complex
coordinates (s, s̄); the real manifold is P(s, s̄), whose imaginary part (rounding) is reported.

Validity: the invariance defect E(s) = ‖G(P(s)) − M DP(s) Λ s‖ / ‖G‖-scale is measured on the
circle |s| = ρ (:meth:`UnstableManifold.defect`); :meth:`validity_radius` returns the largest ρ
with defect below a declared tolerance. Globalization: :meth:`fundamental_domain` gives points
on |s| = ρ (every orbit of the manifold crosses it exactly once, because |s(t)|² grows strictly
under ṡ = Λ s with Re λ > 0), and :func:`globalize` integrates them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..errors import ConvergenceError, OptionError, UnsupportedError
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["UnstableManifold", "unstable_manifold", "polynomial_degree_defect", "globalize"]


def polynomial_degree_defect(system, x0, degree: int, *, par=None, h: float = 0.5, seed: int = 0) -> float:
    """Relative (degree+1)-th finite difference of h ↦ G(x₀ + h u) on a random unit direction u
    (zero to rounding iff G is a polynomial of degree ≤ ``degree`` along that line)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(system.n)
    u *= h / np.linalg.norm(u) * max(1.0, np.linalg.norm(x0))
    m = degree + 1
    vals = [system.G(np.asarray(x0) + j * u, par) for j in range(m + 1)]
    diff = sum((-1) ** (m - j) * _binom(m, j) * vals[j] for j in range(m + 1))
    scale = max(max(np.linalg.norm(v) for v in vals), 1e-300)
    return float(np.linalg.norm(diff) / scale)


def _binom(n, k):
    from math import comb
    return comb(n, k)


def _multi_indices(d: int, K: int):
    """All α ∈ N^d with 1 ≤ |α| ≤ K, ordered by |α|."""
    out = []
    for order in range(1, K + 1):
        for a in itertools.product(range(order + 1), repeat=d):
            if sum(a) == order:
                out.append(tuple(a))
    return out


@dataclass
class UnstableManifold:
    """Parametrization P(s) = Σ a_α s^α of the unstable manifold of x₀ (see module doc)."""
    __repr__ = compact_repr   # one line, arrays by shape
    x0: np.ndarray
    eigenvalues: np.ndarray            # λ₁..λ_d
    coeffs: dict                       # α -> a_α (complex n-vectors)
    order: int
    pairs: list                        # [(j, j+1)] complex-conjugate coordinate pairs
    degree_defect: float
    resonance_margin: float            # min σ_min/σ_max over the homological solves
    system: object = field(repr=False)
    par: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.eigenvalues)

    def _s_full(self, s) -> np.ndarray:
        """Complex coordinate vector from real parameters: a pair (j, j+1) takes s_j = σ_j + iσ_{j+1},
        s_{j+1} = conj(s_j); real eigenvalues take σ_j."""
        s = np.asarray(s, float).reshape(-1)
        z = s.astype(complex)
        for j, k in self.pairs:
            z[j], z[k] = s[j] + 1j * s[k], s[j] - 1j * s[k]
        return z

    def __call__(self, s) -> np.ndarray:
        """Real point P(s) for real parameters ``s`` (length d; see ``_s_full``)."""
        z = self._s_full(s)
        out = self.x0.astype(complex).copy()
        for a, c in self.coeffs.items():
            out += np.prod(z ** np.array(a)) * c
        return out.real

    def imag_part(self, s) -> float:
        z = self._s_full(s)
        out = sum(np.prod(z ** np.array(a)) * c for a, c in self.coeffs.items())
        return float(np.linalg.norm(np.imag(out)))

    def tangent(self, s) -> np.ndarray:
        """DP(s) Λ s (the velocity of the conjugate linear flow pushed forward), real."""
        z = self._s_full(s)
        lam = self.eigenvalues
        out = np.zeros(len(self.x0), complex)
        for a, c in self.coeffs.items():
            mono = np.prod(z ** np.array(a))
            out += (np.dot(a, lam)) * mono * c            # s_j ∂_j s^α = α_j s^α
        return out.real

    def defect(self, s) -> float:
        """‖G(P(s)) − M DP(s) Λ s‖ / ‖M DP(s) Λ s‖ at the real parameter point s."""
        x = self(s)
        lhs = self.system.G(x, self.par)
        rhs = self.system.M @ self.tangent(s)
        return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-300))

    def circle(self, rho: float, m: int) -> np.ndarray:
        """m real parameter points on the fundamental loop |s| = ρ (d = 1: the two points ±ρ;
        d = 2: the circle; d > 2: refused)."""
        if self.dim == 1:
            return np.array([[rho], [-rho]])
        if self.dim == 2:
            th = 2 * np.pi * np.arange(m) / m
            return np.stack([rho * np.cos(th), rho * np.sin(th)], axis=1)
        raise UnsupportedError("fundamental domains of manifolds of dimension > 2 are not implemented")

    def max_defect(self, rho: float, m: int) -> float:
        return max(self.defect(s) for s in self.circle(rho, m))

    def validity_radius(self, tol: float, *, m: int, rho_max: float) -> float:
        """Largest ρ ≤ ρ_max (bisection, 40 halvings) with max defect on |s| = ρ below ``tol``."""
        if self.max_defect(rho_max, m) <= tol:
            return float(rho_max)
        lo, hi = 0.0, float(rho_max)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.max_defect(mid, m) <= tol:
                lo = mid
            else:
                hi = mid
        if lo == 0.0:
            raise ConvergenceError("validity_radius: defect above tol at every radius tried")
        return lo

    def fundamental_domain(self, rho: float, m: int) -> np.ndarray:
        """Initial states P(s) on |s| = ρ (each orbit of the manifold crosses once)."""
        return np.array([self(s) for s in self.circle(rho, m)])

    def explain(self) -> str:
        lam = ", ".join(f"{x:.6g}" for x in self.eigenvalues)
        cn = max(np.linalg.norm(c) for a, c in self.coeffs.items() if sum(a) == self.order)
        lines = [f"UnstableManifold: dim {self.dim}, eigenvalues [{lam}], parametrization order "
                 f"{self.order} ({len(self.coeffs)} coefficients, max ‖a_α‖ at top order {cn:.2e}); "
                 f"polynomial-degree defect {self.degree_defect:.1e}; resonance margin "
                 f"{self.resonance_margin:.2e}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def unstable_manifold(system, x0, *, order: int, degree: int, eigvec_scale: float,
                      unstable_tol: float, par=None, dims: int | None = None,
                      resonance_tol: float = 1e-10, degree_tol: float = 1e-8) -> UnstableManifold:
    """Parametrization of the unstable manifold of the equilibrium ``x0`` of ``system``
    (:class:`~conatus.bifurcation.DynamicalSystem`), to the declared ``order``.

    ``degree`` (2 or 3): declared polynomial degree of G (verified, refused above ``degree_tol``);
    ``eigvec_scale``: length of each eigenvector (a free scaling of s, declared: it sets how fast
    the coefficients decay); ``unstable_tol``: eigenvalues with Re λ > unstable_tol are unstable
    (declared); ``dims``: keep only the ``dims`` most unstable (a pair counts 2; the slow
    sub-manifold is then a *strong*-unstable one only if the kept eigenvalues dominate, stated)."""
    from ..bifurcation.equilibria import generalized_eigenvalues
    if degree not in (2, 3):
        raise OptionError("unstable_manifold: degree must be 2 or 3 (exact B, C truncation)")
    x0 = np.asarray(x0, float)
    dd = polynomial_degree_defect(system, x0, degree, par=par)
    if dd > degree_tol:
        raise UnsupportedError(f"unstable_manifold: G is not a polynomial of degree {degree} "
                               f"(finite-difference defect {dd:.1e} > degree_tol {degree_tol:g})")
    J = system.J(x0, par)
    M = system.M.toarray() if hasattr(system.M, "toarray") else np.asarray(system.M, float)
    w, vr, _ = generalized_eigenvalues(J, M, vectors=True)
    unst = [i for i in range(len(w)) if w[i].real > unstable_tol]
    if not unst:
        raise OptionError(f"unstable_manifold: no eigenvalue with Re λ > {unstable_tol:g}")
    # keep complete conjugate pairs, most unstable first
    lam, vecs, pairs = [], [], []
    used = set()
    for i in unst:
        if i in used:
            continue
        if abs(w[i].imag) > 1e-12 * max(1.0, abs(w[i])):
            j = min((k for k in unst if k not in used and k != i),
                    key=lambda k: abs(w[k] - np.conj(w[i])), default=None)
            if j is None or abs(w[j] - np.conj(w[i])) > 1e-8 * max(1.0, abs(w[i])):
                raise ConvergenceError("unstable_manifold: complex eigenvalue without its conjugate")
            if dims is not None and len(lam) + 2 > dims:
                break
            vi = vr[:, i] / np.linalg.norm(vr[:, i])
            vi = vi * np.exp(-1j * np.angle(vi[np.argmax(np.abs(vi))]))   # fix the free phase
            if w[i].imag < 0:
                vi, wi = np.conj(vi), np.conj(w[i])
            else:
                wi = w[i]
            pairs.append((len(lam), len(lam) + 1))
            lam += [wi, np.conj(wi)]
            vecs += [eigvec_scale * vi, eigvec_scale * np.conj(vi)]
            used |= {i, j}
        else:
            if dims is not None and len(lam) + 1 > dims:
                break
            vi = np.real(vr[:, i])
            vi = vi / np.linalg.norm(vi)
            vi = vi * np.sign(vi[np.argmax(np.abs(vi))])
            lam.append(complex(w[i].real))
            vecs.append(eigvec_scale * vi.astype(complex))
            used.add(i)
    lam = np.array(lam)
    d = len(lam)
    coeffs: dict = {}
    for j in range(d):
        a = [0] * d
        a[j] = 1
        coeffs[tuple(a)] = vecs[j]
    margin = np.inf
    idx = _multi_indices(d, order)
    for a in idx:
        if sum(a) < 2:
            continue
        rhs = np.zeros(len(x0), complex)
        # quadratic terms: Σ_{β+γ=α} ½ B(a_β, a_γ)
        for b in idx:
            if sum(b) >= sum(a):
                break
            g = tuple(np.subtract(a, b))
            if min(g) < 0 or g not in coeffs or sum(g) < 1:
                continue
            rhs += 0.5 * system.B(x0, coeffs[b], coeffs[g], par)
        if degree == 3:
            for b in idx:
                if sum(b) > sum(a) - 2:
                    break
                for c in idx:
                    if sum(b) + sum(c) > sum(a) - 1:
                        break
                    g = tuple(np.subtract(np.subtract(a, b), c))
                    if min(g) < 0 or sum(g) < 1 or g not in coeffs:
                        continue
                    rhs += system.C(x0, coeffs[b], coeffs[c], coeffs[g], par) / 6.0
        A = J - np.dot(a, lam) * M
        sv = np.linalg.svd(A, compute_uv=False)
        margin = min(margin, sv[-1] / sv[0])
        if sv[-1] / sv[0] < resonance_tol:
            raise UnsupportedError(f"unstable_manifold: resonance at α = {a} (α·λ = "
                                   f"{np.dot(a, lam):.6g} is an eigenvalue; σ_min/σ_max = "
                                   f"{sv[-1] / sv[0]:.1e} < resonance_tol)",
                                   fix="lower the order or treat the resonant term (normal form)")
        coeffs[a] = np.linalg.solve(A, -rhs)
    man = UnstableManifold(x0, lam, coeffs, int(order), pairs, dd, float(margin), system,
                           None if par is None else dict(par))
    man.notes.append(f"eigenvector length {eigvec_scale:g} (declared); unstable: Re λ > {unstable_tol:g}")
    man.notes.append(f"homological equations exact for a polynomial G of degree {degree} "
                     f"(measured defect {dd:.1e})")
    n_unst = sum(1 for i in unst)
    if d < n_unst:
        man.notes.append(f"{d} of {n_unst} unstable directions kept (dims = {dims}): "
                         "a sub-manifold of the unstable manifold")
    return man


@renames("conatus.manifolds.parametrization:globalize")
def globalize(integrator, points, until: float) -> list:
    """Integrate every initial point (e.g. ``fundamental_domain``) for ``until``: the manifold as
    a family of trajectories (not arclength-adaptive; the sampling is the caller's, declared)."""
    return [integrator.integrate(p, until) for p in points]
