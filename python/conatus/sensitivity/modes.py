"""Direct and adjoint global modes, structural sensitivity and the wavemaker.

A :class:`ModePair` holds, for one eigenvalue σ of a :class:`~conatus.nonmodal.ModalSystem`
discretized by the tau method (descriptor M x' + L x = 0, i.e. (L + σ M) x = 0):

* the **discrete** right and left vectors of the pencil (x, y with yᴴ(L + σM) = 0): they give the
  *exact* first-order change of the discrete eigenvalue, δσ = −yᴴ(δL + σ δM)x / (yᴴ M x);
* the **physical** direct mode u(y) (state fields) and adjoint mode u†(y) for the *declared*
  product ⟨a, b⟩ = Σ_i w_i ∫ ā_i b_i dy (the energy weights of the system, without the ½),
  normalized ⟨u, u⟩ = 1 and ⟨u†, u⟩ = 1. The adjoint is either the discrete adjoint W⁻¹Lᴴ W
  (``adjoint="discrete"``, from the energy-orthonormal generator) or the discretized continuous
  adjoint (``adjoint="continuous"``, from ``ModalSystem.formal_adjoint`` with declared BCs).
  There is no default: the two differ at finite N and converge together (docs/mathematics/
  nonmodal.md).

Structural sensitivity (Giannetti & Luchini 2007). A localized feedback f = g(y) C₀ u added to
dt u = A u + … moves the eigenvalue by

    δσ = ⟨u†, g C₀ u⟩ / ⟨u†, u⟩ = ∫ g(y) Σ_ij w_i ū†_i(y) (C₀)_ij u_j(y) dy,

so the tensor S_ij(y) = w_i ū†_i(y) u_j(y) / ⟨u†, u⟩ is the structural sensitivity and its norm
S(y) = ‖u†(y)‖_w ‖u(y)‖_w / |⟨u†, u⟩|, with ‖v‖_w² = Σ_i w_i |v_i|², bounds |δσ| ≤ |g| ‖C₀‖ S(y):
the *wavemaker*. All integrals are exact Chebyshev-series integrals (see ``_cheb``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import sympy

from ..errors import OptionError, SpectralError
from ..nonmodal.adjoint import continuous_adjoint_mode
from ..nonmodal.statespace import state_space
from ..nonmodal.system import Descriptor, ModalSystem
from . import _cheb
from .._repr import compact_repr

__all__ = ["ModePair", "mode_pair", "StructuralSensitivity", "structural_sensitivity",
           "feedback_system", "ADJOINT_KINDS"]

ADJOINT_KINDS = ("discrete", "continuous")


def _eig_near(A: np.ndarray, target: complex):
    w, V = np.linalg.eig(A)
    i = int(np.argmin(np.abs(w - target)))
    return complex(w[i]), V[:, i]


@dataclass
class ModePair:
    """Eigenvalue σ with discrete (pencil) and physical (declared-product) direct/adjoint modes."""
    __repr__ = compact_repr   # one line, arrays by shape

    system: ModalSystem
    descriptor: Descriptor
    sigma: complex
    x: np.ndarray                  # pencil right vector (full descriptor unknowns)
    y: np.ndarray                  # pencil left vector: yᴴ (L + σ M) = 0
    direct: np.ndarray             # state coefficients of u, ⟨u, u⟩ = 1
    adjoint: np.ndarray            # state coefficients of u†, ⟨u†, u⟩ = 1
    adjoint_kind: str
    adjoint_eigenvalue: complex    # eigenvalue of the adjoint operator used (→ conj σ)
    residual_right: float          # ‖(L + σM) x‖ / ((‖L‖ + |σ|‖M‖) ‖x‖)
    residual_left: float           # same for yᴴ
    residual_adjoint: float        # ‖Ã† q − λ† q‖ / (‖Ã†‖ ‖q‖) of the physical adjoint
    kappa: float                   # ‖u‖ ‖u†‖ / |⟨u†, u⟩| (eigenvalue condition number)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ products
    @property
    def bounds(self) -> tuple[float, float]:
        return self.system.bounds

    def inner(self, a: np.ndarray, b: np.ndarray) -> complex:
        """⟨a, b⟩ = Σ_i w_i ∫ ā_i b_i dy = 2 aᴴ W b (W: the exact energy Gram, with its ½)."""
        return complex(2.0 * (a.conj() @ self.descriptor.W @ b))

    def fields(self, which: str) -> dict[str, np.polynomial.Chebyshev]:
        """Chebyshev series of each state field of the 'direct' or 'adjoint' mode."""
        v = {"direct": self.direct, "adjoint": self.adjoint}.get(which)
        if v is None:
            raise OptionError(f"which='{which}': expected 'direct' or 'adjoint'")
        N = self.descriptor.N
        return {n: _cheb.series(v[k * N:(k + 1) * N], self.bounds)
                for k, n in enumerate(self.system.state)}

    def evaluate(self, which: str, yg) -> dict[str, np.ndarray]:
        yg = np.asarray(yg, float)
        return {n: s(yg) for n, s in self.fields(which).items()}

    # ------------------------------------------------------------------ discrete drift
    def shift(self, dL: np.ndarray, dM: np.ndarray | None = None) -> complex:
        """Exact first-order change of the *discrete* eigenvalue for L → L + δL, M → M + δM."""
        num = self.y.conj() @ (dL @ self.x)
        if dM is not None:
            num = num + self.sigma * (self.y.conj() @ (dM @ self.x))
        return complex(-num / (self.y.conj() @ (self.descriptor.M @ self.x)))

    def shift_towards(self, other: ModalSystem) -> complex:
        """First-order eigenvalue change for the table change ``other − system`` (the tau
        descriptor is linear in the table coefficients, so δL = L(other) − L(system) exactly)."""
        d2 = other.discretize(self.descriptor.N)
        if d2.M.shape != self.descriptor.M.shape:
            raise OptionError("the other system has a different descriptor layout (fields, "
                              "multipliers or boundary conditions changed)")
        return self.shift(d2.L - self.descriptor.L, d2.M - self.descriptor.M)

    def explain(self) -> str:
        return (f"ModePair of '{self.system.name}' (Chebyshev N = {self.descriptor.N}): "
                f"σ = {self.sigma:.15g}; pencil residuals right {self.residual_right:.1e}, left "
                f"{self.residual_left:.1e}; physical adjoint = {self.adjoint_kind} adjoint "
                f"(eigenvalue {self.adjoint_eigenvalue:.15g}, residual "
                f"{self.residual_adjoint:.1e}); product ⟨a,b⟩ = Σ w_i ∫ ā_i b_i dy, w = "
                f"{self.system.weights}; normalization ⟨u,u⟩ = 1, ⟨u†,u⟩ = 1; κ = "
                f"{self.kappa:.6g}. " + " ".join(self.notes))


def mode_pair(system: ModalSystem, *, N: int, target: complex, adjoint: str,
              infinite_tol: float, adjoint_bcs=None) -> ModePair:
    """Eigenpair of ``system`` nearest ``target`` with its adjoint (kind declared, see module doc).

    ``adjoint="continuous"`` needs the adjoint boundary conditions ``adjoint_bcs`` (as in
    ``ModalSystem.formal_adjoint``; they are checked through the Green concomitant).
    """
    if adjoint not in ADJOINT_KINDS:
        raise OptionError(f"adjoint='{adjoint}': expected one of {ADJOINT_KINDS} (no default: "
                          "the discrete and continuous adjoints differ at finite N)")
    if adjoint == "continuous" and adjoint_bcs is None:
        raise OptionError("adjoint='continuous' needs adjoint_bcs (declared, then checked)")
    desc = system.discretize(N)
    L, M = desc.L, desc.M
    (a, b), VL, VR = sla.eig(-L, M, left=True, right=True, homogeneous_eigvals=True)
    scale = np.linalg.norm(L, 1) / np.linalg.norm(M, 1)
    finite = np.abs(b) * scale > infinite_tol * np.abs(a)
    lam = np.full(a.shape, np.inf + 0j)
    lam[finite] = a[finite] / b[finite]
    i = int(np.argmin(np.where(finite, np.abs(lam - target), np.inf)))
    sigma = complex(lam[i])
    x, y = VR[:, i], VL[:, i]
    nrm = np.linalg.norm(L, 2) + abs(sigma) * np.linalg.norm(M, 2)
    rr = float(np.linalg.norm(L @ x + sigma * (M @ x)) / (nrm * np.linalg.norm(x)))
    rl = float(np.linalg.norm(y.conj() @ L + sigma * (y.conj() @ M)) / (nrm * np.linalg.norm(y)))
    # nearest other finite eigenvalue: an ill-separated pair makes the choice ambiguous
    others = np.abs(lam[finite] - sigma)
    gap = float(np.sort(others)[1]) if others.size > 1 else np.inf
    if gap <= 1e-8 * max(1.0, abs(sigma)):
        raise SpectralError(f"eigenvalue {sigma:.6g} is (nearly) multiple (gap {gap:.1e}): "
                            "first-order sensitivities are not defined for it")

    ss = state_space(desc, infinite_tol=infinite_tol)
    if adjoint == "discrete":
        Ah = ss.A.conj().T
        lam_a, q = _eig_near(Ah, np.conj(sigma))
        res_a = float(np.linalg.norm(Ah @ q - lam_a * q) / (np.linalg.norm(Ah, 2) * np.linalg.norm(q)))
        ua = ss.to_state @ q
    else:
        adj = system.formal_adjoint(adjoint_bcs)
        lam_a, ua, ssa = continuous_adjoint_mode(adj, N, sigma, infinite_tol=infinite_tol)
        qa = np.linalg.lstsq(ssa.to_state, ua, rcond=None)[0]
        res_a = float(np.linalg.norm(ssa.A @ qa - lam_a * qa)
                      / (np.linalg.norm(ssa.A, 2) * np.linalg.norm(qa)))
    u = x[desc.cols]
    pair = ModePair(system, desc, sigma, x, y, u, ua, adjoint, complex(lam_a), rr, rl, res_a, 0.0)
    u = u / np.sqrt(pair.inner(u, u).real)
    ip = pair.inner(ua, u)
    if abs(ip) <= 1e-14 * np.sqrt(pair.inner(ua, ua).real):
        raise SpectralError("⟨u†, u⟩ = 0 to rounding: the adjoint mode does not match the direct "
                            "one (defective eigenvalue, or wrong adjoint boundary conditions)")
    ua = ua / np.conj(ip)
    pair.direct, pair.adjoint = u, ua
    pair.kappa = float(np.sqrt(pair.inner(ua, ua).real))
    pair.notes.append(f"nearest other finite eigenvalue at distance {gap:.3e}")
    return pair


# ---------------------------------------------------------------------- structural sensitivity
@dataclass
class StructuralSensitivity:
    """Giannetti–Luchini structural sensitivity of one mode (see module doc)."""

    pair: ModePair

    def tensor(self, yg) -> np.ndarray:
        """S_ij(y) = w_i ū†_i(y) u_j(y) (⟨u†, u⟩ = 1), shape (len(yg), nfields, nfields)."""
        a = self.pair.evaluate("adjoint", yg)
        u = self.pair.evaluate("direct", yg)
        st, w = self.pair.system.state, self.pair.system.weights
        return np.stack([np.stack([w[i] * np.conj(a[i]) * u[j] for j in st], -1) for i in st], -2)

    def wavemaker(self, yg) -> np.ndarray:
        """S(y) = ‖u†(y)‖_w ‖u(y)‖_w / |⟨u†, u⟩| (weighted local norms, see module doc)."""
        a = self.pair.evaluate("adjoint", yg)
        u = self.pair.evaluate("direct", yg)
        w = self.pair.system.weights
        na = np.sqrt(sum(w[n] * np.abs(a[n]) ** 2 for n in a))
        nu = np.sqrt(sum(w[n] * np.abs(u[n]) ** 2 for n in u))
        return na * nu

    def maximum(self, *, points: int) -> tuple[float, float]:
        """(y*, S(y*)): location and value of the wavemaker maximum, sampled on ``points`` points
        then refined by a bounded Brent search around the best sample."""
        import scipy.optimize as so
        a, b = self.pair.bounds
        yg = np.linspace(a, b, points)
        s = self.wavemaker(yg)
        k = int(np.argmax(s))
        lo, hi = yg[max(k - 1, 0)], yg[min(k + 1, points - 1)]
        r = so.minimize_scalar(lambda v: -self.wavemaker(np.array([v]))[0], bounds=(lo, hi),
                               method="bounded", options={"xatol": 1e-12 * (b - a)})
        return (float(r.x), float(-r.fun)) if -r.fun >= s[k] else (float(yg[k]), float(s[k]))

    def feedback_shift(self, g, C0) -> tuple[complex, dict]:
        """δσ = ∫ g(y) Σ_ij w_i ū†_i (C₀)_ij u_j dy for the feedback f = g(y) C₀ u (first order).

        ``g``: sympy expression in Y, number or callable; ``C0``: {(i, j): complex} over state
        fields. Exact Chebyshev integral; returns (δσ, {"g_degree": adaptive degree of g})."""
        p = self.pair
        gs, deg = _cheb.adaptive(g, p.bounds, what="feedback weight g(y)")
        a, u, w = p.fields("adjoint"), p.fields("direct"), p.system.weights
        h = _cheb.series([0.0], p.bounds)
        for (i, j), c in C0.items():
            if i not in a or j not in u:
                raise OptionError(f"C0 key {(i, j)}: both must be state fields {p.system.state}")
            h = h + complex(c) * w[i] * _cheb.conj(a[i]) * u[j]
        return _cheb.integral(gs * h), {"g_degree": deg}

    def explain(self) -> str:
        return ("Structural sensitivity (Giannetti & Luchini 2007): tensor S_ij(y) = w_i ū†_i u_j, "
                "wavemaker S(y) = ‖u†(y)‖_w ‖u(y)‖_w with ⟨u†, u⟩ = 1; first-order drift of a "
                "localized feedback g(y) C₀ u is ∫ g Σ S_ij C₀_ij dy (exact Chebyshev integral). "
                + self.pair.explain())


def structural_sensitivity(pair: ModePair) -> StructuralSensitivity:
    """Structural sensitivity and wavemaker of the mode in ``pair``."""
    return StructuralSensitivity(pair)


def feedback_system(system: ModalSystem, g, C0: dict, *, epsilon: float) -> ModalSystem:
    """``system`` with the feedback ε g(y) C₀ u added to the state equations (for tests and
    finite-difference checks of :meth:`StructuralSensitivity.feedback_shift`)."""
    from ..nonmodal.system import _add
    A = {k: dict(v) for k, v in system.A.items()}
    gs = sympy.sympify(g)
    for (i, j), c in C0.items():
        A[(i, j)] = _add(A.get((i, j), {}), {0: sympy.nsimplify(epsilon) * sympy.nsimplify(c) * gs})
    return ModalSystem(system.name + f" + {epsilon:g} g C0 feedback", system.bounds, system.state,
                       dict(system.weights), system.multipliers, A, dict(system.G), dict(system.C),
                       system.bcs, list(system.notes) + [f"feedback ε g C0, ε = {epsilon:g}"])
