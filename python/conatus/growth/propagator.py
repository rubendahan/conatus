"""Finite-time propagator Φ(t0, T) of M X' + L0 X = −B(t) X, and the energy gain. :mod:`conatus.growth.periodic` builds the general case from any linear IR problem).

Φ uses the same IMEX Runge–Kutta tableaux as Dedalus and conatus.timestep (``rk_tableau``): L0
implicit, the base-flow terms B(t) explicit and evaluated at the exact stage times. All ARS schemes
have equal H[i,i], so one factor of (M + dt H_ii L0) serves the whole run (a classic pitfall
of IMEX propagators). Φ is the ordered product of the one-step maps, assembled from the harmonics of the
step and multiplied in batches (no Python loop over time steps). Works with NumPy or CuPy
arrays (``xp``): on GPU the whole run is batched dense GEMMs on the device.

Gain: G = σ_max²(F Φ F⁻¹), F = W^{1/2} C the factor of the declared energy norm
(½∫(|u|²+|v|²+|w|² + w_θ|θ|²) dy). Singular values only, never eigenvalues of Φ (cond ~ 1e23).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._deprecation import renames
from ..reference import transforms as rt
from ..timestep.integrator import rk_tableau
from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["propagator", "step_harmonics", "energy_factor", "gain", "GainResult"]


def _xp(backend: str):
    if backend == "cpu":
        return np
    if backend == "cuda":
        import cupy
        return cupy
    raise OptionError("backend must be 'cpu' or 'cuda' (explicit choice)",
                      fix="backend='cpu' or backend='cuda'")


# Memory for one batch of one-step maps (complex n x n each); only sets how many steps are
# multiplied per batched call, never the arithmetic (same products, grouped differently).
_BATCH_BYTES = 128 * 2 ** 20


def _tableau(scheme: str):
    """(A, H, c, H_ii) of an IMEX RK scheme usable for a state-only propagator.

    Two structural requirements, checked (never assumed): equal implicit diagonal H_ii (one
    factor of M + dt H_ii L0 serves every stage), and H[:, 0] = 0 (the initial state enters
    only through M X0 and the explicit F(X0), so pressure, taus and gauges at t0 do not
    matter: the map acts on the physical state alone). The ARS schemes RK111, RK222 and RK443
    satisfy both; RKSMR and RKGFY (Crank-Nicolson-like, L X0 used) are refused, as are the
    multistep schemes (no tableau)."""
    A, H, c = (np.asarray(m, dtype=float) for m in rk_tableau(scheme))
    hii = H[1, 1]
    if not np.all(np.diag(H)[1:] == hii):          # exact constants of the tableau
        raise OptionError(f"{scheme}: unequal H[i,i], one factor per stage is not implemented "
                          "(use RK111, RK222 or RK443)",
                          fix="an IMEX Runge-Kutta scheme with equal diagonal implicit coefficients (RK222, RK443)")
    if np.any(H[:, 0] != 0):
        raise OptionError(f"{scheme}: the implicit part uses L X0 (H[:, 0] != 0), so the map would "
                          "depend on the pressure and taus at t0, which are not part of the "
                          "state; use RK111, RK222 or RK443",
                          fix="an IMEX Runge-Kutta scheme whose implicit part does not use L X0 (RK222, RK443)")
    return A, H, c, hii


def _step_map(xp, Ainv, M, L0, Bs, A, H, c, dt: float, t: float, period: float):
    """One IMEX RK step at time t applied to the identity: the n x n step map S(t).

    Same stage arithmetic as conatus.timestep (L0 implicit, -B(t) explicit at the exact stage
    times t + c_j dt, one factor Ainv = (M + dt H_ii L0)^-1), with
    B(t) = sum_m e^{2 pi i m t / period} B_m (``Bs`` = [(m, B_m), ...])."""
    s = len(c) - 1
    X = xp.eye(M.shape[0], dtype=complex)
    MX0 = M @ X
    Xs, Fs, LXs = [X], [], []
    for i in range(1, s + 1):
        tj = t + c[i - 1] * dt                              # F of the previous stage
        F = None
        for m, Bm in Bs:
            ph = 2 * math.pi * m * tj / period
            term = complex(math.cos(ph), math.sin(ph)) * (Bm @ Xs[-1])
            F = term if F is None else F + term
        Fs.append(-F if F is not None else xp.zeros_like(X))
        LXs.append(L0 @ Xs[-1] if H[i:, i - 1].any() else None)   # L X_0 unused (H[:,0] = 0)
        rhs = MX0.copy()
        for j in range(i):
            if A[i, j]:
                rhs += (dt * A[i, j]) * Fs[j]
            if H[i, j] and LXs[j] is not None:
                rhs -= (dt * H[i, j]) * LXs[j]
        Xs.append(Ainv @ rhs)
    return Xs[-1]


def _harmonics_of(M, L0, B: dict, dt: float, scheme: str, xp, period: float):
    """(S_k for k = -K..K stacked, residual): S(t) = sum_k e^{2 pi i k t / period} S_k.

    Stage i applies B(t) once to stage i-1, so with B of harmonic half-width Mh the step map is
    a trigonometric polynomial of degree K = s Mh (s stages). It is sampled at Q = 2K + 3
    phases t = q period / Q and transformed (DFT); the two extra harmonics |k| = K + 1 must
    vanish: their size relative to the largest S_k is returned as the residual (checked,
    never assumed)."""
    A, H, c, hii = _tableau(scheme)
    s = len(c) - 1
    K = s * max((abs(m) for m in B), default=0)
    Ainv = xp.asarray(sla.inv(M + dt * hii * L0))
    Md, L0d = xp.asarray(M, dtype=complex), xp.asarray(L0, dtype=complex)
    Bs = [(m, xp.asarray(Bm, dtype=complex)) for m, Bm in sorted(B.items(), reverse=True)]
    Q = 2 * K + 3
    samples = xp.stack([_step_map(xp, Ainv, Md, L0d, Bs, A, H, c, dt, q * period / Q, period)
                        for q in range(Q)])
    Shat = xp.fft.fft(samples, axis=0) / Q                 # index k mod Q
    Sk = Shat[xp.asarray([k % Q for k in range(-K, K + 1)])]
    extra = Shat[xp.asarray([(K + 1) % Q, (-K - 1) % Q])]
    scale = float(xp.abs(Sk).max())
    resid = float(xp.abs(extra).max()) / scale if scale else 0.0
    return Sk, resid


def _base_harmonics(op) -> dict:
    """Base-flow harmonics of an operator: ``op.harmonics`` (dict m -> B_m) for the generic
    periodic operator."""
    B = getattr(op, "harmonics", None)
    return dict(B) if B is not None else {1: op.Bp, -1: op.Bm}


def step_harmonics(op: object, dt: float, scheme: str,
                   backend: str = "cpu") -> tuple[np.ndarray, float]:
    """Harmonics S_k (k = -K..K) of the one-step map, and the residual of the degree check
    (size of the harmonics K + 1, relative). Returned on host."""
    xp = _xp(backend)
    Sm, resid = _harmonics_of(op.M, op.L0, _base_harmonics(op), float(dt), scheme, xp,
                              float(getattr(op, "period", 1.0)))
    return (Sm.get() if backend == "cuda" else np.asarray(Sm)), resid


def _ordered_product(xp, S):
    """S[-1] @ ... @ S[1] @ S[0] (steps in time order) by pairwise batched products: log2(len)
    batched matmuls, no loop over steps."""
    while S.shape[0] > 1:
        tail = S[-1:] if S.shape[0] % 2 else None           # latest step, multiplied last
        m = S.shape[0] - (1 if tail is not None else 0)
        S = S[1:m:2] @ S[0:m:2]
        if tail is not None:
            S = xp.concatenate([S, tail])
    return S[0]


@renames("conatus.growth.propagator:propagator")
def propagator(op: object, t0: float, T: float, steps: int, scheme: str,
               backend: str = "cpu") -> np.ndarray:
    """Φ mapping state coefficients at t0 to state coefficients at t0 + T (returned on host).

    ``period`` and ``harmonics`` (dict m -> B_m, B(t) = sum_m e^{2 pi i m t / period} B_m), such
    as :class:`conatus.growth.periodic.PeriodicOperator`.

    The step is dt = T / steps exactly, so the output time is never rounded (a rounded
    or zero horizon gives wrong gains without any error). T > 0 is required.

    Same scheme and arithmetic class as stepping the unit states one step at a time (the
    step-by-step reference is kept as a test oracle, equal to round-off):
    Φ is the ordered product of the one-step maps S(t_k), t_k = t0 + k dt. Each S(t_k) is
    assembled from the harmonics of the step (``step_harmonics``, degree checked to round-off),
    then the maps are multiplied in batches by pairwise batched products. No Python loop over
    time steps: Python only loops over batches of steps and over log2(batch) levels.
    """
    if not (T > 0 and int(steps) == steps and steps > 0):
        raise OptionError(f"need T > 0 and a positive integer steps (got T={T}, steps={steps})",
                          fix="T > 0 and steps a positive integer, e.g. steps=4000")
    steps = int(steps)
    dt = T / steps
    xp = _xp(backend)
    period = float(getattr(op, "period", 1.0))
    Sm, resid = _harmonics_of(op.M, op.L0, _base_harmonics(op), dt, scheme, xp, period)
    if not resid < 1e-10:
        raise RuntimeError(f"step map of {scheme} is not a trigonometric polynomial of degree "
                           f"<= stages x max|m| in e^(2 pi i t / period) (residual {resid:.1e}); "
                           "B(t) must be sum_m e^(2 pi i m t / period) B_m")
    K = (Sm.shape[0] - 1) // 2
    n = op.M.shape[0]
    cols = xp.asarray(op.state_cols)
    batch = max(1, _BATCH_BYTES // (16 * n * n))
    ks = np.arange(-K, K + 1)
    Sm2 = Sm.reshape(2 * K + 1, n * n)
    Phi = None
    for k0 in range(0, steps, batch):                      # batches of steps, not steps
        k = np.arange(k0, min(k0 + batch, steps))
        t = t0 + k * dt                                     # same stage clock as the old loop
        z = np.exp(2j * np.pi * np.outer(t, ks) / period)   # e^{2 pi i k t_j / period}
        S = (xp.asarray(z) @ Sm2).reshape(len(k), n, n)     # S(t_j) = sum_k z^k S_k (one GEMM)
        P = _ordered_product(xp, S)
        Phi = P[:, cols] if Phi is None else P @ Phi        # unit states at t0 = columns cols
    Phi = Phi[cols, :]
    return Phi.get() if backend == "cuda" else np.asarray(Phi)


def energy_factor(op: object, theta_weight: float | None = None) -> np.ndarray:
    """F with ‖F x‖² = ½ Σ_comp w_comp ∫|comp|² dy (Fejér quadrature on the Chebyshev grid).

    ``theta_weight`` (w_θ, 1 or N²) is required when θ is in the state; it enters as √w_θ.
    """
    b, N = op.basis, op.N
    I = np.eye(N)
    C = rt.backward(I, b, axis=0, dealias=False)                     # coeff -> grid
    Cinv = rt.forward(I, b, axis=0, dealias=False)                   # grid -> coeff
    n = np.arange(N)
    intT = np.where(n % 2 == 0, 2.0 / (1.0 - n.astype(float) ** 2 + (n == 1)), 0.0)
    w = (op.Ly / 2) * (intT @ Cinv)                                  # ∫ of each grid delta
    if np.any(w <= 0):
        raise RuntimeError("non-positive quadrature weight on the Chebyshev grid")
    Fc = np.sqrt(0.5 * w)[:, None] * C
    blocks = []
    for comp in op.state:
        if comp == "b":
            if theta_weight is None or theta_weight <= 0:
                raise OptionError("theta is in the state: give theta_weight > 0 (1 or N^2)",
                                  fix="theta_weight=1.0 (or N**2)")
            blocks.append(math.sqrt(theta_weight) * Fc)
        else:
            blocks.append(Fc)
    return sla.block_diag(*blocks)


@dataclass
class GainResult:
    """Optimal gain G = sigma_1^2 over the admissible inputs, the singular values and the
    optimal initial state / response (coefficients)."""
    __repr__ = compact_repr   # one line, arrays by shape

    G: float
    sigmas: np.ndarray
    v_opt: np.ndarray          # optimal initial state (coefficients)
    u_resp: np.ndarray         # response Φ v_opt at t0 + T (coefficients; energy G)

    def explain(self) -> str:
        """The gain, the next singular values and what the structures are."""
        nxt = ", ".join(f"{s ** 2:.6g}" for s in np.asarray(self.sigmas)[1:4])
        return (f"optimal gain G = sigma_1^2 = {self.G:.10g} over the admissible inputs (next "
                f"squared singular values {nxt or 'none'}); v_opt: optimal initial state (unit "
                f"energy), u_resp: its response Φ v_opt (energy G) ({np.size(self.v_opt)} coefficients each)")


def gain(Phi: np.ndarray, F: np.ndarray, Z: np.ndarray | None) -> GainResult:
    """G = max ‖F Φ x‖² / ‖F x‖² over x in range(Z) (SVD), and the optimal structures.

    ``Z`` = ``op.admissible_basis()``: initial states that satisfy continuity and the boundary
    conditions. ``Z=None`` maximizes over all free coefficients (the convention of some
    published codes);
    those inputs are not admissible, the first step projects them in a dt-dependent way and the
    gain then converges only at order ≈ 1 in dt (measured).
    The choice is explicit: there is no default.
    """
    if Z is None:
        Z = np.eye(Phi.shape[1])
    Q, R = np.linalg.qr(F @ Z)                        # energy-orthonormal input basis F Z R⁻¹
    Rinv = np.linalg.inv(R)
    A = F @ Phi @ Z @ Rinv
    try:
        U, s, Vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        U, s, Vh = sla.svd(A, lapack_driver="gesvd")
    v_opt = Z @ Rinv @ Vh[0].conj()                  # unit energy: ‖F v_opt‖ = 1
    # the response is the propagated optimal state Φ v_opt (energy σ_1² = G), not a unit-energy
    # vector along it (finding 93: F⁻¹ u_1 was returned, normalised)
    return GainResult(float(s[0] ** 2), s, v_opt, Phi @ v_opt)
