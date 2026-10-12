"""Dynamic mode decomposition in a declared inner product (data-driven toolkit).

Pairs (x_j, y_j = x_{j+1}) with time step Δt (declared), product ⟨a, b⟩ = aᴴ W b, W = Fᴴ F.

**Exact DMD** (Schmid 2010; Tu, Rowley, Luchtenburg, Brunton & Kutz 2014), in the F-coordinates:

    F X = U Σ Vᴴ (rank r, declared),  Ã = Uᴴ (F Y) V Σ⁻¹,  Ã w_k = μ_k w_k,
    φ_k = F⁻¹ (F Y) V Σ⁻¹ w_k / μ_k   (exact modes; unit W-norm after scaling),
    ω_k = log(μ_k)/Δt  (principal branch: |Im ω| < π/Δt, aliasing declared).

**Residual** of each pair (ResDMD with the linear dictionary, Colbrook & Townsend 2023): with
g_k = V Σ⁻¹ w_k, res_k = ‖F (Y − μ_k X) g_k‖ / ‖F X g_k‖ — the defect of x = X g_k as an eigenvector
of any linear map sending the x_j to the y_j; ≈ rounding for data from a linear system of rank ≤ r.

**Amplitudes** (declared): ``"first"``: b = argmin ‖F(x_0 − Φ b)‖; ``"optimal"``: b minimizing
‖F(X − Φ diag(b) V_μ)‖_F over all snapshots (Jovanović, Schmid & Nichols 2014), V_μ the
Vandermonde matrix μ_k^j.

**Optimized DMD** (``method="optimized"``, Askham & Kutz 2018): variable projection on
min_{ω, B} ‖F X_all − (F Φ B)(e^{ω t_j})‖_F over all snapshots (times may be non-uniform),
started from exact DMD, solved by ``scipy.optimize.least_squares`` in the rank-r POD subspace of
F X_all with declared ``tol`` and ``max_nfev``; reports the final relative misfit. Bagging
(BOP-DMD, Sashidhar & Kutz 2022) is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ConvergenceError, OptionError
from .norms import DataInnerProduct, array_module, to_host
from .._repr import compact_repr

__all__ = ["DMD", "dmd", "DMD_METHODS", "AMPLITUDES"]

DMD_METHODS = ("exact", "optimized")
AMPLITUDES = ("first", "optimal")

#: sigma_r / sigma_1 below which the requested rank is noted as above the numerical rank of the
#: data (note only: the result is unchanged; declared in ``conatus.tolerances.DECLARED``).
RANK_NOTE_RTOL = 1e-12
#: |mu| below which a DMD eigenvalue is treated as zero: the exact modes Y G / mu are undefined
#: and the call is refused with the count (declared in ``conatus.tolerances.DECLARED``).
ZERO_EIGENVALUE_TOL = 1e-14


@dataclass
class DMD:
    __repr__ = compact_repr   # one line, arrays by shape
    eigenvalues: np.ndarray         # μ_k (discrete time)
    omega: np.ndarray               # ω_k = log μ_k / Δt (growth rate + i angular frequency)
    modes: Any                      # n × r, unit W-norm columns (device of the data)
    amplitudes: np.ndarray          # b_k
    residuals: np.ndarray           # res_k (see module doc); NaN for optimized DMD
    dt: float
    rank: int
    method: str
    amplitude_policy: str
    inner_product: DataInnerProduct
    singular_values: np.ndarray
    misfit: float = float("nan")    # ‖F(X − reconstruction)‖_F / ‖F X‖_F over all snapshots
    notes: list[str] = field(default_factory=list)

    @property
    def frequencies(self) -> np.ndarray:
        """Im ω / 2π (cycles per unit time)."""
        return self.omega.imag / (2 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        return self.omega.real

    def reconstruct(self, times) -> Any:
        """Σ_k b_k φ_k e^{ω_k t} at ``times`` (n × len(times))."""
        xp = array_module(self.modes)
        t = np.asarray(times, float)
        E = np.exp(np.outer(self.omega, t)) * self.amplitudes[:, None]
        return self.modes @ xp.asarray(E)

    def explain(self) -> str:
        o = np.argsort(-np.abs(self.amplitudes))[:6]
        rows = "; ".join(f"ω = {self.omega[k]:.10g} (|b| {abs(self.amplitudes[k]):.3g}, res "
                         f"{self.residuals[k]:.1e})" for k in o)
        dflt = " (default: least-squares fit of x_0)" if self.amplitude_policy == "first" else ""
        return (f"DMD ({self.method}, rank {self.rank}, Δt = {self.dt:g}, amplitudes "
                f"'{self.amplitude_policy}'{dflt}, {self.inner_product.kind} product): {rows}; "
                f"misfit {self.misfit:.1e}; principal log (|Im ω| < π/Δt = {np.pi / self.dt:.6g}). "
                + " ".join(self.notes))


def _optimal_amplitudes(FPhi, FX, mu) -> np.ndarray:
    """argmin_b ‖F X − F Φ diag(b) V_μ‖_F (Jovanović et al. 2014, closed form)."""
    m = FX.shape[1]
    Vand = mu[:, None] ** np.arange(m)[None, :]
    G = to_host(FPhi.conj().T @ FPhi)
    P = G * (Vand.conj() @ Vand.T)                   # P_jk = ⟨φ_j, φ_k⟩ Σ_t conj(μ_j^t) μ_k^t
    q = (to_host(FPhi.conj().T @ FX) * Vand.conj()).sum(axis=1)
    return np.linalg.solve(P, q)


def dmd(X, *, dt: float, rank: int, inner_product: DataInnerProduct, amplitudes: str = "first",
        method: str = "exact", Y=None, times=None, tol: float | None = None,
        max_nfev: int | None = None) -> DMD:
    """DMD of the sequence X = [x_0 … x_m] (pairs x_j → x_{j+1}), or of pairs (X, Y).

    ``amplitudes`` (how the amplitudes b are fitted; the modes and eigenvalues do not depend on
    it): declared default ``"first"``, the standard least-squares fit of the first snapshot
    b = argmin ‖F(x_0 − Φ b)‖ (Schmid 2010), valid for sequences and pairs; ``"optimal"`` fits
    every snapshot of a sequence (Jovanović et al. 2014). The choice is printed by ``explain()``.

    ``method="optimized"`` needs ``tol`` and ``max_nfev`` (declared), uses the whole sequence
    (``Y`` not allowed) and optionally non-uniform ``times`` (default: j·Δt, declared by ``dt``).
    """
    if method not in DMD_METHODS:
        raise OptionError(f"method='{method}': expected one of {DMD_METHODS}")
    if amplitudes not in AMPLITUDES:
        raise OptionError(f"amplitudes='{amplitudes}': expected one of {AMPLITUDES}")
    if not dt > 0:
        raise OptionError("dt must be > 0 (declared sampling interval)")
    xp = array_module(X)
    Xall = X
    pairs = Y is not None
    if not pairs:
        X, Y = Xall[:, :-1], Xall[:, 1:]
    elif method == "optimized":
        raise OptionError("optimized DMD fits a sequence: pass X only (no Y)")
    elif amplitudes == "optimal":
        # 'optimal' fits x_j = Φ diag(b) μ^j over a sequence; the columns of X given with Y are
        # pairs, not a sequence (they used to be taken as x_0 ... x_{m-1} without a word)
        raise OptionError("amplitudes='optimal' fits a snapshot sequence: with pairs (X, Y) use "
                          "amplitudes='first' (fit of X[:, 0]), or pass the sequence as X alone")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(X.shape):
        raise OptionError(f"rank = {rank}: expected an integer 1 <= rank <= {min(X.shape)}")
    FX, FY = inner_product.half(X), inner_product.half(Y)
    U, s, Vh = xp.linalg.svd(FX, full_matrices=False)
    s_all = to_host(s)
    U, s, V = U[:, :rank], s_all[:rank], Vh[:rank].conj().T
    VS = V * xp.asarray(1.0 / s)[None, :]
    At = to_host(U.conj().T @ (FY @ VS))
    mu, Wv = np.linalg.eig(At)
    notes = []
    if s[-1] / s[0] < RANK_NOTE_RTOL:
        notes.append(f"σ_r/σ_1 = {s[-1] / s[0]:.1e}: rank above the numerical rank of the data")
    G = VS @ xp.asarray(Wv)                          # g_k (m × r)
    small = np.abs(mu) < ZERO_EIGENVALUE_TOL
    if small.any():
        raise OptionError(f"{int(small.sum())} DMD eigenvalue(s) μ ≈ 0: exact modes undefined "
                          "(reduce the rank)")
    FPhi = (FY @ G) / xp.asarray(mu)[None, :]
    nrm = to_host(xp.linalg.norm(FPhi, axis=0))
    FPhi = FPhi / xp.asarray(nrm)[None, :]
    # ResDMD residual with the linear dictionary
    num = to_host(xp.linalg.norm(FY @ G - (FX @ G) * xp.asarray(mu)[None, :], axis=0))
    den = to_host(xp.linalg.norm(FX @ G, axis=0))
    res = num / den
    omega = np.log(mu.astype(complex)) / dt
    if amplitudes == "first":
        b = to_host(xp.linalg.lstsq(FPhi, FX[:, 0], rcond=None)[0])
    else:
        FXall = inner_product.half(Xall)
        b = _optimal_amplitudes(FPhi, FXall, mu)
    out = DMD(mu, omega, inner_product.unhalf(FPhi), b, res, float(dt), rank, "exact",
              amplitudes, inner_product, s_all, notes=notes)
    if pairs:                   # no sequence to reconstruct: the misfit is not defined (NaN)
        out.notes.append("pairs (X, Y): no snapshot sequence, misfit not defined (NaN); the "
                         "residuals res_k measure each pair")
        return out
    FXall = inner_product.half(Xall)
    tt = np.arange(Xall.shape[1]) * dt
    out.misfit = float(to_host(xp.linalg.norm(FXall - inner_product.half(out.reconstruct(tt))))
                       / to_host(xp.linalg.norm(FXall)))
    if method == "exact":
        return out
    return _optimized(Xall, out, inner_product, times, tol, max_nfev)


def _optimized(Xall, ex: DMD, ip: DataInnerProduct, times, tol, max_nfev) -> DMD:
    """Variable-projection optimized DMD (Askham & Kutz 2018), started from exact DMD."""
    import scipy.optimize as so
    if tol is None or max_nfev is None:
        raise OptionError("optimized DMD needs tol and max_nfev (declared stopping criteria)")
    xp = array_module(Xall)
    FXall = ip.half(Xall)
    t = np.arange(Xall.shape[1]) * ex.dt if times is None else np.asarray(times, float)
    if t.size != Xall.shape[1]:
        raise OptionError("times must have one entry per snapshot")
    U, s, Vh = xp.linalg.svd(FXall, full_matrices=False)
    r = ex.rank
    Ur = U[:, :r]
    Z = to_host(Ur.conj().T @ FXall)                 # r × m: data in its POD subspace
    lost = float(np.sqrt(max(to_host(xp.linalg.norm(FXall)) ** 2 - np.linalg.norm(Z) ** 2, 0.0)))

    def fit(theta):
        om = theta[:r] + 1j * theta[r:]
        E = np.exp(np.outer(t, om))                  # m × r
        B, *_ = np.linalg.lstsq(E, Z.T, rcond=None)  # r × r (rows: modes)
        return om, B, E

    def resid(theta):
        _, B, E = fit(theta)
        R = (E @ B - Z.T).ravel()
        return np.concatenate([R.real, R.imag])

    th0 = np.concatenate([ex.omega.real, ex.omega.imag])
    sol = so.least_squares(resid, th0, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev,
                           method="lm" if 2 * Z.size >= 2 * r else "trf")
    if sol.status <= 0:
        raise ConvergenceError(f"optimized DMD: least_squares did not converge ({sol.message})",
                               fix="raise max_nfev or relax tol")
    om, B, E = fit(sol.x)
    # modes: columns of Ur Bᵀ, normalized; amplitudes = their norms
    FPhi = Ur @ xp.asarray(B.T)
    b = to_host(xp.linalg.norm(FPhi, axis=0))
    FPhi = FPhi / xp.asarray(b)[None, :]
    misfit = float(np.sqrt(np.linalg.norm(E @ B - Z.T) ** 2 + lost ** 2)
                   / to_host(xp.linalg.norm(FXall)))
    dtn = ex.dt
    return DMD(np.exp(om * dtn), om, ip.unhalf(FPhi), b.astype(complex),
               np.full(r, np.nan), dtn, r, "optimized", "variable projection", ip,
               ex.singular_values, misfit,
               ex.notes + [f"variable projection: {sol.nfev} evaluations, status {sol.status} "
                           f"({sol.message}); times {'uniform' if times is None else 'given'}"])
