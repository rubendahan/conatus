"""Spectral proper orthogonal decomposition (Towne, Schmidt & Colonius 2018) (data-driven).

Data q_j = q(j Δt), j = 0 … nt−1 (n × nt), declared product ⟨a, b⟩ = aᴴ W b. Welch estimate of
the cross-spectral density with every choice declared:

    blocks of ``nfft`` samples, overlap ``noverlap``, N_b = ⌊(nt − noverlap)/(nfft − noverlap)⌋,
    window w (name for ``scipy.signal.get_window`` or an array), mean ``"subtract"`` (long-time
    mean) or ``"keep"``,
    q̂^{(l)}(f_k) = √(Δt / Σ_j w_j²) Σ_j w_j q^{(l)}_j e^{−2πi jk/nfft}   (numpy FFT sign),
    S(f_k) ≈ (1/N_b) Q̂_k Q̂_kᴴ   (two-sided density: Σ_k S(f_k) Δf = covariance, Δf = 1/(nfft Δt)),

and with ``onesided=True`` (real data) the frequencies 0 … Nyquist with the interior ones
doubled — the conventions of ``scipy.signal.csd(..., scaling="density")``, used as an
independent reference in the tests. SPOD modes and energies at each frequency: the eigenpairs of
S(f) in the W-product, from the N_b × N_b matrix (1/N_b) Q̂_kᴴ W Q̂_k (method of snapshots):

    λ_1 ≥ λ_2 ≥ …,  Ψ = Q̂_k Θ Λ^{−1/2} / √N_b  (W-orthonormal),

or, when N_b > n, from the n × n weighted CSD F S Fᴴ itself (same nonzero eigenvalues; the size
used is stated in ``explain``); eigenvalues beyond the rank are 0. A mode is defined only up to
the numerical rank of S(f) (eigenvalues above ``RANK_RTOL`` x max(n, N_b) x eps x λ_1(f)):
``nmodes`` above the smallest rank over the frequencies is refused (it used to come back as zero
or arbitrary columns beside a zero eigenvalue).

Confidence interval of λ (Schmidt & Colonius 2020, eq. 20): with ν degrees of freedom,
[ν λ / χ²_ν(1 − α/2), ν λ / χ²_ν(α/2)]. This is the law of a *scalar*: one direction fixed in
advance, independent Gaussian blocks. ``confidence(alpha)`` therefore holds for n = 1 and
``noverlap = 0`` and refuses anything else. ν = 2N_b where the block coefficient q̂ is circular
complex (two equal quadratures per block). For real data q̂ of a white block has the relative
pseudo-variance ρ_k = |Σ_j w_j² e^{−4πi jk/nfft}| / Σ_j w_j², and ν = 2N_b / (1 + ρ_k²) (Satterthwaite):
ν = N_b at f = 0 and at the Nyquist frequency, where q̂ is real (ρ = 1), and for an impulse window at
every frequency; ν = 2N_b for a boxcar inside. Real values in a complex array count as real data.
Complex data are assumed circular. ``mean="subtract"`` removes the long-time mean, which takes one
more degree of freedom from the f = 0 estimate; it is not corrected for.

For n > 1 the eigenvalues are the ordered eigenvalues of a sample Wishart matrix. They are biased
apart (the largest upwards, the smallest downwards), so the scalar law is only an approximation,
reached for N_b ≫ n and separated population eigenvalues. ``confidence(alpha, approximate=True)``
returns that formula, also with overlap. ``APPROX_COVERAGE`` holds its measured pointwise coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import OptionError
from .norms import DataInnerProduct, array_module, to_host
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["SPOD", "spod", "welch_blocks", "APPROX_COVERAGE"]

#: Measured coverage of the nominal 95 % ``confidence(alpha, approximate=True)`` on independent
#: real white Gaussian datasets, boxcar window, no overlap, nfft = 8, interior bin k = 1, 2000
#: trials (run script benchmarks/datadriven/spod_coverage_table.py). Keys: population
#: spectrum, n, N_b; values: coverage of the leading and of the smallest eigenvalue. "repeated": all
#: population eigenvalues equal; "separated": 32, 16, ..., 0.25. The scalar ``confidence(alpha)``
#: assumes circular complex data: q = x + 0.5 i y (improper) measured 0.906 at f = 0, 0.944 inside and
#: 0.923 at Nyquist on the same setting.
APPROX_COVERAGE = {
    ("repeated", 1, 16): (0.956, 0.956),
    ("repeated", 2, 16): (0.84, 0.814),
    ("repeated", 4, 16): (0.312, 0.11),
    ("repeated", 8, 16): (0.001, 0.0),
    ("repeated", 8, 128): (0.0, 0.0),
    ("separated", 8, 16): (0.951, 0.218),
    ("separated", 8, 128): (0.952, 0.891),
}

#: Numerical rank of the CSD at one frequency: eigenvalues λ > RANK_RTOL x max(n, N_b) x eps x λ_1
#: (the rounding of a Hermitian eigensolve of that size, relative to the largest); a mode beyond the
#: rank is not defined and is refused, never returned as a zero column (declared in
#: ``conatus.tolerances.DECLARED``).
RANK_RTOL = 10.0


def welch_blocks(Q, *, dt: float, nfft: int, noverlap: int, window, mean: str):
    """(Q̂ with shape (nfreq=nfft, n, N_b), frequencies (fftfreq order), window, N_b)."""
    import scipy.signal as ss
    xp = array_module(Q)
    n, nt = Q.shape
    if not 2 <= nfft <= nt:
        raise OptionError(f"nfft = {nfft}: expected 2 <= nfft <= nt = {nt}")
    if not 0 <= noverlap < nfft:
        raise OptionError(f"noverlap = {noverlap}: expected 0 <= noverlap < nfft")
    if mean not in ("subtract", "keep"):
        raise OptionError(f"mean='{mean}': expected 'subtract' or 'keep' (declared)")
    from ..errors import check_real
    dt = check_real(dt, "dt", "welch_blocks()", positive=True)
    w = (ss.get_window(window, nfft) if isinstance(window, (str, tuple))
         else np.asarray(window, float))
    if w.shape != (nfft,):
        raise OptionError(f"window of length {w.shape} for nfft = {nfft}")
    if not (np.all(np.isfinite(w)) and np.sum(w ** 2) > 0):
        raise OptionError("window: finite values, not all zero (Σw² normalizes the density)")
    step = nfft - noverlap
    nb = (nt - noverlap) // step
    if nb < 1:
        raise OptionError("not a single block: nt too small for nfft")
    Qc = Q - Q.mean(axis=1, keepdims=True) if mean == "subtract" else Q
    scale = np.sqrt(dt / np.sum(w ** 2))
    wx = xp.asarray(w * scale)
    idx = np.arange(nb)[:, None] * step + np.arange(nfft)[None, :]      # N_b × nfft
    blocks = Qc[:, xp.asarray(idx)] * wx[None, None, :]                 # n × N_b × nfft
    Qh = xp.fft.fft(blocks, axis=2)                                     # n × N_b × nfft
    Qh = xp.moveaxis(Qh, 2, 0)                                          # nfft × n × N_b
    return Qh, np.fft.fftfreq(nfft, dt), w, nb


@dataclass
class SPOD:
    __repr__ = compact_repr   # one line, arrays by shape
    frequencies: np.ndarray        # f_k (Hz, i.e. cycles per time unit)
    eigenvalues: np.ndarray        # (nfreq, N_b), descending at each frequency
    modes: Any                     # (nfreq, n, nmodes), W-orthonormal (device of the data)
    nblocks: int
    nfft: int
    noverlap: int
    dt: float
    window: str
    onesided: bool
    mean_policy: str
    inner_product: DataInnerProduct
    notes: list[str] = field(default_factory=list)
    #: the data were real: the coefficients at f = 0 and at the Nyquist frequency are real
    real_data: bool = False
    #: the window of the blocks (its squared DFT gives the pseudo-variance of q̂ used by ``dof``)
    window_values: Any = None

    def dof(self) -> np.ndarray:
        """Degrees of freedom ν of the χ² law of ν λ̂ / λ at each frequency (nfreq,): 2 N_b; for
        real data 2 N_b / (1 + ρ_k²) with ρ_k the relative pseudo-variance of the windowed
        coefficient (N_b at f = 0 and at the Nyquist frequency; module docstring)."""
        nu = np.full(self.frequencies.shape, 2 * self.nblocks, dtype=float)
        if self.real_data:
            pseudo = np.abs(np.fft.fft(np.asarray(self.window_values) ** 2))
            # index k of the frequency in both layouts (one-sided: 0 ... nfft//2; two-sided: fft)
            rho = pseudo[(2 * np.arange(nu.size)) % self.nfft] / pseudo[0]
            nu = nu / (1 + rho ** 2)
        return nu

    def confidence(self, alpha: float, *, approximate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) (1 − α) bounds on every eigenvalue: ν λ / χ²_ν(1 − α/2) and
        ν λ / χ²_ν(α/2) with ν = :meth:`dof` at each frequency.

        Holds for one entry (n = 1) and independent blocks (``noverlap = 0``), and is refused
        otherwise. ``approximate=True`` returns the same formula for any n and overlap: a pointwise
        approximation for N_b ≫ n and separated eigenvalues, whose coverage can be far below 1 − α
        (``APPROX_COVERAGE``, module docstring)."""
        from scipy.stats import chi2
        from ..errors import check_real
        alpha = check_real(alpha, "alpha", "SPOD.confidence()", positive=True)
        if not alpha < 1:
            raise OptionError(f"SPOD.confidence: alpha = {alpha:g}: expected 0 < alpha < 1")
        n = self.modes.shape[1]
        if not approximate and (n != 1 or self.noverlap != 0):
            what = f"n = {n} entries" if n != 1 else f"overlapping blocks (noverlap = {self.noverlap})"
            raise OptionError(
                f"SPOD.confidence: the χ² law holds for one entry and independent blocks; this "
                f"estimate has {what}",
                why="the ordered eigenvalues of a sample CSD matrix do not follow a scalar χ² law: "
                    "with 8 white channels and N_b = 16 the nominal 95 % interval of λ_1 covers "
                    "below 1 % of the time",
                fix="confidence(alpha, approximate=True) for the pointwise approximation "
                    "(measured coverage: conatus.datadriven.spod.APPROX_COVERAGE)")
        nu = self.dof()[:, None]
        return (nu * self.eigenvalues / chi2.ppf(1 - alpha / 2, nu),
                nu * self.eigenvalues / chi2.ppf(alpha / 2, nu))

    def explain(self) -> str:
        k = int(np.argmax(self.eigenvalues[:, 0]))
        dof = ("N_b dof at f = 0 and Nyquist (real data), up to 2 N_b elsewhere" if self.real_data
               else "2 N_b dof, circular data assumed")
        n = self.modes.shape[1]
        conf = ("χ² confidence for independent Gaussian blocks" if n == 1 and self.noverlap == 0
                else "χ² confidence only as an approximate=True pointwise approximation for "
                     "N_b >> n and separated eigenvalues")
        return (f"SPOD (Towne et al. 2018): N_b = {self.nblocks} blocks of nfft = {self.nfft} "
                f"(overlap {self.noverlap}; {conf}, {dof}), window "
                f"{self.window}, Δt = {self.dt:g}, mean: {self.mean_policy}, "
                f"{'one' if self.onesided else 'two'}-sided density (scipy.signal.csd "
                f"conventions), {self.inner_product.kind} product; largest λ_1 = "
                f"{self.eigenvalues[k, 0]:.6g} at f = {self.frequencies[k]:.6g}. "
                + " ".join(self.notes))


@renames("conatus.datadriven.spod:spod")
def spod(Q, *, dt: float, nfft: int, noverlap: int, window, mean: str, onesided: bool,
         inner_product: DataInnerProduct, n_modes: int) -> SPOD:
    """SPOD of the time series ``Q`` (n × nt); every estimation choice is declared."""
    xp = array_module(Q)
    if onesided and xp.iscomplexobj(Q):
        raise OptionError("onesided=True is for real data (the negative frequencies of complex "
                          "data carry independent information)")
    Qh, f, w, nb = welch_blocks(Q, dt=dt, nfft=nfft, noverlap=noverlap, window=window, mean=mean)
    n_space = Q.shape[0]
    if not 1 <= n_modes <= min(nb, n_space):
        # n < n_modes <= N_b used to crash in the n x n branch (n modes into n_modes columns)
        raise OptionError(f"n_modes = {n_modes}: expected 1 <= n_modes <= min(N_b, n) = "
                          f"{min(nb, n_space)} (N_b = {nb} blocks, n = {n_space} entries)")
    if onesided and not inner_product.is_real:
        # z = q̂(f), q̂(−f) = conj(z): zᴴ W z and zᵀ W conj(z) differ when Im W ≠ 0, so doubling the
        # positive side does not give the energy
        raise OptionError(
            "onesided=True with a complex Gram matrix: the energies at +f and -f of real data "
            "differ, and doubling one side does not keep the energy",
            fix="onesided=False; or Re(W), which gives real data the same time-domain energy")
    if onesided:
        nf = nfft // 2 + 1
        Qh, f = Qh[:nf], np.abs(f[:nf])
        dbl = np.ones(nf)
        dbl[1:nf - 1 if nfft % 2 == 0 else nf] = 2.0
        Qh = Qh * xp.asarray(np.sqrt(dbl))[:, None, None]
    nf = Qh.shape[0]
    n = Qh.shape[1]
    # the nonzero eigenvalues of (1/N_b) (F Q̂)ᴴ(F Q̂) (N_b × N_b) and of (1/N_b) (F Q̂)(F Q̂)ᴴ
    # (n × n, the weighted CSD itself) coincide: the smaller matrix is used (reported)
    small = "blocks" if nb <= n else "space"
    lam = np.zeros((nf, nb))
    modes = xp.zeros((nf, n, n_modes), dtype=complex)
    ranks = np.zeros(nf, dtype=int)
    for k in range(nf):                               # loop over frequencies, not time steps
        FQ = inner_product.half(Qh[k])
        if small == "blocks":
            M = to_host(FQ.conj().T @ FQ) / nb
            ev, Th = np.linalg.eigh((M + M.conj().T) / 2)
            o = np.argsort(ev)[::-1]
            ev, Th = np.clip(ev[o], 0.0, None), Th[:, o]
            lam[k] = ev
            ranks[k] = _rank(ev, n, nb)
            if ranks[k] < n_modes:
                continue                              # refused below: no division by λ ≈ 0
            sq = np.sqrt(np.where(ev[:n_modes] > 0, ev[:n_modes], np.inf))
            modes[k] = Qh[k] @ xp.asarray(Th[:, :n_modes] / sq[None, :] / np.sqrt(nb))
        else:
            M = to_host(FQ @ FQ.conj().T) / nb
            ev, U = np.linalg.eigh((M + M.conj().T) / 2)
            o = np.argsort(ev)[::-1]
            lam[k, :n] = np.clip(ev[o], 0.0, None)
            ranks[k] = _rank(lam[k, :n], n, nb)
            modes[k] = inner_product.unhalf(xp.asarray(U[:, o[:n_modes]]))
    if ranks.min() < n_modes:
        k = int(np.argmin(ranks))
        raise OptionError(
            f"n_modes = {n_modes}: the cross-spectral density has numerical rank {ranks[k]} at "
            f"f = {f[k]:.6g} ({int(np.sum(ranks < n_modes))} of {nf} frequencies below {n_modes}; "
            f"rank: eigenvalues above {RANK_RTOL:g} x max(n, N_b) x eps x λ_1(f))",
            why="a SPOD mode beyond the rank is not defined (its eigenvalue is 0); it is never "
                "returned as a zero or arbitrary column",
            fix=f"n_modes={max(int(ranks.min()), 1)}" if ranks.min() >= 1 else
                "data with energy at every frequency (e.g. mean='keep', or a longer record)")
    wname = window if isinstance(window, (str, tuple)) else f"user array (Σw² = {np.sum(w**2):.6g})"
    note = ("eigenproblem of size N_b (method of snapshots)" if small == "blocks" else
            f"eigenproblem of size n = {n} < N_b (the weighted CSD; same eigenvalues)")
    return SPOD(f, lam, modes, nb, nfft, noverlap, float(dt), str(wname), onesided, mean,
                inner_product, [note],
                real_data=not bool(xp.iscomplexobj(Q)) or bool(xp.all(Q.imag == 0)),
                window_values=w)


def _rank(ev: np.ndarray, n: int, nb: int) -> int:
    """Numerical rank of a CSD from its eigenvalues (descending, >= 0); see RANK_RTOL."""
    if ev.size == 0 or ev[0] <= 0:
        return 0
    return int(np.sum(ev > RANK_RTOL * max(n, nb) * np.finfo(float).eps * ev[0]))
