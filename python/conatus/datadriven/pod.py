"""Proper orthogonal decomposition in a declared inner product (data-driven toolkit).

Snapshots X = [x_1 … x_m] (n × m), inner product ⟨a, b⟩ = aᴴ W b (``DataInnerProduct``). With
F the square root of W (W = Fᴴ F) the POD is the SVD of F X' (X' = X, or X minus its mean, as
*declared*): F X' = U Σ Vᴴ, and

    modes φ_k = F⁻¹ u_k (⟨φ_j, φ_k⟩ = δ_jk),  energies λ_k = σ_k² / m,  coefficients a_k(t_j) = σ_k v̄_jk,

so X' = Σ_k φ_k a_k and λ_k is the time average of |a_k|² (uniform snapshot weights). Two
declared methods (Sirovich 1987; Holmes, Lumley & Berkooz 1996):

* ``"snapshots"``: eigen-decomposition of the m × m correlation (F X')ᴴ(F X'); cheap when
  m ≪ n; relative accuracy of σ_k ≈ eps · (σ_1/σ_k)² (squared condition): ``explain`` flags kept
  modes with σ_k/σ_1 < 1e-7;
* ``"svd"``: thin SVD of F X' (accurate to eps · σ_1 for every σ_k).

Arrays may be NumPy or CuPy (the heavy products run on the data's device).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import OptionError
from .norms import DataInnerProduct, array_module, to_host
from .._repr import compact_repr

__all__ = ["POD", "pod", "POD_METHODS", "MEAN_POLICIES"]

POD_METHODS = ("snapshots", "svd")
MEAN_POLICIES = ("subtract", "keep")


@dataclass
class POD:
    __repr__ = compact_repr   # one line, arrays by shape
    energies: np.ndarray            # λ_k = σ_k² / m
    singular_values: np.ndarray     # σ_k of F X'
    modes: Any                      # n × r, W-orthonormal (device of the data)
    coefficients: np.ndarray        # r × m, a_k(t_j)
    mean: Any                       # n-vector (zeros if mean="keep")
    inner_product: DataInnerProduct
    method: str
    mean_policy: str
    orthonormality: float           # ‖Φᴴ W Φ − I‖_max
    reconstruction: float           # ‖F (X' − Φ A)‖_F / ‖F X'‖_F with the kept rank
    notes: list[str] = field(default_factory=list)
    total_energy: float = float("nan")  # ‖F X'‖_F² / m: the energy of all the snapshots

    @property
    def energy_fraction(self) -> float:
        """Fraction of the snapshots' total energy captured by the kept modes, Σ_k λ_k / E_total
        (finding 92: explain() used to normalise on the kept modes, so it
        always printed 1)."""
        return float(self.energies.sum() / self.total_energy) if self.total_energy > 0 else 1.0

    @property
    def rank(self) -> int:
        return self.energies.size

    def project(self, X) -> np.ndarray:
        """Coefficients ⟨φ_k, x − mean⟩ of new snapshots."""
        xp = array_module(X)
        Xc = X - xp.asarray(self.mean)[:, None]
        return to_host(self.inner_product.inner(self.modes, Xc))

    def explain(self) -> str:
        return (f"POD ({self.method}, mean: {self.mean_policy}, {self.inner_product.kind} product) "
                f"rank {self.rank}: energies λ_k = σ_k²/m = "
                + ", ".join(f"{e:.6g}" for e in self.energies[:6])
                + (" …" if self.rank > 6 else "")
                + f"; kept fraction of the total snapshot energy {self.energy_fraction:.6g}; "
                f"orthonormality defect {self.orthonormality:.1e}; relative reconstruction "
                f"residual {self.reconstruction:.1e}. " + " ".join(self.notes))


def pod(X, *, inner_product: DataInnerProduct, rank: int, method: str, mean: str) -> POD:
    """POD of the snapshot matrix ``X`` (n × m); every choice is declared (see module doc)."""
    if method not in POD_METHODS:
        raise OptionError(f"method='{method}': expected one of {POD_METHODS}")
    if mean not in MEAN_POLICIES:
        raise OptionError(f"mean='{mean}': expected one of {MEAN_POLICIES} (no default: the mean "
                          "is part of the decomposition)")
    xp = array_module(X)
    n, m = X.shape
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(n, m):
        raise OptionError(f"rank = {rank}: expected an integer 1 <= rank <= min(n, m) = {min(n, m)}")
    mu = X.mean(axis=1) if mean == "subtract" else xp.zeros(n, dtype=X.dtype)
    Xc = X - mu[:, None]
    from .norms import check_data
    check_data(inner_product, X)       # audit inputs: NaN was a bare LinAlgError
    FX = inner_product.half(Xc)
    notes = []
    if method == "svd":
        U, s, Vh = xp.linalg.svd(FX, full_matrices=False)
        U, s, Vh = U[:, :rank], to_host(s)[:rank], Vh[:rank]
        coeffs = s[:, None] * to_host(Vh)
        modes = inner_product.unhalf(U)
    else:
        C = to_host(FX.conj().T @ FX)                   # m × m correlation
        ev, V = np.linalg.eigh((C + C.conj().T) / 2)
        order = np.argsort(ev)[::-1][:rank]
        ev, V = ev[order], V[:, order]
        if np.any(ev <= 0):
            raise OptionError(f"rank {rank} exceeds the numerical rank of the snapshots "
                              "(non-positive correlation eigenvalue)")
        s = np.sqrt(ev)
        U = FX @ xp.asarray(V / s[None, :])
        coeffs = s[:, None] * V.conj().T
        modes = inner_product.unhalf(U)
        small = s[-1] / s[0]
        if small < 1e-7:
            notes.append(f"method 'snapshots' with σ_r/σ_1 = {small:.1e} < 1e-7: the smallest "
                         "kept singular values lose accuracy (squared condition); use 'svd'")
    G = to_host(inner_product.inner(modes, modes))
    ortho = float(np.abs(G - np.eye(rank)).max())
    R = FX - inner_product.half(modes) @ xp.asarray(coeffs)
    nFX = float(to_host(xp.linalg.norm(FX)))
    rec = float(to_host(xp.linalg.norm(R)) / max(nFX, 1e-300))
    return POD(s ** 2 / m, s, modes, coeffs, mu, inner_product, method, mean, ortho, rec, notes,
               total_energy=nFX ** 2 / m)
