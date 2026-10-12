"""Matrix-free Newton–Krylov with hookstep trust region.

Solves R(z) = 0 given R and the Jacobian-vector product A(z)·v. Each Newton iteration runs
Arnoldi–GMRES (modified Gram–Schmidt) to build V_{m+1}, H̄ (m+1)×m. The Newton step
y_N = argmin ‖β e₁ − H̄ y‖ is taken if ‖y_N‖ ≤ Δ; otherwise the **hookstep** (Viswanath 2007,
J. Fluid Mech. 580; Channelflow) minimizes the same least squares on the ball ‖y‖ = Δ through
the SVD of H̄: y(μ) = W diag(σ/(σ²+μ)) Uᵀβe₁, with μ ≥ 0 set so that ‖y(μ)‖ = Δ.
The trust radius follows the ratio ρ = actual / predicted reduction of ‖R‖ (shrink if ρ < ρ_bad,
grow if ρ > ρ_good at the boundary). Every tolerance, radius and budget is an explicit argument,
reported by :meth:`NewtonKrylovHookstep.explain` and in the iteration history.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .._deprecation import renames
from ..errors import OptionError, check_count
from ..solvers.nlbvp import NewtonNotConverged
from ..tolerances import REQUIRED, require
from .._repr import compact_repr

__all__ = ["NewtonKrylovHookstep", "NewtonKrylovResult", "arnoldi_gmres", "NewtonNotConverged"]

_RADII = {"absolute": "(absolute)", "relative": "(× ‖z0‖)"}


@renames("conatus.ecs.newton:arnoldi_gmres")
def arnoldi_gmres(apply_A: Callable[[np.ndarray], np.ndarray], b: np.ndarray, *,
                  rtol: float, max_iterations: int):
    """Arnoldi–GMRES from x₀ = 0. Returns (V, H̄, β, y, relres) with ‖b − A V_m y‖ = relres ‖b‖."""
    beta = float(np.linalg.norm(b))
    if beta == 0:
        raise OptionError("GMRES: zero right-hand side",
                          why="x = 0 solves it; the Krylov basis cannot start from b = 0",
                          fix="test the residual before calling arnoldi_gmres")
    n = b.size
    V = np.zeros((n, max_iterations + 1))
    H = np.zeros((max_iterations + 1, max_iterations))
    V[:, 0] = b / beta
    m, relres, y = 0, 1.0, np.zeros(0)
    for j in range(max_iterations):
        w = apply_A(V[:, j])
        for i in range(j + 1):                       # modified Gram–Schmidt (+ one reorthog.)
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        for i in range(j + 1):
            c = V[:, i] @ w
            H[i, j] += c
            w = w - c * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        m = j + 1
        e1 = np.zeros(m + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[:m + 1, :m], e1, rcond=None)
        relres = float(np.linalg.norm(e1 - H[:m + 1, :m] @ y) / beta)
        if relres < rtol or H[j + 1, j] < 1e-14 * beta:
            break
        V[:, j + 1] = w / H[j + 1, j]
    return V[:, :m + 1], H[:m + 1, :m], beta, y, relres


def _hook(H: np.ndarray, beta: float, delta: float) -> tuple[np.ndarray, float]:
    """y minimizing ‖βe₁ − H y‖ subject to ‖y‖ ≤ Δ; returns (y, μ)."""
    U, s, Wt = np.linalg.svd(H, full_matrices=False)
    p = beta * U[0, :]                               # Uᵀ β e₁
    def y_of(mu):
        return Wt.T @ (s * p / (s ** 2 + mu))
    if np.linalg.norm(y_of(0.0)) <= delta:
        return y_of(0.0), 0.0
    lo, hi = 0.0, 1.0
    while np.linalg.norm(y_of(hi)) > delta:
        hi *= 4.0
    for _ in range(100):                             # bisection on log-free μ (monotone ‖y(μ)‖)
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(y_of(mid)) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, hi):
            break
    return y_of(hi), hi


@dataclass
class NewtonKrylovResult:
    __repr__ = compact_repr   # one line, arrays by shape
    z: np.ndarray
    residual: float                   # ‖R(z)‖ (absolute)
    relative_residual: float          # ‖R(z)‖ / scale (scale declared by the caller)
    iterations: int
    history: list[dict] = field(default_factory=list)
    #: how the trust radii were read (absolute, or relative to ‖z0‖ with the values used)
    trust: str = ""

    def explain(self) -> str:
        lines = [f"Newton–Krylov–hookstep: {self.iterations} iterations, ‖R‖ = {self.residual:.3e} "
                 f"(relative {self.relative_residual:.3e})"]
        if self.trust:
            lines.append(f"  trust radii: {self.trust}")
        for h in self.history:
            lines.append(f"  it {h['it']:3d}  ‖R‖={h['res']:.3e}  gmres {h['gmres_iters']:3d} "
                         f"(relres {h['gmres_relres']:.1e})  Δ={h['delta']:.2e}  "
                         f"step={h['step']:.2e}  {h['kind']}  ρ={h['rho']:.2f}")
        return "\n".join(lines)


#: Declared defaults of the algorithm controls of NewtonKrylovHookstep (the rule of
#: conatus.tolerances: algorithm controls default and are declared, accuracy tolerances are
#: required). They change how fast Newton converges, never the digits of the answer: the
#: residual is tested against the required ``tol`` and returned. Values = the ones used across
#: the tests and tutorials (KS, Kolmogorov, channel and Couette solutions).
#:
#: Trust radii: ``radii="absolute"`` (the v0.1 meaning, kept as the
#: default so that v0.1 results are unchanged) reads delta0/delta_min/delta_max in the norm of the
#: unknowns; ``radii="relative"`` reads them as fractions of ‖z0‖, the norm of the starting
#: unknowns (state and, for orbits, T and the shift), so the same numbers serve a state of norm
#: 1e-3 or 1e3. Absolute default radii on a start whose norm lies outside HOOKSTEP_NORM_MIN..MAX
#: give a TrustRadiusWarning (they were tuned on states of norm O(1)); explain() of the result
#: prints the radii used.
HOOKSTEP_DEFAULTS = dict(max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60, delta0=1.0,
                         delta_min=1e-10, delta_max=10.0, rho_bad=0.1, rho_good=0.75,
                         radii="absolute")
#: Range [HOOKSTEP_NORM_MIN, HOOKSTEP_NORM_MAX] of ‖z0‖ in which the default absolute radii
#: (Δ0 = 1 in [1e-10, 10]) are on the scale of the state; outside it, TrustRadiusWarning (a
#: detection: nothing is changed). Two decades either side of the norm the defaults were tuned on.
HOOKSTEP_NORM_MIN, HOOKSTEP_NORM_MAX = 1e-2, 1e2


class TrustRadiusWarning(UserWarning):
    """Default absolute trust radii used on a starting state far from norm 1: the
    radii are not on the scale of the unknowns; pass ``radii="relative"`` or explicit radii."""


class NewtonKrylovHookstep:
    """Newton–Krylov with hookstep. ``tol`` (the accuracy target, on ‖R‖/scale) is required;
    the algorithm controls (budget, inexact-GMRES tolerance and Krylov budget, trust radii and
    how they are read, acceptance ratios) have declared defaults (``HOOKSTEP_DEFAULTS``), printed
    by ``explain()``. ``radii="relative"`` reads delta0, delta_min and delta_max as fractions of
    the norm of the starting unknowns (``"absolute"``, the default: in the norm of the unknowns)."""

    @renames("conatus.ecs.newton:NewtonKrylovHookstep.__init__")
    def __init__(self, *, tol: float = REQUIRED, max_iterations: int = 40,
                 gmres_rtol: float = 1e-3, gmres_max_iterations: int = 60, delta0: float = 1.0,
                 delta_min: float = 1e-10, delta_max: float = 10.0,
                 rho_bad: float = 0.1, rho_good: float = 0.75, radii: str = "absolute") -> None:
        tol = require(tol, "tol", where="NewtonKrylovHookstep()", suggest="tol=1e-12",
                      why="tol is the accuracy of the returned solution (‖R‖/scale < tol): how "
                          "many digits you get is your choice; the other settings only change "
                          "how fast it is reached")
        self._given = {k for k, v in dict(max_iterations=max_iterations, gmres_rtol=gmres_rtol,
                                          gmres_max_iterations=gmres_max_iterations, delta0=delta0,
                                          delta_min=delta_min, delta_max=delta_max,
                                          rho_bad=rho_bad, rho_good=rho_good, radii=radii).items()
                       if v != HOOKSTEP_DEFAULTS[k]}
        where = "NewtonKrylovHookstep()"
        if radii not in ("absolute", "relative"):
            raise OptionError(f"{where}: radii={radii!r}",
                              why="the trust radii are read in the norm of the unknowns "
                                  "('absolute') or as fractions of the starting norm ‖z0‖ "
                                  "('relative')", fix="radii='relative' (or 'absolute', the default)")
        self.tol = float(tol)
        self.max_iterations = check_count(max_iterations, "max_iterations", where)
        self.gmres_rtol = float(gmres_rtol)
        self.gmres_max_iterations = check_count(gmres_max_iterations, "gmres_max_iterations", where)
        self.delta0, self.delta_min, self.delta_max = float(delta0), float(delta_min), float(delta_max)
        self.rho_bad, self.rho_good = float(rho_bad), float(rho_good)
        self.radii = radii

    @property
    def gmres_maxiter(self) -> int:
        """Former name of :attr:`gmres_max_iterations` (read-only, kept for v0.1 readers)."""
        return self.gmres_max_iterations

    def explain(self) -> str:
        """Accuracy target (given) and every algorithm control, marked given or default."""
        given = getattr(self, "_given", set(HOOKSTEP_DEFAULTS))
        d = {k: ("" if k in given else " (default)") for k in HOOKSTEP_DEFAULTS}
        return (f"NewtonKrylovHookstep(tol={self.tol:g} on ‖R‖/scale (given), max_iterations="
                f"{self.max_iterations}{d['max_iterations']}, GMRES rtol={self.gmres_rtol:g}"
                f"{d['gmres_rtol']} max_iterations={self.gmres_max_iterations}"
                f"{d['gmres_max_iterations']}, trust radius Δ0={self.delta0:g}{d['delta0']} in "
                f"[{self.delta_min:g}{d['delta_min']}, {self.delta_max:g}{d['delta_max']}] "
                f"{_RADII[self.radii]}{d['radii']}, ρ_bad={self.rho_bad}{d['rho_bad']}, "
                f"ρ_good={self.rho_good}{d['rho_good']})")

    def _radii(self, z0: np.ndarray) -> tuple[float, float, float, str]:
        """(Δ0, Δmin, Δmax) in the norm of the unknowns for the start ``z0``, and the sentence
        that says how they were read (kept in the result's explain())."""
        n0 = float(np.linalg.norm(z0))
        if self.radii == "relative":
            if not n0 > 0:
                raise OptionError("NewtonKrylovHookstep(radii='relative'): the starting unknowns "
                                  "are zero, so radii relative to ‖z0‖ are zero",
                                  fix="radii='absolute' with explicit delta0/delta_min/delta_max")
            return (self.delta0 * n0, self.delta_min * n0, self.delta_max * n0,
                    f"relative to ‖z0‖ = {n0:.3e}: Δ0 = {self.delta0 * n0:.3e} in "
                    f"[{self.delta_min * n0:.3e}, {self.delta_max * n0:.3e}]")
        text = f"absolute (‖z0‖ = {n0:.3e})"
        lo, hi = HOOKSTEP_NORM_MIN, HOOKSTEP_NORM_MAX
        given = getattr(self, "_given", set())
        if not {"delta0", "delta_min", "delta_max", "radii"} & given and not lo <= n0 <= hi:
            msg = (f"NewtonKrylovHookstep: the default trust radii are absolute (Δ0 = "
                   f"{self.delta0:g} in [{self.delta_min:g}, {self.delta_max:g}], tuned on states "
                   f"of norm O(1)) and the starting unknowns have norm {n0:.3e}; pass "
                   "radii='relative' (fractions of ‖z0‖) or explicit radii")
            from ..errors import _warn_user       # the user's line, whatever the finder above
            _warn_user(msg, TrustRadiusWarning)
            text += f"; outside [{lo:g}, {hi:g}] of the default radii (TrustRadiusWarning)"
        return self.delta0, self.delta_min, self.delta_max, text

    def solve(self, residual: Callable[[np.ndarray], np.ndarray],
              jvp: Callable[[np.ndarray, np.ndarray], np.ndarray], z0: np.ndarray, *,
              scale: float, log: Callable[[str], None] | None = None,
              on_iteration: Callable[[dict], None] | None = None) -> NewtonKrylovResult:
        """Iterate from z0. ``scale`` normalizes ‖R‖ for the stopping test (declared by caller).
        ``on_iteration`` (optional, explicit) receives each iteration's record, e.g. a monitor
        telemetry ``Telemetry.newton_callback()``; it never changes the iteration."""
        z = np.array(z0, dtype=float)
        delta0, delta_min, delta_max, trust = self._radii(z)
        R = residual(z)
        res = float(np.linalg.norm(R))
        delta = delta0
        hist: list[dict] = []
        for it in range(1, self.max_iterations + 1):
            if res / scale < self.tol:
                return NewtonKrylovResult(z, res, res / scale, it - 1, hist, trust)
            V, H, beta, yN, gres = arnoldi_gmres(lambda v, z=z: jvp(z, v), -R, rtol=self.gmres_rtol,
                                                 max_iterations=self.gmres_max_iterations)
            while True:
                y, mu = _hook(H, beta, delta)
                kind = "newton" if mu == 0.0 else "hook"
                dz = V[:, :y.size] @ y
                e1 = np.zeros(H.shape[0])
                e1[0] = beta
                pred = res - float(np.linalg.norm(e1 - H @ y))
                Rt = residual(z + dz)
                rt = float(np.linalg.norm(Rt))
                rho = (res - rt) / pred if pred > 0 else -1.0
                if rho < self.rho_bad and delta > delta_min:
                    delta = max(delta_min, 0.5 * min(delta, float(np.linalg.norm(y))))
                    continue                             # retry on the same Krylov space
                break
            step = float(np.linalg.norm(dz))
            hist.append(dict(it=it, res=res, gmres_iters=H.shape[1], gmres_relres=gres,
                             delta=delta, step=step, kind=kind, rho=rho))
            if on_iteration is not None:
                on_iteration(dict(hist[-1], res_trial=rt, scale=scale))
            if log:
                log(f"it {it}: ‖R‖={res:.3e} → {rt:.3e}, gmres {H.shape[1]} ({gres:.1e}), "
                    f"{kind}, Δ={delta:.2e}, ρ={rho:.2f}")
            if rt >= res and delta <= delta_min:
                raise self._not_converged(
                    f"no decrease at the minimal trust radius Δ={delta:g} (‖R‖={res:.3e}); "
                    + NewtonKrylovResult(z, res, res / scale, it, hist, trust).explain(),
                    res / scale, stalled=True, z=z)
            if rt < res:
                z, R, res = z + dz, Rt, rt
            if rho > self.rho_good and kind == "hook":
                delta = min(delta_max, 2.0 * delta)
        if res / scale < self.tol:
            return NewtonKrylovResult(z, res, res / scale, self.max_iterations, hist, trust)
        raise self._not_converged(
            f"budget of {self.max_iterations} iterations exhausted: "
            + NewtonKrylovResult(z, res, res / scale, self.max_iterations, hist, trust).explain(),
            res / scale, stalled=bool(hist) and res > 0.5 * hist[-1]["res"], z=z)

    def _not_converged(self, message: str, rel: float, *, stalled: bool,
                       z: np.ndarray | None = None) -> NewtonNotConverged:
        """The failure, with the stagnation diagnosis as attributes (``relative_residual``,
        ``tol``, ``stagnated_near_tol``, ``stalled``, ``last_iterate``) and, when it applies,
        why/fix lines.

        ``stalled``: the last iteration no longer halved ‖R‖ (or no step decreased it at the
        minimal trust radius); a search cut by its budget while ‖R‖ still fell is not stalled.
        ``last_iterate``: the unknowns at the failure (a copy), e.g. the wave speed of a
        travelling-wave search.

        ``stagnated_near_tol``: the last iteration no longer halved ‖R‖ (``stalled``) and ‖R‖/scale
        is within 10x of tol: the tolerance is below the residual floor this problem can reach
        (Krylov tolerance, rounding at this size; seen on the 3D Couette continuations stalled at 2-4e-10 against 1e-10). A run still converging when its budget ends is not
        labelled so. Said, never loosened for you."""
        floor = stalled and rel < 10 * self.tol
        e = NewtonNotConverged(
            message,
            why=(f"‖R‖/scale = {rel:.2e} stagnates within 10x of tol = {self.tol:g}: the tolerance "
                 "is probably below the attainable residual floor" if floor else None),
            fix=(f"declare a reachable tolerance (e.g. tol={3 * rel:.0e}) or a tighter GMRES "
                 f"tolerance (gmres_rtol < {self.gmres_rtol:g}); check the reported full-space "
                 "residual of the result" if floor else None))
        e.relative_residual, e.tol, e.stagnated_near_tol = float(rel), self.tol, floor
        e.stalled = bool(stalled)
        e.last_iterate = None if z is None else np.array(z, copy=True)
        return e
