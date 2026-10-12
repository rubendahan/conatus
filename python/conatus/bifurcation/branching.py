"""Automatic branch switching (§branching).

* **Branch point on an equilibrium branch** (transcritical, pitchfork): H(z) = G(x; p), z = (x, p).
  At the located point H_z has a two-dimensional kernel span{t0, φ} (t0: tangent of the known
  branch) and a left null vector ψ. The tangents of the branches through the point are the
  directions α t0 + β φ solving the *algebraic branching equation* (Keller 1977; Allgower & Georg
  1990, §8.3)  c11 α² + 2 c12 α β + c22 β² = 0,  c_ij = ψ·H_zz[e_i, e_j], with the exact second
  derivatives of the IR (B, ∂J/∂p, ∂²G/∂p²). The root not aligned with t0 is the new branch; its
  alignment with t0 (the other root) is reported as a check. With a declared symmetry S (a linear
  map with S G(x) = G(S x)), the kernel parity S φ = ±φ classifies the point: S φ = −φ on an
  S-invariant branch is a **symmetry-breaking pitchfork** (the two new half-branches are S-images of
  each other: one is continued, the other is its image), S φ = +φ a symmetric transcritical/pitchfork.
* **Hopf point → periodic orbits**: the cycle is predicted by the Hopf normal form (radius from
  the declared amplitude, Δp = −Re c1 |z|² / α', period 2π/(ω + ...)), x(τ) = x0 + 2 Re(z q e^{2πiτ}),
  corrected at fixed p by collocation Newton, then continued (``continue_orbits``).
* **Period doubling → doubled orbit**: on a mesh of 2·ntst intervals, the orbit is repeated
  (period 2T) and perturbed by ε w, w = the multiplier −1 Floquet vector v on [0, T] and −v on
  [T, 2T] (periodic of period 2T); corrected on the hyperplane orthogonal to w, then continued.

Every switch records the amplitude/step used (declared), the checks, and the classification in
``Curve.notes`` and ``Curve.switch``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError, OptionError, SymmetryError
from .driver import Curve, CurveEvent
from .equilibria import continue_equilibria
from .normal_form import normal_form, parity_under
from .orbits import CollocationMesh, PeriodicOrbitProblem, continue_orbits

__all__ = ["branch_switch", "hopf_orbit_start", "branching_directions"]

# Thresholds of the branch switch (detection/classification only, declared in
# ``conatus.tolerances.DECLARED``; the ABE coefficients and the kernel are reported in
# ``Curve.switch``, so each decision can be checked against the numbers it used).
#: φ is taken from the first kernel basis vector unless its projection orthogonal to t0 has norm
#: below KERNEL_SWITCH_TOL (then from the second, of norm >= 1 − 1e-6): a choice of basis vector,
#: which also fixes the sign of φ (see ``_kernel_complement``), not an accuracy threshold.
#: Why 1e-3: the two projections are parallel and, for an
#: orthonormal kernel basis with t0 in its span, their norms n1, n2 satisfy n1² + n2² = 1. Any
#: threshold far above the round-off of a projection (~1e-16 x the condition of the SVD) and far
#: below 1 gives the same φ up to sign; the sign is taken from the vector used, i.e. from the
#: LAPACK SVD's orientation of the kernel basis. 1e-3 keeps 13 decades from round-off and leaves
#: the second vector with n2 >= 1 − 5e-7. Which branch ``direction=+1`` follows is therefore a
#: convention of the SVD, not of the physics: Curve.switch reports φ and the tangent used, and the
#: other branch is ``direction=-1``.
KERNEL_SWITCH_TOL = 1e-3
#: c22 of the ABE is treated as zero (one root is β/α = ∞, i.e. α = 0) below ABE_ZERO_TOL x max|c|.
ABE_ZERO_TOL = 1e-12
#: Transcritical when |c22| > TRANSCRITICAL_TOL x max|c_ij|, else pitchfork (classification note).
TRANSCRITICAL_TOL = 1e-8
#: The located point is S-invariant when |S x − x| <= INVARIANT_TOL x max(1, max|x|).
INVARIANT_TOL = 1e-9
#: Kernel parity S φ = ±φ to PARITY_TOL x max|φ| (looser than normal_form.PARITY_TOL: φ comes from
#: the SVD of the bordered Jacobian at a located point, which may carry the chord of the location).
PARITY_TOL = 1e-7


def _Hzz(system, x, par, pname, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Second derivative of H(x, p) = G(x; p) in directions a, b ∈ R^{n+1} (exact IR forms)."""
    ax, ap, bx, bp = a[:-1], a[-1], b[:-1], b[-1]
    out = system.B(x, ax, bx, par)
    if ap:
        out = out + ap * system.Jp(x, bx, pname, par)
    if bp:
        out = out + bp * system.Jp(x, ax, pname, par)
    if ap and bp:
        out = out + ap * bp * system.Gp(x, pname, par, order=2)
    return np.real(out)


def _kernel_complement(null: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """φ: the unit kernel direction orthogonal to t0 (t0 unit, in the span of the 2 columns of
    ``null``). Both projections null[:, k] − (t0·null[:, k]) t0 are parallel to φ; the first is used
    unless its norm is below KERNEL_SWITCH_TOL. The rule fixes the sign of φ, hence the sense of
    ``direction=+1`` at a pitchfork (new tangent = ±φ): a threshold-free choice (the larger
    projection oriented along the first) would take that sign from the round-off of a first
    projection of norm ~1e-17 (review of the)."""
    cand = null[:, 0] - t0 * (t0 @ null[:, 0])
    if np.linalg.norm(cand) < KERNEL_SWITCH_TOL:
        cand = null[:, 1] - t0 * (t0 @ null[:, 1])
    return cand / np.linalg.norm(cand)


def branching_directions(system, x: np.ndarray, par: Mapping[str, float], pname: str,
                         t0: np.ndarray) -> dict:
    """Tangents of the branches through a simple branch point (ABE); see the module docstring."""
    n = system.n
    Hz = np.hstack([system.J(x, par), system.Gp(x, pname, par)[:, None]])
    U, sv, Vh = np.linalg.svd(Hz)
    null = Vh[-2:].T                        # 2-dim kernel of the n x (n+1) Jacobian
    psi = U[:, -1]
    t0 = t0 / np.linalg.norm(t0)
    # the known branch tangent, projected on the kernel (the located point may carry a chord)
    proj = null @ (null.T @ t0)
    kernel_gap = float(np.linalg.norm(t0 - proj))
    t0 = proj / np.linalg.norm(proj)
    phi = _kernel_complement(null, t0)
    c11 = float(psi @ _Hzz(system, x, par, pname, t0, t0))
    c12 = float(psi @ _Hzz(system, x, par, pname, t0, phi))
    c22 = float(psi @ _Hzz(system, x, par, pname, phi, phi))
    disc = c12 ** 2 - c11 * c22
    # directions (α, β) with c11 α² + 2 c12 α β + c22 β² = 0
    dirs = []
    if abs(c22) > ABE_ZERO_TOL * max(abs(c11), abs(c12), 1e-300):
        for sgn in (1, -1):
            r = (-c12 + sgn * np.sqrt(max(disc, 0.0))) / c22     # β / α
            d = t0 + r * phi
            dirs.append(d / np.linalg.norm(d))
    else:
        dirs.append(phi.copy())                                 # α = 0
        if abs(c12) > 0:
            d = -c11 / (2 * c12) * t0 + phi                     # c11 α + 2 c12 β = 0
            d = d if abs(c11) > 0 else t0
            dirs.append(d / np.linalg.norm(d))
        else:
            dirs.append(t0.copy())
    align = [abs(float(d @ t0)) for d in dirs]
    k_new = int(np.argmin(align))
    return {"new": dirs[k_new], "old_alignment": max(align), "phi": phi, "psi": psi,
            "c": (c11, c12, c22), "discriminant": disc, "sigma": sv[-2:],
            "kernel_gap": kernel_gap, "n": n}


def hopf_orbit_start(system, x0: np.ndarray, par: Mapping[str, float], parameter: str, *,
                     amplitude: float, mesh: CollocationMesh):
    """(problem, z_pred, z_hopf, nf): collocation problem and the normal-form cycle predictor."""
    nf = normal_form(system, x0, kind="hopf", par=par, parameter=parameter)
    if nf.dlambda is None or nf.dlambda.real == 0:
        raise ConvergenceError("hopf_orbit_start: transversality fails (α' = 0)")
    if nf.c1.real == 0:
        raise ConvergenceError("hopf_orbit_start: degenerate Hopf (l1 = 0): use continue_codim2")
    r = amplitude / (2 * np.abs(nf.q).max())
    dp = -nf.c1.real * r ** 2 / nf.dlambda.real
    pb = PeriodicOrbitProblem(system, parameter, mesh, par=par)
    taus = mesh.times()
    U = nf.orbit(dp, taus)
    T = nf.period(dp)
    z_pred = np.concatenate([U.ravel(), [T, par[parameter] + dp]])
    z_h = np.concatenate([np.tile(x0, taus.size), [2 * np.pi / nf.omega, par[parameter]]])
    return pb, z_pred, z_h, nf


def branch_switch(curve: Curve, event: CurveEvent, settings: ContinuationSettings, *,
                  step: float | None = None, amplitude: float | None = None,
                  mesh: CollocationMesh | None = None, direction: int = 1, symmetry: Any = None,
                  bounds=None, marginal_tol: float | None = None, **orbit_kw) -> Curve:
    """Continue the branch emanating from ``event`` of ``curve``.

    * equilibrium branch point: ``step`` (declared predictor length along the new tangent);
      ``symmetry`` (optional S) classifies the pitchfork; ``direction`` = ±1 picks the half-branch.
    * equilibrium Hopf point: ``amplitude`` (max-norm amplitude of the first cycle, declared) and
      ``mesh`` (collocation mesh) are required.
    * orbit period doubling: ``amplitude`` (ε, max-norm of the perturbation, declared); the mesh is
      the doubled one.

    ``bounds``: (lo, hi) on the continuation parameter; for an orbit branch also a dict
    {parameter and/or "T": (lo, hi)} (see :func:`continue_orbits`).

    Returns the new :class:`Curve` (``curve.switch`` holds the classification and checks).
    """
    if event.point is None or not event.converged:
        raise ConvergenceError(f"branch_switch: the {event.kind} event is not located "
                               f"({event.reason})",
                               fix="refine the location (event budget/tolerance)")
    if curve.kind == "equilibrium" and event.kind == "branch point":
        return _switch_bp(curve, event, settings, step, direction, symmetry, bounds, marginal_tol)
    if bounds is not None and curve.kind == "equilibrium" and event.kind == "hopf" \
            and not isinstance(bounds, Mapping):
        # (lo, hi) on the parameter, the form continue_equilibria takes: continue_orbits takes a
        # dict (a tuple used to fail with "'tuple' object has no attribute 'items'")
        bounds = {curve.context["parameter"]: tuple(bounds)}
    if bounds is not None and curve.kind == "periodic orbit" and not isinstance(bounds, Mapping):
        bounds = {curve.context.parameter: tuple(bounds)}
    if curve.kind == "equilibrium" and event.kind == "hopf":
        if amplitude is None or mesh is None:
            raise OptionError("branch_switch at a Hopf point: amplitude= and mesh= are required (declared)")
        return _switch_hopf(curve, event, settings, amplitude, mesh, bounds, **orbit_kw)
    if curve.kind == "periodic orbit" and event.kind == "pd":
        if amplitude is None:
            raise OptionError("branch_switch at a period doubling: amplitude= (ε) is required")
        return _switch_pd(curve, event, settings, amplitude, direction, bounds, **orbit_kw)
    raise OptionError(f"branch_switch: no switch from a {event.kind!r} event on a {curve.kind} curve",
                      fix="branch points and Hopf points of equilibria, period doublings of orbits")


def _switch_bp(curve, event, cfg, step, direction, symmetry, bounds, marginal_tol):
    ctx = curve.context
    system, pname, base = ctx["system"], ctx["parameter"], ctx["base"]
    if step is None:
        raise OptionError("branch_switch at a branch point: step= (predictor length) is required")
    z = event.point.z
    x, par = z[:-1], {**base, pname: float(z[-1])}
    info = branching_directions(system, x, par, pname, event.point.tangent)
    d = info["new"] * (1 if direction >= 0 else -1)
    kind = ("transcritical" if abs(info["c"][2]) > TRANSCRITICAL_TOL * max(map(abs, info["c"]))
            else "pitchfork")
    sym_note = ""
    if symmetry is not None:
        S = symmetry if callable(symmetry) else (lambda v, S=np.asarray(symmetry): S @ v)  # noqa: B008
        phx = info["phi"][:-1]
        # absolute bounds only (rtol=0: np.allclose's default rtol = 1e-5 overrode both
        # declared tolerances)
        on_sym = np.allclose(S(x), x, rtol=0.0, atol=INVARIANT_TOL * max(1.0, np.abs(x).max()))
        parity = parity_under(S(phx), phx, PARITY_TOL)
        if parity is None:
            raise SymmetryError("branch_switch: kernel neither S-symmetric nor S-antisymmetric")
        if parity == -1 and on_sym:
            kind = "pitchfork (symmetry-breaking)"
            sym_note = "S φ = −φ on an S-invariant branch: the other half-branch is the S-image"
        else:
            sym_note = f"S φ = {'+' if parity > 0 else '−'}φ (symmetric bifurcation)"
    drv = ctx["driver"]
    zp = z + step * d
    z1, rel, its = drv.correct(zp, d)
    new = continue_equilibria(system, z1[:-1], pname, cfg, par={**base, pname: float(z1[-1])},
                              marginal_tol=marginal_tol if marginal_tol is not None else ctx["marginal_tol"],
                              tangent=d, bounds=bounds)
    new.switch = {"from": event.kind, "classification": kind, "abe": info["c"],
                  "discriminant": info["discriminant"], "old_alignment": info["old_alignment"],
                  "kernel_singular_values": info["sigma"], "symmetry": sym_note}
    new.notes.insert(0, f"branch switch at {pname} = {z[-1]:.12g}: {kind}; ABE c = "
                        f"({info['c'][0]:.3e}, {info['c'][1]:.3e}, {info['c'][2]:.3e}), "
                        f"known-branch alignment {info['old_alignment']:.12f}, step {step:g} "
                        f"(direction {'+1' if direction >= 0 else '-1'}); first corrected point "
                        f"residual {rel:.1e}"
                        + (f"; {sym_note}" if sym_note else ""))
    return new


def _switch_hopf(curve, event, cfg, amplitude, mesh, bounds, **kw):
    ctx = curve.context
    system, pname, base = ctx["system"], ctx["parameter"], ctx["base"]
    z = event.point.z
    par = {**base, pname: float(z[-1])}
    pb, z_pred, z_h, nf = hopf_orbit_start(system, z[:-1], par, pname, amplitude=amplitude, mesh=mesh)
    new = continue_orbits(pb, z_pred, cfg, direction=z_pred - z_h, bounds=bounds, **kw)
    z1 = new.points[0].z
    U1 = pb.split(z1)[0] if pb.mesh.tmesh.size == mesh.tmesh.size else None
    amp1 = float(np.abs(U1 - z[:-1][None, :]).max()) if U1 is not None else float("nan")
    new.switch = {"from": "hopf", "normal_form": nf, "amplitude": amplitude,
                  "predicted_dp": float(z_pred[-1] - par[pname]),
                  "corrected_amplitude": amp1, "predicted_T": float(z_pred[-2]),
                  "corrected_T": float(z1[pb.nu])}
    new.notes.insert(0, f"Hopf → cycles at {pname} = {par[pname]:.12g} (ω = {nf.omega:.10g}, "
                        f"l1 = {nf.l1:.6g}, {nf.criticality}); predictor: amplitude {amplitude:g} "
                        f"(declared), Δ{pname} = {z_pred[-1] - par[pname]:.4e}, T = {z_pred[-2]:.8g}; "
                        f"corrected at fixed {pname}: T = {z1[pb.nu]:.8g}")
    return new


def _switch_pd(curve, event, cfg, eps, direction, bounds, **kw):
    pb: PeriodicOrbitProblem = curve.context
    z = event.point.z
    # the problem's mesh may have been changed by remeshing after this point: rebuild from z size
    if z.size != pb.nu + 2:
        raise OptionError("branch_switch (PD): the orbit problem was remeshed after this event")
    U, T, p = pb.split(z)
    v = pb.floquet_vector(z, -1.0)
    end_check = float(np.abs(v[-1] + v[0]).max() / np.abs(v[0]).max())
    m2 = CollocationMesh(2 * pb.mesh.ntst, pb.mesh.ncol,
                         np.concatenate([pb.mesh.tmesh / 2, 0.5 + pb.mesh.tmesh[1:] / 2]))
    U2 = np.vstack([U, U[1:]])
    w = np.vstack([v, -v[1:]])
    w = w / np.abs(w).max()
    pb2 = PeriodicOrbitProblem(pb.sys, pb.parameter, m2, par=pb.base)
    sgn = 1 if direction >= 0 else -1
    z_star = np.concatenate([U2.ravel(), [2 * T, p]])
    t = np.concatenate([sgn * w.ravel(), [0.0, 0.0]])
    zp = z_star + eps * t
    pb2.set_reference(U2)
    z1 = _correct_hyperplane(pb2, zp, t, cfg)
    new = continue_orbits(pb2, z1, cfg, direction=t, bounds=bounds, fix_start=False, **kw)
    mu = event.data.get("multiplier", np.nan)
    new.switch = {"from": "pd", "epsilon": eps, "antiperiodicity_check": end_check, "multiplier": mu}
    new.notes.insert(0, f"period doubling at {pb.parameter} = {p:.12g} (μ = {np.real(mu):+.10f}): "
                        f"doubled orbit T = {2 * T:.10g}, ε = {eps:g} (declared) along the μ = −1 "
                        f"Floquet vector (|v(T) + v(0)|/|v(0)| = {end_check:.1e})")
    return new


def _correct_hyperplane(pb, zp, t, cfg):
    """Newton on [R(z); t·(z − zp)] = 0 (square)."""
    z = zp.copy()
    tn = t / np.linalg.norm(t)
    for k in range(cfg.max_newton + 1):
        r = np.concatenate([pb.residual(z), [tn @ (z - zp)]])
        err = float(np.max(np.abs(r)))
        if err <= cfg.tolerance:
            return z
        if k == cfg.max_newton:
            break
        A = sp.vstack([pb.jacobian(z), sp.csr_matrix(tn[None, :])]).tocsc()
        z = z + spla.spsolve(A, -r)
    raise ConvergenceError(f"PD switch: corrector not converged (max|R| = {err:.2e})",
                           fix="smaller amplitude ε, finer mesh")
