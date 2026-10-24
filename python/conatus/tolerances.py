"""Tolerances of Conatus: which ones you must give, which ones have a declared default.

Rule:

* a tolerance that **decides when an iterative solver stops** (and so how many digits the
  returned value has: Arnoldi/Lanczos, Brent/Newton on a critical parameter), that **filters the
  returned set** (resolution-drift filter of an eigenvalue spectrum) or that **pilots a numerical
  choice** (safety factor of an adaptive or CFL time step) has **no default**. Its default is
  :data:`REQUIRED`; leaving it out raises :class:`~conatus.errors.OptionError` with a value to
  write and why it matters (:func:`require`);
* a threshold that only **detects or refuses** (a residual check, a rank decision reported with
  its singular values, an iteration cap that raises, a classification note), or a stopping test
  at the attainable floor whose residual is always returned, keeps a **declared default**: a
  named value listed in :data:`DECLARED` (with the reason), printed by :func:`explain` and by the
  ``explain()`` of the object that uses it. The development test suite fails if a public
  signature gains a numeric tolerance default that is not declared here, or if a declared value
  changes without this table.

Refinement: **algorithm controls default and
are declared; accuracy tolerances are required.** A setting that only changes *how fast* or *by
which path* an iteration reaches a target whose residual is tested and returned (trust radii,
step growth factors, inexact-Newton/GMRES tolerances and Krylov budgets, corrector budgets, the
finite-difference step rule of a parameter derivative, how DMD amplitudes are fitted, the
finite-state check of the time steppers) has a declared default, listed in :data:`DECLARED`
(names matching the tolerance pattern) or :data:`ALGORITHM_DEFAULTS` (the others), printed by
:func:`explain` and marked "(default)" by the ``explain()`` of the object. The accuracy targets
(``tol`` of a Newton solve, ``tolerance`` of a corrector, ``xtol`` of a critical value,
``drift_tol``, ...) stay required. Thresholds that decide where a bifurcation event is located
(``fold_tol``, ``crossing_tol``, ``hopf_imag_tol``) keep their declared defaults and are printed in
the result of each located event.

Nothing here changes a value for you: the code suggests, it does not modify.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# IMAG_ROUNDOFF is importable and declared, but not in __all__: the v0.1 API surface is frozen
# Documented names only (M5, 0.2); the others stay importable from this module.
__all__ = ["Declared", "NCC_TAIL_TOL", "NCC_ROUNDING", "ncc_floor_text", "SPURIOUS_TAIL_TOL",
           "SPURIOUS_TAIL_FRACTION", "SPURIOUS_SIGNIFICANT", "UNRESOLVED_TAIL_TOL", "HILL_TAIL_TOL",
           "BASE_RESIDUAL_TOL", "DRIFT_TOL_MAX", "explain"]


class _Required:
    """Sentinel default of an argument that must be given (``help()`` shows ``REQUIRED``)."""

    _instance: _Required | None = None

    def __new__(cls) -> _Required:              # a single instance, also across copies/pickles
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REQUIRED"

    def __reduce__(self) -> str:
        return "REQUIRED"

    def __bool__(self) -> bool:
        raise TypeError("REQUIRED has no truth value: test `x is REQUIRED`")


REQUIRED: Any = _Required()


def is_required(value: Any) -> bool:
    """True if ``value`` is the :data:`REQUIRED` sentinel (the argument was not given)."""
    return value is REQUIRED


def require(value: Any, name: str, *, where: str, suggest: str, why: str,
            doc: str | None = None) -> Any:
    """``value`` unchanged, or :class:`~conatus.errors.OptionError` if it is :data:`REQUIRED`.

    The message names the argument, says what to write (``suggest``: a complete ``name=value``)
    and why the value is the user's choice (``why``)."""
    if value is REQUIRED:
        from .errors import OptionError
        raise OptionError(f"{where}: {name} must be given explicitly (no default)",
                          why=why, fix=f"write {suggest}",
                          doc=doc or "docs/site/guide/errors.md#optionerror")
    return value


@dataclass(frozen=True)
class Declared:
    """A tolerance with a declared default: where it is, its value, its role and why a default."""

    where: str        # "module:Qualname(param)" for a signature default, "module.NAME" for a constant
    value: Any        # a number, or a named rule ("auto", "first") for an algorithm control
    role: str         # "refusal" | "detection" | "stop at the floor" | "iteration cap" | ...
    why: str
    used_by: tuple[str, ...] = ()    # signature defaults "module:Qualname(param)" set to this value


#: NCC resolution: refusal threshold of the relative Chebyshev tail of a non-constant
#: coefficient sampled on the grid (refused above, with the resolution to use or the cutoff to
#: declare). Same value as ``conatus.resolution.TAIL_TARGET``, the tail Conatus calls "resolved"
#: for a solution field: one meaning of "resolved" for fields and coefficients. Measured:
#: the coefficients of the test suite and tutorials are either resolved to 1e-10..4e-9 or
#: clearly under-resolved (1e-7..1e-1); 1e-12 (the former warning) flagged correct profiles.
NCC_TAIL_TOL = 1e-8
#: NCC rounding floor: a tail below NCC_ROUNDING x eps x (data scale) x (derivative
#: amplification) is the rounding of the coefficient's computation, not under-resolution. The
#: derivative amplification applies only when the differentiated data are themselves exact to
#: NCC_ROUNDING x eps (relative tail); otherwise the amplified tail is interpolation error, judged
#: like any coefficient.
NCC_ROUNDING = 1e3
#: Round-off of an imaginary part: a complex array that stands for a real
#: quantity (a starting state of a real continuation, a grid property of a ComplexFourier problem)
#: is accepted as real only when max|Im| <= IMAG_ROUNDOFF x max(1, max|a|); above it, it is refused,
#: never truncated. Same 1e3 x eps convention as NCC_ROUNDING and ``solvers.evp.INF_TOL``; not a
#: solver tolerance: an imaginary part of 1e-7 is not round-off, whatever the requested accuracy.
IMAG_ROUNDOFF = 1e3 * 2.220446049250313e-16
#: Spurious-mode flag of eigenvalue results (conatus.solvers.spurious):
#: a mode whose relative energy in the last SPURIOUS_TAIL_FRACTION of its wall-normal coefficients
#: exceeds SPURIOUS_TAIL_TOL is flagged spurious (kept, listed, ranked last; never selected as the
#: least stable). Measured on Orr-Sommerfeld N = 64: tau modes 0.96..1.0, the least
#: stable physical mode 3e-17, damped resolved modes 1e-13..1e-9; 1e-3 (a tail amplitude of about
#: 3 % of the mode) lies far from both. A detection, not a filter: drift= removes modes.
SPURIOUS_TAIL_TOL = 1e-3
#: Fraction of the coefficients of the non-periodic axis (per piece, at least two) whose energy
#: is the "tail" of the test above.
SPURIOUS_TAIL_FRACTION = 0.1
#: An unknown of the mode is judged only if its energy is at least this fraction of the energy of
#: the most excited unknown (a field the mode does not excite holds round-off; amplitude 1e-6).
SPURIOUS_SIGNIFICANT = 1e-12
#: Resolution check of the SELECTED mode: the mode that a result
#: without a drift test selects (``order()[0]``, ``least_stable``, the mode of ``critical``) is
#: called under-resolved when its relative tail energy (same measure as SPURIOUS_TAIL_TOL) exceeds
#: this level: a warning in the result, its explain() and one Python ResolutionWarning; nothing is
#: removed or re-ranked. Measured on Orr-Sommerfeld Re = 1e4, alpha = 1 (tail of the
#: least stable mode / error of c): N = 16: 8.3e-4 / 0.31; N = 24: 3.7e-4 / 0.34 (a wrong mode,
#: and a critical Reynolds number 5037.6 instead of 5772.2); N = 32: 1.0e-8 / 4.4e-5; N = 40:
#: 1.7e-10 / 5.4e-6; N = 48: 1.2e-12 / 8.8e-8; N = 64: 2.6e-17 / 1.3e-9. The eigenvalue error
#: follows about 0.1-0.4 x sqrt(tail), so 1e-10 asks for about five correct digits of the selected
#: eigenvalue; the under-resolved modes (4e-4..8e-4) pass SPURIOUS_TAIL_TOL = 1e-3 but sit six
#: orders of magnitude above this level.
UNRESOLVED_TAIL_TOL = 1e-10
#: Harmonic truncation of a Floquet-Hill mode: relative weight
#: of the outermost harmonics (the two at each end of -Nh..Nh: sum of their norms over the sum of
#: all harmonic norms) of a returned exponent's eigenvector, above which the exponent is reported
# : truncated in Nh (ResolutionWarning, line in explain()).
#: alpha = 0.3, N = 64 (weight / error of the leading exponent against -0.0935006):
#: Nh = 2: 2.0e-1 / 5.9e-6; Nh = 20: 3.1e-3 / 3.1e-5; Nh = 30: 8.5e-4 / 6.5e-6; Nh = 40: 1.4e-4 /
#: 1.5e-5; Nh = 50: 8.6e-6 / 2.6e-7; Nh = 60: 4.6e-8 / 3.1e-8. The error stays below about 0.1 x
#: the weight but is not monotone in Nh while the weight is large (Nh = 2 is right to 6e-6 by
#: cancellation; every residual is 1e-14, the eigenproblem is well conditioned: this is
#: truncation); 1e-6 separates the converged Nh = 60 from every truncated one.
HILL_TAIL_TOL = 1e-6
#: Base state of a linearization: largest relative residual of
#: the steady equations and boundary conditions at the base state (per equation, after a least
#: squares fit of the unknowns without a base value: pressure, taus) above which the base is not a
#: steady solution. Above it, a base not declared as such (``linearize(..., base_residual=
#: "accept")``) gets a BaseStateWarning in 0.1 and 0.2 (refused from 0.3) and a line in explain().
#: Measured: the RB conduction state and the forced Poiseuille flow at rounding (0);
#: RB about T = +z: 2.0 (in T(z=-0.5) = 0.5); unforced Poiseuille: 1.0 (momentum equation).
BASE_RESIDUAL_TOL = 1e-6
#: Largest accepted drift tolerance: drift_tol is a relative
#: distance to the nearest eigenvalue at the finer resolution; above 1e-2 most modes, spurious
#: ones included, find a neighbour, so the test no longer filters (refused, with why).
DRIFT_TOL_MAX = 1e-2


def _d(where: str, value: float, role: str, why: str,
       used_by: tuple[str, ...] = ()) -> tuple[str, Declared]:
    return where, Declared(where, value, role, why, used_by)


_NCC_USERS = tuple(f"conatus.{s}(ncc_tail_tol)" for s in (
    "discretize.compiler:Compiler.__init__", "discretize.plan:Plan.__init__",
    "discretize.plan:discretize", "discretize.polar_plan:PolarPlan.__init__",
    "discretize.sphere_plan:SpherePlan.__init__", "discretize.terrain_plan:TerrainPlan.__init__",
    "model.linearization:Linearization.periodic_operator",
    "solvers.nlbvp:NewtonSolver.__init__", "solvers.parametric:ParametricEVP.__init__",
    "growth.periodic:periodic_operator", "growth.periodic:optimal_gain",
    "growth.this case:this case_operator")) + (
    "conatus.discretize.ultraspherical:NCCReport.__init__(tail_tol)",)


_FLOOR = "stopping test at the attainable floor; the residual of the result is always returned"
_CHECK = "verification only: above it the input is refused, with the measured value in the message"
_RANK = "rank decision, reported with the singular values it separated"
_CAP = "iteration cap: reaching it raises ConvergenceError, never returns an unconverged result"
_NOTE = "classification note only (the returned object is unchanged, the note cites the value)"
_VERIFY = "pass threshold of a verify() check; the measured value is printed beside it, never hidden"

#: Every declared default (key: :attr:`Declared.where`). Required tolerances are not listed: their
#: default is REQUIRED.
DECLARED: dict[str, Declared] = dict([
    # discretization
    _d("conatus.tolerances.NCC_TAIL_TOL", NCC_TAIL_TOL, "refusal",
       "an under-resolved coefficient is refused; ncc_tail_tol= or ncc_cutoff= declare more",
       _NCC_USERS),
    _d("conatus.tolerances.NCC_ROUNDING", NCC_ROUNDING, "detection",
       "an NCC tail below NCC_ROUNDING x eps x data scale (x N^2 2/L per derivative, only if the "
       "differentiated data are themselves exact to NCC_ROUNDING x eps) is rounding (e.g. U'' of "
       "a linear U), reported as such, not judged"),
    _d("conatus.tolerances.IMAG_ROUNDOFF", IMAG_ROUNDOFF, "refusal",
       "a complex array standing for a real quantity (starting state of a real continuation, grid "
       "property of FlowProperties) is refused when max|Im| exceeds it x max(1, max|a|) (below: "
       "rounding, dropped); the measured value is in the message"),
    _d("conatus.tolerances.SPURIOUS_TAIL_TOL", SPURIOUS_TAIL_TOL, "detection",
       "eigenvalue results: a mode whose relative tail energy (last SPURIOUS_TAIL_FRACTION of its "
       "wall-normal coefficients) exceeds it is flagged spurious, kept, listed by explain() and "
       "ranked last (spurious='none' switches it off; drift= is the filter that removes modes)"),
    _d("conatus.tolerances.SPURIOUS_TAIL_FRACTION", SPURIOUS_TAIL_FRACTION, "detection",
       "part of the wall-normal coefficients (per piece, at least two) taken as the tail of the "
       "spurious-mode test"),
    _d("conatus.tolerances.SPURIOUS_SIGNIFICANT", SPURIOUS_SIGNIFICANT, "detection",
       "an unknown is judged by the spurious-mode test only if its energy is at least this "
       "fraction of the most excited unknown's (round-off is never judged)"),
    _d("conatus.tolerances.UNRESOLVED_TAIL_TOL", UNRESOLVED_TAIL_TOL, "detection",
       "the mode a result without a drift test selects (order()[0], least_stable, critical) is "
       "reported under-resolved above it (warning in the result, explain() and one "
       "ResolutionWarning; nothing removed or re-ranked)"),
    _d("conatus.tolerances.HILL_TAIL_TOL", HILL_TAIL_TOL, "detection",
       "a Floquet-Hill exponent whose eigenvector holds more than this relative weight in its "
       "outermost harmonics is reported truncated in Nh (ResolutionWarning, explain())"),
    _d("conatus.tolerances.BASE_RESIDUAL_TOL", BASE_RESIDUAL_TOL, "detection",
       "a linearization base state whose steady residual (equations and boundary conditions, "
       "per equation) exceeds it is not a steady solution: BaseStateWarning in 0.1 and 0.2 (refused "
       "from 0.3) unless declared with base_residual='accept'; the residual is in explain()"),
    _d("conatus.tolerances.DRIFT_TOL_MAX", DRIFT_TOL_MAX, "refusal",
       "drift_tol above it no longer filters (most modes find a neighbour): refused"),
    # projections of the curvilinear step (m = 0 real part, shift-reflect subspace)
    _d("conatus.timestep.real_projection.PROBE_TOL", 1e-12, "refusal",
       "floor of the bound to which the m = 0 map J read off the transforms is verified on an "
       "independent probe (projection off above it, reason in explain())"),
    _d("conatus.timestep.real_projection.RT_FACTOR", 100.0, "refusal",
       "verification bound of the inferred projection maps = max(PROBE_TOL, RT_FACTOR x the "
       "transforms' round-trip rounding measured on the probe), printed by explain()"),
    _d("conatus.timestep.real_projection.RT_QUANTILE", 0.9, "detection",
       "quantile of the per-entry round-trip change taken as the transforms' rounding (a defect "
       "confined to a few entries is not taken for rounding)"),
    _d("conatus.timestep.real_projection.RT_M0_FACTOR", 10.0, "refusal",
       "verification bound of the m = 0 map J on the m = 0 entries = max(bound elsewhere, "
       "RT_M0_FACTOR x the largest round-trip change of those entries on a real probe), printed by "
       "explain()"),
    _d("conatus.timestep.real_projection.RT_MAX", 1e-10, "refusal",
       "round-trip rounding above which the transforms do not define the projection maps (m = 0: "
       "projection off, declared; shift-reflect: refused)"),
    _d("conatus.timestep.symmetry_projection.PROBE_TOL", 1e-12, "refusal",
       "floor of the verification bound of the shift-reflect map (refused above, with the value)"),
    _d("conatus.timestep.symmetry_projection.LEAK_MAX", 1e-9, "refusal",
       "part of the state removed by the shift-reflect projection in one step (relative), checked "
       "at the end of every run/run_graph: above it the run is refused (operators not equivariant)"),
    _d("conatus.ecs.pipe.NEUTRAL_OVERLAP_MIN", 0.99, "detection",
       "alignment with d_z x of the eigenvector (or degenerate eigenspace) taken as the neutral "
       "translation mode of a pipe travelling wave; reported with the eigenvalues, never counted"),
    # mapped domains (terrain-following map)
    _d("conatus.ir.terrain.METRIC_ROUNDING", 1e-15, "detection",
       "metric Fourier coefficients below this fraction of the largest are FFT rounding (fine grid) "
       "and left out of the coupled operator; the count kept is printed by explain()"),
    _d("conatus.ir.terrain.SURFACE_TAIL", 1e-13, "refusal",
       "a wall surface whose Fourier tail on the fine grid exceeds it (not smooth or not periodic) "
       "is refused: its derivatives would not be spectral"),
    _d("conatus.discretize.terrain_plan.MAX_COUPLED_UNKNOWNS", 100_000, "refusal",
       "size of the coupled sparse system of a mapped domain, refused above (before assembly) "
       "with the size in the message; a resource limit of the direct LU, not of accuracy"),
    _d("conatus.discretize.terrain_plan.MAX_COUPLED_NNZ", 10_000_000, "refusal",
       "nnz(L) + nnz(M) of the coupled system of a mapped domain, refused above (before any "
       "factorization; the LU fills in far beyond it)"),
    _d("conatus.solvers.evp.INF_TOL", 1e3 * 2.220446049250313e-16, "detection",
       "infinite eigenvalues: |beta| <= INF_TOL |alpha| (scaled) with M singular, or beta = 0; "
       "counted and logged. With M of full numerical rank every pair is finite"),
    _d("conatus.solvers.evp.DRIFT_FLOOR", 1e-3, "detection",
       "floor of the drift ratio as a fraction of the median nonzero |sigma| of the finer spectrum: the "
       "drift test is relative, and a mode below this fraction of the spectrum's scale is near "
       "zero for it (judged against the floor); no unit of the eigenvalue enters (audit "
       "invariance; was max(|sigma|, 1))"),
    _d("conatus._scales.SPECTRAL_FLOOR", 1e-3, "detection",
       "near zero for a relative test on an eigenvalue: below this fraction of the median modulus "
       "of the nonzero eigenvalues of the spectrum (Hopf / neutral-saddle split of equilibria, "
       "codim-2 and continuation, the continuation crossing test, the mode-switch check of "
       "critical(), the sensitivity gap); replaces the unit-carrying max(|x|, 1) (critic review of the invariance audit, "
       ")"),
    _d("conatus.solvers.evp.ARNOLDI_TOL", 0.0, "stop at the floor",
       "ARPACK tol = 0 is machine precision (the strictest), each mode's residual is returned",
       ("conatus.solvers.evp:solve_sparse(tol)",)),
    # parametric pencils
    _d("conatus.solvers.parametric.RECON_TOL", 1e-10, "refusal", _CHECK),
    _d("conatus.solvers.parametric.ENTRY_DROP_TOL", 1e-13, "detection",
       "a Laurent coefficient entry at most this x the largest modulus of that matrix entry over "
       "the sample plans is rounding of the DFT and discarded (counted); an evaluation where the "
       "discarded entries could exceed RECON_TOL of the pencil is refused (a global amplitude "
       "threshold would delete a real parameter term)"),
    _d("conatus.solvers.parametric.TRACK_TOL", 1e-13, "stop at the floor", _FLOOR,
       ("conatus.solvers.parametric:ParametricEVP.track(tol)",)),
    _d("conatus.solvers.parametric.TRACK_MAX_ITERATIONS", 30, "iteration cap", _CAP,
       ("conatus.solvers.parametric:ParametricEVP.track(max_iterations)",)),
    _d("conatus.solvers.parametric.MULTIPLICITY_TOL", 1e-8, "detection",
       "multiple eigenvalue detection; the multiplicity and the declared choice are in the note"),
    _d("conatus.solvers.parametric.MODE_SWITCH_TOL", 1e-8, "refusal",
       "mode switch at the critical point: refused, never returned"),
    _d("conatus.solvers.parametric.CRITICAL_MAX_ITERATIONS", 50, "iteration cap", _CAP),
    # Floquet
    _d("conatus.floquet.cascade.RESONANCE_TOL", 1e-12, "refusal", _CHECK),
    _d("conatus.floquet.roots.SECANT_STEP", 1e-4 * (1 + 1j), "start offset",
       "second point seed + SECANT_STEP of the determinant secants (secant_root(step), "
       "this caseHill.find_root(step).find_root(step)); the stop is the "
       "caller's tol, stalls are reported"),
    _d("conatus.growth.periodic:operator_floquet(residual_max)", 1e-8, "refusal",
       "an exponent whose Hill residual exceeds it is not returned (each one carries its residual)"),
    _d("conatus.growth.periodic.LEAK_MAX", 1e-8, "refusal", _CHECK,
       ("conatus.growth.periodic:operator_floquet(leak_max)",)),
    _d("conatus.growth.periodic.FOLD_ROUNDING", 1e3 * 2.220446049250313e-16, "detection",
       "an exponent within it x 2 pi / P of the lower end of (-pi/P, pi/P] lies on that end to the "
       "rounding of the fold and moves by one period (nothing else is moved)"),
    _d("conatus.floquet.hill_general:harmonic_matrices(linearity_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.floquet.hill_general:hill_from_plan(linearity_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.floquet.hill_gpu:HillGPU.__init__(affine_check_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.growth.periodic:periodic_operator_from_plan(linearity_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.growth.periodic:periodic_operator(linearity_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.growth.periodic:periodic_operator_from_plan(rank_tol)", 1e-10, "rank", _RANK),
    _d("conatus.growth.periodic:periodic_operator(rank_tol)", 1e-10, "rank", _RANK),
    _d("conatus.growth.periodic:optimal_gain(rank_tol)", 1e-10, "rank", _RANK),
    _d("conatus.model.linearization:Linearization.periodic_operator(rank_tol)", 1e-10, "rank", _RANK),
    # adjoint
    _d("conatus.adjoint.nlop:AdmissibleProjector.__init__(rank_tol)", 1e-10, "rank", _RANK),
    _d("conatus.adjoint.sphere_energy:SphereAdmissibleProjector.__init__(rank_tol)", 1e-10, "rank",
       _RANK),
    # continuation (the corrector tolerance and budgets are required fields)
    _d("conatus.continuation.core:ContinuationSettings.__init__(event_max_iterations)", 30,
       "iteration cap", "event secant budget: an event not located is flagged converged=False"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(fold_tol)", 1e-12, "detection",
       "event location only (the branch points are unchanged); flagged converged=False outside it"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(crossing_tol)", 1e-11, "detection",
       "event location only (the branch points are unchanged); flagged converged=False outside it"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(hopf_imag_tol)", 1e-8, "detection",
       "Hopf / real crossing classification of a located event (note in the Event)"),
    # invariant solutions
    _d("conatus.ecs.dae:find_relative_equilibrium_dae(c_zero_tol)", 1e-10, "detection", _NOTE),
    _d("conatus.ecs.dae:stability_dae(translation_kernel_tol)", 1e-8, "detection",
       "marginal translation mode identification, counted in the result"),
    _d("conatus.ecs.dae:stability_dae(translation_overlap_min)", 0.99, "detection",
       "marginal translation mode identification, counted in the result"),
    _d("conatus.ecs.solutions:symmetry_breaking_check(symmetry_tol)", 1e-10, "detection", _NOTE),
    _d("conatus.ecs.solutions:floquet_multipliers(marginal_overlap_min)", 0.99, "detection",
       "marginal time/translation multipliers identified by overlap, declared and returned"),
    _d("conatus.ecs.solutions.TRIVIAL_TOL", 1e-8, "detection", _NOTE),
    _d("conatus.ecs.solutions.STEADY_TOL", 1e3, "detection",
       "an orbit whose shape change over its period (max_j min_s ‖τ_s u_j − u_0‖/‖u_0‖) is below "
       "STEADY_TOL x the accuracy of the solve is a steady state reached by the orbit search: "
       "retracted, value in the notes (relative to the orbit amplitude)"),
    _d("conatus.ecs.solutions.FULL_RESIDUAL_FACTOR", 10.0, "refusal",
       "an invariant solution is returned only if its full residual (frozen mean and symmetry "
       "rows included) is <= FULL_RESIDUAL_FACTOR x the Newton tol x scale; otherwise "
       "NewtonNotConverged"),
    _d("conatus.ecs.solutions.PERIOD_FD_REL", 1e-5, "algorithm control",
       "relative central-difference step of the period column dPhi_T/dT of the orbit Newton "
       "(the dT_rel of the 3-D shooting solver)"),
    _d("conatus.ecs.solutions.STEADY_SAMPLES", 64, "detection",
       "states of the period (evenly spaced) at which the steady-state check measures the shape "
       "change of an orbit"),
    _d("conatus.ecs.solutions.TW_SPEED_TOL", 1.0, "detection",
       "a travelling wave with |c| ||dx U|| below TW_SPEED_TOL x the Newton tol x scale is "
       "relabelled an equilibrium (c = 0 within the accuracy of the solve), value in the notes"),
    # runs and solves: detections and refusals of wrong-looking results
    _d("conatus.model.initial_check.INITIAL_TAIL_TOL", 1e-6, "detection",
       "callable initial data whose relative spectral tail beyond the resolution (sampled on "
       "twice the grid, per axis) exceeds it are not fully resolved: InitialDataWarning once, "
       "recorded by explain() (initial(..., check=False) silences it, declared)"),
    _d("conatus.model.initial_check.INITIAL_TAIL_REFUSE", 1e-2, "refusal",
       "callable initial data whose relative tail beyond the resolution, or whose jump between "
       "the ends of a periodic axis, exceeds it are clearly wrong and refused: an aliased mode or "
       "a function not periodic on the box (initial(..., check=False) accepts them, declared)"),
    _d("conatus.timestep.integrator.RUN_TAIL_TOL", 1e-3, "detection",
       "relative spectral tail (amplitude) of an unknown along a direction, measured at each "
       "finite-state check: above it the run is under-resolved, warned once and printed by "
       "explain() (1e-4 warned on the resolved KS run L = 22, N = 32)"),
    _d("conatus.timestep.integrator.RUN_TAIL_MIN_MODES", 3, "detection",
       "a direction whose tail window holds fewer coefficients is not judged by the run-time "
       "resolution check (one top mode is not a tail); explain() lists it"),
    _d("conatus.timestep.integrator.WHOLE_STEPS_TOL", 1e-9, "refusal",
       "one rule for a horizon that must be a whole number of steps (run(until=), the watch's dt "
       "setter, flowmap(T), shell Floquet, pipe orbits, edge tracking): |T/dt - round| <= it x "
       "max(1, T/dt), refused otherwise with the nearest whole choice"),
    _d("conatus.timestep.integrator.RUN_TAIL_FLOOR", 1e-12, "detection",
       "an unknown whose largest coefficient is below this fraction of the largest of the state "
       "is round-off-sized and not judged by the run-time resolution check"),
    _d("conatus.solvers.nlbvp.NEWTON_TRIVIAL_FRACTION", 1e-8, "detection",
       "a converged NewtonSolver solution whose nonlinear unknowns end below this fraction of "
       "the guess is flagged trivial=True (the trivial state u = 0), noted in explain()"),
    # data-driven
    _d("conatus.datadriven.spod.RANK_RTOL", 10.0, "refusal",
       "numerical rank of the SPOD cross-spectral density at a frequency (eigenvalues above it x "
       "max(n, N_b) x eps x λ_1(f)); nmodes above the smallest rank is refused"),
    _d("conatus.datadriven.norms.HERMITIAN_TOL", 1e-13, "refusal", _CHECK),
    _d("conatus.datadriven.dmd.RANK_NOTE_RTOL", 1e-12, "detection", _NOTE),
    _d("conatus.datadriven.dmd.ZERO_EIGENVALUE_TOL", 1e-14, "refusal",
       "DMD eigenvalue |μ| below it: the exact modes Y G / μ are undefined, refused with the count"),
    # verification report (pass thresholds, printed beside each measured value)
    _d("conatus.verify.report.RECONSTRUCTION_TOL", 1e-10, "check", _VERIFY),
    _d("conatus.verify.report.RESIDUAL_TOL", 1e-12, "check", _VERIFY),
    _d("conatus.verify.report.TAIL_TOL", 1e-8, "check", _VERIFY),
    _d("conatus.verify.report.TAYLOR_TOL", 1e-5, "check", _VERIFY),
    _d("conatus.verify.report.CRITICAL_GROWTH_TOL", 1e-10, "check", _VERIFY),
    # time stepping
    _d("conatus.timestep.control.DT_FACTOR_MIN", 0.2, "limiter",
       "adaptive step: smallest change factor per step (EmbeddedError.explain() prints it)"),
    _d("conatus.timestep.control.DT_FACTOR_MAX", 2.0, "limiter",
       "adaptive step: largest change factor per step (EmbeddedError.explain() prints it)"),
    # bifurcation and manifolds
    _d("conatus.bifurcation.normal_form:normal_form(kernel_tol)", 1e-8, "detection",
       "kernel of the Jacobian at the bifurcation point; its singular values are in the result"),
    _d("conatus.bifurcation.orbits.MARGINAL_MULTIPLIER_TOL", 1e-6, "detection",
       "orbit stability count: nontrivial |μ| > 1 + tol is unstable, ||μ| - 1| <= tol near-marginal "
       "(stored per point and listed by explain(), never counted); the multipliers are returned"),
    _d("conatus.bifurcation.codim2.L1_KERNEL_TOL", 1e-6, "detection",
       "kernel tolerance of the first Lyapunov coefficient along a Hopf curve (diagnostic column)"),
    _d("conatus.bifurcation.codim2.HOPF_PAIR_TOL", 1e-6, "refusal",
       "a point of a Hopf curve is kept only when the eigenvalues of J (QZ, independent of the "
       "augmented system) hold ±iω, ω = √κ, within it x max(1, |λ|max); κ <= 0 ends the curve"),
    _d("conatus.bifurcation.codim2.BT_KAPPA_TOL", 1e-6, "detection",
       "a zero-Hopf test zero with |ω²| below it x max(1, |λ|max²) is reported as the "
       "Bogdanov–Takens point, not a fold–Hopf point (the event keeps κ = ω²)"),
    _d("conatus.bifurcation.orbits.RESOLVED_REL", 1e-8, "detection",
       "orbit multipliers below it x max(1, |μ|max) are unresolved by the collocation product: "
       "counted and left out of the stability tests (explain() prints the count)"),
    _d("conatus.bifurcation.branching.KERNEL_SWITCH_TOL", 1e-3, "detection",
       "choice of the kernel basis vector giving φ ⟂ t0 at a branch point (the second when the "
       "first projects below it); fixes the sign of φ, hence the sense of direction=+1 (a "
       "convention of the SVD's kernel orientation: any value far from round-off and from 1 "
       "gives the same φ up to that sign; the other branch is direction=-1)"),
    _d("conatus.bifurcation.branching.ABE_ZERO_TOL", 1e-12, "detection",
       "c22 of the algebraic branching equation treated as zero (root α = 0); c is in Curve.switch"),
    _d("conatus.bifurcation.branching.TRANSCRITICAL_TOL", 1e-8, "detection", _NOTE),
    _d("conatus.bifurcation.branching.INVARIANT_TOL", 1e-9, "detection",
       "branch point on an S-invariant branch (symmetry-breaking pitchfork note)"),
    _d("conatus.bifurcation.branching.PARITY_TOL", 1e-7, "refusal",
       "kernel neither S-symmetric nor S-antisymmetric at a branch point: SymmetryError"),
    _d("conatus.bifurcation.normal_form.PARITY_TOL", 1e-8, "refusal",
       "kernel neither S-symmetric nor S-antisymmetric in normal_form: SymmetryError"),
    _d("conatus.manifolds.ecs_system:ecs_system(degree_tol)", 1e-8, "refusal", _CHECK),
    _d("conatus.manifolds.parametrization:unstable_manifold(degree_tol)", 1e-8, "refusal", _CHECK),
    _d("conatus.manifolds.parametrization:unstable_manifold(resonance_tol)", 1e-10, "refusal", _CHECK),
    _d("conatus.manifolds.snaking:restrict_system(invariance_tol)", 1e-12, "refusal", _CHECK),
    _d("conatus.manifolds.connections:find_connection(max_iterations)", 50, "iteration cap", _CAP),
    # algorithm controls: how fast, never how accurate (the residual is tested
    # against the required tol and returned)
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(max_iterations)", 40, "iteration cap",
       _CAP + " (NewtonNotConverged, with the stagnation diagnosis)"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(gmres_rtol)", 1e-3, "algorithm control",
       "inexact-Newton forcing term: relative GMRES tolerance of each Newton step (the Newton "
       "residual is tested against the required tol)"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(gmres_max_iterations)", 60,
       "algorithm control",
       "Krylov budget of each Newton step (a short step is hooked or retried, never accepted "
       "unless the residual decreases)"),
    _d("conatus.bifurcation.equilibria:newton_equilibrium(max_iterations)", 50, "iteration cap",
       _CAP),
    # budgets with declared defaults (rule the accuracy stays required)
    _d("conatus.io.checkpoint.CHECKPOINT_KEEP", 2, "storage count",
       "periodic checkpoints kept by sim.checkpoints / cn.io.CheckpointSeries (keep=): the latest "
       "and the one before it; changes no result"),
    _d("conatus.chaos.poincare.SECTION_MAX_ITERATIONS", 50, "iteration cap",
       "regula-falsi location of one section crossing (g_tol required); a crossing not located "
       "within it is refused (ConvergenceError)",
       ("conatus.chaos.poincare:poincare_section(max_iterations)",)),
    _d("conatus.floquet.roots.SECANT_MAX_ITERATIONS", 50, "iteration cap",
       "determinant secant of the Floquet root searches (tol required); a search ending on the "
       "cap raises ConvergenceError with the last iterate (allow_unconverged=True returns it "
       "flagged converged=False), never a converged status",
       ("conatus.floquet.hill_general:secant_root(max_iterations)")),
    _d("conatus.floquet.roots.SECANT_STALL_STEPS", 6, "iteration cap",
       "find_root (Hill, cascade): the secant stops when its step has not halved for this many "
       "consecutive iterations, reported as a stall (not converged), never as convergence"),
    _d("conatus.solvers.nlbvp.NEWTON_MAX_ITERATIONS", 50, "iteration cap",
       "Newton iterations of a boundary-value solve (tol required); reaching it raises "
       "NewtonNotConverged with the residual of every iterate",
       ("conatus.solvers.nlbvp:NewtonSolver.__init__(max_iterations)",
        "conatus.solvers.periodic:PeriodicNewtonSolver.__init__(max_iterations)")),
    # contour methods (conatus.contour)
    _d("conatus.contour.beyn.RESIDUAL_TOL", 1e-13, "refusal",
       "acceptance of every root of beyn() and of m.eigenvalues(region=): residual "
       "||T(z) v|| / (s(z) ||v||) at most this (s(z) per operator: BeynResult.residual_definition), "
       "else the root is not returned (and the call is refused if the accepted roots do not add "
       "up to the count)",
       ("conatus.contour.beyn:beyn(residual_tol)",
        "conatus.contour.evp:region_eigenvalues(residual_tol)")),
    _d("conatus.contour.beyn.POLISH_MAX_ITERATIONS", 20, "iteration cap",
       "nonlinear inverse iterations of one Ritz value; the residual of every root is checked "
       "against residual_tol whatever the number of iterations",
       ("conatus.contour.beyn:beyn(polish_max_iterations)",)),
])

_ALG = "algorithm control: changes the path or the speed, not the accuracy target"

#: Declared defaults of algorithm controls whose names are not tolerance-like:
#: trust radii, step growth, budgets, FD step rule, DMD amplitudes, finite-state cadence.
ALGORITHM_DEFAULTS: dict[str, Declared] = dict([
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(delta0)", 1.0, "algorithm control",
       "initial trust radius of the hookstep, in the norm of the unknowns (radii='absolute') or "
       "as a fraction of ‖z0‖ (radii='relative'); the radii used are in the result's explain()"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(delta_min)", 1e-10, "algorithm control",
       "smallest trust radius (read as delta0); no decrease there raises NewtonNotConverged"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(delta_max)", 10.0, "algorithm control",
       "largest trust radius (read as delta0)"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(radii)", "absolute", "algorithm control",
       "how the trust radii are read: 'absolute' (v0.1 meaning, unchanged results) or 'relative' "
       "to the norm of the starting unknowns (the absolute defaults "
       "suit states of norm O(1) only)"),
    _d("conatus.ecs.newton.HOOKSTEP_NORM_MIN", 1e-2, "detection",
       "default absolute radii on a start of norm ‖z0‖ below it: TrustRadiusWarning (pass "
       "radii='relative' or explicit radii; nothing is changed)"),
    _d("conatus.ecs.newton.HOOKSTEP_NORM_MAX", 1e2, "detection",
       "default absolute radii on a start of norm ‖z0‖ above it: TrustRadiusWarning"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(rho_bad)", 0.1, "algorithm control",
       "actual/predicted decrease below it: the trust radius is halved and the step retried"),
    _d("conatus.ecs.newton:NewtonKrylovHookstep.__init__(rho_good)", 0.75, "algorithm control",
       "actual/predicted decrease above it after a hook: the trust radius doubles"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(adapt)", 1.3, "algorithm control",
       "arclength step growth factor after a fast corrector (halved on failure)"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(fast_iterations)", 3,
       "algorithm control", "corrector iterations counted as fast (the step then grows)"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(max_newton)", 12,
       "iteration cap", "corrector budget: a failure halves the step, never accepts the point"),
    _d("conatus.continuation.core:ContinuationSettings.__init__(dp_fd)", "auto",
       "algorithm control", "central finite-difference step of G_p: eps^(1/3) max(1, |p|) "
       "(FD_STEP_REL), the balance of truncation and rounding; an analytic G_p is used when given"),
    _d("conatus.continuation.krylov:KrylovContinuationSettings.__init__(adapt)", 1.3,
       "algorithm control", "arclength step growth factor"),
    _d("conatus.continuation.krylov:KrylovContinuationSettings.__init__(fast_iterations)", 3,
       "algorithm control", "corrector iterations counted as fast"),
    _d("conatus.continuation.krylov:KrylovContinuationSettings.__init__(p_weight)", 1.0,
       "algorithm control", "weight of the parameter in the arclength norm"),
    _d("conatus.continuation.krylov:KrylovContinuationSettings.__init__(fold_refinements)", 3,
       "algorithm control", "step refinements around a detected fold"),
    _d("conatus.continuation.krylov:KrylovContinuationSettings.__init__(fold_refine_factor)", 4.0,
       "algorithm control", "step division per fold refinement"),
    _d("conatus.datadriven.dmd:dmd(amplitudes)", "first", "algorithm control",
       "least-squares fit of the amplitudes to the first snapshot (Schmid 2010); the modes and "
       "eigenvalues do not depend on it; 'optimal' fits every snapshot"),
    _d("conatus.solvers.evp_modes.NORMALIZATION_ZERO_REL", 1e-10, "detection",
       "res.mode(i): an unknown whose largest |coefficient| is below this fraction of the "
       "eigenvector's is zero for the normalisation (round-off is never scaled to 1)"),
    _d("conatus.solvers.nlbvp:NewtonSolver.__init__(damping)", 1.0, "algorithm control",
       "Newton step factor of a boundary-value solve: the full step; a smaller "
       "one only changes the path to the required tol"),
    _d("conatus.solvers.periodic:PeriodicNewtonSolver.__init__(damping)", 1.0,
       "algorithm control", "as NewtonSolver"),
    _d("conatus.nonmodal.energy:max_growth(n_scan)", 64, "algorithm control",
       "points of the uniform scan bracketing the maximum of G(t) before bounded Brent to the "
       "required xtol (MAX_GROWTH_N_SCAN); a narrow peak needs more, stated with n_scan="),
    _d("conatus.growth.optimize:optimal_growth(max_evals)", 200, "iteration cap",
       "Nelder-Mead evaluation budget; reaching it gives the status "
       "INCOMPLETE_BUDGET, never JOINT_CONVERGED"),
    _d("conatus.growth.optimize:optimal_growth(simplex_step)", (0.005, 0.001, 0.005),
       "algorithm control", "initial Nelder-Mead simplex edges in (log kx, t0, log T): relative "
       "for kx and T, in units of the forcing period for t0 (OPTIMAL_GROWTH_SIMPLEX_STEP); the "
       "path of the search only, xtol and fatol are required; printed by OptimalGrowth.explain()"),
    _d("conatus.ecs.solutions:equilibrium_eigenvalues(dense_max)", 2000, "algorithm control",
       "dense QZ up to this active size (every eigenvalue, marginal ones included), ARPACK at "
       "the required arnoldi_tol above it"),
    _d("conatus.ecs.solutions:floquet_multipliers(dense_max)", 2000, "algorithm control",
       "as equilibrium_eigenvalues"),
    _d("conatus.ecs.solutions:symmetry_breaking_check(dense_max)", 2000, "algorithm control",
       "largest size of this dense-only check; above it the check is refused"),
    _d("conatus.timestep.integrator.FINITE_CHECK_EVERY", 100, "detection",
       "finite-state check of the time steppers every 100 steps, at the end of each run call and "
       "before each write (check_finite=False turns it off); NaN/inf raises NumericalError"),
    _d("conatus.solvers.parametric.OVER_EDGE_FRACTION", 1e-6, "detection",
       "critical(over=): a minimum of the neutral value within this fraction of the width of the "
       "over interval from one of its ends, or within the bounded minimizer's own tolerance "
       "3 (xtol max|end| + sqrt(eps) |end|) if larger, is at the edge (a constrained minimum): "
       "CorrectnessWarning, and the CriticalPoint's repr and explain() say it"),
    _d("conatus.solvers.parametric.OVER_SCAN_POINTS", 9, "detection",
       "critical(over=): the neutral value is computed at this many equally spaced points of the "
       "over interval (ends included, the least stable mode selected afresh at each); every "
       "local minimum of the scan is refined by bounded Brent and the smallest refined value is "
       "returned; a minimum narrower than the scan spacing is not seen (said in the log)"),
    _d("conatus.solvers.parametric.BRACKET_SCAN_POINTS", 8, "detection",
       "critical(): when the growth of the least stable admissible mode has the same sign at both "
       "ends of the bracket (refused), this many interior points are scanned (geometric for a "
       "positive bracket wider than a factor 4) to name the sub-brackets holding neutral points"),
    # contour methods (conatus.contour)
    _d("conatus.contour.count.PHASE_STEP_MAX", math.pi / 4, "detection",
       "principal-increment bound of the argument-principle walk: a step (and each of its halves) "
       "is accepted only when det T turns by at most pi/4 over it and log det T is resolved by it; "
       "else it is bisected (count(phase_step=))"),
    _d("conatus.contour.count.INITIAL_POINTS", 16, "algorithm control",
       "points per edge of the box before any refinement of the walk (count(initial_points=))"),
    _d("conatus.contour.count.MAX_EVALUATIONS", 20000, "iteration cap",
       "determinant evaluations of one walk; reaching it refuses the count with the place where "
       "the walk stalled (an eigenvalue on the boundary, a discontinuous determinant)"),
    _d("conatus.contour.beyn.CLUSTER_RADII", (1e-6, 1e-4, 1e-2, 1e-1), "detection",
       "linkage radii (fractions of the box half-diagonal) tried in turn to group Beyn's polished "
       "roots: 1e-6 merges duplicates and a double root split by rounding (~ sqrt(eps)), the "
       "larger radii a defective eigenvalue of order m split on a circle of radius ~ eps^(1/m). "
       "Every group, a single root included, is counted by the argument principle on its own "
       "square; a grouping is accepted only when every count is >= 1 and they add up to Z, "
       "else the round fails (never a root reported without its own count)"),
    _d("conatus.contour.beyn.SAME_OPERATOR_TOL", 1e-8, "refusal",
       "beyn(counted=...): det T at the boundary points the count recorded must agree with the "
       "counted operator's (unit phase absolute, log modulus relative) to it, else the count is "
       "of another operator and refused"),
    _d("conatus.contour.beyn.SEPARATION_ROUNDING", 1e3, "detection",
       "Beyn's close roots are separated on squares of their own; a square below this x eps x "
       "max(box radius, largest |root|) is not tried (det T is rounding noise there), the "
       "group stays a cluster"),
    _d("conatus.contour.beyn.QUADRATURE_POINTS", 32, "algorithm control",
       "Gauss-Legendre nodes per edge for Beyn's moments, doubled at each new round"),
    _d("conatus.contour.beyn.BEYN_MARGIN", 2, "algorithm control",
       "probe columns x moments of Beyn's method above the count Z (doubled at each new round)"),
    _d("conatus.contour.beyn.BEYN_MAX_ROUNDS", 3, "iteration cap",
       "rounds of moments on one box (more nodes, columns and moments each) before it is cut"),
    _d("conatus.contour.beyn.BEYN_MAX_SPLITS", 3, "iteration cap",
       "levels of cuts of a box whose rounds did not account for its count (each half counted, the "
       "counts adding up, and solved) before beyn() is refused"),
    _d("conatus.contour.operators.DERIVATIVE_STEP", 6e-6, "algorithm control",
       "relative step of the central difference for T'(z) of a callable given without its "
       "derivative: used by the polish only (convergence rate), not by the acceptance test"),
])


#: Internal deciding literals (float literals inside the library's comparisons, G4): a second
#: table, by meaning, with the functions that use each value (``conatus._tolerances_internal``;
#: the development test suite). Not part of ``__all__``.
from ._tolerances_internal import INTERNAL  # noqa: E402,F401


def ncc_floor_text(plan: Any, residual: float | None = None) -> str | None:
    """The accuracy floor that the coefficients of ``plan`` set on a result (P4.4), or None when every coefficient is exact or at rounding level.

    Why NCC_TAIL_TOL = 1e-8 coexists with residuals of 1e-12: a residual measures how well the
    *discrete* problem is solved, with the coefficients as represented (the truncated Chebyshev
    series of each NCC); the NCC tail bounds how well that representation matches the given
    coefficient. The answer of the continuous problem is therefore known to about the larger of the
    two (times the conditioning of the problem, not estimated here), never to the residual alone.
    1e-8 is the level at which the coefficients of the suite are either clearly resolved
    (1e-10..4e-9) or clearly not (1e-7..1e-1); a result that needs more digits declares
    ``ncc_tail_tol=`` smaller and a resolution that reaches it."""
    reps = getattr(plan, "ncc_reports", None) or {}
    tails = [(max(float(r.tail), float(getattr(r, "dropped_max", 0.0) or 0.0)), r.name)
             for r in reps.values()
             if not getattr(r, "exact", False) and not getattr(r, "rounding", 0.0) > 0]
    if not tails:
        return None
    t, name = max(tails)
    res = f"the residual {residual:.1e} measures the discrete problem; " if residual is not None else ""
    return (f"accuracy floor: {res}the coefficients are represented to a relative {t:.1e} "
            f"(largest NCC tail, {name}; refused above NCC_TAIL_TOL = {NCC_TAIL_TOL:g} unless "
            "declared): digits of the continuous problem beyond about that level are not claimed")


def explain() -> str:
    """The rule and the table of declared defaults (required tolerances have no default)."""
    lines = ["Tolerances (v0.1 rule): stopping, filtering and time-step tolerances are REQUIRED (an "
             "OptionError says what to write); the declared defaults below only detect, refuse, "
             "cap iterations, or stop at the attainable floor with the residual returned:"]
    for d in DECLARED.values():
        lines.append(f"  {d.where} = {_fmt(d.value)}  [{d.role}] {d.why}")
    lines.append("Algorithm controls (they default and are declared; accuracy "
                 "tolerances are required):")
    for d in ALGORITHM_DEFAULTS.values():
        lines.append(f"  {d.where} = {_fmt(d.value)}  [{d.role}] {d.why}")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return repr(v) if isinstance(v, str) else f"{v:g}"
