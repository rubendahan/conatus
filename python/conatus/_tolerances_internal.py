"""Internal deciding constants: the float literals inside comparisons of the library (G4).

:data:`conatus.tolerances.DECLARED` lists the declared defaults of public arguments and the named
module constants. The library also decides with literals written inside its comparisons
(``abs(L - 2 * pi) > 1e-12``, ``sv[-1] <= 1e-14 * sv[0]``): whether two lengths are the same,
whether a matrix is singular, whether a duration is a whole number of steps. They are listed here,
grouped by what they decide, so that every such number has a stated reason.

Each :class:`Internal` entry names one value and the functions (``module:qualname``) whose
comparisons use it with that meaning. The development test suite reads every
comparison of ``python/conatus`` (and the ``rtol``/``atol`` of ``np.allclose``/``np.isclose``,
which must be written out: NumPy's implicit ``rtol = 1e-5`` is a hidden literal) and fails when a
float literal other than 0 and 1 is neither listed here for that function nor exempted in the
test with a reason (underflow guards, sentinels, display rules), and when an entry here no longer
matches the code. Changing such a literal means changing this table in the same commit.

Nothing here is read by the library at run time: the table documents the literals where they are.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Internal", "INTERNAL", "explain"]


@dataclass(frozen=True)
class Internal:
    """One internal deciding literal: its value, what it decides, why, and where it is used."""

    name: str
    value: float
    role: str          # "rounding equality" | "structural zero" | "rank" | "integer ratio" | ...
    why: str
    where: tuple[str, ...]   # "conatus.module:Qualname" of every function using it this way


_EQ = ("two numbers that are equal by construction (lengths, bounds, wavenumbers, positions, "
       "symmetry centres, period = n dt, a stage time inside its step) are compared to this "
       "relative bound: the rounding of the few operations that produced them, never a physical "
       "tolerance; a difference above it is a different input and is refused or reported")
_ZERO = ("a part that vanishes by construction (the imaginary part of real data, the component "
         "outside a declared subspace or layout, an entry outside the structure of a block) is "
         "accepted as zero below this fraction of the scale of the data, which is the rounding "
         "of its computation; above it the input is refused, never truncated")
_RANK = ("rank or breakdown decision on singular values, pivots or a Krylov subdiagonal relative "
         "to the largest: below it the matrix is singular to working precision")

_ENTRIES = (
    Internal("ROUNDING_EQUAL", 1e-12, "rounding equality", _EQ, (
        "conatus.adjoint.stagetime:StageTimeFused.coefficient_time_of",
        "conatus.bifurcation.orbits:CollocationMesh.explain",
        "conatus.discretize.curvilinear_symmetry:_check_generator",
        "conatus.discretize.plan:Plan._reflection_kind",
        "conatus.discretize.polar_plan:PolarPlan.group",
        "conatus.discretize.ultraspherical:_round_up",
        "conatus.ecs.edge:EdgeTracker.fate",
        "conatus.ecs.fourier_ops:AxisOps.__init__",
        "conatus.ecs.pipe:_neutral_mode",
        "conatus.ir.domain:_check_curvilinear",
        "conatus.ir.domain:_check_spherical",
        "conatus.ir.symmetry:AzimuthalMFold.centre",
        "conatus.ir.symmetry:Reflection.__init__",
        "conatus.ir.symmetry:Reflection.centre",
        "conatus.ir.symmetry:Reflection.is_fixed",
        "conatus.ir.terrain:_unresolved_metric",
        "conatus.manifolds.connections:find_connection",
        "conatus.manifolds.trajectory:SolverIntegrator.integrate.restep",
        "conatus.methods:_frac",
        "conatus.model.model:_select_group",
        "conatus.model.linearization:Linearization.group",
        "conatus.model.linearization:Linearization.parametric.group_of",
        "conatus.model.linearization:sincos_eigenvalues",
        "conatus.model.sincos:extend_to_period.ext",
        "conatus.solvers.evp:EVPResult.explain",
        "conatus.solvers.evp_modes:_evaluation_matrix",
        "conatus.timestep.tau_modes:check_tau_modes",
    )),
    Internal("ROUNDING_EQUAL_14", 1e-14, "rounding equality",
             _EQ + " (data produced by one or two operations: a bound at 0, a symmetric "
             "interval, a fraction written in decimal)", (
                 "conatus.ecs.subspace:AxisAction.signs",
                 "conatus.ir.bases:CompoundChebyshev.piece_of",
                 "conatus.ir.bases:_frac",
                 "conatus.ir.domain:_check_curvilinear",
                 "conatus.ir.domain:_check_spherical",
             )),
    Internal("RELATIVE_FLOOR", 1e-14, "floor",
             "the smallest relative step or tolerance a double-precision iteration can resolve "
             "(about 50 eps): a bisection stops there, a smaller requested rtol is refused", (
                 "conatus.ecs.edge:ThresholdBisection.__init__",
                 "conatus.ecs.newton:_hook",
             )),
    Internal("SEARCH_WIDTH_13", 1e-13, "floor",
             "a golden-section search on a periodic shift stops when its bracket is narrower than "
             "this fraction of the period (about 500 eps: the position is resolved to rounding); "
             "the 80-iteration budget is the other stop", (
                 "conatus.ecs.solutions:_min_shift_distance",
             )),
    Internal("STRUCTURAL_ZERO", 1e-12, "structural zero", _ZERO, (
        "conatus.continuation.ks:KSDomainContinuation.run",
        "conatus.discretize.curvilinear_symmetry:_ball_reg_sign",
        "conatus.discretize.curvilinear_symmetry:check_rows_in_subspace",
        "conatus.discretize.plan:Plan._ncc_coeffs",
        "conatus.discretize.plan:Plan.solve",
        "conatus.discretize.polar_plan:PolarPlan._axisymmetric",
        "conatus.discretize.polar_plan:PolarPlan.boundary_tensor",
        "conatus.discretize.sphere_blocks:ball_map",
        "conatus.discretize.sphere_plan:SpherePlan.ncc_data",
        "conatus.discretize.sphere_plan:SpherePlan.surface_tensor",
        "conatus.ecs.shooting:ShootingRPO.__init__",
        "conatus.ecs.solutions:find_equilibrium",
        "conatus.growth.periodic:PeriodicOperator.energy_factor",
        "conatus.manifolds.connections:unstable_left_basis",
        "conatus.manifolds.orbits:orbit_manifold_seeds",
        "conatus.manifolds.parametrization:unstable_manifold",
        "conatus.model.run:Simulation.evaluate",
        "conatus.timestep.polar:PolarGlobalSystem.gather_state",
        "conatus.timestep.real_projection:M0RealProjection.__init__",
        "conatus.timestep.system:GlobalSystem.gather_state",
    )),
    Internal("STRUCTURAL_ZERO_13", 1e-13, "structural zero",
             _ZERO + " (complex views and real projections of assembled blocks)", (
                 "conatus.discretize.polar_eval:PolarSpace.embed",
                 "conatus.timestep.spectral:SpectralEvaluator.to_coeffs",
                 "conatus.timestep.system:GlobalSystem._complex_view",
                 "conatus.timestep.system:GlobalSystem._multi_complex_views",
             )),
    Internal("STRUCTURAL_ZERO_14", 1e-14, "structural zero",
             _ZERO + " (coefficients read off exact operators: traces, radial NCC expansions, "
             "cross blocks of a pencil, a pairing that vanishes by symmetry)", (
                 "conatus.discretize.polar_plan:PolarPlan._axisymmetric",
                 "conatus.discretize.polar_plan:PolarPlan.ncc_radial",
                 "conatus.discretize.polar_plan:PolarPlan.ncc_spin_coeffs",
                 "conatus.discretize.polar_plan:_PolarCompiler._trace_dot",
                 "conatus.discretize.sphere_compiler:_SphereCompiler._trace_dot",
                 "conatus.discretize.sphere_compiler:_SphereCompiler._trace_dot.f",
                 "conatus.discretize.sphere_plan:SpherePlan._angular",
                 "conatus.discretize.sphere_plan:SpherePlan.ball_ncc_legendre",
                 "conatus.discretize.sphere_plan:SpherePlan.ncc_data",
                 "conatus.sensitivity.modes:mode_pair",
             )),
    Internal("RANK_12", 1e-12, "rank", _RANK, (
        "conatus.ecs.shooting:span_fractions",
        "conatus.growth.periodic:operator_gain",
        "conatus.manifolds.orbits:section_projector",
    )),
    Internal("RANK_13", 1e-13, "rank", _RANK + " (energy weights: a Gram matrix must be definite)", (
        "conatus.nonmodal.pencil:reduce_pencil",
    )),
    Internal("RANK_14", 1e-14, "rank", _RANK + " (Arnoldi breakdown, LU pivots of a weight)", (
        "conatus.adjoint.flowmap:FlowMap.vjp_physical",
        "conatus.ecs.newton:arnoldi_gmres",
        "conatus.ecs.shooting:arnoldi_eigs",
        "conatus.linearize.inner_product:InnerProduct.adjoint",
    )),
    Internal("RANK_10", 1e-10, "rank",
             "the radiation selector of the Hill far field is degenerate below it (refused: the "
             "boundary rows would not select the decaying solutions)",
             ()),
    Internal("NULL_VECTOR_SUPPORT", 1e-6, "report",
             "entries of a null vector above this fraction of its largest are the unknowns named "
             "in a rank-deficiency report (a hint only; the rank itself is decided by the SVD)",
             ("conatus.discretize.plan:Plan.rank_report",)),
    Internal("INTEGER_RATIO", 1e-9, "integer ratio",
             "a duration that must be a whole number of steps (T / dt), or a wavenumber that must "
             "be a whole multiple of 2 pi / L, is accepted within this relative bound of the "
             "nearest integer (the ratio of two decimal inputs, a few hundred eps at most); "
             "otherwise refused, dt or k is never changed (also the upper end of a stage time "
             "inside its step; the whole-steps refusals of run(until=), NLOP, FTLE and the "
             "terrain run use timestep.integrator.is_whole, WHOLE_STEPS_TOL, the same value)", (
                 "conatus.adjoint.stagetime:StageTimeFused.coefficient_time_of",
                 "conatus.manifolds.trajectory:SolverIntegrator.flow",
                 "conatus.manifolds.trajectory:SolverIntegrator.integrate",
                 "conatus.model.linearization:Linearization._N",
                 "conatus.translate.dedalus_result:DedalusTranslation.steps",
             )),
    Internal("SAME_VALUE", 1e-9, "classification",
             "two computed values taken as the same (uniform output spacing, equal ratios of a "
             "temporal convergence study, an optimum on a bound of the search box); a label of "
             "the result, never a change of a value",
             ("conatus.growth.optimize:optimal_growth", "conatus.model.watch:_is_uniform",
              "conatus.verify.uncertainty:_temporal")),
    Internal("SAME_STAGNATION", 1e-3, "classification",
             "two stalled travelling-wave searches of one solver whose final relative residuals "
             "and final wave speeds both agree to this relative bound stopped at the same "
             "stagnation point (the residual is printed to 4 digits; a search cut by its budget "
             "while the residual still fell is not compared); said in the refusal, nothing "
             "retried",
             ("conatus.ecs.solutions:_note_repeated_stagnation",)),
    Internal("BORDER_RTOL", 1e-5, "classification",
             "optimal_growth reports BORDER when the optimum lies within this relative distance "
             "of a bound of the box (NumPy's default rtol, written out)",
             ("conatus.growth.optimize:optimal_growth",)),
    Internal("CHECK_10", 1e-10, "verification",
             "an object built from exact data is verified to this relative bound before use "
             "(ball regularity map, the harmonic step map of a periodic propagator, the parity "
             "of data on a sine/cosine axis, a first-Fourier-mode slice template); refused above",
             ("conatus.discretize.sphere_blocks:ball_map", "conatus.growth.propagator:propagator",
              "conatus.model.sincos:data_parity", "conatus.model.sincos:extend_to_period.ext",
              "conatus.slices.fourier:FirstFourierModeSlice.__init__")),
    Internal("CHECK_11", 1e-11, "verification",
             "the interpolated assembly of a group is checked against the direct assembly of "
             "one group to this relative bound (refused above)",
             ("conatus.discretize.plan:Plan._assemble_interpolated",)),
    Internal("CHECK_8", 1e-8, "verification",
             "a computed property that holds exactly in theory is accepted to this bound (a real "
             "field computed through complex arithmetic, the leak of a projection, a symmetry "
             "eigenvalue on the unit circle, a match of two exponents, a conjugate pair, a "
             "multiple eigenvalue); the measured value is in the message or the note", (
                 "conatus.discretize.polar:Disk.bessel_modes",
                 "conatus.discretize.terrain_plan:TerrainPlan.to_grid",
                 "conatus.growth.periodic:operator_floquet",
                 "conatus.manifolds.parametrization:unstable_manifold",
                 "conatus.nonmodal.energy:galerkin_numerical_abscissa",
                 "conatus.sensitivity.modes:mode_pair",
                 "conatus.timestep.real_projection:M0RealProjection.__init__",
                 "conatus.timestep.symmetry_projection:ShiftReflectProjection.__init__",
             )),
    Internal("POD_CONDITION_NOTE", 1e-7, "detection",
             "method of snapshots: sigma_r / sigma_1 below it (about sqrt(eps)) means the "
             "smallest modes lose digits (squared condition); a note, the modes are returned",
             ("conatus.datadriven.pod:pod",)),
    Internal("PENCIL_GAP", 1e4, "refusal",
             "the finite and infinite eigenvalues of a reduced pencil must be separated by this "
             "factor; a smaller gap is refused (the split would be arbitrary)",
             ("conatus.nonmodal.pencil:reduce_pencil",)),
    Internal("ROUNDING_FACTOR", 1e3, "floor",
             "1e3 x eps (or 1e3 x the requested tol) is the rounding level of a computed "
             "quantity, the convention of NCC_ROUNDING and IMAG_ROUNDOFF: an imaginary part "
             "below it is rounding; a Newton or tracking update below it has converged",
             ("conatus.solvers.nlbvp:_real_part", "conatus.solvers.parametric:ParametricEVP.track",
              "conatus.solvers.periodic:PeriodicNewtonSolver.solve",
              "conatus.solvers.periodic:_negligible_imag")),
    Internal("DECAY_SLOPE", 1e-3, "detection",
             "a spectral envelope whose fitted log10 slope is not below -1e-3 per mode is not "
             "decaying: no resolution can be suggested from it (said, never guessed)",
             ("conatus.discretize.ultraspherical:suggested_size", "conatus.ir.terrain:_unresolved_metric")),
    Internal("DISTINCT_EXPONENT", 1e-3, "detection",
             "two folded Floquet exponents closer than this (relative) are candidates for the "
             "same mode found from two targets; their eigenvector overlap decides (OVERLAP)",
             ("conatus.growth.periodic:operator_floquet.shifted_copy",)),
    Internal("OVERLAP", 0.99, "detection",
             "eigenvector overlap above which two candidates are the same mode (as "
             "NEUTRAL_OVERLAP_MIN and translation_overlap_min)",
             ("conatus.growth.periodic:operator_floquet.shifted_copy",)),
    Internal("METRIC_KEEP", 0.9, "refusal",
             "a coordinate map whose metric needs more than 90 % of the Chebyshev coefficients "
             "sampled is not resolved by them and is refused",
             ("conatus.ir.maps:CoordinateMap.metric_coeffs",)),
    Internal("STALL_FACTOR", 0.5, "detection",
             "an iteration that does not halve its residual (Newton-Krylov hookstep) or its step "
             "(determinant secants) is stalling; reported (the hookstep shrinks its trust "
             "region), never accepted as converged",
             ("conatus.ecs.newton:NewtonKrylovHookstep.solve",
              "conatus.floquet.roots:determinant_secant")),
    Internal("WINDING_INTEGER", 1e-6, "refusal",
             "the argument-principle winding (the sum of the phase increments of det T along the "
             "boundary, closed on the first point evaluated again) is an integer up to rounding by "
             "construction (about 1e-13); further than 1e-6 from one, det T does not return the "
             "same value at the same point and the count is refused",
             ("conatus.contour.count:count",)),
    Internal("WALK_MIN_STEP", 1e-12, "refusal",
             "a boundary step shorter than 1e-12 of the perimeter whose phase still jumps: a zero "
             "on the boundary within rounding or a discontinuous determinant; the count is refused",
             ("conatus.contour.count:_walk",)),
    Internal("WALK_BEND_MIN_STEP", 1e-10, "floor",
             "the bend test of the boundary walk (operators that may have poles) "
             "halves steps down to 1e-10 of the perimeter and no further, so that rounding noise "
             "in log det T never drives the walk to the 1e-12 refusal by itself",
             ("conatus.contour.count:_walk",)),
    Internal("HANKEL_RANK", 1e-10, "rank",
             "Beyn's method: singular values of the block Hankel moment matrix above 1e-10 of the "
             "largest are its numerical rank (the Gauss-Legendre moments are accurate to about "
             "1e-12..1e-14 with eigenvalues clear of the edges); every Ritz value of that rank is "
             "polished and checked by its residual, so a noise direction kept costs time",
             ("conatus.contour.beyn:_rounds",)),
    # --- invariance battery (tests/invariance, audit invariance): the bounds of the
    # metamorphic relations. They decide no library result; each says why an equivalent
    # computation may differ from the original by that much and no more. The infra test checks
    # that every function named here reads the entry by name.
    Internal("INVARIANCE_ORDER", 0.3, "invariance test",
             "observed global order log2(e(dt) / e(dt/2)) against the stated order: three step "
             "sizes in the asymptotic range of a smooth problem, the pre-asymptotic drift of the "
             "ratio is below 0.2 for the problems of the battery; an order off by one (a start-up "
             "or a boundary convention) is 3 times this bound", (
                 "tests.invariance.test_inv_time_order:test_logistic_observed_order_is_the_stated_order",
                 "tests.invariance.test_inv_time_order:test_moving_walls_observed_order_is_the_stated_order",
                 "tests.invariance.test_inv_time_order:test_moving_walls_stated_order_is_achieved",
                 "tests.invariance.test_inv_time_order:test_mapped_box_observed_order_is_the_stated_order",
             )),
    Internal("INVARIANCE_ROUNDING", 1e-11, "invariance test",
             "relative difference of two computations equal in exact arithmetic that execute "
             "different floating-point operations (an equation multiplied by a constant, a "
             "rescaled unit, a reordered batch): about 1e3 eps of accumulated rounding over a "
             "few hundred steps or one conditioned solve; a numerical difference of method is "
             "orders of magnitude above it", (
                 "tests.invariance.test_inv_dynamics:test_lyapunov_reorthonormalisation_and_blocks",
                 "tests.invariance.test_inv_dynamics:test_lyapunov_trajectory_independent_of_batch_place",
                 "tests.invariance.test_inv_dynamics:test_pod_relations",
                 "tests.invariance.test_inv_scaling:test_ivp_equation_scaling",
                 "tests.invariance.test_inv_scaling:test_ivp_amplitude_scaling",
                 "tests.invariance.test_inv_scaling:test_nlbvp_equation_scaling",
                 "tests.invariance.test_inv_scaling:test_units_heat_space_time",
                 "tests.invariance.test_inv_solver_paths:test_ivp_paths_rb2d",
                 "tests.invariance.test_inv_symmetry:test_ks_translation_and_reflection",
                 "tests.invariance.test_inv_symmetry:test_rb_reflection_x",
                 "tests.invariance.test_inv_symmetry:test_disk_rotation",
             )),
    Internal("INVARIANCE_EIGENVALUE", 1e-9, "invariance test",
             "relative difference of eigenvalues of two equivalent pencils (scaled rows, rescaled "
             "eigenvalue, another solver path): backward error 1e-15 times the eigenvalue "
             "condition number of the non-normal Orr-Sommerfeld operator (1e4..1e5 at Re = 1e4, "
             "tests/evp/test_orr_sommerfeld.py)", (
                 "tests.invariance.test_inv_dynamics:test_dmd_relations",
                 "tests.invariance.test_inv_dynamics:test_floquet_phase_and_paths",
                 "tests.invariance.test_inv_scaling:test_evp_equation_scaling",
                 "tests.invariance.test_inv_scaling:test_evp_eigenvalue_unit",
                 "tests.invariance.test_inv_scaling:test_generalized_eigenvalues_scaling",
                 "tests.invariance.test_inv_solver_paths:test_evp_dense_shift_invert_contour",
                 "tests.invariance.test_inv_solver_paths:test_polar_dense_vs_shift_invert",
             )),
    Internal("INVARIANCE_ROW_SCALING", 1e-7, "invariance test",
             "relative difference (grid values of T and u against max |T|) of a 2-D "
             "Rayleigh-Benard run with its momentum equation multiplied by 1e-2..1e5: the "
             "bordered banded solve does not equilibrate the rows of the system, so their "
             "relative scale enters the rounding (measured 3.0e-8 for 1e-2 and 7.5e-10 for "
             "1e2..1e5 after 40 steps); a fragile spot of docs/audits/invariance.md, not a "
             "tolerance of the library", (
                 "tests.invariance.test_inv_scaling:test_ivp_equation_scaling_rb2d",
             )),
    Internal("INVARIANCE_ADJOINT", 1e-9, "invariance test",
             "relative defect of <J V, W> = <V, J^T W> and of two transposes of one map (matrix-"
             "free and dense oracle): both sides are sums of products over 7..9 steps of a "
             "2-D map with 64 unknowns, measured up to 3e-10 (CNAB2 with the RK443 start-up, "
             "time-dependent data; tests/adjoint use 1e-11..1e-9); a wrong term of the transpose "
             "is O(1)", (
                 "tests.invariance.test_inv_adjoint:test_flowmap_identities_every_scheme",
                 "tests.invariance.test_inv_adjoint:test_flowmap_time_dependent_from_t0",
                 "tests.invariance.test_inv_adjoint:test_flowmap_polynomial_vs_dense_transpose",
             )),
    Internal("INVARIANCE_TAYLOR_ORDER", 1.8, "invariance test",
             "Taylor remainder |Phi(X + e V) - Phi(X) - e J V| halves twice: observed order "
             "above 1.8 (exactly 2 for an exact tangent; a wrong tangent gives 1)", (
                 "tests.invariance.test_inv_adjoint:test_flowmap_identities_every_scheme",
                 "tests.invariance.test_inv_adjoint:test_flowmap_time_dependent_from_t0",
             )),
    Internal("INFINITE_EIGENVALUE", 1e14, "detection",
             "eigenvalues of a tau pencil above it in modulus are its infinite eigenvalues "
             "(beta ~ 0), not tau modes to report",
             ("conatus.timestep.tau_modes:_find_tau_modes",)),
)

#: Every internal deciding literal, by name (see the module docstring).
INTERNAL: dict[str, Internal] = {e.name: e for e in _ENTRIES}


def explain() -> str:
    """The table, one line per entry with its functions."""
    lines = ["Internal deciding literals of Conatus (conatus._tolerances_internal):"]
    for e in INTERNAL.values():
        lines.append(f"  {e.name} = {e.value:g}  [{e.role}] {e.why}")
        lines += [f"      {w}" for w in e.where]
    return "\n".join(lines)
