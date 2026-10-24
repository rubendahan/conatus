"""EVP entry point on a Plan: group/subspace selection, dense or shift-invert, drift test."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import structural_rank

from ..discretize.plan import Plan
from ..ir.problem import EVP
from .._deprecation import renames
from ..tolerances import REQUIRED
from .evp import EVPResult, check_drift_tol, drift_filter, solve_dense, solve_sparse

PENCIL_SHIFT = 0.3719 + 0.6153j     # generic point for the regularity test of L + s M
NUMERICAL_MAX = 2000
#: Number of shift-invert modes when n_modes is not given (the v0.1 default of k).
DEFAULT_N_MODES = 6


def pencil_rank(gm) -> str:
    """Rank report of the pencil L + s M, refusing singular pencils.

    L alone is singular whenever 0 is an eigenvalue (e.g. exactly at onset), so
    the test is on the pencil at a generic shift s: a pencil singular for every s
    (missing BC or gauge) has no well-defined spectrum.
    """
    from ..discretize.plan import RankError
    n, m = gm.L.shape
    if n != m:
        raise RankError(f"EVP refused before solving [{gm.label}]: {n} equations for {m} unknowns",
                        where=f"group [{gm.label}]",
                        why="more equations than unknowns means boundary conditions without tau "
                            "terms (or extra conditions); fewer means missing conditions",
                        fix="count the conditions per non-periodic direction in pb.explain() (rank report): "
                 "a total differential order k needs k conditions, each with its tau "
                 "(pb.add_bc places them)")
    pattern = sp.csr_matrix((abs(gm.L) + abs(gm.M)) > 0, dtype=float)
    srank = int(structural_rank(pattern)) if n else 0
    if srank < n:
        raise RankError(f"EVP refused before solving [{gm.label}]: pencil structural rank "
                        f"{srank} / {n} (missing boundary condition or gauge)",
                        where=f"group [{gm.label}]",
                        why="a pencil singular for every eigenvalue has no well-defined spectrum",
                        fix="count the conditions per non-periodic direction in pb.explain() (rank report): "
                 "a total differential order k needs k conditions, each with its tau "
                 "(pb.add_bc places them)")
    if n > NUMERICAL_MAX:
        return f"pencil L + sM: structural rank {n}/{n}; numerical test skipped (size {n})"
    nL = abs(gm.L).sum(axis=0).max() if gm.L.nnz else 1.0
    nM = abs(gm.M).sum(axis=0).max() if gm.M.nnz else 1.0
    A = (gm.L + PENCIL_SHIFT * (nL / max(nM, 1e-300)) * gm.M).toarray()
    # two sweeps of row/column equilibration: the rank test must not see bad scaling
    # (e.g. 1/Re in front of D^4 next to O(1) boundary rows) as singularity
    for _ in range(2):
        A = A / np.maximum(np.abs(A).max(axis=1, keepdims=True), 1e-300)
        A = A / np.maximum(np.abs(A).max(axis=0, keepdims=True), 1e-300)
    sv = np.linalg.svd(A, compute_uv=False)
    ratio = sv[-1] / sv[0]
    if ratio < n * np.finfo(float).eps:
        raise RankError(f"EVP refused before solving [{gm.label}]: pencil L + sM numerically "
                        f"singular at a generic s (sigma_min/sigma_max = {ratio:.1e})",
                        where=f"group [{gm.label}]",
                        why="a null space present for every s: usually an undetermined constant "
                            "(pressure level, pure Neumann) or two conditions fixing the same "
                            "thing",
                        fix="add a gauge (integ(p) = 0 with a constant tau) or replace the "
                            "redundant condition; pb.explain() shows the counts")
    return (f"pencil L + sM: structural rank {n}/{n}, regular at a generic s "
            f"(equilibrated sigma_min/sigma_max = {ratio:.1e})")


def _group_key(plan, group) -> tuple:
    """What identifies a group across resolutions: its wavenumbers (Cartesian Fourier items) or
    its label (curvilinear groups: m, k_z, l), never its position in ``plan.groups``."""
    if isinstance(group, tuple) and all(hasattr(it, "k") for it in group):
        return tuple(round(float(it.k), 12) for it in group)
    return (type(group).__name__, getattr(group, "label", repr(group)))


def _matching_group(plan, group, hp):
    """The group of the drift plan ``hp`` with the wavenumbers of ``group`` (it was taken at the same *position*, which is another wavenumber as soon as a
    Fourier size differs, e.g. k = -1 at N = 8 against k = 6 at N = 16 in ComplexFourier order:
    every mode was then removed as 'spurious')."""
    key = _group_key(plan, group)
    for g in hp.groups:
        if _group_key(hp, g) == key:
            return g
    from ..errors import ResolutionError
    raise ResolutionError(f"drift_plan has no group with the wavenumbers of [{plan.matrices(group).label}]",
                          why="the drift test compares the spectrum of one wavenumber at two "
                              "resolutions",
                          fix="give the drift plan Fourier sizes that hold this wavenumber (refine "
                              "the non-periodic direction, keep or enlarge the Fourier sizes)")


@renames("conatus.solvers.plan_evp:eigenvalues")
def eigenvalues(plan: Plan, group=None, *, reduced: bool | None = None, method: str | None = None,
                target: complex | None = None, n_modes: int | None = None, left: bool = False,
                drift_plan: Plan | Callable[[], Plan] | None = None,
                drift_tol: float = REQUIRED, spurious: str = "tail", region=None) -> EVPResult:
    """Eigenvalues sigma of (L + sigma M) x = 0 in one group of ``plan``.

    The rank report runs first (refuses ill-posed systems). ``method`` is 'dense'
    (all modes, QZ) or 'shift-invert' (``n_modes`` modes nearest ``target``, default
    ``DEFAULT_N_MODES``; ``k`` is its deprecated name: ``k`` is a wavenumber
    everywhere else). With
    ``drift_plan`` (a finer-resolution plan of the same problem) only modes that
    reappear there within ``drift_tol`` are kept, and the count of removed modes
    is logged in the result. ``drift_tol`` has no default: required with ``drift_plan``.

    ``spurious``: ``"tail"`` (default) flags the modes whose eigenvector has its
    energy in the last wall-normal coefficients (spurious tau modes; declared tolerances
    ``conatus.tolerances.SPURIOUS_TAIL_TOL``): they are kept, listed by ``explain()`` and ranked
    last by ``order()``; ``"none"`` switches the flag off (said by the result).

    ``region``: a ``cn.box(re=(a, b), im=(c, d))`` of the eigenvalue plane:
    every eigenvalue of the group in it, counted by the argument principle and computed by
    Beyn's contour-integral method, each checked by its residual
    (:class:`conatus.contour.RegionEVPResult`: ``count``, ``count_method``, ``multiplicity``);
    ``method`` is then ``"contour"``. ``method=None`` (the default) means ``"dense"`` without a region
    and ``"contour"`` with one; any other method given with ``region=`` (``"dense"`` too: it would
    have been replaced without a word) is refused, as are ``target`` and ``left=True``.
    """
    from ..errors import check_flag
    from .spurious import check_spurious
    check_spurious(spurious, "eigenvalues")
    left = check_flag(left, "left", "eigenvalues()")      # audit inputs: "no" reached LAPACK
    check_eig_method(method, target, region)          # before any assembly (review, inputs)
    method = _region_method(region, method, target, left)
    if n_modes is not None and method != "shift-invert":
        # it used to be ignored without a word: said, refused from 0.3
        from ..errors import _warn_user       # the user's line (a fixed level hid it)
        _warn_user(f"eigenvalues(n_modes={n_modes!r}) has no effect with method={method!r} "
                   "(every mode is returned); drop it (refused from 0.3)",
                   DeprecationWarning)
    if method == "shift-invert":
        from ..errors import check_count
        n_modes = DEFAULT_N_MODES if n_modes is None else check_count(
            n_modes, "n_modes", "eigenvalues(method='shift-invert')")
    if not isinstance(plan.problem, EVP):
        from ..errors import InputTypeError
        raise InputTypeError(f"eigenvalues() needs a plan of an EVP, got a {plan.problem.kind}",
                             fix="discretize an EVP (eigenvalue=...); an IVP is linearized first: "
                                 "pb.linearize(base, eigenvalue) or m.linearize(about=...)")
    plan.problem.check()
    if drift_plan is not None:            # checked before any solve
        check_drift_tol(drift_tol, "eigenvalues(drift_plan=...)")      # given, in range
    if group is not None and not any(_same_group(group, g) for g in plan.groups):
        # audit inputs: a foreign group was a bare ValueError of zip() in the assembly
        from ..errors import OptionError
        raise OptionError(f"eigenvalues(group={group!r}): not a group of this plan",
                          fix="a member of plan.groups (or plan.group(k=...) / m=... on the "
                              "native layer)")
    group = plan.groups[0] if group is None else group
    gm = plan.matrices(group, reduced)
    check = pencil_rank(gm)

    def run(gm_):
        if method == "dense":
            return solve_dense(gm_.L, gm_.M, left=left, label=gm_.label)
        if method == "contour":
            from ..contour.evp import region_eigenvalues
            return region_eigenvalues(gm_.L, gm_.M, region, label=gm_.label)
        if method == "shift-invert":
            if target is None:
                from ..errors import OptionError
                raise OptionError("eigenvalues(method='shift-invert') needs a target",
                                  fix="target=0.1+0.2j (the eigenvalues nearest it are returned)")
            return solve_sparse(gm_.L, gm_.M, target, n_modes, label=gm_.label)
        from ..errors import OptionError, did_you_mean
        raise OptionError(f"eigenvalues(method={method!r})",
                          fix="method='dense' (all modes), 'shift-invert' (n_modes modes near "
                              "target) or 'contour' (every mode in region=cn.box(...))"
                              + did_you_mean(str(method), ("dense", "shift-invert", "contour")))

    res = run(gm)
    res.log.insert(0, check)
    if drift_plan is not None:
        hp = drift_plan() if callable(drift_plan) else drift_plan
        hgm = hp.matrices(_matching_group(plan, group, hp), reduced)
        if hgm.L.shape[0] <= gm.L.shape[0]:
            # Y7: drift=N kept everything and read as filtered;
            # drift < N removed every mode without a word
            from ..errors import ResolutionError
            raise ResolutionError(
                f"the drift resolution is not finer: [{hgm.label}] has {hgm.L.shape[0]} unknowns "
                f"against {gm.L.shape[0]} at the resolution solved",
                why="the drift test keeps the modes found again at a FINER resolution; at the same "
                    "resolution every mode is found again (nothing is filtered) and at a coarser "
                    "one the resolved modes move too",
                fix="drift=N2 with N2 > N in the wall-normal direction, e.g. N2 = 3N/2",
                doc="docs/site/guide/dynamics/eigenvalues.md")
        hi = run(hgm)
        res = drift_filter(res, hi, drift_tol)
        if res.values.size == 0:
            from ..errors import ResolutionError
            raise ResolutionError(
                f"the drift test removed every mode of [{gm.label}]: no eigenvalue at this "
                f"resolution is found again within drift_tol = {drift_tol:g} at the finer one "
                f"([{hgm.label}])",
                why="no mode is resolved to drift_tol at the resolution solved (or drift_tol is "
                    "tighter than the accuracy N can give); an empty result would only fail later "
                    "(order()[0] -> IndexError)",
                fix="raise N (and the drift resolution with it), or loosen drift_tol (1e-6 is the "
                    "usual value); eigenvalues without drift= lists the modes and their tails",
                doc="docs/site/guide/dynamics/eigenvalues.md")
    # what res.mode(i) needs to put the eigenvectors on the grid; the perturbation
    # variables of a linearized model ("du") are shown under the unknown's name ("u")
    rename = getattr(plan.problem, "mode_names", None) or {}
    res.context = (plan, group, bool(gm.reduced) if hasattr(gm, "reduced") else False, rename)
    _annotate(res, plan, drift_plan is not None)
    from .spurious import flag, tail_layout
    if spurious == "none":
        flag(res, None, "" if res.filter_note is None else
             "spectral-tail spurious-mode flag off (spurious='none')")
    else:
        try:
            layout, why = tail_layout(plan, group, bool(res.context[2]))
        except Exception as exc:  # noqa: BLE001 -- the flag is a detection: say why it did not run
            layout, why = None, f"spectral-tail test not run ({type(exc).__name__}: {exc})"
        flag(res, layout, why)
    from .spurious import check_selected
    check_selected(res)                    # Y1: resolution of the selected mode
    return res


#: The methods of :func:`eigenvalues` (``None``: "dense", or "contour" with ``region=``).
EIG_METHODS = ("dense", "shift-invert", "contour")


def check_eig_method(method: Any, target: Any = None, region: Any = None,
                     where: str = "eigenvalues") -> None:
    """``method`` one of :data:`EIG_METHODS` (or None), and a finite ``target`` with
    ``method="shift-invert"`` (unless ``region=`` is given, which refuses both later with its own
    message), checked before anything is compiled or assembled (review of the inputs audit: a typo of the method, or a shift-invert call without target, was refused only after
    the group had been assembled and its rank report computed, and on the native layer after the
    model had been compiled and discretized)."""
    from ..errors import OptionError, did_you_mean
    if method is not None and (not isinstance(method, str) or method not in EIG_METHODS):
        raise OptionError(f"{where}(method={method!r})",
                          fix="method='dense' (all modes), 'shift-invert' (n_modes modes near "
                              "target) or 'contour' (every mode in region=cn.box(...))"
                              + did_you_mean(str(method), EIG_METHODS))
    if method == "shift-invert" and region is None:
        if target is None:
            raise OptionError(f"{where}(method='shift-invert') needs a target",
                              fix="target=0.1+0.2j (the eigenvalues nearest it are returned)")
        from .evp import check_target
        check_target(target)


def _region_method(region, method: str | None, target, left: bool) -> str:
    """The method of a call: ``None`` is ``"dense"`` without ``region=`` and ``"contour"`` with it;
    a method that cannot go with ``region=`` is refused, never replaced (``method="dense"`` with a
    region used to be switched to the contour method without a word: critic review of L10)."""
    from ..errors import OptionError
    doc = "docs/site/guide/dynamics/contour.md"
    if region is None:
        if method == "contour":
            raise OptionError("eigenvalues(method='contour') needs region=cn.box(re=..., im=...)",
                              why="a contour method computes the eigenvalues inside a region",
                              fix="region=cn.box(re=(a, b), im=(c, d)) in the eigenvalue plane",
                              doc=doc)
        return "dense" if method is None else method
    from ..contour.region import ContourBox
    if not isinstance(region, ContourBox):
        raise OptionError(f"eigenvalues(region={region!r}): a cn.box(re=(a, b), im=(c, d))",
                          fix="region=cn.box(re=(0.0, 0.5), im=(-0.1, 0.1))", doc=doc)
    if method not in (None, "contour"):
        raise OptionError(f"eigenvalues(region=..., method={method!r})",
                          why="region= computes every eigenvalue in the box by the contour method "
                              "(argument-principle count + Beyn); the dense solver computes the "
                              "whole spectrum and shift-invert the modes near a target, so "
                              f"method={method!r} contradicts region=",
                          fix="drop method= (or method='contour') to keep region=; or drop "
                              f"region= for method={method!r}", doc=doc)
    if target is not None or left:
        raise OptionError("eigenvalues(region=...) with " + ("target=" if target is not None
                                                             else "left=True"),
                          why="the region decides which modes are computed; left eigenvectors "
                              "are not computed by the contour method",
                          fix="drop target= / left=; left eigenvectors: the dense method",
                          doc=doc)
    return "contour"


#: Said by an unfiltered result of a problem with taus: once, in
#: explain() and the one-line summary, never as a warning.
NO_FILTER_NOTE = ("no spurious-mode filter: tau modes may appear among these eigenvalues; use "
                  "drift=N2, drift_tol=1e-6 (native) or drift_plan=, drift_tol= (expert) to keep "
                  "only the modes found again at a finer resolution")


def _annotate(res: EVPResult, plan: Plan, filtered: bool) -> None:
    """Growth convention of the eigenvalue (ordering, explain) and the spurious-mode sentence."""
    from .evp_modes import _tau_names
    from .parametric import growth_convention
    ev = plan.problem.eigenvalue
    name = ev.name if ev is not None else "sigma"
    growth, note = growth_convention(name, getattr(plan.problem, "growth", None),
                                     getattr(plan.problem, "growth_why", None))
    res.eigenvalue, res.growth, res.growth_note = name, growth, note
    try:
        taus = bool(_tau_names(plan))
    except AttributeError:                    # a plan without the IR layout has no tau to report
        taus = False
    except Exception as err:  # noqa: BLE001 -- reported on the result, not hidden
        res.filter_note = ("spurious-mode filter: unknown whether this problem has tau modes (the "
                           f"tau variables could not be identified: {type(err).__name__}: {err})")
        return
    if taus:
        res.filter_note = ("spurious-mode filter: resolution drift (see below)" if filtered
                           else NO_FILTER_NOTE)


def _same_group(a: object, b: object) -> bool:
    """``a`` is the group ``b`` of a plan (groups compare by value; an array-valued comparison
    or an incomparable object is not a match)."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False
