"""Every eigenvalue of ``T(z) v = 0`` in a region: Beyn's contour-integral method, certified.

Beyn (2012, Linear Algebra Appl. 436, 3839) for linear and nonlinear eigenproblems: with a random
probe block ``V`` (n x l) and the scaled variable ``zeta = (z - c) / rho`` (box centre ``c``,
half-diagonal ``rho``), the moments

    A_k = (1 / 2 pi i) \\oint zeta^k T(z)^{-1} V dz,      k = 0 .. 2p - 1,

hold the eigenvalues inside the contour: by Keldysh's theorem ``T(z)^{-1}`` is a sum of pole
terms at the eigenvalues plus a holomorphic part, which the integral removes. The block Hankel
matrices ``B0 = [A_{i+j}]``, ``B1 = [A_{i+j+1}]`` (i, j < p) give, after a rank-revealing SVD
``B0 = U S W^H`` truncated to its rank ``K``, the small matrix ``U_K^H B1 W_K S_K^{-1}`` whose
eigenvalues are the ``zeta`` of the eigenvalues inside (and whose eigenvectors give theirs).
The integrals are Gauss-Legendre sums on each edge of the box (exponentially convergent for the
piecewise-analytic integrand; slower when an eigenvalue lies just outside an edge).

What is returned is certified, root by root and as a whole:

1. the number of eigenvalues inside, ``Z``, is counted first by the argument principle
   (:func:`conatus.contour.count`); ``l p`` is chosen above it;
2. every Ritz value is polished by two-sided nonlinear inverse iteration (right and left vectors,
   ``x <- T(z)^{-1} T'(z) x``, ``y <- T(z)^{-H} T'(z)^H y``, Newton step on ``y^H T(z) x = 0``: the
   two-sided Rayleigh quotient of a pencil), or one-sided ``u = T(z)^{-1} T'(z) v``,
   ``z <- z - (c^H v) / (c^H u)`` when no adjoint solve is available (Hill operators); a polish
   never leaves half the distance to the nearest other Ritz value;
3. a root is kept when it lies in the box and its relative backward residual
   ``||T(z) v|| / (||T(z)|| ||v||)`` is at most ``residual_tol``;
4. roots closer than a millionth of the box size are one root; the multiplicity of such a
   cluster (a double eigenvalue, a coalescence, two Ritz values polished onto one root) is its
   own argument-principle count on a small box around it;
5. the multiplicities must add up to ``Z``. If they do not, the moments are recomputed with more
   quadrature nodes, columns and moments (``max_rounds``); if they still do not, the call is
   refused with what was found. A partial or uncertified list is never returned.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla

from ..errors import ConvergenceError, OptionError
from .count import CountResult, _walk, count
from .count import MAX_EVALUATIONS, PHASE_STEP_MAX
from .operators import Operator, SingularPoint, as_operator
from .region import ContourBox

__all__ = ["beyn", "BeynResult", "RESIDUAL_TOL", "QUADRATURE_POINTS", "POLISH_MAX_ITERATIONS",
           "BEYN_MARGIN", "BEYN_MAX_ROUNDS", "BEYN_MAX_SPLITS"]

_DOC = "docs/site/guide/dynamics/contour.md"
#: Certification threshold of a root: relative backward residual ||T(z) v|| / (||T(z)|| ||v||).
#: The roots of the test suite (Orr-Sommerfeld, Hill, nonlinear) polish to 1e-19..1e-15 (dense QZ:
#: about 1e-15). 1e-10 was too loose for a strongly non-normal operator: on Orr-Sommerfeld at
#: Re = 1e4, N = 128, points 0.01..0.08 away from the S-branch eigenvalues passed it (residuals
#: 1e-10..1e-11, pseudospectrum); at 1e-13 every certified root agrees with QZ to 5e-8.
RESIDUAL_TOL = 1e-13
#: Gauss-Legendre nodes per edge of the box for the moments (doubled at every new round).
QUADRATURE_POINTS = 32
#: Iteration cap of the polish of one Ritz value (a cap, not a convergence claim: the residual
#: of every root is checked against residual_tol).
POLISH_MAX_ITERATIONS = 20
#: Probe columns x moments above the count: l p >= Z + BEYN_MARGIN (doubled at every new round).
BEYN_MARGIN = 2
#: Rounds of moments on one box before it is cut in two.
BEYN_MAX_ROUNDS = 3
#: Levels of cuts of a box whose rounds did not certify its roots (each half counted and solved;
#: the counts of the halves must add up), before the method is refused.
BEYN_MAX_SPLITS = 3


@dataclass
class BeynResult:
    """Certified eigenvalues of ``T(z) v = 0`` in a region (:func:`beyn`)."""

    values: np.ndarray            # distinct roots in the box
    vectors: np.ndarray           # their right eigenvectors (columns, unit 2-norm)
    residuals: np.ndarray         # relative backward residual of each root
    multiplicity: np.ndarray      # algebraic multiplicity (argument-principle count of a cluster)
    counted: CountResult          # Z, by the argument principle on the whole box
    region: ContourBox
    residual_tol: float
    columns: int = 0
    moments: int = 0
    quadrature: int = 0
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank: int = 0
    rounds: int = 0
    seed: int = 0
    operator: str = ""
    iterations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    log: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Z: the number of eigenvalues in the box, multiplicities included (= multiplicity.sum())."""
        return int(self.counted.count)

    @property
    def count_method(self) -> str:
        """How the count was certified, in words (the argument principle, matched by the roots)."""
        return self.counted.count_method + "; matched by the certified roots of Beyn's method"

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return (f"BeynResult({self.values.size} certified root(s), count Z = {self.count} in "
                f"{self.region.describe()}; .explain())")

    def explain(self, n: int = 12) -> str:
        """The count, the moments, every root with its residual and multiplicity, and the log."""
        lines = [f"Beyn contour integral in {self.region.describe()}: {self.values.size} distinct "
                 f"root(s), multiplicities adding up to the count Z = {self.count}",
                 f"  count: {self.counted.count_method}",
                 f"  operator: {self.operator}"]
        if self.count:
            sv = self.singular_values
            lines.append(f"  moments: up to {self.columns} probe column(s) x {self.moments} "
                         f"block(s) (zeta^0 .. zeta^{2 * self.moments - 1}), Gauss-Legendre up to "
                         f"{self.quadrature} nodes per edge, seed {self.seed}, up to round "
                         f"{self.rounds}")
            if sv.size >= self.count and not any("cut at" in s for s in self.log):
                gap = (f"; sigma_{self.count + 1}/sigma_1 = {sv[self.count] / sv[0]:.1e}"
                       if sv.size > self.count else "")
                lines.append(f"  Hankel singular values: rank {self.rank} above 1e-10 sigma_1, "
                             f"sigma_Z/sigma_1 = {sv[self.count - 1] / sv[0]:.1e}{gap}")
            lines += [
                      "  polish: inverse iteration on T(z)^-1 T'(z) (two-sided when adjoint solves exist); certified when "
                      f"||T(z) v|| / (||T(z)|| ||v||) <= {self.residual_tol:g} and z in the box"]
            lines.append("  roots (multiplicity > 1: a cluster counted on its own small box):")
            for i in np.argsort(-self.values.imag, kind="stable")[:n]:
                lines.append(f"    z = {self.values[i]: .12e}  residual {self.residuals[i]:.1e}  "
                             f"multiplicity {int(self.multiplicity[i])}  "
                             f"[{int(self.iterations[i])} polish iteration(s)]")
            if self.values.size > n:
                lines.append(f"    ... {self.values.size - n} more (explain(n=...))")
        lines += ["  " + s for s in self.log]
        return "\n".join(lines)


def _check_tol(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
            or not 0 < value < 1):
        raise OptionError(f"{name}={value!r}: a relative residual in (0, 1)",
                          fix=f"{name}=1e-13 (the declared default)", doc=_DOC)
    return float(value)


def _moments(op: Operator, region: ContourBox, V: np.ndarray, n_moments: int, q: int):
    """``A_k = (1/2 pi i) oint zeta^k T(z)^{-1} V dz`` by Gauss-Legendre on the four edges."""
    t, w = np.polynomial.legendre.leggauss(q)
    c, rho = region.center, region.radius
    corners = region.corners
    A = [np.zeros(V.shape, dtype=complex) for _ in range(n_moments)]
    for k in range(4):
        z0, z1 = corners[k], corners[(k + 1) % 4]
        for tj, wj in zip(t, w, strict=True):
            z = z0 + 0.5 * (tj + 1.0) * (z1 - z0)
            try:
                X = op.factor(z).solve(V)
            except SingularPoint:
                raise ConvergenceError(f"a quadrature node z = {z:.12g} is an eigenvalue",
                                       fix="move the box slightly", doc=_DOC) from None
            if not np.all(np.isfinite(X)):
                raise ConvergenceError(f"a quadrature node z = {z:.12g} is an eigenvalue",
                                       fix="move the box slightly", doc=_DOC)
            f = 0.5 * wj * (z1 - z0) / (2j * math.pi)
            zeta, pw = (z - c) / rho, 1.0 + 0j
            for m in range(n_moments):
                A[m] += (f * pw) * X
                pw *= zeta
    return A


def _residual(op: Operator, z: complex, x: np.ndarray) -> float:
    r = op.apply(z, x.reshape(-1, 1))[:, 0]
    s = op.scale(z) * float(np.linalg.norm(x))
    return float(np.linalg.norm(r)) / s if s > 0 else math.inf


def _polish(op: Operator, z: complex, x: np.ndarray, max_iterations: int, reach: float = math.inf):
    """Polish of a Ritz value (z, x); the best iterate by residual, its vector and the iterations.

    Two-sided when the operator gives adjoint solves (pencils, matrix functions): right and left
    vectors by inverse iteration, ``x <- T(z)^-1 T'(z) x``, ``y <- T(z)^-H T'(z)^H y``, and the
    Newton step on ``y^H T(z) x = 0``, ``z <- z - (y^H T(z) x) / (y^H T'(z) x)`` (the two-sided
    Rayleigh quotient of a pencil: cubic for a simple eigenvalue, and accurate for a strongly
    non-normal operator, where the one-sided update stalls). Otherwise one-sided nonlinear inverse
    iteration (``c`` the starting vector): ``u = T(z)^-1 T'(z) x``, ``z <- z - (c^H x)/(c^H u)``.

    ``reach``: iterates farther than this from the starting Ritz value are not accepted (half the
    distance to the nearest other Ritz value): the polish refines the eigenvalue the moments
    found, it never moves onto a neighbour (two Ritz values polished onto one root of a close
    pair would lose the other)."""
    eps = np.finfo(float).eps
    z0 = z
    nx = float(np.linalg.norm(x))
    x = x / nx if nx > 0 else np.ones(op.n, complex) / math.sqrt(op.n)
    c = x.copy()                                   # normalisation c^H x = 1 (one-sided)
    y = x.copy()
    two_sided = hasattr(op, "dapply_h")
    best = (_residual(op, z, x), z, x, 0)
    for it in range(1, max_iterations + 1):
        try:
            F = op.factor(z)
            u = F.solve(op.dapply(z, x.reshape(-1, 1)))[:, 0]
            if two_sided:
                w = F.solve_h(op.dapply_h(z, y.reshape(-1, 1)))[:, 0]
        except SingularPoint:
            break                                  # z is an eigenvalue to working precision
        if not np.all(np.isfinite(u)) or (two_sided and not np.all(np.isfinite(w))):
            break
        if two_sided:
            nu, nw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
            if nu == 0 or nw == 0:
                break
            x, y = u / nu, w / nw
            den = complex(np.vdot(y, op.dapply(z, x.reshape(-1, 1))[:, 0]))
            if den == 0:
                break
            step = complex(np.vdot(y, op.apply(z, x.reshape(-1, 1))[:, 0])) / den
            z = z - step
        else:
            cu = complex(np.vdot(c, u))
            if cu == 0:
                break
            step = complex(np.vdot(c, x)) / cu
            z, x = z - step, u / cu
        if not abs(z - z0) <= reach:
            break                                  # heading for another eigenvalue: stop here
        r = _residual(op, z, x)
        if r < best[0]:
            best = (r, z, x, it)
        if abs(step) <= 4 * eps * max(1.0, abs(z)):
            break
    r, z, x, it = best
    return z, x / np.linalg.norm(x), r, it


def _clusters(z: np.ndarray, radius: float) -> list[list[int]]:
    """Indices of the roots grouped by single linkage at distance ``radius``."""
    groups: list[list[int]] = []
    for i in np.argsort(z.real, kind="stable"):
        hit = [g for g in groups if np.min(np.abs(z[g] - z[i])) <= radius]
        merged = [int(i)] + [k for g in hit for k in g]
        groups = [g for g in groups if g not in hit] + [merged]
    return groups


#: Linkage radii (fractions of the box half-diagonal) tried in turn to group the polished roots
#: before their multiplicities are counted: 1e-6 merges duplicates of one simple root and a double
#: root split by rounding (~ sqrt(eps)); the larger radii gather a defective eigenvalue of higher
#: order, which rounding splits into m values on a circle of radius ~ eps^(1/m).
CLUSTER_RADII = (1e-6, 1e-4, 1e-2, 1e-1)


def _local_count(op: Operator, region: ContourBox, z: complex, half: float) -> int | None:
    """Argument-principle count on the square of half-side ``half`` around ``z`` (None when the
    walk is refused: a root on its edge)."""
    try:
        w = _walk(op, region.around(z, half), PHASE_STEP_MAX, 4, MAX_EVALUATIONS)
    except ConvergenceError:
        return None
    return round(w["winding"])


def _vector_at(op: Operator, z: complex, x: np.ndarray, rho: float) -> np.ndarray:
    """A null vector estimate of ``T(z)``: one inverse-iteration step from ``x`` (at a point
    shifted by 1e-15 rho, then more, when ``z`` is singular to working precision: the shift
    bounds the residual of the vector at ``z``)."""
    for shift in (0.0, 1e-15 * rho, 1e-13 * rho, 1e-11 * rho):
        try:
            y = op.factor(z + shift).solve(np.asarray(x, complex).reshape(-1, 1))[:, 0]
        except SingularPoint:
            continue
        if not np.all(np.isfinite(y)):
            continue
        big = float(np.max(np.abs(y)))           # rescale first: near a defective eigenvalue
        if big > 0:                              # the solve is huge, its 2-norm would overflow
            y = y / big
            return y / float(np.linalg.norm(y))
    return np.asarray(x, complex) / float(np.linalg.norm(x))


def _certified_groups(op: Operator, region: ContourBox, roots: np.ndarray, Z: int,
                      obstacles: np.ndarray, cache: dict):
    """(groups, multiplicities, half-sides) of the certified roots ``roots``, every multiplicity
    an argument-principle count on a small square around its group, the squares disjoint, inside
    the region and clear of every other polished root (``obstacles``); a grouping is accepted
    when every count is >= 1 and they add up to ``Z`` (else the last one tried is returned, and
    the round fails).

    Every group, a single root too, is counted (critic review of L10: a defective eigenvalue of
    order m >= 3 was returned as m "certified" simple roots spread over eps^(1/m), each of
    backward residual 1e-16 but none an eigenvalue, multiplicity 1 each, adding up to Z). A
    group's square has half-side ``max(2 r_g, 1e-3 rho)`` capped by a quarter of the distance to
    the other roots and 0.9 of the distance to the boundary (``r_g``: the group's radius about
    its mean). The linkage radii ``CLUSTER_RADII`` are tried in turn, a larger one only while a
    failed group has a neighbour within the largest one (else merging cannot change the
    outcome)."""
    rho = region.radius
    last = ([], np.zeros(0, dtype=int), np.zeros(0))
    for level, frac in enumerate(CLUSTER_RADII):
        groups = _clusters(roots, frac * rho)
        centres = np.array([roots[g].mean() for g in groups])
        radii = np.array([float(np.max(np.abs(roots[g] - c))) for g, c in zip(groups, centres,
                                                                                  strict=True)])
        mult = np.zeros(len(groups), dtype=int)
        halves = np.zeros(len(groups))
        failed = []
        for i, c in enumerate(centres):
            own = radii[i] * (1 + 1e-12) + 1e-300    # the group's own members (rounding)
            mine = np.abs(obstacles - c) <= own
            others = obstacles[~mine]
            half = max(2.0 * radii[i], 1e-3 * rho)
            if others.size:
                half = min(half, 0.25 * float(np.min(np.abs(others - c))))
            half = min(half, 0.9 * float(region.distance_to_boundary(c)))
            halves[i] = half
            if not half > radii[i]:            # the square would not hold its own group
                failed.append(i)
                continue
            key = (complex(c), float(half))
            if key not in cache:
                cache[key] = _local_count(op, region, complex(c), half)
            k = cache[key]
            if k is None or k < 1:
                failed.append(i)
                continue
            mult[i] = k
        last = (groups, mult, halves)
        if not failed and int(mult.sum()) == Z:
            return last
        if level + 1 < len(CLUSTER_RADII):
            nxt = CLUSTER_RADII[-1] * rho      # a neighbour any larger radius could merge
            near = [i for i in failed
                    if np.any(np.abs(np.delete(centres, i) - centres[i]) <= nxt)]
            if not near:
                break       # no failed group has a neighbour to merge with: no larger radius helps
    return last


def beyn(T: Any, region: ContourBox, *, poles: int | None = None, derivative: Any = None,
         counted: CountResult | None = None, control: tuple[ContourBox, int] | None = None,
         residual_tol: float = RESIDUAL_TOL, quadrature: int = QUADRATURE_POINTS, seed: int = 0,
         max_rounds: int = BEYN_MAX_ROUNDS,
         polish_max_iterations: int = POLISH_MAX_ITERATIONS) -> BeynResult:
    """Every eigenvalue of ``T(z) v = 0`` in ``region``, certified (module docstring).

    ``T``: a pair ``(L, M)`` (the pencil ``L + z M``), a :class:`conatus.floquet.HillBlocks`, or a
    callable ``z -> matrix`` (or scalar) with, optionally, its ``derivative`` ``z -> T'(z)``.
    ``poles``: the poles of ``det T`` inside the region; required for a callable (0 when ``T`` is
    holomorphic there), since the count of eigenvalues is ``Z = N_Gamma + P``. ``counted``: a
    :class:`CountResult` of the same ``T`` and region already computed (else :func:`count` runs,
    with ``control=``). ``residual_tol``: certification threshold of every root (declared default
    ``RESIDUAL_TOL``). ``quadrature``: Gauss-Legendre nodes per edge; ``seed``: of the random
    probe block (the result is reproducible). Raises ConvergenceError when the certified roots do
    not account for the count after ``max_rounds`` rounds."""
    residual_tol = _check_tol(residual_tol, "residual_tol")
    for name, v in (("quadrature", quadrature), ("max_rounds", max_rounds),
                    ("polish_max_iterations", polish_max_iterations)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise OptionError(f"{name}={v!r}: a positive integer", doc=_DOC)
    op = as_operator(T, poles=poles, derivative=derivative)
    if counted is None:
        counted = count(op, region, control=control)
    elif counted.region != region:
        raise OptionError("beyn(counted=...): that count is of another region", doc=_DOC)
    if counted.count is None:
        raise OptionError("beyn(): the number of eigenvalues inside is not known: the winding "
                          f"of det T is Z - P = {counted.winding} and P (the poles of det T in "
                          "the box) was not declared",
                          why="the roots are certified by adding up to the count Z",
                          fix="poles=0 if T(z) is holomorphic in the box, poles=k if det T has "
                              "k poles there (multiplicities included)", doc=_DOC)
    Z = int(counted.count)
    base = dict(counted=counted, region=region, residual_tol=residual_tol, seed=seed,
                operator=op.explain())
    if Z < 0:
        raise ConvergenceError(f"count Z = {Z} < 0 in {region.describe()}: the declared poles "
                               f"({counted.poles}) exceed what the winding allows", doc=_DOC)
    if Z == 0:
        return BeynResult(np.zeros(0, complex), np.zeros((op.n, 0), complex), np.zeros(0),
                          np.zeros(0, dtype=int), log=["no eigenvalue in the box: nothing to compute"],
                          **base)
    log: list[str] = []
    if op.needs_reference:
        # residual scale of an operator without a structural norm: never below the typical size
        # of T on the boundary (a scalar f vanishes at its roots: |f(z)| / |f(z)| = 1)
        op.reference = 0.0
        op.reference = float(np.median([op.scale(z) for z in region.point(np.arange(8) / 2)]))
        log.append(f"residual scale: max(||T(z)||_1, {op.reference:.3e}), the median ||T||_1 at "
                   "the corners and edge midpoints of the box")
    opts = dict(residual_tol=residual_tol, quadrature=int(quadrature), max_rounds=int(max_rounds),
                polish_max_iterations=int(polish_max_iterations))
    out = _solve(op, region, Z, np.random.default_rng(seed), opts, BEYN_MAX_SPLITS)
    out.log[:0] = log
    return BeynResult(**out.__dict__ | base)


@dataclass
class _Found:
    values: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray
    multiplicity: np.ndarray
    columns: int = 0
    moments: int = 0
    quadrature: int = 0
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank: int = 0
    rounds: int = 0
    iterations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    log: list[str] = field(default_factory=list)


def _rounds(op: Operator, region: ContourBox, Z: int, rng, opts: dict):
    """Beyn's rounds on one box: (_Found or None, log, the roots and Ritz values seen)."""
    residual_tol, max_rounds = opts["residual_tol"], opts["max_rounds"]
    margin, q, log = BEYN_MARGIN, opts["quadrature"], []
    c, rho = region.center, region.radius
    seen = np.zeros(0, complex)
    found: tuple = ()
    cache: dict = {}                  # local counts already walked: (centre, half) -> count
    for rnd in range(1, max_rounds + 1):
        ell = min(op.n, Z + margin)
        p = -(-(Z + margin) // ell)
        V = rng.standard_normal((op.n, ell)) + 1j * rng.standard_normal((op.n, ell))
        A = _moments(op, region, V, 2 * p, q)
        B0 = np.block([[A[i + j] for j in range(p)] for i in range(p)])
        B1 = np.block([[A[i + j + 1] for j in range(p)] for i in range(p)])
        U, s, Wh = sla.svd(B0, full_matrices=False)
        rank = int(np.count_nonzero(s > 1e-10 * s[0])) if s[0] > 0 else 0
        if rank < Z:
            log.append(f"round {rnd}: Hankel rank {rank} < Z = {Z} (l = {ell}, p = {p}, "
                       f"{q} nodes per edge): more moments")
            margin, q = 2 * margin, 2 * q
            continue
        K = rank
        Bk = (U[:, :K].conj().T @ B1 @ Wh[:K].conj().T) / s[:K]
        zeta, Y = sla.eig(Bk)
        X = (U[:, :K] @ Y)[:op.n]
        cand = c + rho * zeta
        seen = np.r_[seen, cand]
        roots, vecs, res, its = [], [], [], []
        outside = 0
        gaps = np.abs(cand[:, None] - cand[None, :]) + np.diag(np.full(K, np.inf))
        reach = 0.5 * gaps.min(axis=1) if K > 1 else np.full(K, np.inf)
        for j in range(K):
            zj, xj, rj, itj = _polish(op, complex(cand[j]), X[:, j], opts["polish_max_iterations"],
                                      float(reach[j]))
            if not rj <= residual_tol:
                # not certified within its reach: polish on without the bound (a duplicate of a
                # neighbour is merged by the clusters; a root it leaves behind is found again
                # by the next round or a cut)
                zf, xf, rf, itf = _polish(op, zj, xj, opts["polish_max_iterations"])
                if rf < rj:
                    zj, xj, rj, itj = zf, xf, rf, itj + itf
            if not bool(region.contains(zj)):
                outside += 1
                continue
            roots.append(zj)
            vecs.append(xj)
            res.append(rj)
            its.append(itj)
        roots_a = np.array(roots, complex)
        # only certified roots are grouped and reported (an uncertified one cannot be returned),
        # but every polished root bounds the squares of the others: a square must not reach a
        # root whose polish merely failed, else its eigenvalue would be counted as a multiplicity
        ok_idx = [k for k, r in enumerate(res) if r <= residual_tol]
        if ok_idx:
            sub, mult, half = _certified_groups(op, region, roots_a[ok_idx], Z, roots_a, cache)
            groups = [[ok_idx[i] for i in g] for g in sub]
        else:
            groups, mult, half = [], np.zeros(0, int), np.zeros(0)
        # one representative per group: the root of smallest residual, or, for a group of
        # several values, their mean when its own residual passes too: a defective eigenvalue
        # split by rounding on a circle of radius ~ eps^(1/m) has its eigenvalue at the mean to
        # rounding (the trace of the split block is preserved); two distinct close roots (a
        # coalescence, +-sqrt(eps)) do not, and keep their best member
        keep = [g[int(np.argmin([res[k] for k in g]))] for g in groups]
        for g, k in zip(groups, keep, strict=True):
            if len(g) > 1:
                zc = complex(roots_a[g].mean())
                xc = _vector_at(op, zc, vecs[k], rho)
                rc = _residual(op, zc, xc)
                if rc <= residual_tol:
                    roots_a[k], vecs[k], res[k] = zc, xc, rc
        uncert = [k for k in keep if not res[k] <= residual_tol]
        msg = (f"round {rnd}: l = {ell} column(s), p = {p}, {q} nodes per edge; Hankel rank "
               f"{rank}; {K} Ritz value(s) polished, {outside} left the box, "
               f"{len(roots) - len(ok_idx)} above residual_tol, {len(keep)} distinct certified "
               f"root(s) with multiplicities adding up to {int(mult.sum())}")
        for gi, k in enumerate(keep):
            if mult[gi] > 1:
                msg += (f"; {roots_a[k]:.10g}: {int(mult[gi])} eigenvalues within the square of "
                        f"half-side {half[gi]:.1e} around it (a multiple eigenvalue, or a cluster "
                        "tighter than that)")
        log.append(msg)
        found = (roots_a, res, mult, keep)
        if int(mult.sum()) == Z and not uncert:
            k = np.array(keep, dtype=int)
            return _Found(roots_a[k], np.stack([vecs[i] for i in keep], axis=1),
                          np.array([res[i] for i in keep]), mult, columns=ell, moments=p,
                          quadrature=q, singular_values=s, rank=rank, rounds=rnd,
                          iterations=np.array([its[i] for i in keep], dtype=int), log=log), log, seen, found
        margin, q = 2 * margin, 2 * q
    return None, log, seen, found


def _split_boxes(region: ContourBox, avoid: np.ndarray):
    """Two halves of ``region`` across its longer side, the cut placed among a few positions near
    the middle as far as possible from the points ``avoid`` (eigenvalue estimates); candidates in
    order of preference."""
    (a, b), (c, d) = region.re, region.im
    horizontal = (b - a) >= (d - c)
    lo, hi = (a, b) if horizontal else (c, d)
    fracs = np.array([0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65])
    cuts = lo + fracs * (hi - lo)
    coord = avoid.real if horizontal else avoid.imag
    dist = np.array([np.min(np.abs(coord - x), initial=np.inf) for x in cuts])
    out = []
    for x in cuts[np.argsort(-dist, kind="stable")]:
        if horizontal:
            out.append((x, ContourBox((a, x), (c, d)), ContourBox((x, b), (c, d))))
        else:
            out.append((x, ContourBox((a, b), (c, x)), ContourBox((a, b), (x, d))))
    return out


def _solve(op: Operator, region: ContourBox, Z: int, rng, opts: dict, splits: int) -> _Found:
    """Beyn on ``region`` holding ``Z`` eigenvalues; if its rounds do not certify them, the box is
    cut in two (each half counted, the counts adding up to Z) and each half solved, at most
    ``splits`` times deep. Raises ConvergenceError when that fails too."""
    if Z == 0:
        return _Found(np.zeros(0, complex), np.zeros((op.n, 0), complex), np.zeros(0),
                      np.zeros(0, dtype=int), log=[f"{region.describe()}: no eigenvalue"])
    got, log, seen, found = _rounds(op, region, Z, rng, opts)
    if got is not None:
        return got
    # a cut only where P = 0 holds by construction: with poles, or a declared P, a half's own P
    # is not known (a pole and a zero in one half would read as an empty half)
    if splits > 0 and op.poles == 0 and op.structural_poles:
        for x, b1, b2 in _split_boxes(region, seen):
            try:
                z1, z2 = count(op, b1).winding, count(op, b2).winding
            except ConvergenceError:
                continue                       # an eigenvalue on this cut: the next position
            if z1 + z2 != Z:
                continue
            parts = [_solve(op, bb, zz, rng, opts, splits - 1) for bb, zz in ((b1, z1), (b2, z2))]
            note = (f"{region.describe()}: the rounds did not certify the Z = {Z} roots "
                    f"({log[-1] if log else 'no round'}); cut at "
                    f"{'Re' if b1.re[1] != region.re[1] else 'Im'} = {x:.6g} into boxes of "
                    f"{z1} and {z2}, each counted and solved")
            return _Found(np.r_[parts[0].values, parts[1].values],
                          np.hstack([parts[0].vectors, parts[1].vectors]),
                          np.r_[parts[0].residuals, parts[1].residuals],
                          np.r_[parts[0].multiplicity, parts[1].multiplicity],
                          columns=max(pp.columns for pp in parts),
                          moments=max(pp.moments for pp in parts),
                          quadrature=max(pp.quadrature for pp in parts),
                          singular_values=parts[0].singular_values,
                          rank=sum(pp.rank for pp in parts),
                          rounds=max(pp.rounds for pp in parts),
                          iterations=np.r_[parts[0].iterations, parts[1].iterations],
                          log=[note] + parts[0].log + parts[1].log)
    roots_a, res, mult, keep = found or (np.zeros(0), [], np.zeros(0, int), [])
    listing = "; ".join(f"{roots_a[k]:.10g} (residual {res[k]:.1e}, multiplicity {m})"
                        for k, m in zip(keep, mult, strict=True)) or "none"
    raise ConvergenceError(
        f"Beyn's method refused in {region.describe()}: the certified roots do not account for "
        f"the count Z = {Z} after {opts['max_rounds']} round(s) and {BEYN_MAX_SPLITS} level(s) of "
        "box cuts",
        why="a root list that does not match the argument-principle count, or a root whose "
            f"residual exceeds residual_tol = {opts['residual_tol']:g}, is not returned. Found: "
            + listing + ". Log: " + " | ".join(log),
        fix="an eigenvalue just outside an edge slows the quadrature: move the box edge away "
            "from it, or raise quadrature= and max_rounds=; check T'(z) (derivative=) if the "
            "polish does not converge", doc=_DOC)
