"""Projection of the m = 0 coefficients of a curvilinear state onto real fields ( e216).

Real fields on the disk, the sphere, the shell and the ball store m >= 0 only; the m = 0 rows
are complex and a real field needs them to satisfy c = J c, with J the antilinear map
"coefficients of the complex conjugate field" (J c = sigma * conj(c[partner]): spin component
s <-> -s, same l and radial index, sigma = +-1 or a phase). The tau solves create a non-real
m = 0 part at the rounding level, which the real linear operator never removes: relative to a
decaying field it grows, until ``solver.grid`` refuses a healthy state.

``M0RealProjection`` projects X onto the real fields, P X = (X + J X) / 2 on the m = 0 entries
(identity elsewhere), after every solve of the IVP step (``IVPSolver._solve_state``). P is exact
(Re of the grid values), real-linear and self-adjoint for the real pairing Re(a^H b), so the
tangent of the projected step is P (dS) and its adjoint applies P before the transposed solve
(``conatus.adjoint.FlowMap``). P is not complex-linear: it is applied by the time step only,
never inside ``system.solve`` (Krylov methods with complex scalars use the plain solves).

J is not written by hand for each geometry: it is read off the space's own transforms at
construction (two random probes through backward -> Re -> forward, the definition of the
projection) and kept only if it is a unit-modulus signed permutation of the m = 0 entries and an
involution, with the identity on the other entries (checked on an independent probe to
max(1e-12, 100 x the transforms' round-trip rounding, 90th percentile of the entries), declared);
otherwise the projection is off, ``explain()`` says why and an ``M0ProjectionWarning`` is issued
when the solver is built (never a silent change: the unprojected step is a substitution the user
did not ask for)."""
from __future__ import annotations

import numpy as np

__all__ = ["M0ProjectionWarning", "M0RealProjection", "PROBE_TOL", "RT_FACTOR", "RT_M0_FACTOR", "RT_MAX", "RT_QUANTILE", "m0_projection",
           "m0_projection_text"]

PROBE_TOL = 1e-12
"""Relative agreement required between the inferred map J and the transforms' projection on an
independent random probe (rounding of one backward/forward pair is ~1e-15 at small sizes). The
bound actually used is max(PROBE_TOL, RT_FACTOR x the transforms' own round-trip rounding measured
on the entries the projection leaves unchanged), declared by ``explain()``: the radial transforms
round at ~1e-12 from N_r = 32 on."""

RT_FACTOR = 100.0
"""Margin of the verification bound over the measured round-trip rounding of the transforms."""

RT_QUANTILE = 0.9
"""Quantile of the per-entry round-trip change taken as the transforms' rounding (robust to a
defect confined to a few entries)."""

RT_MAX = 1e-10
"""Round-trip rounding above which the transforms are not trusted to define J (projection off)."""

RT_M0_FACTOR = 10.0
"""Margin of the m = 0 verification bound over the largest round-trip change of the m = 0 entries
of a real probe (P z, a real field: the round trip must leave it unchanged up to rounding). The
m = 0 entries round more than the 90th percentile of the m > 0 entries (shell 256x127x64: the m = 0
map was verified to 5.8e-12 and its unit factors to 1.9e-11, against a bound of 4.7e-12 read on
the m > 0 entries; gpu2): their bound is measured on themselves, as a maximum (the residual
it bounds is a maximum). The m > 0 entries keep the quantile bound, so a defect confined to a few
of them is still caught."""


class M0ProjectionWarning(UserWarning):
    """The m = 0 real projection (on by default) turned itself off: the transforms' projection did
    not pass the check of J, so the solver keeps the unprojected step."""


class M0RealProjection:
    """P X = (X + J X) / 2 on the m = 0 entries of the stacked state of a Polar/SphereGlobalSystem.

    Attributes: ``active`` (bool), ``reason`` (why it is off), ``idx`` (positions of the m = 0
    entries in X), ``partner`` (positions of their partners), ``sigma`` (unit factors), ``err``
    (verification residual of the inferred J)."""

    def __init__(self, host, seed: int = 20261007) -> None:
        self.active, self.reason, self.err, self.n_null = False, "", float("nan"), 0
        # True once the check of J has run and refused it (a real basis with m = 0 entries: the
        # projection was wanted); False for "nothing to project" (complex basis, no m = 0 entry)
        self.failed = False
        self.rt, self.tol = float("nan"), PROBE_TOL
        self.rt_m0, self.tol_m0 = float("nan"), PROBE_TOL
        self.err_m0 = self.err_rest = float("nan")
        self.idx = self.partner = self.sigma = None
        self.n = int(host.n)
        S = host.plan.space
        if not getattr(S, "real", False):
            self.reason = "complex Fourier basis in phi (complex fields: nothing to project)"
            return
        rng = np.random.default_rng(seed)
        n = self.n

        def proj(z):                          # exact projection by the space's own transforms
            out = []
            for v, c in zip(host.variables, host.scatter(z), strict=True):
                # Re of the values on the dealiased grid (the round trip is exact there; on the
                # plain grid the top radial modes of a spin component are not, disk / ball)
                g = S.backward(c, v.domain, v.rank, True, real_tol=np.inf)
                out.append(S.forward(g, v.domain, v.rank, True))
            return host.gather_state(out, check_subspace=False)

        z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        Pz = proj(z)
        scale0 = float(np.abs(z).max())
        # changed entries, measured against the probe's scale: a per-entry relative test flagged an
        # m > 0 entry whose random value happened to be small (its round-trip rounding, ~1e-12 of
        # the scale at N_r = 32, exceeded 1e-10 of the entry), which then had no partner and turned
        # the projection off (pipe at (32, 32, 8))
        mask = np.abs(Pz - z) > 1e-8 * scale0
        # slots with no image on the grid (ball: spin +-1 of a vector at l = 0) are mapped to 0 by
        # the transforms' projection; they do not belong to any field value: left as they are
        null = mask & (np.abs(Pz) <= 1e-12 * np.abs(z))
        self.n_null = int(null.sum())
        mask &= ~null
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            self.reason = "no m = 0 entry changed by the projection"
            return
        # the transforms' own rounding, read on the entries the projection leaves unchanged: the
        # 90th percentile, so that a defect confined to a few entries is not taken for rounding
        # (it is then caught by the identity check of the independent probe)
        keep = ~mask & ~null
        self.rt = (float(np.quantile(np.abs(Pz[keep] - z[keep]), RT_QUANTILE)) / scale0
                   if keep.any() else 0.0)
        if self.rt > RT_MAX:
            self.failed = True
            self.reason = (f"the transforms' round trip rounds at {self.rt:.1e} of the probe "
                           f"(> {RT_MAX:g}): J is not defined to that accuracy")
            return
        self.tol = max(PROBE_TOL, RT_FACTOR * self.rt)
        tol = self.tol
        # rounding of the m = 0 entries themselves: a real probe (P z) is left unchanged by the
        # transforms' projection up to rounding; its largest change on the m = 0 entries sets their
        # bound (they round more than the m > 0 quantile at large sizes; gpu2)
        Pzr = proj(Pz)
        self.rt_m0 = float(np.abs(Pzr[idx] - Pz[idx]).max()) / scale0
        if self.rt_m0 > RT_MAX:
            self.failed = True
            self.reason = (f"the transforms' round trip of a real field rounds at {self.rt_m0:.1e} on "
                           f"the m = 0 entries (> {RT_MAX:g}): J is not defined to that accuracy")
            return
        self.tol_m0 = tol0 = max(tol, RT_M0_FACTOR * self.rt_m0)
        k = idx.size
        # partners by distinct moduli: |J z|_i = |z|_partner(i) for a unit-modulus permutation
        r = 1.0 + np.arange(k) / k
        zp = np.zeros(n, complex)
        zp[idx] = r * np.exp(1j * rng.uniform(0.0, 2 * np.pi, k))
        Jz = 2.0 * proj(zp) - zp
        loc = np.clip(np.rint((np.abs(Jz[idx]) - 1.0) * k).astype(int), 0, k - 1)
        sigma = Jz[idx] / np.conj(zp[idx][loc])
        # independent probe on every entry (m > 0 included): the transforms' projection must be
        # (z + J z)/2 on the m = 0 entries and the identity elsewhere, both to PROBE_TOL (a probe
        # zero outside the m = 0 entries would not test the identity part e216 review)
        z2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        z2[null] = 0.0
        Pz2 = proj(z2)
        pred = 0.5 * (z2[idx] + sigma * np.conj(z2[idx][loc]))
        scale = float(np.abs(z2).max())
        rest = ~mask & ~null                   # entries P must leave unchanged (m > 0)
        self.err_m0 = float(np.abs(Pz2[idx] - pred).max()) / scale
        self.err_rest = float(np.abs(Pz2[rest] - z2[rest]).max()) / scale if rest.any() else 0.0
        self.err = max(self.err_m0, self.err_rest)
        # J an involution (J^2 = I: partner of the partner, sigma_i = sigma_partner(i)), the
        # condition for P to be idempotent and self-adjoint for Re(a^H b)
        # (sigma = Jz / conj(zp) carries the transforms' rounding of the m = 0 entries divided by
        # |zp| >= 1, doubled by Jz = 2 P zp - zp: the m = 0 bound, twice)
        ok = (self.err_m0 <= tol0 and self.err_rest <= tol and np.all(loc[loc] == np.arange(k))
              and np.abs(np.abs(sigma) - 1.0).max() <= 2 * tol0
              and np.abs(sigma - sigma[loc]).max() <= 2 * tol0)
        if not ok:
            self.failed = True
            self.reason = (f"the transforms' projection is not (I + J)/2 with J a unit signed "
                           f"permutation and involution of the m = 0 entries, identity elsewhere "
                           f"(probe residual {self.err_m0:.1e} on the m = 0 entries, bound {tol0:.1e}; "
                           f"{self.err_rest:.1e} elsewhere, bound {tol:.1e}; unit factors within "
                           f"{np.abs(np.abs(sigma) - 1.0).max():.1e})")
            return
        self.active = True
        self.idx, self.partner = idx, idx[loc]
        # exact unit factors where they are +-1 (the usual case): no rounding added by P
        sr = np.where(np.abs(sigma - np.rint(sigma.real)) <= 2 * tol0, np.rint(sigma.real), sigma)
        self.sigma = sr.astype(complex)
        self._dev: dict = {}

    # ------------------------------------------------------------------ application
    def _arrays(self, xp):
        if xp is np:
            return self.idx, self.partner, self.sigma
        a = self._dev.get(id(xp))
        if a is None:
            a = self._dev[id(xp)] = (xp.asarray(self.idx), xp.asarray(self.partner), xp.asarray(self.sigma))
        return a

    def apply(self, X):
        """P X in place (NumPy or CuPy X of length n; graph-capturable on the GPU). Returns X."""
        if not self.active:
            return X
        from ..discretize._xp import array_module
        xp = array_module(X)
        idx, par, sig = self._arrays(xp)
        X[idx] = 0.5 * (X[idx] + sig * xp.conj(X[par]))
        return X

    def non_real(self, X) -> float:
        """max |X - P X| on the m = 0 entries (0 for an exactly real state)."""
        if not self.active:
            return 0.0
        from ..discretize._xp import array_module
        xp = array_module(X)
        idx, par, sig = self._arrays(xp)
        return float(xp.abs(0.5 * (X[idx] - sig * xp.conj(X[par]))).max())

    def explain(self) -> str:
        if not self.active:
            return f"m = 0 real projection: off ({self.reason})"
        npair = int(np.sum(self.partner != self.idx)) // 2
        return (f"m = 0 real projection after every solve of the step: {self.idx.size} m = 0 "
                f"entries ({npair} conjugate pairs"
                + (f"; {self.n_null} slot(s) with no grid image left as they are" if self.n_null else "")
                + "), P X = (X + J X)/2, exact, real-linear, "
                f"self-adjoint for Re(a^H b); J read off the transforms (verified to {self.err_m0:.1e} "
                f"on the m = 0 entries, bound {self.tol_m0:.0e} = max(bound elsewhere, {RT_M0_FACTOR:g} x "
                f"their round-trip rounding {self.rt_m0:.1e}); to {self.err_rest:.1e} elsewhere, bound "
                f"{self.tol:.0e} = max({PROBE_TOL:g}, {RT_FACTOR:g} x round-trip rounding {self.rt:.1e})); "
                "project_m0=False keeps the unprojected step")


def m0_projection(host, project_m0) -> M0RealProjection | None:
    """The projection used by a Polar/SphereIVPSolver (``project_m0``: True by default, declared);
    None when it is off (``project_m0=False``, complex fields, or a map that failed its check)
    -- the reason is kept for ``explain()`` (:func:`m0_projection_text`); a failed check also
    issues an ``M0ProjectionWarning`` (the user asked for the projection)."""
    from ..errors import OptionError
    if not isinstance(project_m0, (bool, np.bool_)):
        raise OptionError(f"project_m0 must be True or False, got {project_m0!r}",
                          fix="project_m0=True (default: m = 0 coefficients projected onto real "
                              "fields after every solve) or False (the unprojected step)")
    if not project_m0:
        return None
    pr = M0RealProjection(host)
    if pr.active:
        return pr
    if pr.failed:
        from ..errors import _warn_user       # the user's line, whatever the solver depth
        _warn_user(f"m = 0 real projection requested (project_m0=True, the default) but off: "
                   f"{pr.reason}. The step runs unprojected (non-real m = 0 rounding may grow "
                   f"relative to a decaying field); project_m0=False silences this.",
                   M0ProjectionWarning)
    return _Off(pr.reason)


class _Off:
    """An inactive projection (reason kept for explain); _solve_state treats it as None."""

    active = False

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def apply(self, X):
        return X

    def explain(self) -> str:
        return f"m = 0 real projection: off ({self.reason})"


def m0_projection_text(solver) -> str:
    """One explain() line on the m = 0 real projection of a curvilinear solver."""
    pr = getattr(solver, "_m0proj", None)
    if pr is None:
        return "\n  m = 0 real projection: off (project_m0=False, as requested)"
    return "\n  " + pr.explain()
