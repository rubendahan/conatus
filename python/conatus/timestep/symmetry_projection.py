"""Projection of a cylindrical IVP state onto the shift-reflect subspace (pipe travelling waves).

``ShiftReflect(phi, z)``: (u_phi, u_r, u_z)(phi, r, z) -> (-u_phi, u_r, u_z)(-phi, r, z + Lz/2),
times the declared parity p of each field. On the stored coefficients of a real field (m >= 0) it
is the antilinear map

    (J X)[a] = sigma[a] conj(X[partner[a]]),      partner: (m, k_z) -> (m, -k_z) (same radial index),

so the fixed subspace is a *real* subspace of the complex group vectors: it is not a reduction of
the complex group systems (as ``AzimuthalMFold`` and ``Reflection(z)`` are), and the IVP enforces it
by the projection P X = (X + J X) / 2 after every solve of the step, together with the m = 0 real
projection (``real_projection.py``). The operators being equivariant, P removes only rounding;
the removed part (the *leak*, max |X - J X| / max |X|) is recorded on every step (device scalar on
the GPU, no synchronisation) and reported (``solver.symmetry_leak()``); initial data outside the
subspace are refused (never projected in silence).

J is not written by hand: the partner of every entry is (same object, same m, same radial index,
k_z -> -k_z, the same or the spin-flipped tensor component) and the unit factors sigma are read off
the space's own transforms (backward on the dealiased grid -> the grid map phi_j -> phi_-j,
z_l -> z_{l + Mz/2}, sign p (-1)^(number of phi indices) -> forward) on random real fields, then
checked on an independent probe to ``PROBE_TOL``; otherwise the solver is refused (an internal
error, never a silent change). Cost: one gather, one conjugation and one average per step (no
transform at run time)."""
from __future__ import annotations

import numpy as np

from ..discretize.spin import flip as _flip_comp
from ..errors import SymmetryError
from .real_projection import RT_FACTOR, RT_MAX, RT_QUANTILE

__all__ = ["ShiftReflectProjection", "ProjectionChain", "shift_reflect_projection", "PROBE_TOL",
           "LEAK_MAX"]

PROBE_TOL = 1e-12
"""Relative agreement between the inferred antilinear map and the transforms (one backward/forward
pair of a random real field rounds to ~1e-15)."""

LEAK_MAX = 1e-9
"""Declared bound on the relative part removed by the projection in one step (equivariant operators:
rounding, ~1e-15). Initial data are refused above the verification bound of the inferred map,
max(PROBE_TOL, RT_FACTOR x the transforms' round-trip rounding) (``tol``, in explain())."""


class ShiftReflectProjection:
    """P X = (X + J X) / 2 for ``ShiftReflect`` on the stacked state of a PolarGlobalSystem
    (cylinder, RealFourier in phi). Attributes: ``idx`` (all entries), ``partner``, ``sigma``,
    ``err`` (verification residual), ``R`` (the generator)."""

    active = True

    def __init__(self, host, R, m0proj=None, seed: int = 20261007) -> None:
        plan = host.plan
        S = plan.space
        sym = plan.problem.symmetry
        if not (S.cyl and S.real):
            raise SymmetryError(f"{R!r}: needs a cylinder with a RealFourier azimuth (real fields)")
        Mz = S.sizes(True)[2]
        if Mz % 2:
            raise SymmetryError(f"{R!r}: the dealiased axial grid has {Mz} points (odd): the half-"
                                "period shift is not a grid permutation; use an even Nz * dealias")
        self.R, self.host, self.n = R, host, int(host.n)
        self.parity = [int(sym.parity(v, R)) for v in host.variables]
        self._m0 = m0proj if getattr(m0proj, "active", False) else None
        # candidate partners: same / spin-flipped component, k_z -> -k_z
        zf = S.zfreq
        look = {int(f): j for j, f in enumerate(zf)}
        zmir = np.array([look.get(int(-f), j) for j, f in enumerate(zf)], int)
        where = {(int(o), int(f)): a for a, (o, f) in enumerate(zip(host.cols_obj, host.cols_flat, strict=True))}
        cand = np.full((2, self.n), -1, int)
        for a, (o, f) in enumerate(zip(host.cols_obj, host.cols_flat, strict=True)):
            v = host.variables[int(o)]
            shape = host.var_shapes[int(o)]
            ix = list(np.unravel_index(int(f), shape))
            rank = v.rank
            if shape[-1] > 1:
                ix[-1] = zmir[ix[-1]]
            for c, comp in enumerate((tuple(ix[:rank]), _flip_comp(tuple(ix[:rank])))):
                jx = list(comp) + ix[rank:]
                cand[c, a] = where.get((int(o), int(np.ravel_multi_index(tuple(jx), shape))), -1)
        rng = np.random.default_rng(seed)
        z1, z2, z3 = (self._real(rng.standard_normal(self.n) + 1j * rng.standard_normal(self.n))
                      for _ in range(3))
        # the grid round trip must be the identity on real fields (else the probe is meaningless)
        rt = self._grid_map(z1, identity=True)
        e_rt = float(np.abs(rt - z1).max() / np.abs(z1).max())
        if e_rt > RT_MAX:
            raise SymmetryError(f"{R!r}: the transforms' round trip is not exact on this plan "
                                f"({e_rt:.1e} > {RT_MAX:g}); projection refused")
        # verification bound: max(PROBE_TOL, RT_FACTOR x the round-trip rounding (90th percentile)), the
        # radial transforms rounding at ~1e-12 from N_r = 32 on; declared
        q_rt = float(np.quantile(np.abs(rt - z1), RT_QUANTILE) / np.abs(z1).max())
        self.rt, self.tol = q_rt, max(PROBE_TOL, RT_FACTOR * q_rt)
        tol = self.tol
        J1, J2, J3 = (self._grid_map(z) for z in (z1, z2, z3))
        partner = np.full(self.n, -1, int)
        sigma = np.zeros(self.n, complex)
        scale = float(max(np.abs(z1).max(), np.abs(z2).max()))
        for c in (0, 1):
            todo = (partner < 0) & (cand[c] >= 0)
            if not todo.any():
                continue
            a = np.nonzero(todo)[0]
            w1, w2 = np.conj(z1[cand[c, a]]), np.conj(z2[cand[c, a]])
            # least squares over both probes (a single quotient / conj(z1) amplifies the
            # rounding of the transforms where the probe entry happens to be small)
            sg = (J1[a] * np.conj(w1) + J2[a] * np.conj(w2)) / (np.abs(w1) ** 2 + np.abs(w2) ** 2)
            ok = (np.abs(J1[a] - sg * w1) <= tol * scale) \
                & (np.abs(J2[a] - sg * w2) <= tol * scale) & (np.abs(np.abs(sg) - 1) <= 1e-8)
            partner[a[ok]] = cand[c, a[ok]]
            sigma[a[ok]] = sg[ok]
        if (partner < 0).any():
            bad = int(np.sum(partner < 0))
            raise SymmetryError(f"{R!r}: {bad} entries of the state have no conjugate partner with a "
                                "unit factor (the transforms do not act as the declared map): refused")
        sr = np.where(np.abs(sigma - np.rint(sigma.real)) <= 1e-8, np.rint(sigma.real), sigma)
        self.partner, self.sigma = partner, sr.astype(complex)
        pred = self.sigma * np.conj(z3[self.partner])
        self.err = float(np.abs(J3 - pred).max() / np.abs(z3).max())
        if self.err > tol:
            raise SymmetryError(f"{R!r}: the inferred map misses an independent probe by "
                                f"{self.err:.1e} > {tol:.1e}: refused")
        self._dev: dict = {}
        self._leak = 0.0
        self._leak_dev = None

    # ------------------------------------------------------------------ probes
    def _real(self, z):
        if self._m0 is None:
            return z
        return self._m0.apply(z.copy())

    def _grid_map(self, z, identity: bool = False):
        """The transforms' image of state z under the generator (or the plain round trip)."""
        host = self.host
        S = host.plan.space
        Mp, _, Mz = S.sizes(True)
        out = []
        for v, c, p in zip(host.variables, host.scatter(z), self.parity, strict=True):
            g = S.backward(c, v.domain, v.rank, True, real_tol=np.inf)
            if not identity:
                g = np.array(g)
                ax_p, ax_z = v.rank, v.rank + 2
                if g.shape[ax_p] > 1:
                    g = np.take(g, (-np.arange(Mp)) % Mp, axis=ax_p)
                if g.shape[ax_z] > 1:
                    g = np.take(g, (np.arange(Mz) + Mz // 2) % Mz, axis=ax_z)
                if v.rank:
                    sign = np.ones((3,) * v.rank)
                    for comp in np.ndindex(*(3,) * v.rank):
                        sign[comp] = (-1) ** sum(1 for i in comp if i == 0)
                    g = g * sign.reshape(sign.shape + (1, 1, 1))
                g = p * g
            out.append(S.forward(g, v.domain, v.rank, True))
        return host.gather_state(out, check_subspace=False)

    # ------------------------------------------------------------------ application
    def _arrays(self, xp):
        if xp is np:
            return self.partner, self.sigma
        a = self._dev.get(id(xp))
        if a is None:
            a = self._dev[id(xp)] = (xp.asarray(self.partner), xp.asarray(self.sigma))
        return a

    def image(self, X):
        """J X (NumPy or CuPy)."""
        from ..discretize._xp import array_module
        xp = array_module(X)
        par, sig = self._arrays(xp)
        return sig * xp.conj(X[par])

    def apply(self, X, record: bool = True):
        """P X in place; the removed part is recorded (``leak``) unless ``record`` is False (a
        projection of data on purpose, ``symmetrize_state``). Returns X."""
        from ..discretize._xp import array_module
        xp = array_module(X)
        JX = self.image(X)
        if not record:
            X[...] = 0.5 * (X + JX)
            return X
        d = xp.abs(X - JX).max() / 2
        s = xp.abs(X).max()
        rel = d / xp.maximum(s, 1e-300)
        if xp is np:
            self._leak = max(self._leak, float(rel))
        else:                                         # device scalar: no synchronisation
            self._leak_dev = rel if self._leak_dev is None else xp.maximum(self._leak_dev, rel)
        X[...] = 0.5 * (X + JX)
        return X

    def distance(self, X) -> float:
        """max |X - J X| / 2 relative to max |X| (0 for a state of the subspace)."""
        from ..discretize._xp import array_module
        xp = array_module(X)
        s = float(xp.abs(X).max())
        return float(xp.abs(X - self.image(X)).max()) / 2 / max(s, 1e-300)

    def leak(self, reset: bool = False) -> float:
        """Largest relative part removed by one projection since the last reset."""
        out = self._leak
        if self._leak_dev is not None:
            out = max(out, float(self._leak_dev))
        if reset:
            self._leak, self._leak_dev = 0.0, None
        return out

    def explain(self) -> str:
        """The projection, its inferred map, verification and leak bound."""
        npair = int(np.sum(self.partner != np.arange(self.n)))
        return (f"{self.R!r}: projection P X = (X + J X)/2 after every solve of the step, J antilinear "
                f"((m, k_z) <-> conj (m, -k_z), {npair} paired entries of {self.n}), read off the "
                f"transforms (verified to {self.err:.1e}, bound {self.tol:.0e}); not a reduction of the complex groups; "
                f"leak (part removed per step) recorded, bound {LEAK_MAX:g} checked at the end of every "
                f"run/run_graph and by solver.symmetry_leak(); initial data outside the subspace (> "
                f"{self.tol:.0e}) refused")


class ProjectionChain:
    """m = 0 real projection, then the shift-reflect projection, then the m = 0 projection again
    (removes the rounding of the average): the declared projection of the step of a cylindrical
    IVP in a shift-reflect subspace. Same interface as ``M0RealProjection`` for the solver."""

    active = True

    def __init__(self, m0proj, sproj: ShiftReflectProjection) -> None:
        self.m0 = m0proj if getattr(m0proj, "active", False) else None
        self.m0_any = m0proj
        self.s = sproj

    def apply(self, X, record: bool = True):
        """m = 0 projection, shift-reflect projection (leak recorded unless record=False), m = 0 again."""
        if self.m0 is not None:
            self.m0.apply(X)
        self.s.apply(X, record)
        if self.m0 is not None:
            self.m0.apply(X)
        return X

    def non_real(self, X) -> float:
        """Non-real part of the m = 0 entries of X (0 without the m = 0 projection)."""
        return 0.0 if self.m0 is None else self.m0.non_real(X)

    def explain(self) -> str:
        """Both projections of the step, in order."""
        m0 = ("m = 0 real projection: off (project_m0=False, as requested)" if self.m0_any is None
              else self.m0_any.explain())
        return m0 + "\n  " + self.s.explain()


def shift_reflect_projection(solver, host, m0proj):
    """The projection of a PolarIVPSolver whose problem declares ``ShiftReflect`` (else m0proj)."""
    from ..ir.symmetry import ShiftReflect
    sym = solver.plan.problem.symmetry
    gens = [R for R in (sym.generators if sym is not None else ()) if isinstance(R, ShiftReflect)]
    if not gens:
        return m0proj
    if len(gens) > 1:
        raise SymmetryError("one ShiftReflect generator at most")
    if m0proj is None:
        raise SymmetryError(f"{gens[0]!r} needs the m = 0 real projection (project_m0=True): its map "
                            "is defined on real fields")
    return ProjectionChain(m0proj, ShiftReflectProjection(host, gens[0], m0proj))
