"""Symmetry operators on the state vector of a 1D RealFourier problem.

The state ``X`` is the solver vector (``GlobalSystem`` ordering). Operators act on the RealFourier
coefficients ``[cos_0, msin_0, cos_1, msin_1, ...]`` (see ``reference/transforms.py``) of each field,
f = Σ c_n cos(k_n (x-a)) + s_n (-sin(k_n (x-a))), and are mapped back through ``scatter``/``gather``.

- translation τ_ℓ f(x) = f(x + ℓ): (c, s) → (c cos kℓ − s sin kℓ, c sin kℓ + s cos kℓ);
- generator ∂x (d/dℓ τ_ℓ at ℓ = 0): (c, s) → (−k s, k c);
- reflection R f(x) = −f(−x) about x = a (the KS symmetry u → −u(−x)): (c, s) → (−c, s).

Only one periodic RealFourier direction is supported; anything else is refused explicitly.
"""
from __future__ import annotations

import numpy as np

from ..errors import OptionError
from ..ir.bases import RealFourier

__all__ = ["FourierOps", "AxisOps"]


class FourierOps:
    """Translation, generator and reflection on the state of a 1D RealFourier IVP."""

    def __init__(self, solver) -> None:
        dom = solver.plan.domain
        bases = [b for b in dom.bases if b is not None]
        if len(bases) != 1 or not isinstance(bases[0], RealFourier):
            raise OptionError("FourierOps: exactly one RealFourier direction is supported "
                              f"(got {[type(b).__name__ for b in bases]})",
                              fix="use AxisOps(solver, coord) for several dimensions")
        self.basis = b = bases[0]
        self.system = solver.system
        N = b.size
        lo, hi = b.bounds
        self.L = float(hi - lo)
        self.k = 2 * np.pi * np.arange(N // 2) / self.L          # wavenumber of pair n
        # index of the mean (cos_0) in X: frozen by the ECS problems (conserved quantity)
        e = [np.zeros(N) for _ in self.system.variables]
        e[0][0] = 1.0
        self.mean_index = int(np.flatnonzero(self.system.gather_state(e, check_subspace=False))[0])

    # -- coefficient-space kernels -------------------------------------------------------
    def _map(self, X: np.ndarray, fn) -> np.ndarray:
        out = []
        for c in self.system.scatter(np.asarray(X)):
            c = np.array(c, copy=True)
            out.append(fn(c[0::2].copy(), c[1::2].copy(), c))
        return self.system.gather_state(out, check_subspace=False)

    def translate(self, X: np.ndarray, ell: float) -> np.ndarray:
        """τ_ℓ X, i.e. f(x) → f(x + ℓ)."""
        ck, sk = np.cos(self.k * ell), np.sin(self.k * ell)

        def fn(c, s, out):
            out[0::2], out[1::2] = c * ck - s * sk, c * sk + s * ck
            return out
        return self._map(X, fn)

    def ddx(self, X: np.ndarray) -> np.ndarray:
        """∂x X (generator of translations)."""
        def fn(c, s, out):
            out[0::2], out[1::2] = -self.k * s, self.k * c
            return out
        return self._map(X, fn)

    def reflect(self, X: np.ndarray) -> np.ndarray:
        """R X, i.e. f(x) → −f(−x) about the left end of the interval."""
        def fn(c, s, out):
            out[0::2], out[1::2] = -c, s
            return out
        return self._map(X, fn)

    def odd_mask(self, X_like: np.ndarray) -> np.ndarray:
        """Boolean mask of the dofs of the antisymmetric subspace R X = X (sin components)."""
        ones = np.ones_like(np.asarray(X_like, dtype=float))
        return np.abs(self.reflect(ones) - ones) < 0.5

    def explain(self) -> str:
        return (f"FourierOps(RealFourier N={self.basis.size}, L={self.L:.6g}): τ_ℓ f(x)=f(x+ℓ), "
                "∂x generator, R f(x)=−f(−x); mean (cos_0) index "
                f"{self.mean_index} is conserved and frozen by ECS solvers")


class AxisOps:
    """Translation f(·) → f(· + ℓ ê) and its generator along one RealFourier coordinate, for any
    number of variables and dimensions (variables without that coordinate are unchanged, their
    derivative is zero). Same (cos, −sin) pair convention as :class:`FourierOps`."""

    def __init__(self, solver, coord: str) -> None:
        self.system = solver.system
        self.coord = coord
        self.axes, self.k = [], None
        for v in self.system.variables:
            names = [b.coord.name for b in v.domain.bases]
            if coord in names:
                b = v.domain.bases[names.index(coord)]
                if not isinstance(b, RealFourier):
                    raise OptionError(f"AxisOps: {coord} is not a RealFourier direction of {v.name}",
                                      fix="translate along a periodic (RealFourier) coordinate")
                lo, hi = b.bounds
                k = 2 * np.pi * np.arange(b.size // 2) / (hi - lo)
                if self.k is not None and not np.allclose(k, self.k, rtol=1e-12, atol=0.0):
                    raise OptionError("AxisOps: variables disagree on the wavenumbers of " + coord,
                                      fix=f"give every variable the same bounds along {coord}")
                self.k = k
                self.axes.append(names.index(coord))
            else:
                self.axes.append(None)
        if self.k is None:
            raise OptionError(f"AxisOps: no variable has a RealFourier basis along {coord}",
                              fix="name a periodic coordinate of the domain")

    def _map(self, X, fn):
        out = []
        for c, ax in zip(self.system.scatter(np.asarray(X)), self.axes, strict=True):
            c = np.array(c, dtype=float, copy=True)
            if ax is None:
                out.append(fn(None, None, c, None))
                continue
            c = np.moveaxis(c, ax, 0)
            shape = (-1,) + (1,) * (c.ndim - 1)
            res = fn(c[0::2].copy(), c[1::2].copy(), c, shape)
            out.append(np.moveaxis(res, 0, ax))
        return self.system.gather_state(out, check_subspace=False)

    def translate(self, X, ell: float):
        ck, sk = np.cos(self.k * ell), np.sin(self.k * ell)

        def fn(c, s, out, shape):
            if c is None:
                return out
            out[0::2] = c * ck.reshape(shape) - s * sk.reshape(shape)
            out[1::2] = c * sk.reshape(shape) + s * ck.reshape(shape)
            return out
        return self._map(X, fn)

    def ddx(self, X):
        def fn(c, s, out, shape):
            if c is None:
                return np.zeros_like(out)
            k = self.k.reshape(shape)
            out[0::2], out[1::2] = -k * s, k * c
            return out
        return self._map(X, fn)

    def explain(self) -> str:
        return (f"AxisOps({self.coord}): τ_ℓ f = f(· + ℓ ê_{self.coord}) and generator ∂{self.coord}, "
                f"{self.k.size} wavenumbers")
