"""Numerical dye: a passive scalar c advected by a given velocity, with a declared diffusivity.

    dc/dt + u . grad c = kappa lap c          on a fully periodic grid

Pseudo-spectral, read-only with respect to the flow (the velocity is given, never modified):

* the diffusion is integrated **exactly** by an integrating factor exp(-kappa |k|^2 t);
* the advection -u . grad c is integrated by classical RK4 on the integrating-factor variable
  (IF-RK4; Cox & Matthews 2002 call it the "integrating factor RK4" scheme): 4th order in dt;
* products are formed on the grid; ``dealias="2/3"`` (declared) truncates the dye spectrum to
  |k_i| < n_i / 3 on every axis after each stage, ``dealias=None`` keeps the full spectrum
  (aliasing errors possible, declared);
* the velocity is frozen (arrays) or a callable ``u(t)`` returning the component arrays.

The mean of c is conserved exactly by the scheme when the velocity is divergence-free and the
product is resolved (dealiased): the k = 0 mode of u . grad c equals -<c div u> = 0.
``kappa = 0`` is allowed (pure advection) but declared, like everything else, in :meth:`Dye.explain`.
Grids with a Chebyshev axis are refused (:class:`~conatus.errors.UnsupportedError`): wall
boundary conditions for the dye are not implemented.

This is the **standalone** dye (a velocity you give, periodic grids only); ``sim.dye(...)``
(:mod:`conatus.model.dye`) is the one inside a running native simulation. Contracts compared in
:mod:`conatus.diagnostics.lagrangian` (finding 63).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import CoordinateError, OptionError, UnsupportedError
from .grid import Grid, Periodic

__all__ = ["Dye"]


class Dye:
    """Passive scalar on a periodic :class:`Grid` (``c0``: initial values, grid shape).

    ``kappa`` >= 0 (required, no default); ``dealias``: ``"2/3"`` or ``None`` (required)."""

    def __init__(self, grid: Grid, c0, *, kappa: float, dealias: str | None, t0: float = 0.0) -> None:
        for i, ax in enumerate(grid.axes):
            if not isinstance(ax, Periodic):
                raise UnsupportedError(f"dye on axis {i} ({ax!r}): only fully periodic grids",
                                       why="no wall boundary condition for the dye is implemented",
                                       fix="use Lagrangian streaklines/timelines near walls, or a "
                                           "periodic sub-box")
        if not kappa >= 0:
            raise OptionError(f"kappa = {kappa}: a diffusivity >= 0 is required")
        if dealias not in ("2/3", None):
            raise OptionError(f"dealias={dealias!r}: '2/3' or None (declared)")
        self.grid, self.xp = grid, grid.xp
        self.kappa, self.dealias, self.t, self.nsteps = float(kappa), dealias, float(t0), 0
        xp = self.xp
        c0 = xp.asarray(c0, dtype=float)
        if c0.shape != grid.shape:
            raise CoordinateError(f"c0 of shape {c0.shape} on a grid of shape {grid.shape}")
        d = grid.ndim
        ks = []
        for i, ax in enumerate(grid.axes):             # rfftn: last axis is the half spectrum
            real = i == d - 1
            f = np.fft.rfftfreq(ax.n, 1.0 / ax.n) if real else np.fft.fftfreq(ax.n, 1.0 / ax.n)
            ks.append((2 * np.pi * f / ax.L, f))
        K = np.meshgrid(*[k for k, _ in ks], indexing="ij")
        self._k = [xp.asarray(k) for k in K]
        self._k2 = xp.asarray(sum(k**2 for k in K))
        mask = np.ones(K[0].shape)
        if dealias == "2/3":
            I = np.meshgrid(*[f for _, f in ks], indexing="ij")
            for i, ax in enumerate(grid.axes):
                mask *= np.abs(I[i]) < ax.n / 3.0
        self._mask = xp.asarray(mask)
        self._axes = tuple(range(d))
        self.chat = self._fft(c0) * self._mask
        self.n_nonfinite = 0

    # ---------------------------------------------------------------- transforms
    def _fft(self, f):
        return self.xp.fft.rfftn(f, axes=self._axes)

    def _ifft(self, F):
        return self.xp.fft.irfftn(F, s=self.grid.shape, axes=self._axes)

    @property
    def c(self):
        """Dye values on the grid."""
        return self._ifft(self.chat)

    def mean(self) -> float:
        """Volume average of c."""
        return float(self.grid.mean(self.c))

    def variance(self) -> float:
        """<c^2> - <c>^2 (volume averages)."""
        c = self.c
        m = self.grid.mean(c)
        return float(self.grid.mean((c - m) ** 2))

    # ---------------------------------------------------------------- stepping
    def _N(self, chat, u: Sequence[Any]):
        """Fourier transform of -u . grad c (dealiased as declared)."""
        adv = 0
        for ui, ki in zip(u, self._k, strict=True):
            adv = adv + ui * self._ifft(1j * ki * chat)
        return -self._fft(adv) * self._mask

    def step(self, dt: float, u: Sequence[Any] | Callable[[float], Sequence[Any]], nsteps: int = 1) -> None:
        """``nsteps`` IF-RK4 steps of ``dt``; ``u``: d component arrays (frozen) or ``u(t)``."""
        xp = self.xp
        E = xp.exp(-self.kappa * self._k2 * dt / 2)     # exact diffusion over dt/2
        E2 = E * E
        from ..timestep.integrator import check_steps
        for _ in range(check_steps(nsteps, "Dye.step")):
            t = self.t
            uf = (lambda s: u(s)) if callable(u) else (lambda s: u)
            c = self.chat
            k1 = self._N(c, uf(t))
            k2 = self._N(E * (c + dt / 2 * k1), uf(t + dt / 2))
            k3 = self._N(E * c + dt / 2 * k2, uf(t + dt / 2))
            k4 = self._N(E2 * c + dt * E * k3, uf(t + dt))
            self.chat = E2 * c + dt / 6 * (E2 * k1 + 2 * E * (k2 + k3) + k4)
            self.t += dt
            self.nsteps += 1
        if not bool(xp.all(xp.isfinite(self.chat))):
            self.n_nonfinite += 1
            raise OptionError(f"dye became non-finite at t = {self.t:g}",
                              fix="reduce dt (advective CFL) or use dealias='2/3'")

    def explain(self) -> str:
        """Equation, scheme, diffusivity and dealiasing actually used."""
        return (f"Dye(dc/dt + u.grad c = {self.kappa:g} lap c"
                + (" (kappa = 0: pure advection, declared)" if self.kappa == 0 else "")
                + f"; pseudo-spectral IF-RK4 (diffusion exact, advection RK4), dealias "
                f"{self.dealias or 'none (aliasing possible)'}; t = {self.t:g} after {self.nsteps} "
                f"steps; {self.grid.explain()})")
