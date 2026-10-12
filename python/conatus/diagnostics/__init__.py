"""Declared, read-only diagnostics of a flow: Eulerian, Lagrangian, mixing.

Nothing here modifies a computation: every function reads physical values on a declared
:class:`Grid` (NumPy or CuPy, same code) and returns new arrays. Spec, conventions and tests:
 ``explain()`` lists every diagnostic with its definition.

    from conatus import diagnostics as dg
    g = dg.Grid([dg.Periodic(4 * pi, 192), dg.Chebyshev(-1, 1, 129), dg.Periodic(2 * pi, 160)])
    Q = dg.q_criterion(g, [u, v, w])
"""
from . import dye, euler, lagrangian, mixing
from .dye import Dye
from .euler import (dissipation, divergence, energy_spectrum, enstrophy, helicity, kinetic_energy, lambda2,
                    poloidal_toroidal, pressure, q_criterion, spectrum_1d, strain_rotation, streamfunction,
                    streamfunction_from_vorticity, sym3_eigvals, vector_potential, velocity_gradient,
                    vorticity)
from .grid import Chebyshev, Grid, Periodic, array_module
from .lagrangian import (AnalyticVelocity, Emitter, GridVelocity, Interpolator, MaxeyRiley, Particles,
                         PathRecorder, Streaklines, Timelines, ftle)
from .mixing import mix_norm, mixing_efficiency, mixing_rate, potential_energies, variance

__all__ = ["Chebyshev", "Grid", "Periodic", "array_module", "euler", "mixing", "lagrangian", "dye", "explain",
           "AnalyticVelocity", "Emitter", "GridVelocity", "Interpolator", "MaxeyRiley", "Particles",
           "PathRecorder", "Streaklines", "Timelines", "ftle", "Dye",
           "divergence", "dissipation", "energy_spectrum", "enstrophy", "helicity",
           "kinetic_energy", "lambda2", "poloidal_toroidal", "pressure", "q_criterion",
           "spectrum_1d", "strain_rotation", "streamfunction", "streamfunction_from_vorticity",
           "sym3_eigvals", "vector_potential", "velocity_gradient", "vorticity",
           "mix_norm", "mixing_efficiency", "mixing_rate", "potential_energies", "variance"]

_CATALOGUE = [
    ("vorticity", "2D: dx v - dy u; 3D: curl u (spectral derivatives)"),
    ("streamfunction", "u = dy psi, v = -dx psi: periodic^2 (Poisson + mean flow) or periodic x Chebyshev (mode by mode)"),
    ("vector_potential", "curl A = u - <u>, div A = 0, periodic 3D box"),
    ("poloidal_toroidal", "u = curl curl(phi e_w) + curl(psi e_w) + means (Conatus channel convention)"),
    ("q_criterion", "(|W|^2 - |S|^2)/2 (Hunt, Wray & Moin 1988)"),
    ("lambda2", "middle eigenvalue of S^2 + W^2 (Jeong & Hussain 1995); eigvalsh or closed form"),
    ("helicity", "u . omega"),
    ("dissipation", "2 nu S:S (nu declared)"),
    ("pressure", "lap p = -d_i u_j d_j u_i, periodic box, zero mean"),
    ("energy_spectrum / spectrum_1d", "shell sums (dk = min 2pi/L) / one-sided 1D spectra, Parseval exact"),
    ("Interpolator", "field at points: exact spectral series, or Lagrange of declared order"),
    ("Particles", "tracers / inertial (Maxey-Riley simplified, neglected terms listed), RK4 or RK2"),
    ("PathRecorder / Streaklines / Timelines", "pathlines, smoke streaklines, hydrogen-bubble timelines (NaN = not emitted)"),
    ("ftle", "ln sqrt(lambda_max(C))/|T|, Cauchy-Green by centred neighbour differences"),
    ("Dye", "passive scalar, declared kappa, IF-RK4 pseudo-spectral (periodic box)"),
    ("variance / mix_norm", "<c'^2>; H^{-s} norm of c' (Mathew, Mezic & Petzold 2005), s = 1/2 default"),
    ("potential_energies", "E_p, E_b by sorting, APE = E_p - E_b >= 0 (Winters et al. 1995)"),
    ("mixing_rate / mixing_efficiency", "Phi_d = dE_b/dt - Phi_i (Phi_i declared); eta = Phi_d/(Phi_d + eps)"),
]


def explain() -> str:
    """Every diagnostic of this package with its definition (one line each)."""
    w = max(len(n) for n, _ in _CATALOGUE)
    return "conatus.diagnostics (read-only, NumPy/CuPy):\n" + "\n".join(
        f"  {n:<{w}}  {d}" for n, d in _CATALOGUE)
