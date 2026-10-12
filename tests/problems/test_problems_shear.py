"""conatus.problems shear flows: Orszag (1971) for the channel, Romanov (1973) stability for
plane Couette, and the laminar profiles as steady states of the 3D IVPs."""
import numpy as np

from conatus import problems as P
from conatus.discretize import discretize
from conatus.solvers import eigenvalues
from conatus.timestep import IVPSolver

C_ORSZAG = 0.23752649 + 0.00373967j        # Orszag (1971), Re = 1e4, alpha = 1


def _physical(r, cut=2.0):
    """Converged-looking eigenvalues (|c| < cut: discards tau modes) and their residuals."""
    res = eigenvalues(discretize(r.problem, ncc=r.ncc))
    keep = np.abs(res.values) < cut
    return res.values[keep], res.residuals[keep]


def test_channel_orr_sommerfeld_orszag():
    c, resid = _physical(P.channel_evp(N=64, Re=1e4, alpha=1.0))
    i = int(np.argmax(c.imag))
    assert abs(c[i] - C_ORSZAG) < 1e-8, c[i]
    assert resid[i] < 1e-10


def test_plane_couette_stable_and_symmetric():
    """Romanov (1973): no unstable mode; spectrum symmetric under c -> -conj(c); the
    least-stable mode is converged in N (64 vs 96) at Re = 1e4."""
    for Re in (1e3, 1e5):
        c, _ = _physical(P.plane_couette_evp(N=96, Re=Re, alpha=1.0), cut=1.0)
        assert c.imag.max() < 0, (Re, c.imag.max())
    c64, _ = _physical(P.plane_couette_evp(N=64, Re=1e4, alpha=1.0), cut=1.0)
    c96, _ = _physical(P.plane_couette_evp(N=96, Re=1e4, alpha=1.0), cut=1.0)
    lead64, lead96 = c64[np.argsort(-c64.imag)[:2]], c96[np.argsort(-c96.imag)[:2]]
    assert np.abs(np.sort_complex(lead64) - np.sort_complex(lead96)).max() < 1e-8
    top = c96[np.argsort(-c96.imag)[:6]]
    assert all(np.min(np.abs(c96 + np.conj(z))) < 1e-8 for z in top)


def _laminar_drift(r, u, init, steps):
    """max |u(t) - u(0)| after ``steps`` RK443 steps of dt = 0.01 from the laminar state."""
    s = IVPSolver(discretize(r.problem, ncc=r.ncc), "RK443", 0.01)
    s.set_state({u: init})
    g0 = s.grid("u").copy()
    s.run(steps)
    return float(np.abs(s.grid("u") - g0).max())


def test_channel_and_couette_ivp_keep_laminar_profile():
    Re = 100.0
    ch = P.channel(Nx=4, Ny=12, Nz=4, Lx=2 * np.pi, Lz=np.pi, Re=Re, pressure_gradient=2 / Re,
                   dealias=1.5)
    drift = _laminar_drift(ch, ch.fields["u"],
                           lambda X, Y, Z: np.array([1 - Y ** 2, 0 * Y, 0 * Y]) + 0 * X + 0 * Z, 10)
    assert drift < 1e-12, drift
    pc = P.plane_couette(Nx=4, Ny=12, Nz=4, Lx=2 * np.pi, Lz=np.pi, Re=Re, dealias=1.5)
    drift = _laminar_drift(pc, pc.fields["u"],
                           lambda X, Y, Z: np.array([Y, 0 * Y, 0 * Y]) + 0 * X + 0 * Z, 10)
    assert drift < 1e-12, drift
