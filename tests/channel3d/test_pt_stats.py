"""Spectral plane averages of the PT channel (pt_stats.PTStats) = averages of the velocity on the
grid computed by the reference evaluator (pt_velocity_grid). The xz-mean of a product of two
band-limited fields is exact on the N-point Fourier grid (no aliasing into k = 0)."""
import numpy as np

from ch3d_cases import channel_3d_pt, pt_velocity_grid
from conatus.reference.transforms import grid as grid1d
from pt_stats import PTStats


def test_spectral_moments_match_grid_averages():
    Nx, Ny, Nz, Lx, Lz = 8, 13, 10, 2 * np.pi, np.pi
    plan, f = channel_3d_pt(Nx, Ny, Nz, 100.0, Lx, Lz, pressure_gradient=0.02)
    rng = np.random.default_rng(3)
    co = {k: rng.standard_normal((Nx, Ny, Nz)) for k in ("phi", "psi")}
    for k in co:                              # (0, 0) group carries no velocity; sin(0) = 0
        co[k][:2, :, :2] = 0.0
        co[k][1, :, :] = 0.0
        co[k][:, :, 1] = 0.0
    co["U"] = rng.standard_normal((1, Ny, 1))
    co["W"] = np.zeros((1, Ny, 1))
    vel = pt_velocity_grid(plan, f, co)                        # (3, Nx, Ny, Nz)
    y = grid1d(plan.domain.basis(f["y"]), dealias=False)
    st = PTStats(Nx, Ny, Nz, Lx, Lz, y)
    m = st.moments(co["phi"], co["psi"])
    up = vel[0] - vel[0].mean(axis=(0, 2), keepdims=True)
    vp, wp = vel[1], vel[2]
    ref = dict(uu=(up * up).mean(axis=(0, 2)), vv=(vp * vp).mean(axis=(0, 2)),
               ww=(wp * wp).mean(axis=(0, 2)), uv=(up * vp).mean(axis=(0, 2)))
    for k in ref:
        assert np.max(np.abs(m[k] - ref[k])) < 1e-11 * np.max(np.abs(ref[k])), k
    assert np.max(np.abs(st.mean(co["U"]) - vel[0].mean(axis=(0, 2)))) < 1e-11
