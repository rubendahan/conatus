"""Nonzero dynamic quantities against exact values, one per way a term reaches M or L.

The defect (a bare unknown compiled into the mass matrix) left every threshold and steady-state
check green: sigma = 0 and L u = f do not see a wrong scale of M. Each test below measures a growth
or decay rate that is *not* zero and compares it with its closed form, for one way a term enters
the pencil (L + sigma M) or the IVP (M dt X + L X = F):

  coefficient on dt (c dt(u), dt(c u)), scaled unknown in L (a u), explicit RHS (b u, evaluator and
  fused F), lifts / tau rows (Dirichlet walls), NCC in L (f(z) u) and in M (f(z) dt(u)),
  reflection subspace (reduced solve), EVP with a bare term and with a scaled sigma term,
  linearize() (dt -> sigma), parametric pencil (sigma and d sigma / d p), disk, annulus, cylinder,
  sphere S^2.

References are closed forms only: Fourier mode e^{ikx}, sin(n pi (z + 1) / 2) on (-1, 1), zeros of
J_m (scipy.special.jn_zeros, Abramowitz & Stegun Table 9.5), the ODE u' = -u / f(z), l (l + 1) on S^2.
Every LHS carries a *bare* unknown ``+ u`` (coefficient 1, the case: ``a * u`` was always
compiled correctly), so every test fails if is reintroduced; checked in the by
monkeypatching ``Compiler._c``, ``_PolarCompiler._c``, ``_SphereCompiler._c`` (and the EVP split
``compiler._esplit``) to compile a bare Field into M again: 24 failures out of 24.
"""
import numpy as np
import pytest
from scipy.special import jn, jn_zeros

import conatus.expert as C
from conatus.ir import Parity
from conatus.discretize import discretize
from conatus.reference import transforms as T

K = 3                      # Fourier wavenumber on (0, 2 pi)
A, B, DIFF = 0.4, -0.25, 0.3   # coefficient of a u (L), of b u (RHS), diffusivity; + u is bare


def _a():
    """The coefficient a as a Parameter: u + a u stays two terms (u + 0.4 u would fold into 1.4 u
    and hide the bare unknown this file is about)."""
    return C.Parameter("a", A)


def _rate(s, name, u0_max, t):
    """Observed exponential rate log(max|u(t)| / max|u(0)|) / t."""
    return np.log(np.abs(s.grid(name)).max() / u0_max) / t


# ----------------------------------------------------------------------------------------------
# Cartesian 1D Fourier: c dt(u) + u - a u - D u_xx = b u  ->  rate (a + b - 1 - D k^2) / c
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["c*dt(u)", "dt(c*u)"])
@pytest.mark.parametrize("scheme,dt,tol", [("RK443", 2e-3, 1e-8), ("SBDF2", 1e-3, 5e-6)])
@pytest.mark.parametrize("f_backend", ["evaluator", "fused"])
def test_fourier_rate_with_mass_coefficient_bare_term_and_rhs(form, scheme, dt, tol, f_backend):
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    b = C.RealFourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)
    u = C.Field("u", C.Domain(cs, [b]))
    c = C.Parameter("c", 2.0)
    pb = C.IVP([u])
    mass = c * C.dt(u) if form == "c*dt(u)" else C.dt(c * u)
    pb.add_equation((mass + u - _a() * u - DIFF * C.lap(u), B * u))
    s = C.IVPSolver(discretize(pb), scheme, dt, f_backend=f_backend)
    s.set_state({u: lambda X: np.cos(K * X)})
    s.run(int(round(0.5 / dt)))
    exact = (A + B - 1 - DIFF * K ** 2) / 2.0
    assert abs(_rate(s, "u", 1.0, s.t) - exact) < tol


# ----------------------------------------------------------------------------------------------
# Chebyshev with lifts (Dirichlet): c dt(u) + u + a u - D u'' + taus = 0
#   -> rate -(1 + a + D (n pi/2)^2) / c
# ----------------------------------------------------------------------------------------------

def _cheb_ivp(c=2.0, symmetric=False, N=24):
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, N, (-1, 1), dealias=3 / 2)
    D = C.Domain(cs, [b])
    u, t1, t2 = C.Field("u", D), C.Field("t1", D.without(z)), C.Field("t2", D.without(z))
    G = C.SymmetrySubspace(C.Reflection(z, about=0), u=C.EVEN, t1=C.ODD, t2=C.EVEN) if symmetric else None
    pb = C.IVP([u, t1, t2], symmetry=G)
    pb.add_equation((c * C.dt(u) + u + _a() * u - DIFF * C.lap(u) + C.lift(t1, b, -1) + C.lift(t2, b, -2), 0))
    pb.add_equation((C.interp(u, z, "left"), 0))
    pb.add_equation((C.interp(u, z, "right"), 0))
    return discretize(pb), u, b


@pytest.mark.parametrize("n", [1, 2])
def test_chebyshev_dirichlet_rate_with_lifts(n):
    plan, u, b = _cheb_ivp()
    s = C.IVPSolver(plan, "RK443", 2e-3)
    s.set_state({u: lambda Z: np.sin(n * np.pi * (Z + 1) / 2)})
    u0 = np.abs(s.grid("u")).max()          # grid maximum (z = 0 is not a Gauss point)
    s.run(250)
    exact = -(1 + A + DIFF * (n * np.pi / 2) ** 2) / 2.0
    assert abs(_rate(s, "u", u0, s.t) - exact) < 1e-8


def test_chebyshev_rate_in_reflection_subspace():
    plan, u, b = _cheb_ivp(symmetric=True)
    s = C.IVPSolver(plan, "RK443", 2e-3)
    assert s.system.reduced
    s.set_state({u: lambda Z: np.cos(np.pi * Z / 2)})
    u0 = np.abs(s.grid("u")).max()          # grid maximum (z = 0 is not a Gauss point)
    s.run(250)
    exact = -(1 + A + DIFF * (np.pi / 2) ** 2) / 2.0
    assert abs(_rate(s, "u", u0, s.t) - exact) < 1e-8


# ----------------------------------------------------------------------------------------------
# NCC: f(z) u in L, f(z) dt(u) in M (pointwise ODEs with exact solutions)
# ----------------------------------------------------------------------------------------------

def _ncc_problem(where):
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, 16, (-1, 1), dealias=1)
    D = C.Domain(cs, [b])
    u, f = C.Field("u", D), C.Field("f", D)
    pb = C.IVP([u])
    if where == "L":            # dt(u) + u - z u = 0: u = e^{(z - 1) t}
        pb.add_equation((C.dt(u) + u - f * u, 0))
        return discretize(pb, ncc={f: lambda Z: Z}), u, b, lambda Z, t: np.exp((Z - 1) * t)
    if where == "M":            # (2 + z) dt(u) + u = 0: u = e^{-t / (2 + z)}
        pb.add_equation((f * C.dt(u) + u, 0))
    else:                       # dt((2 + z) u) + u = 0: same solution
        pb.add_equation((C.dt(f * u) + u, 0))
    return discretize(pb, ncc={f: lambda Z: 2 + Z}), u, b, lambda Z, t: np.exp(-t / (2 + Z))


@pytest.mark.parametrize("where", ["L", "M", "dt(f u)"])
def test_ncc_rate_is_pointwise_exact(where):
    plan, u, b, exact = _ncc_problem(where)
    s = C.IVPSolver(plan, "RK443", 2e-3)
    s.set_state({u: lambda Z: 1.0 + 0 * Z})
    s.run(500)
    Z = T.grid(b, dealias=False)
    assert np.abs(s.grid("u") - exact(Z, s.t)).max() < 1e-9


# ----------------------------------------------------------------------------------------------
# EVP: bare term, scaled sigma term, linearize(), parametric pencil and sensitivities
# ----------------------------------------------------------------------------------------------

def _cheb_evp(c_value=2.0):
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, 32, (-1, 1), dealias=1)
    D = C.Domain(cs, [b])
    u, t1, t2 = C.Field("u", D), C.Field("t1", D.without(z)), C.Field("t2", D.without(z))
    s, a, d, c = C.Parameter("s"), C.Parameter("a", A), C.Parameter("D", DIFF), C.Parameter("c", c_value)
    pb = C.EVP([u, t1, t2], eigenvalue=s)
    pb.add_equation((s * (c * u) + u + a * u - d * C.lap(u) + C.lift(t1, b, -1) + C.lift(t2, b, -2), 0))
    pb.add_equation((C.interp(u, z, "left"), 0))
    pb.add_equation((C.interp(u, z, "right"), 0))
    return pb, (a, d, c)


def _exact_sigmas(n, c=2.0):
    return -(1 + A + DIFF * (np.arange(1, n + 1) * np.pi / 2) ** 2) / c


def test_evp_bare_term_and_scaled_sigma():
    pb, _ = _cheb_evp()
    plan = discretize(pb)
    res = C.eigenvalues(plan, plan.groups[0])
    w = np.sort(res.values.real[np.isfinite(res.values)])[::-1][:4]
    assert np.abs(w - _exact_sigmas(4)).max() < 1e-10
    assert res.residuals.max() < 1e-10


def test_linearize_turns_dt_into_sigma_with_the_right_scale():
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, 32, (-1, 1), dealias=3 / 2)
    D = C.Domain(cs, [b])
    u, t1, t2 = C.Field("u", D), C.Field("t1", D.without(z)), C.Field("t2", D.without(z))
    ivp = C.IVP([u, t1, t2])
    ivp.add_equation((2.0 * C.dt(u) + u + _a() * u - DIFF * C.lap(u) + C.lift(t1, b, -1) + C.lift(t2, b, -2),
                      -u * u))
    ivp.add_equation((C.interp(u, z, "left"), 0))
    ivp.add_equation((C.interp(u, z, "right"), 0))
    zero = C.Field("zero", D)
    evp = C.linearize(ivp, {u: zero, t1: C.Field("z1", D.without(z)), t2: C.Field("z2", D.without(z))},
                      C.Parameter("s"))
    plan = discretize(evp, ncc={zero: lambda Z: 0 * Z})
    vals = C.eigenvalues(plan, plan.groups[0]).values
    w = np.sort(vals.real[np.isfinite(vals)])[::-1][:3]
    assert np.abs(w - _exact_sigmas(3)).max() < 1e-10


def test_parametric_pencil_sigma_and_sensitivities():
    pb, (a, d, c) = _cheb_evp()
    pe = C.ParametricEVP(pb, [a, d, c])
    for vals in ({"a": A, "D": DIFF, "c": 2.0}, {"a": 1.3, "D": 0.7, "c": 0.5}):
        m = pe.least_stable(**vals)
        sig = -(1 + vals["a"] + vals["D"] * (np.pi / 2) ** 2) / vals["c"]
        assert abs(m.sigma - sig) < 1e-10
        assert abs(pe.sensitivity(m, "a") - (-1 / vals["c"])) < 1e-9
        assert abs(pe.sensitivity(m, "D") - (-(np.pi / 2) ** 2 / vals["c"])) < 1e-9
        assert abs(pe.sensitivity(m, "c") - (-sig / vals["c"])) < 1e-9


# ----------------------------------------------------------------------------------------------
# Polar: disk, annulus, cylinder (PolarPlan has its own compiler: was also there)
# ----------------------------------------------------------------------------------------------

PC = C.PolarCoordinates("phi", "r")
PHI, R = PC["phi"], PC["r"]


def test_polar_evp_bare_term_is_not_in_mass():
    D = C.Domain(PC, [C.ComplexFourier(PHI, 8, (0, 2 * np.pi), dealias=1), C.DiskBasis(R, 12, 1.0, dealias=1)])
    u, s = C.Field("u", D, dtype=complex), C.Parameter("s", dtype=complex)
    pb = C.EVP([u], eigenvalue=s)
    pb.add_equation((s * u + u, 0))
    plan = discretize(pb)
    for m in (0, 1, -3):
        assert np.allclose(C.eigenvalues(plan, plan.group(m)).values, -1.0)


@pytest.mark.parametrize("m", [0, 2])
def test_disk_dirichlet_decay_with_bare_term(m):
    """2 dt(u) + u + a u - lap u + tau = 0, u = J_m(j r) cos(m phi): rate -(1 + a + j^2) / 2."""
    br = C.DiskBasis(R, 20, 1.0, dealias=3 / 2)
    D = C.Domain(PC, [C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=3 / 2), br])
    u, t1 = C.Field("u", D), C.Field("t1", D.without(R))
    pb = C.IVP([u, t1])
    pb.add_equation((2.0 * C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2), 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    j = jn_zeros(m, 1)[0]
    s = C.IVPSolver(discretize(pb), "RK443", 5e-4)     # |sigma| dt <= 7e-3: RK443 error ~ 1e-7
    s.set_state({u: lambda P, Rr: jn(m, j * Rr) * np.cos(m * P)})
    u0 = np.abs(s.grid("u")).max()
    s.run(400)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A + j ** 2) / 2) < 1e-7


def test_annulus_evp_bare_term_and_scaled_sigma():
    """s (2 u) + u + a u - lap u + taus = 0 on 0.5 < r < 1.5, m = 0: s = -(1 + a + l^2) / 2, l the first
    root of J_0(l a) Y_0(l b) - J_0(l b) Y_0(l a) (A&S 9.5.27; 2.9811... for a = 0.5, b = 1.5)."""
    from scipy.optimize import brentq
    from scipy.special import yn
    br = C.AnnulusBasis(R, 32, (0.5, 1.5), dealias=1)
    D = C.Domain(PC, [C.RealFourier(PHI, 4, (0, 2 * np.pi), dealias=1), br])
    u = C.Field("u", D, dtype=complex)
    t1, t2 = C.Field("t1", D.without(R), dtype=complex), C.Field("t2", D.without(R), dtype=complex)
    s = C.Parameter("s")
    pb = C.EVP([u, t1, t2], eigenvalue=s)
    pb.add_equation((s * (2.0 * u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2)
                     + C.lift(t2, br, -2, lam=2), 0))
    pb.add_equation((C.interp(u, R, "left"), 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    plan = discretize(pb)
    vals = C.eigenvalues(plan, plan.group(0)).values
    lead = np.sort(vals.real[np.isfinite(vals)])[::-1][0]
    lam = brentq(lambda l: jn(0, 0.5 * l) * yn(0, 1.5 * l) - jn(0, 1.5 * l) * yn(0, 0.5 * l), 2.0, 4.0)
    assert abs(lead + (1 + A + lam ** 2) / 2) < 1e-9


def test_cylinder_decay_with_bare_term():
    """dt(u) + u + a u - lap u + tau = 0, u = J_0(j r) sin(z): rate -(1 + a + j^2 + 1)."""
    CC = C.CylindricalCoordinates("phi", "r", "z")
    phi, r, zc = CC["phi"], CC["r"], CC["z"]
    br = C.DiskBasis(r, 16, 1.0, dealias=3 / 2)
    D = C.Domain(CC, [C.RealFourier(phi, 4, (0, 2 * np.pi), dealias=3 / 2), br,
                      C.ComplexFourier(zc, 8, (0, 2 * np.pi), dealias=3 / 2)])
    u, t1 = C.Field("u", D), C.Field("t1", D.without(r))
    pb = C.IVP([u, t1])
    pb.add_equation((C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2), 0))
    pb.add_equation((C.interp(u, r, "right"), 0))
    j = jn_zeros(0, 1)[0]
    s = C.IVPSolver(discretize(pb), "RK443", 1e-3)
    s.set_state({u: lambda P, Rr, Zz: jn(0, j * Rr) * np.sin(Zz)})
    u0 = np.abs(s.grid("u")).max()
    s.run(100)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A + j ** 2 + 1)) < 1e-7


# ----------------------------------------------------------------------------------------------
# Sphere S^2 (SpherePlan has its own compiler, with the guard): exact decay of Y_l^0
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ell,Y", [(1, lambda T: np.cos(T)), (2, lambda T: 3 * np.cos(T) ** 2 - 1)])
def test_sphere_decay_with_bare_term(ell, Y):
    """2 dt(u) + u + a u - lap u = 0 on the unit sphere, u = Y_l^0: rate -(1 + a + l (l + 1)) / 2."""
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(s2["theta"], 8, radius=1.0, dealias=3 / 2)])
    u = C.Field("u", D)
    pb = C.IVP([u])
    pb.add_equation((2.0 * C.dt(u) + u + _a() * u - C.lap(u), 0))
    s = C.IVPSolver(discretize(pb), "RK443", 1e-3)
    s.set_state({u: lambda P, T: Y(T) + 0 * P})
    u0 = np.abs(s.grid("u")).max()
    s.run(200)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A + ell * (ell + 1)) / 2) < 1e-8


# ----------------------------------------------------------------------------------------------
# Curvilinear IVPs on the GPU (backend='cuda', timestep/curvilinear_device.py): the same exact
# rates, with the explicit RHS b u evaluated by the CuPy evaluator (disk, cylinder, S^2, ball,
# shell). An agreement between backends does not see an error common to both: these do.
# ----------------------------------------------------------------------------------------------

def _gpu_or_skip():
    cp = pytest.importorskip("cupy")
    from conatus.runtime import cuda
    if not cuda.available() or cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")


@pytest.mark.gpu
@pytest.mark.parametrize("geom", ["disk", "cylinder"])
def test_gpu_polar_decay_with_bare_term_and_rhs(geom):
    """2 dt(u) + u + a u - lap u + tau = b u on the GPU: disk u = J_2(j r) cos(2 phi), rate
    -(1 + a - b + j^2) / 2; cylinder u = J_0(j r) sin(z), rate -(1 + a - b + j^2 + 1) / 2."""
    _gpu_or_skip()
    if geom == "disk":
        m, br = 2, C.DiskBasis(R, 20, 1.0, dealias=3 / 2)
        D = C.Domain(PC, [C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=3 / 2), br])
        r, kz2 = R, 0.0
        ic = lambda P, Rr: jn(m, j * Rr) * np.cos(m * P)                  # noqa: E731
    else:
        CC = C.CylindricalCoordinates("phi", "r", "z")
        r, m, kz2 = CC["r"], 0, 1.0
        br = C.DiskBasis(r, 16, 1.0, dealias=3 / 2)
        D = C.Domain(CC, [C.RealFourier(CC["phi"], 4, (0, 2 * np.pi), dealias=3 / 2), br,
                          C.ComplexFourier(CC["z"], 8, (0, 2 * np.pi), dealias=3 / 2)])
        ic = lambda P, Rr, Zz: jn(0, j * Rr) * np.sin(Zz)                 # noqa: E731
    j = jn_zeros(m, 1)[0]
    u, t1 = C.Field("u", D), C.Field("t1", D.without(r))
    pb = C.IVP([u, t1])
    pb.add_equation((2.0 * C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2), B * u))
    pb.add_equation((C.interp(u, r, "right"), 0))
    s = C.IVPSolver(discretize(pb), "RK443", 5e-4, backend="cuda")
    s.set_state({u: ic})
    u0 = np.abs(s.grid("u")).max()
    s.run(400)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A - B + j ** 2 + kz2) / 2) < 1e-7


@pytest.mark.gpu
def test_gpu_sphere_decay_with_bare_term_and_rhs():
    """2 dt(u) + u + a u - lap u = b u on S^2 (GPU), u = Y_2^0: rate -(1 + a - b + 6) / 2."""
    _gpu_or_skip()
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(s2["theta"], 8, radius=1.0, dealias=3 / 2)])
    u = C.Field("u", D)
    pb = C.IVP([u])
    pb.add_equation((2.0 * C.dt(u) + u + _a() * u - C.lap(u), B * u))
    s = C.IVPSolver(discretize(pb), "RK443", 1e-3, backend="cuda")
    s.set_state({u: lambda P, T: 3 * np.cos(T) ** 2 - 1 + 0 * P})
    u0 = np.abs(s.grid("u")).max()
    s.run(200)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A - B + 6) / 2) < 1e-8


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["ball", "shell"])
def test_gpu_ball_shell_dirichlet_decay_with_rhs(kind):
    """dt(u) + u + a u - lap u + taus = b u (GPU), Dirichlet: ball u = sin(pi r)/r, shell
    [0.5, 1.5] u = sin(pi (r - 1/2))/r (both -lap u = pi^2 u): rate -(1 + a - b + pi^2)."""
    _gpu_or_skip()
    sc = C.SphericalCoordinates("phi", "theta", "r")
    r = sc["r"]
    br = (C.BallBasis(r, 16, 1.0, dealias=3 / 2) if kind == "ball"
          else C.ShellBasis(r, 16, (0.5, 1.5), dealias=3 / 2))
    D = C.Domain(sc, [C.RealFourier(sc["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(sc["theta"], 5, dealias=3 / 2), br])
    u, t1 = C.Field("u", D), C.Field("t1", D.without(r))
    pb = C.IVP([u, t1] if kind == "ball" else [u, t1, C.Field("t2", D.without(r))])
    if kind == "ball":
        pb.add_equation((C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2), B * u))
        pb.add_equation((C.interp(u, r, "right"), 0))
        ic = lambda P, T, Rr: np.sin(np.pi * Rr) / Rr + 0 * P * T          # noqa: E731
    else:
        t2 = pb.variables[2]
        pb.add_equation((C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2)
                         + C.lift(t2, br, -2, lam=2), B * u))
        pb.add_equation((C.interp(u, r, "left"), 0))
        pb.add_equation((C.interp(u, r, "right"), 0))
        ic = lambda P, T, Rr: np.sin(np.pi * (Rr - 0.5)) / Rr + 0 * P * T  # noqa: E731
    s = C.IVPSolver(discretize(pb), "RK443", 5e-4, backend="cuda")
    s.set_state({u: ic})
    u0 = np.abs(s.grid("u")).max()
    s.run(200)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A - B + np.pi ** 2)) < 1e-7


# ----------------------------------------------------------------------------------------------
# Curvilinear symmetry subspaces (curvilinear_symmetry): the same exact rates computed in
# the reduced system with the folded evaluator (explicit RHS b u on the northern hemisphere /
# azimuthal sector). A reduced-vs-full agreement does not see an error common to both.
# ----------------------------------------------------------------------------------------------

def _sym_decay(D, radial, r, gens, par, ic, rate_exact, dt=5e-4, steps=400, backend="cpu",
               walls=("right",), coef=2.0):
    """coef dt(u) + u + a u - lap u + taus = b u in the subspace of ``gens`` (parity ``par`` for u
    and its taus); returns (observed rate - exact rate, reduced size, full size)."""
    u = C.Field("u", D)
    taus = [C.Field(f"t{i}", D.without(r)) for i in range(len(walls))] if walls else []
    G = C.SymmetrySubspace(*gens, u={g: par for g in gens}, **{t.name: {g: par for g in gens} for t in taus})
    pb = C.IVP([u, *taus], symmetry=G)
    lhs = coef * C.dt(u) + u + _a() * u - C.lap(u)
    for i, t in enumerate(taus):
        lhs = lhs + C.lift(t, radial, -1 - i, lam=2)
    pb.add_equation((lhs, B * u))
    for w in walls:
        pb.add_equation((C.interp(u, r, w), 0))
    plan = discretize(pb)
    s = C.IVPSolver(plan, "RK443", dt, backend=backend)
    assert s.system.n < plan.subspace.col_off[-1]            # really reduced
    s.set_state({u: ic})
    u0 = np.abs(s.grid("u")).max()
    s.run(steps)
    return _rate(s, "u", u0, s.t) - rate_exact, s.system.n, plan.subspace.col_off[-1]


@pytest.mark.parametrize("m0,par,m", [(2, 1, 2), (2, -1, 1), (4, 1, 4)])
def test_disk_mfold_subspace_decay_with_bare_term_and_rhs(m0, par, m):
    """AzimuthalMFold(phi, m0): u = J_m(j r) cos(m phi) in the class of parity par, rate
    -(1 + a - b + j^2) / 2 (evaluator folded to the sector 2 pi / m0)."""
    br = C.DiskBasis(R, 20, 1.0, dealias=3 / 2)
    D = C.Domain(PC, [C.RealFourier(PHI, 16, (0, 2 * np.pi), dealias=3 / 2), br])     # m <= 7
    j = jn_zeros(m, 1)[0]
    err, n, nf = _sym_decay(D, br, R, [C.AzimuthalMFold(PHI, m0)], Parity(par),
                            lambda P, Rr: jn(m, j * Rr) * np.cos(m * P), -(1 + A - B + j ** 2) / 2,
                            dt=5e-4 if m < 4 else 2.5e-4, steps=400 if m < 4 else 800)   # |sigma| dt <= 8e-3
    # time error of RK443, O(dt^3): 2.5e-7 at m = 4 (sigma = -29); an defect is O(1)
    assert abs(err) < (1e-7 if m < 4 else 1e-6) and n <= nf / m0 + 30


@pytest.mark.parametrize("ell,par,Y", [(2, 1, lambda T: 3 * np.cos(T) ** 2 - 1), (1, -1, lambda T: np.cos(T))])
def test_sphere_equatorial_subspace_decay_with_bare_term_and_rhs(ell, par, Y):
    """EquatorialParity on S^2: u = Y_l^0 in the class (-1)^l = par, rate -(1 + a - b + l(l+1)) / 2,
    F on the northern hemisphere."""
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(s2["theta"], 8, radius=1.0, dealias=3 / 2)])
    err, n, nf = _sym_decay(D, None, None, [C.EquatorialParity(s2)], Parity(par),
                            lambda P, T: Y(T) + 0 * P, -(1 + A - B + ell * (ell + 1)) / 2,
                            dt=1e-3, steps=200, walls=())
    assert abs(err) < 1e-8 and n < 0.6 * nf


@pytest.mark.parametrize("par", [1, -1])
def test_ball_equatorial_mfold_subspace_decay(par):
    """Ball, equatorial parity x 2-fold rotation: par = +1, u = sin(pi r)/r (l = 0), rate
    -(1 + a - b + pi^2) / 2; par = -1, u = j_1(k r) cos(theta) (l = 1, m = 0, k the first zero of
    j_1), rate -(1 + a - b + k^2) / 2 (the rotation class stays even: m = 0)."""
    from scipy.optimize import brentq
    from scipy.special import spherical_jn
    sc = C.SphericalCoordinates("phi", "theta", "r")
    r = sc["r"]
    br = C.BallBasis(r, 16, 1.0, dealias=3 / 2)
    D = C.Domain(sc, [C.RealFourier(sc["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(sc["theta"], 5, dealias=3 / 2), br])
    eq, rot = C.EquatorialParity(sc), C.AzimuthalMFold(sc["phi"], 2)
    if par == 1:
        k, ic = np.pi, (lambda P, T, Rr: np.sin(np.pi * Rr) / Rr + 0 * P * T)
    else:
        k = brentq(lambda x: spherical_jn(1, x), 4.0, 5.0)
        ic = (lambda P, T, Rr: spherical_jn(1, k * Rr) * np.cos(T) + 0 * P)
    u = C.Field("u", D)
    t1 = C.Field("t1", D.without(r))
    G = C.SymmetrySubspace(eq, rot, u={eq: Parity(par), rot: C.EVEN}, t1={eq: Parity(par), rot: C.EVEN})
    pb = C.IVP([u, t1], symmetry=G)
    pb.add_equation((2.0 * C.dt(u) + u + _a() * u - C.lap(u) + C.lift(t1, br, -1, lam=2), B * u))
    pb.add_equation((C.interp(u, r, "right"), 0))
    plan = discretize(pb)
    s = C.IVPSolver(plan, "RK443", 5e-4)
    assert s.system.n < 0.4 * plan.subspace.col_off[-1]       # 79 / 223 (l = 0 and taus)
    s.set_state({u: ic})
    u0 = np.abs(s.grid("u")).max()
    s.run(200)
    assert abs(_rate(s, "u", u0, s.t) + (1 + A - B + k ** 2) / 2) < 1e-7


def test_cylinder_axial_reflection_subspace_decay():
    """Reflection(z) on the cylinder: u = J_0(j r) cos(z) (even), rate -(1 + a - b + j^2 + 1) / 2;
    the -k_z groups are empty."""
    CC = C.CylindricalCoordinates("phi", "r", "z")
    phi, r, zc = CC["phi"], CC["r"], CC["z"]
    br = C.DiskBasis(r, 16, 1.0, dealias=3 / 2)
    D = C.Domain(CC, [C.RealFourier(phi, 4, (0, 2 * np.pi), dealias=3 / 2), br,
                      C.ComplexFourier(zc, 8, (0, 2 * np.pi), dealias=3 / 2)])
    j = jn_zeros(0, 1)[0]
    err, n, nf = _sym_decay(D, br, r, [C.Reflection(zc, about=0.0)], C.EVEN,
                            lambda P, Rr, Zz: jn(0, j * Rr) * np.cos(Zz), -(1 + A - B + j ** 2 + 1) / 2,
                            dt=1e-3, steps=100)
    assert abs(err) < 1e-7 and n < 0.6 * nf


def test_sphere_equatorial_subspace_evp_scaled_sigma():
    """EVP s (2 u) + u + a u - lap u = 0 on S^2 in the odd class: s = -(1 + a + l(l+1)) / 2 for the
    degrees l + m odd only (bare term and scaled sigma)."""
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.ComplexFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=1),
                      C.SphereBasis(s2["theta"], 8, radius=1.0, dealias=1)])
    u, s = C.Field("u", D, dtype=complex), C.Parameter("s")
    pb = C.EVP([u], eigenvalue=s, symmetry=C.SymmetrySubspace(C.EquatorialParity(s2), u=C.ODD))
    pb.add_equation((s * (2.0 * u) + u + _a() * u - C.lap(u), 0))
    plan = discretize(pb)
    for m in (0, 1, 2):
        ev = np.sort(C.eigenvalues(plan, plan.group(m)).values.real)[::-1]
        ls = np.array([l for l in range(abs(m), 8) if (l + m) % 2 == 1])
        assert np.abs(ev - np.sort(-(1 + A + ls * (ls + 1)) / 2)[::-1]).max() < 1e-12


@pytest.mark.gpu
def test_gpu_subspace_decay_disk_and_sphere():
    """The folded evaluator on the GPU (sector, hemisphere): disk 2-fold class u = J_2(j r)
    cos(2 phi) and S^2 equatorial u = Y_2^0, exact rates as on the CPU."""
    _gpu_or_skip()
    br = C.DiskBasis(R, 20, 1.0, dealias=3 / 2)
    D = C.Domain(PC, [C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=3 / 2), br])
    j = jn_zeros(2, 1)[0]
    err, _, _ = _sym_decay(D, br, R, [C.AzimuthalMFold(PHI, 2)], C.EVEN,
                           lambda P, Rr: jn(2, j * Rr) * np.cos(2 * P), -(1 + A - B + j ** 2) / 2,
                           backend="cuda")
    assert abs(err) < 1e-7
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(s2["theta"], 8, radius=1.0, dealias=3 / 2)])
    err, _, _ = _sym_decay(D, None, None, [C.EquatorialParity(s2)], C.EVEN,
                           lambda P, T: 3 * np.cos(T) ** 2 - 1 + 0 * P, -(1 + A - B + 6) / 2,
                           dt=1e-3, steps=200, walls=(), backend="cuda")
    assert abs(err) < 1e-8
