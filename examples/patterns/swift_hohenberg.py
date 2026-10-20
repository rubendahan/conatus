"""Swift–Hohenberg: onset of rolls, weakly nonlinear amplitude, roll branch by continuation, 2D rolls.

    du/dt = r u - (1 + lap)^2 u - u^3                         (Swift & Hohenberg 1977)

(a) Linear threshold. A normal mode exp(i q x + s t) of the state u = 0 grows at
    s = r - (1 - q^2)^2, so the neutral curve is r = (1 - q^2)^2 and the onset is r_c = 0 at
    q_c = 1 (Cross & Hohenberg 1993, Rev. Mod. Phys. 65, 851, section on the SH model). Computed
    here by linearize() of the IVP around u = 0 and one parametric pencil in (r, q): the Fourier
    wavenumber q is a declared pencil parameter (``wavenumbers={"x": q}``), so critical() minimizes
    over q without rediscretizing (one pencil per q before). RealFourier: each group
    holds the (cos, sin) pair, a double eigenvalue (translation invariance); track() follows a
    declared member of the eigenspace (it raised LinAlgError before).

(b) Roll branch. Steady rolls u = A sin x + B sin 3x + ... obey, projecting on sin x and sin 3x,
    B = A^3 / (4 (64 - r)) and r = 3 A^2 / 4 - 3 A B / 4 + ..., i.e.

        A^2 = 4 r / 3 + r^2 / 144 + O(r^3)

    (leading term: Cross & Hohenberg 1993; the r^2/144 correction is the third-harmonic feedback
    derived above). The rolls are solved as a periodic NLBVP (PeriodicNewtonSolver: dense coupled
    Jacobian, translation fixed by a declared phase condition, the -sin coefficient of mode 1 set
    to zero, with its multiplier reported), and, independently, in the odd subspace (the banded
    NewtonSolver refuses the periodic form: "varies along a Fourier direction"), *exactly
    equivalent* for the periodic rolls and followed in r by continuation: on x in [0, pi] with
    u = u'' = 0 at both ends, every solution extends to an odd 2 pi-periodic one (sin(n x) modes;
    (1 + d_xx)^2 and u^3 preserve oddness), and conversely. Chebyshev in x, four BCs for the
    fourth-order operator. The amplitude A = (2/pi) int_0^pi u sin x dx is an extra unknown of the
    NLBVP, so it is carried by Newton and by the continuation.

(c) 2D rolls. The same equation as an IVP on a doubly periodic 2 pi x 2 pi box (RealFourier x
    RealFourier, dealias = 2 declared for the cubic term), started from a mixture of x- and
    y-rolls: the y-rolls decay, the x-rolls saturate at the amplitude of the branch (b).

Run:  python examples/patterns/swift_hohenberg.py
"""
import numpy as np

from conatus.expert import (IVP, NLBVP, CartesianCoordinates, Chebyshev, ContinuationSettings,
                            Domain, Field, IVPSolver, NewtonSolver, NLBVPContinuation, Parameter,
                            ParametricEVP, PeriodicNewtonSolver, RealFourier, critical, diff,
                            discretize, dt, integ, interp, lap, linearize)
from conatus.ir import Zero
from conatus.solvers import PhaseCondition


def swift_hohenberg_ivp(domain, lap_):
    """du/dt - r u + (1 + lap)^2 u = -u^3 on ``domain`` (lap_ = the Laplacian of that domain)."""
    u, r = Field("u", domain), Parameter("r", 0.1)

    def one_plus_lap(f):
        return f + lap_(f)

    ivp = IVP([u], name="Swift-Hohenberg")
    ivp.add_equation((dt(u) - r * u + one_plus_lap(one_plus_lap(u)), -u * u * u))
    return ivp, u, r


# ------------------------------------------------------------------ (a) linear threshold
def growth_pencil():
    """Pencil in (r, q) of the perturbation of u = 0; q = wavenumber of x (a pencil parameter,
    so the period of the domain plays no role)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 8, (0, 2 * np.pi), dealias=1)])
    ivp, u, r = swift_hohenberg_ivp(D, lambda f: diff(diff(f, x), x))
    evp = linearize(ivp, around={u: Zero(())}, eigenvalue=Parameter("s", dtype=complex))
    return ParametricEVP(evp, [r], wavenumbers={"x": Parameter("q", 1.0)}), evp


def onset():
    """(q_c, r_c): neutral r of the least stable mode, minimized over q on one pencil."""
    cp = critical(growth_pencil()[0], "r", bracket=(-0.5, 1.0), over={"q": (0.5, 1.5)}, xtol=1e-10)
    return cp.over["q"], cp.value


# ------------------------------------------------------------------ (b) roll branch (NLBVP)
def rolls(N: int):
    """Steady rolls in the odd subspace: x in [0, pi], u = u'' = 0; unknowns u and amplitude A."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, N, (0, np.pi), dealias=1)])
    u, A, sinx, r = Field("u", D), Field("A", D.without(x)), Field("sinx", D), Parameter("r", 0.1)

    def one_plus_lap(f):
        return f + diff(diff(f, x), x)

    pb = NLBVP([u, A], name="Swift-Hohenberg rolls (odd subspace)")
    pb.add_equation((r * u - one_plus_lap(one_plus_lap(u)) - u * u * u, 0))
    pb.add_equation((A - (2 / np.pi) * integ(sinx * u), 0))      # A = first sine coefficient
    uxx = diff(diff(u, x), x)
    pb.add_bc(interp(u, x, 0), interp(u, x, np.pi), interp(uxx, x, 0), interp(uxx, x, np.pi), into=0)
    return pb, u, {sinx: np.sin}


def rolls_periodic(N: int, r: float):
    """Steady rolls on the 2 pi-periodic line (RealFourier, dealias 2 for the cubic term), phase
    fixed by msin_1 = 0; returns the solution (first harmonic via .harmonic("u", x, 1))."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    u, rr = Field("u", Domain(cs, [RealFourier(x, N, (0, 2 * np.pi), dealias=2)])), Parameter("r", r)

    def one_plus_lap(f):
        return f + diff(diff(f, x), x)

    pb = NLBVP([u], name="Swift-Hohenberg rolls (periodic)")
    pb.add_equation((rr * u - one_plus_lap(one_plus_lap(u)) - u * u * u, 0))
    solver = PeriodicNewtonSolver(pb, tol=1e-13, damping=1.0, max_iterations=20,
                                  phase=PhaseCondition(u, x, kind="sin1"))
    return solver.solve({u: lambda X: np.sqrt(4 * r / 3) * np.cos(X)}), x


def roll_solver(N: int, r: float):
    """Newton for the rolls at r, started from the leading-order amplitude sqrt(4 r / 3)."""
    pb, u, ncc = rolls(N)
    solver = NewtonSolver(pb, tol=1e-13, damping=1.0, max_iterations=30, params={"r": r}, ncc=ncc)
    return solver, solver.solve({u: lambda X: np.sqrt(4 * r / 3) * np.sin(X)})


def roll_branch(N: int, r0: float, r1: float, ds_max: float):
    """Pseudo-arclength continuation of the rolls in r from r0 to r1: [(r, A, residual)]."""
    solver, sol = roll_solver(N, r0)
    cfg = ContinuationSettings(ds=0.02, ds_min=1e-6, ds_max=ds_max, adapt=1.5, fast_iterations=3,
                               tolerance=1e-12, max_newton=15, max_points=40, dp_fd=1e-6)
    cont = NLBVPContinuation(solver, "r", cfg)
    branch = cont.run(sol.coeffs, r0, +1, p_bounds=(r0, r1))
    pts = [(q.p, float(np.real(cont.coeffs_of(q.x)["A"]).ravel()[0]), q.residual) for q in branch.points]
    return pts, branch


# ------------------------------------------------------------------ (c) 2D rolls (IVP)
def rolls_2d(N: int, r: float, timestep: float, steps: int):
    """IVP on the 2 pi-periodic square; returns (A_x, A_y, |du/dt| at the end, solver)."""
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    D = Domain(cs, [RealFourier(x, N, (0, 2 * np.pi), dealias=2), RealFourier(y, N, (0, 2 * np.pi), dealias=2)])
    ivp, u, _ = swift_hohenberg_ivp(D, lap)
    solver = IVPSolver(discretize(ivp, params={"r": r}), "SBDF2", timestep)
    solver.set_state({u: lambda X, Y: 0.05 * np.cos(X) + 0.04 * np.cos(Y) + 0.01 * np.sin(2 * X + Y)})
    solver.run(steps - 1)
    before = solver.grid("u")
    solver.run(1)
    g = solver.grid("u")
    e = np.exp(-1j * 2 * np.pi * np.arange(N) / N)
    A_x = 2 * abs(np.mean(g * e[:, None]))           # |cos x, sin x| content (phase free)
    A_y = 2 * abs(np.mean(g * e[None, :]))
    return A_x, A_y, float(np.abs(g - before).max() / timestep), solver


if __name__ == "__main__":
    q_c, r_c = onset()
    print(f"(a) onset: q_c = {q_c:.8f}, r_c = {r_c:.2e}  (theory: q_c = 1, r_c = 0)")
    assert abs(q_c - 1) < 1e-4 and abs(r_c) < 1e-8, (q_c, r_c)    # Swift & Hohenberg (1977)
    P, evp = growth_pencil()
    print(evp.explain())
    print(P.explain())
    print("(b) rolls at r = 0.1, convergence in N (A = first sine coefficient):")
    for N in (16, 20, 24, 32):
        _, sol = roll_solver(N, 0.1)
        print(f"  N = {N}: A = {sol.value('A'):.15f}, Newton residual {sol.residual:.1e}")
    A = sol.value("A")                                             # N = 32, converged in N
    sol, x = rolls_periodic(16, 0.1)
    print(f"  periodic NLBVP (N = 16, phase msin_1 = 0): A = {sol.harmonic('u', x, 1):.15f}")
    assert abs(sol.harmonic("u", x, 1) - A) < 1e-12               # periodic NLBVP = odd subspace
    print(sol.explain())
    pts, branch = roll_branch(24, 0.02, 0.5, 0.05)
    print(branch.explain())
    print("  r         A              A^2 - 4r/3     (A^2 - 4r/3)/r^2  (theory 1/144 = 0.006944)")
    for r, A_r, res in pts:
        gap = A_r * A_r - 4 * r / 3
        print(f"  {r:.5f}  {A_r:.12f}  {gap:.3e}      {gap / r ** 2:.6f}   residual {res:.1e}")
        assert abs(gap) < 0.01 * r ** 2, (r, gap)                 # A^2 = 4r/3 + O(r^2) (Cross & Hohenberg 1993)
    A_x, A_y, rate, s = rolls_2d(16, 0.1, 0.5, 800)
    print(f"(c) 2D IVP at r = 0.1, t = {s.t}: A_x = {A_x:.12f}, A_y = {A_y:.1e}, |du/dt| = {rate:.1e}")
    assert rate < 1e-8 and A_y < 1e-6 and abs(A_x - A) < 1e-9, (rate, A_y, A_x, A)   # 2D IVP = 1D rolls
    print(s.explain())
