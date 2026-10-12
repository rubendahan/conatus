"""Phase-space geometry in three steps.

1. Symmetry reduction: the two-mode SO(2) model of Budanur, Borrero-Echeverry & Cvitanović
   (arXiv:1411.3303) in the first Fourier mode slice; its relative periodic orbit "1" becomes a
   periodic orbit of the reduced flow (Table I: T = 3.64151221, θ = 0.08096967, Λ = −1.48372354).
2. Poincaré section of the slice through the relative equilibrium, located on the declared dense
   output (DOP853 continuous extension, Brent).
3. Lorenz: the homoclinic orbit of the origin found as a *connection* (unstable manifold by the
   parametrization method, projection condition on a sphere around the origin), r ≈ 13.926.

Run: python examples/manifolds/phase_space_geometry.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import (DynamicalSystem, FirstFourierModeSlice, ODEIntegrator, PoincareSection,
                            RotationOps, Trajectory, find_connection, find_rpo_in_slice, ir_system,
                            ode_problem, poincare_crossings, two_mode_callables, unstable_left_basis,
                            unstable_manifold)


def two_mode_rpo():
    f, jac = two_mode_callables()                      # eq. (21) with the parameters of eq. (22)
    ds = DynamicalSystem.from_callables(f, 4, {}, jac=jac, fd_x=1e-5, fd_p=1e-6, name="two-mode")
    ops = RotationOps([1, 2])                           # (z1, z2) -> (e^{iθ} z1, e^{2iθ} z2)
    slc = FirstFourierModeSlice(ops, border_tol=1e-8)   # template (1, 0, 0, 0): Im z1 = 0, Re z1 > 0
    # relative equilibrium = equilibrium of the in-slice flow v̂ = v − θ̇ T x̂
    tw3 = so.fsolve(lambda y: slc.velocity(np.array([y[0], 0, y[1], y[2]]), ds.G)[0][[0, 2, 3]],
                    [0.44, -0.39, 0.07], xtol=1e-14)
    x_tw = np.array([tw3[0], 0, tw3[1], tw3[2]])
    print("TW in the slice:", x_tw)
    T = np.array([ops.ddx(e) for e in np.eye(4)]).T
    w, V = np.linalg.eig(ds.J(x_tw) - slc.velocity(x_tw, ds.G)[1] * T)
    vu = V[:, np.argmax(w.real)]
    integ = ODEIntegrator(ds, method="DOP853", rtol=1e-12, atol=1e-13)
    tr = integ.integrate(x_tw + 1e-3 * np.real(vu), 250.0)
    st = slc.reduce_trajectory(tr.t, tr.X)             # unwrapped phase, border distance
    reduced = Trajectory(tr.t, st.X_hat, "reduced DOP853 dense output",
                         lambda i, t: slc.reduce(tr._dense(i, t))[0])
    a = np.imag(vu / np.linalg.norm(vu))
    a[1] = 0.0
    n = np.cross(a[[0, 2, 3]], [0, 0, 1.0])            # plane ∋ TW spanned by Im v_u and ŷ2
    sec = PoincareSection.hyperplane([n[0], 0, n[1], n[2]], x_tw, direction=-1)
    cr = poincare_crossings(reduced, sec, xtol=1e-13, t_min=100.0)
    print(cr.explain())
    # closest one-return recurrences, best first; a seed that falls back onto the relative
    # equilibrium is refused by find_rpo_in_slice (steady_tol) and the next one is tried
    d = [np.linalg.norm(cr.X[j + 1] - cr.X[j]) for j in range(len(cr.t) - 1)]
    for k in np.argsort(d):
        try:
            rpo = find_rpo_in_slice(slc, integ, sec, cr.X[k], cr.t[k + 1] - cr.t[k], tol=1e-11,
                                    fd_step=1e-6, max_iterations=10, steady_tol=1e-6)
        except Exception as e:  # noqa: BLE001 -- a refused seed (any reason) is printed; try the next return
            print("seed refused:", str(e).splitlines()[0])
            continue
        print(rpo.explain())
        return rpo


def lorenz_homoclinic():
    pb = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 13.9, "b": 8 / 3},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]), P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], "Lorenz")
    ds = ir_system(pb, ["r"])
    origin = np.zeros(3)

    def setup(q):                                       # q = [r]
        par = {"r": float(q[0])}
        man = unstable_manifold(ds, origin, order=10, degree=2, eigvec_scale=1.0,
                                unstable_tol=1e-9, par=par)
        return dict(x0=man([0.5]), x_plus=origin, M=ds.M, end_radius=0.1,
                    integrator=ODEIntegrator(ds, method="DOP853", rtol=1e-11, atol=1e-13, par=par),
                    Psi=unstable_left_basis(ds, origin, unstable_tol=1e-9, par=par),
                    start_defect=man.defect([0.5]))
    con = find_connection(setup, 8.0, bracket=(13.92, 13.93), xtol=1e-11)
    print(con.explain())
    return con


if __name__ == "__main__":
    two_mode_rpo()
    lorenz_homoclinic()
