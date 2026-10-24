"""Equilibria and travelling waves of the Kuramoto-Sivashinsky equation for L = 22, with their
leading stability eigenvalues (Cvitanovic, Davidchack & Siminos 2010).

Reference: P. Cvitanovic, R. L. Davidchack & E. Siminos (2010), On the state space geometry of
the Kuramoto-Sivashinsky flow in a periodic domain, SIAM J. Appl. Dyn. Syst. 9, 1-33,
doi:10.1137/070705623, arXiv:0709.2944. Table 5.1: leading eigenvalues mu_j +- i nu_j of the
equilibria E1, E2, E3 and of the relative equilibria TW1, TW2 (4 decimals); Table 5.2: energies E
(4 decimals); section 5: the speeds c = 0.73699 and 0.34954 of TW1 and TW2.

What is computed: u_t + u u_x + u_xx + u_xxxx = 0 on a periodic interval of length 22. The
equilibria and travelling waves are found by Newton-Krylov-hookstep from fixed seeds at 32 Fourier
modes, then converged again at 64 modes (residual below 1e-12); their eigenvalues are those of the Jacobian of the right-hand side (in the
frame moving with the wave for TW1, TW2), computed densely. Each tabulated eigenvalue is matched
to the nearest unused computed one; the gap is |lambda_Conatus - lambda_published|, and a value is
"within rounding" when both its real and imaginary parts are within half a unit of the 4th
decimal (5e-5) of the printed value. Energies and speeds are compared the same way.

Run: python examples/published/kuramoto_sivashinsky_l22/run.py   (writes result.json)
"""
import json

import numpy as np

import conatus as cn
from conatus.expert import NewtonKrylovHookstep, equilibrium_eigenvalues, find_equilibrium, find_travelling_wave

# Cvitanovic, Davidchack & Siminos (2010), Table 5.1: (mu_j, nu_j); a pair mu +- i nu is one row.
TABLE_51 = {
    "E1": [(0.1308, 0.3341), (0.0824, 0.3402), (0.0, 0.0), (-0.2287, 0.1963), (-0.2455, 0.0),
           (-2.0554, 0.0), (-2.0619, 0.0)],
    "E2": [(0.1390, 0.2384), (0.0, 0.0), (-0.0840, 0.1602), (-0.1194, 0.0), (-0.2711, 0.3563),
           (-2.0130, 0.0), (-2.0378, 0.0)],
    "E3": [(0.0933, 0.0), (0.0933, 0.0), (0.0, 0.0), (-0.4128, 0.0), (-0.6108, 0.3759),
           (-0.6108, 0.3759), (-1.6641, 0.0), (-1.6641, 0.0)],
    "TW1": [(0.1156, 0.8173), (0.0337, 0.4189), (0.0, 0.0), (-0.2457, 0.0), (-0.3213, 0.9813)],
    "TW2": [(0.3370, 0.0), (0.0, 0.0), (-0.0096, 0.6288), (-0.2619, 0.5591), (-0.3067, 0.0725)],
}
ENERGY = {"E1": 0.2609, "E2": 0.4382, "E3": 1.5876, "TW1": 0.4649, "TW2": 0.6048}   # Table 5.2
SPEED = {"TW1": 0.73699, "TW2": 0.34954}                                             # section 5
HALF = 5e-5                                                                         # half a unit of the 4th decimal

L, N0, N = 22.0, 32, 64


def ks_flow(n):
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky L=22")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=n, scheme="RK443", dt=0.01)
    return sim, sim.solver              # the right-hand side and the state layout used by the searches


def newton(max_it, gmres_rtol=1e-3):
    """Trust-region Newton-Krylov: GMRES to gmres_rtol with at most 130 vectors, trust radius from 1."""
    return NewtonKrylovHookstep(tol=1e-12, max_iterations=max_it, gmres_rtol=gmres_rtol, gmres_max_iterations=130,
                                delta0=1.0, delta_min=1e-10, delta_max=10.0)


sim0, ks0 = ks_flow(N0)                 # the searches start at 32 modes ...
sim, ks = ks_flow(N)                    # ... and each solution is converged again at 64 modes


def sine(n, a):
    sim0.initial(u=lambda x: a * np.sin(2 * np.pi * n * x / L))
    return ks0.X.copy()


def random_seed(draw):
    """Fourier modes 1 to 4 drawn from a fixed random stream (draw number `draw`)."""
    rng = np.random.default_rng(0)
    for _ in range(draw + 1):
        c = np.zeros(N0)
        c[2:10] = rng.normal(size=8) * 0.8
    return ks0.system.gather_state([c])


def refine(sol):
    """The solution found at N0 modes, its Fourier coefficients padded with zeros to N modes, then
    converged again by Newton at N modes (GMRES to 1e-10: Newton then converges quadratically)."""
    ks0.X = sol.X
    c = np.zeros(N)
    c[:N0] = ks0.coeffs("u")
    X = ks.system.gather_state([c])
    if sol.kind == "equilibrium":
        return find_equilibrium(ks, X, newton(20, 1e-10))
    return find_travelling_wave(ks, X, sol.params["c"], newton(20, 1e-10))


sols = {"E1": find_equilibrium(ks0, sine(1, 1.0), newton(40)),
        "E2": find_equilibrium(ks0, sine(2, 1.0), newton(40)),
        "E3": find_equilibrium(ks0, sine(3, 2.0), newton(40)),
        "TW1": find_travelling_wave(ks0, random_seed(7), 0.737, newton(60)),
        "TW2": find_travelling_wave(ks0, random_seed(3), 0.35, newton(60))}
sols = {name: refine(sol) for name, sol in sols.items()}

out = {"case": "kuramoto_sivashinsky_l22", "solutions": {}}
n_in = n_all = 0
gaps = []
for name, sol in sols.items():
    ev = equilibrium_eigenvalues(ks, sol, n_modes=14, dense_max=500, arnoldi_tol=1e-12)
    free = list(ev)
    line = f"{name}: residual {sol.residual:.1e}, energy {sol.energy:.6f} (published {ENERGY[name]})"
    e_ok = abs(sol.energy - ENERGY[name]) <= HALF
    n_in += e_ok
    n_all += 1
    gaps.append(abs(sol.energy - ENERGY[name]))
    if name in SPEED:
        c = sol.params["c"]
        c_ok = abs(abs(c) - SPEED[name]) <= 5e-6
        n_in += c_ok
        n_all += 1
        gaps.append(abs(abs(c) - SPEED[name]))
        line += f", speed {abs(c):.7f} (published {SPEED[name]})"
    print(line)
    rows = []
    for mu, nu in TABLE_51[name]:
        ref = complex(mu, nu)
        cand = [z for z in free if (z.imag > 1e-9 if nu > 0 else abs(z.imag) < 1e-9)] or free
        z = min(cand, key=lambda w: abs(w - ref))
        free.remove(z)
        ok = abs(z.real - mu) <= HALF and abs(z.imag - nu) <= HALF
        n_in += ok
        n_all += 1
        gaps.append(abs(z - ref))
        rows.append({"ref": [mu, nu], "conatus": [z.real, z.imag], "gap": abs(z - ref), "within_rounding": bool(ok)})
        print(f"   {mu:+.4f} {nu:+.4f}i   Conatus {z.real:+.6f} {z.imag:+.6f}i   gap {abs(z - ref):.1e}"
              f"   {'yes' if ok else 'no'}")
    out["solutions"][name] = {"energy": sol.energy, "residual": sol.residual, "speed": sol.params.get("c"),
                              "eigenvalues": rows, "all": [[w.real, w.imag] for w in ev]}

print(f"published values (eigenvalues, energies, speeds): {n_all}; within rounding: {n_in} of {n_all}")
print(f"largest gap {max(gaps):.1e}")
out.update({"within_rounding": n_in, "n_ref": n_all, "max_gap": max(gaps)})
if __name__ == "__main__":
    # what this case establishes: every published value within its rounding, converged solutions
    assert n_in == n_all and all(s.residual < 1e-12 for s in sols.values())
    with open("result.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
