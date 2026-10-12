# Gallery "fluids everywhere"

Each example is a short, readable script, written as in the paper (equation, physical
comments, `explain()`), and each checks its result against a
published value or a closed form. No value without a residual or convergence in N.

Run: `python examples/<family>/<case>.py` (with the `tutorials` extra installed).
User lines = lines of code excluding docstrings, comments and blank lines (functions + `__main__` block).

| Case | Script | Value obtained | Reference | Lines |
|---|---|---|---|---|
| Kelvin–Helmholtz (tanh), Rayleigh + Orr–Sommerfeld | `shear/kelvin_helmholtz.py` | max ω_i = 0.189702 at α = 0.44492 (L = 20, N = 384); viscous → inviscid in O(1/Re) | Michalke 1964, JFM 19, Table 1: 0.1897 at α = 0.4446 | 31 + 15 |
| Non-modal Poiseuille: transient growth and continuous adjoint | `shear/nonmodal_poiseuille.py` | G_max = 4896.948 at t = 379.075 (Re = 5000, kz = 2.044; N = 48–96 identical to 1e-10); continuous vs discrete adjoint | Butler & Farrell 1992; Reddy & Henningson 1993: 4897 at t = 379 | 6 + 24 |
| Taylor–Goldstein tanh/tanh (Ri) | `shear/taylor_goldstein.py` | whole line (tanh map, L = 2/α) + shifted complex path: **J_n(1/2) = 1/4 to 2.5e-12** (N = 192) and J_n(0.3) = 0.21 to 3.8e-9 by Newton on c_i(J) = 0, without extrapolation, independent of L and δ; J = 0.3: every mode damped | Hazel 1972, JFM 51 (R = 1); Miles 1961 / Howard 1961: J_c = 1/4 | 20 + 14 |
| Poiseuille: wavemaker, sensitivity to the base flow and to steady forcing (Orszag mode) | `shear/sensitivity_wavemaker.py` | wavemaker maximum S = 11.4856 at y = ±0.87387 (critical layer U = c_r); derivative of the pencil = Richardson finite difference to 5e-11; density (continuous adjoint) → pencil to 2e-12 (N = 80) | no exact published value in a channel; closed forms of Chomaz (2005) for Ginzburg–Landau (`tests/sensitivity`) | 10 |
| Womersley (plane channel): LBVP, IVP, Floquet–Hill | `oscillatory/womersley.py` | profile to 9e-16 (N = 24); RK443 order 2.85, SBDF2 1.95; Floquet ε = 0 = Orszag to 1e-9, O(ε²) deviation | Womersley 1955; Orszag 1971, JFM 50, Table 3 | 52 + 23 |
| Swift–Hohenberg 1D/2D | `patterns/swift_hohenberg.py` | threshold (r, q) on a single pencil (q a parameter, RealFourier); A(r = 0.1) = 0.3652433 (A² − 4r/3 → r²/144); periodic NLBVP (declared phase) = odd subspace to 1e-12; 2D IVP on the branch to 1e-9 | Swift & Hohenberg 1977; Cross & Hohenberg 1993 | 75 + 22 |
| Complex Ginzburg–Landau | `patterns/ginzburg_landau.py` | complex form (conj() on the LHS, doubled system declared); BFN threshold: 1 + c₁c₃ < 1e-12 (Richardson k → 0); Eckhaus q_E² to 1e-7 | Newell 1974; Aranson & Kramer 2002, RMP 74 | 30 + 17 |
| Brusselator (Turing) | `patterns/brusselator.py` | b_c to ≤ 9e-16 relative (3 sets) | Prigogine & Lefever 1968; Peña & Pérez-García 2001, PRE 64 | 26 + 8 |
| Rotating Rayleigh–Bénard (f-plane) | `convection/rotating_rayleigh_benard.py` | one pencil for all Ta (√Ta declared); free walls: Ra_c(Ta) = closed form to ≤ 2e-15 (Ta = 0…1e5); rigid: 1756.347 / 2151.341 / 4712.042 (Ta = 1e2, 1e3, 1e4) | Chandrasekhar 1961, ch. III | 34 + 16 |
| Salt fingers (double diffusion) | `convection/salt_fingers.py` | R_S/τ − R_T = 27π⁴/4 to ≤ 4e-16 (4 pairs) | Stern 1960; Baines & Gill 1969, JFM 37 | 25 + 9 |
| 3D Rayleigh–Bénard (onset) | `convection/rayleigh_benard_3d.py` | Ra = 1707.76178 at |k| = 3.1163, isotropic to 1e-10 | Reid & Harris 1958 | 43 + 9 |
| Phase-space geometry: slice, Poincaré section, connection | `manifolds/phase_space_geometry.py` | two-mode model: RPO "1" in the slice, T = 3.6415119771, θ = 0.0809697054; Lorenz homoclinic orbit as a connection, r = 13.9265574075 | Budanur, Borrero-Echeverry & Cvitanović (arXiv:1411.3303), Table I: 3.64151221; Sparrow 1982: 13.926 | 56 + 3 |

The API gaps encountered are in (entry "gallery, wave 1"); those that are
closed are struck through there with their commit (2026-09-28: `discretize` in `conatus.expert`, (U − c)·L, conj() on the LHS,
wavenumber as a parameter, √Ta declared, `critical()` start/xtol, double eigenvalue, aliasing alert,
periodic NLBVP, mapped domains); the bugs found along the way are fixed ("gallery" entries of
).
