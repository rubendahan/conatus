# Resolution

The resolution report says, for each unknown and each direction, whether the spectral
coefficients have decayed, and how many modes a target accuracy would need.

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
sim.run(steps=10)
rep = sim.plan.resolution_report(sim.solver, velocity="u")   # a simulation, an LBVP solution, or data
print(rep.explain())
rep.ok                          # every (variable, direction) resolved
rep.recommend(target=1e-10)     # declared recommendations, never applied
rep.entry("T", "z").modes_for(1e-12)
```

For each variable and each direction, the report measures the **spectral tail** of the
coefficients:

- the mode number is the polynomial degree (Chebyshev, Legendre, Jacobi) or $|n|$ (Fourier; ±n
  merged, Nyquist excluded);
- the envelope $e[m]$ is the maximum of $|c|$ over the other axes and components, made
  monotone from the right;
- the **tail** is $E[M-n_\text{tail}]/\max E$ with
  $n_\text{tail} = \max(2, \lceil 0.1 M\rceil)$.

All thresholds are named constants of `conatus.resolution`, and each report prints them:

| Verdict | Tail |
|---|---|
| resolved | ≤ `TAIL_TARGET` = 1e-8 |
| marginal | ≤ `TAIL_MARGINAL` = 1e-4 |
| under-resolved | above |

The decay rate σ is fitted on the last half of the modes above the round-off floor. The **"how
many modes?"** estimate gives the smallest size whose predicted tail reaches the target. Its
stated assumption: the fitted exponential decay continues beyond the resolved modes.
`cn.resolution.estimate_modes(coeffs, basis, target=...)` does the same for one array.

The report also gives:

- the dealias factor of each basis;
- when a velocity and `dt` are known, the advective CFL number $\Delta t\,\max\sum_i |u_i|/\Delta
  x_i$, labelled as a **stability indicator, not an error control**.

A recommendation such as "increase Nz from 24 to ~36" is text. Nothing changes the plan, the
bases or the solver (tested).

For example, $\tanh(10x)$ on 24 Chebyshev polynomials is reported under-resolved with an
estimate of about 170 modes; $\sin(\pi x)$ on 32 is resolved, with no recommendation.

Tau variables are skipped unless `include_taus=True`. Polar and spherical plans are refused with
`UnsupportedError` rather than approximated.
