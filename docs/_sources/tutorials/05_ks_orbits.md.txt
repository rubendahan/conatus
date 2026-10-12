# 5. Invariant solutions: Kuramoto–Sivashinsky, L = 22

**Goal.** Find the equilibria, a travelling wave and the shortest pre-periodic orbit of the
Kuramoto–Sivashinsky equation on the $L = 22$ domain by Newton–Krylov–hookstep on the time-stepper's
flow map, with their stability.
**Expected.** Equilibria $E_1$, $E_2$ (energies $0.2609$, $0.4382$; leading eigenvalues of $E_1$
$0.1308 \pm 0.3341i$, $0.0824 \pm 0.3402i$), travelling wave $TW_1$ ($c = 0.73699$, energy $0.4649$):
Cvitanović, Davidchack & Siminos (2010), *SIAM J. Appl. Dyn. Syst.* 9, 1–33, Tables 5.1–5.2 and §5.
Pre-periodic orbit $\overline{pp}_{10.25}$: leading Floquet exponent $\mu_1 = 0.033209$,
$\theta_1 = 2.0079$ (Ding & Cvitanović 2016, *SIAM J. Appl. Dyn. Syst.* 15, Table 1).
**Time:** about a minute on one CPU core (the orbit's dense monodromy dominates).
**Script:** [`t05_ks_orbits.py`](code/t05_ks_orbits.py) · **Tests behind it:** `tests/ecs/test_ecs_ks.py`.

## The system

$$
u_t + u u_x + u_{xx} + u_{xxxx} = 0, \qquad x \in (0, 22) \ \text{periodic}.
$$

It is equivariant under translations $\tau_\ell u(x) = u(x + \ell)$ and the reflection
$R u(x) = -u(-x)$. Invariant solutions come in families: equilibria ($\Phi_t u = u$), travelling waves
($\Phi_t u = \tau_{-ct} u$), relative periodic orbits ($\tau_\ell \Phi_T u = u$) and pre-periodic orbits
($R\,\Phi_T u = u$).

## The equation, as written

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

The model is one line: `m.explain()` would show the split (the linear terms implicit, $u u_x$
explicit) and the declared default dealiasing 3/2. The searches never see the equation: they see the
flow map $\Phi_T$ of the simulation's time stepper (`sim.solver`, RK443 at $dt = 0.01$) and its exact
tangent (Jacobian–vector products through the same stepper). Any model with `dt()` can be searched
the same way, within the scope of these finders: one periodic direction (1D problems such as KS;
flows with walls use the drivers listed in {doc}`../guide/dynamics/invariant_solutions`).
`NewtonKrylovHookstep` is a trust-region (hookstep) Newton with GMRES: its accuracy target `tol`
is required, its algorithm controls (GMRES tolerance and Krylov budget, trust radii) have declared
defaults that `nk.explain()` prints and marks. `state(u0)` turns an initial condition written as a function of $x$ into the state vector
the searches start from.

## Equilibria and their stability

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [equilibria]"
:end-before: "# [/equilibria]"
```

The seeds are single sine modes, $u_0 = \sin(2\pi n x/L)$. A seed that falls back to $u = 0$ is marked `retracted` (the
trivial-state threshold is declared in `explain()`), never reported as a new solution. The stability
eigenvalues come from the linearised flow at the equilibrium.

## A travelling wave

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [wave]"
:end-before: "# [/wave]"
```

The speed $c$ is an unknown of the Newton system, with a phase condition that removes the translation
degeneracy.

## The shortest pre-periodic orbit and its Floquet exponents

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [orbit]"
:end-before: "# [/orbit]"
```

The seed is a state stored by an earlier search (near-recurrence of a long run, then Newton), written in the script itself
(32 coefficients), so that the tutorial runs from any copy. Newton re-converges it on the
*discrete* map with `steps` RK443 steps, so the reported residual is that of the orbit actually
computed. The Floquet multipliers come from the dense monodromy matrix; the time step is halved
relative to the stored seed because $\mu_1$ moves by $2\times10^{-6}$ between $dt \approx 0.04$ and
$0.02$ (time-discretisation error).

## Check against the references

Energies and leading eigenvalues of $E_1$ and $E_2$, the speed and energy of $TW_1$, and the
leading Floquet exponent of $\overline{pp}_{10.25}$ agree with the published digits; every solution
is reported with its Newton residual.

```{literalinclude} code/t05_ks_orbits.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t05_ks_orbits.txt
:language: text
```

## Going further

- Continuation of the equilibria in $L$ and their bifurcation points (`KSDomainContinuation`,
  `tests/continuation/test_ks_L.py`).
- Is an instability of a symmetric solution symmetry-breaking? `symmetry_breaking_check`
  ({doc}`03_symmetries`).
- Kolmogorov flow in 2D: `tests/ecs/test_kolmogorov.py`.
