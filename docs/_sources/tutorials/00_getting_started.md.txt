# Convection between two plates: from onset to simulation

One problem from beginning to end: a layer of fluid heated from below. You write its equations as
on paper, ask whether the fluid at rest is stable, find where convection starts (the published
value to $2\times10^{-4}$ in $Ra_c$ and $7\times10^{-4}$ in $k_c$), then simulate the rolls that appear above onset and measure the heat they
carry. New to Conatus? The ten-minute {doc}`first_steps` comes before this one.

**Expected.** Onset at $Ra_c = 1707.762$, $k_c = 3.117$ between rigid plates (Reid & Harris 1958;
Chandrasekhar 1961); the script finds $Ra_c = 1707.7618$ (absolute difference $2\times10^{-4}$,
relative $10^{-7}$: the published value is rounded) and $k_c = 3.11632$ (difference $7\times10^{-4}$,
relative $2\times10^{-4}$: the tabulated 3.117 differs in its fourth digit) and
checks both to $10^{-3}$; the simulation grows at the eigenvalue's rate; the saturated rolls carry the
same heat through both plates and across the layer.
**Time:** about two minutes of CPU (136 s on a cluster CPU node). **Needs:** the `tutorials` extra (`pip install "conatus[tutorials]"`, {doc}`../install`). **Script:** [`t00_getting_started.py`](code/t00_getting_started.py).

## The model

In thermal-diffusion units, $z \in [-1/2, 1/2]$, rigid plates at fixed temperatures:

$$
\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + Pr\,\nabla^2 \mathbf{u} + Pr\,Ra\,T\,\mathbf{e}_z,\qquad
\nabla\cdot\mathbf{u} = 0,\qquad
\partial_t T + \mathbf{u}\cdot\nabla T = \nabla^2 T,
$$

with $\mathbf{u} = 0$ and $T = \pm 1/2$ on the plates $z = \mp 1/2$.

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [model]"
:end-before: "# [/model]"
```

The domain names its boundaries (`bottom`, `top`), and conditions are stated where they hold. The
native layer chooses the bases and imposes the conditions by fixed rules. `m.explain()` sums up the
implicit–explicit split, the conditions by wall, the pressure gauge and the dealiasing (see the
output at the end of the page). `m.explain(full=True)` lists the compiler decisions tau by tau and
term by term.

## Is the fluid at rest stable?

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [stability]"
:end-before: "# [/stability]"
```

`linearize` derives the linear equations about the conduction state $T = -z$; `eigenvalues`
solves them for the roll $k = \pi$ with the resolution you choose. At $Ra = 1500$ the computed modes
of this roll decay; at $Ra = 2500$ one grows.

## Where convection starts

A disturbance $\propto e^{ikx + st}$ obeys two ordinary differential equations in $z$, written
here as a model on a line with the eigenvalue $s$ declared:

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [onset]"
:end-before: "# [/onset]"
```

`parametric` keeps $Ra$ and $k$ as live parameters; `critical` finds the neutral $Ra$ and minimises
it over $k$. `drift=48` keeps a mode only if the same pencil at 48 modes has it too, within
`drift_tol`. A mode that moves is not converged by this test and is set aside: the spurious modes of
the boundary treatment jump, and so does a physical mode that 32 modes do not resolve. The growth of $s$ is its real part, read off its name and
printed by `crit.explain()`.

```{figure} media/t00_neutral_curve.png
:alt: The neutral curve in the plane of wavenumber and Rayleigh number: a U-shaped red line with its minimum marked at Ra about 1708 and k about 3.12; the region above it, where modes grow, is shaded; two crosses at k = pi mark Ra = 1500 and 2500.
:width: 100%

Rayleigh-Bénard onset between rigid plates, Pr = 1: growth rate of the least stable mode over a grid of (k, Ra), from the parametric pencil of the onset block. The red line is where it vanishes; modes grow in the shaded region. Its minimum is the critical point Ra_c = 1707.762 at k_c = 3.1163. The crosses are the two runs of the page, Ra = 1500 (stable) and 2500 (unstable) at k = π.
```

## Simulate above onset

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [simulate]"
:end-before: "# [/simulate]"
```

The simulation and the eigenvalue problem come from the same model; while the disturbance is small
they agree.

```{figure} media/t00_growth_saturation.png
:alt: Left: the roll amplitude on a log scale rises along a straight dashed line, then levels off. Right: the Nusselt number stays at 1, rises to a peak and settles to a constant.
:width: 100%

The simulation of the page at Ra = 2500, Pr = 1; time in thermal diffusion units d²/κ. Left: the roll amplitude grows at the rate σ = 5.543 of the leading eigenvalue (dashed line, drawn through the run at t = 0.3) until the nonlinear terms stop it. Right: the heat flux across the layer in units of conduction, Nu = 1.474 at t = 6 (it varies by 4×10⁻¹⁵ over the last time unit).
```

## Steady rolls and the heat they carry

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [nusselt]"
:end-before: "# [/nusselt]"
```

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [saturate]"
:end-before: "# [/saturate]"
```

In a steady state the heat entering through the hot plate crosses every level and leaves through
the cold one: the three values of $Nu$ agree.

```{image} outputs/t00_getting_started_rolls.png
:alt: The saturated rolls in the (x, z) plane: warm fluid rising at the sides (red), cool fluid sinking in the middle (blue), isotherms, and a colour bar of the departure from conduction T + z.
:width: 100%
```

## The checks

```{literalinclude} code/t00_getting_started.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output

```{literalinclude} outputs/t00_getting_started.txt
:language: text
```

Next: tutorials 1 to 14, starting with {doc}`01_orr_sommerfeld`.
