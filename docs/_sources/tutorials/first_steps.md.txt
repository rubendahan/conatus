# Getting started in ten minutes

One equation, one plot, one physical number. You write the viscous Burgers equation as on paper,
simulate it, draw the solution and measure its kinetic energy, which the script checks against the
exact solution. Then four small changes show what Conatus decides for you, and what it never does.
The longer path through one problem, from stability to a simulation of convection rolls, is
{doc}`00_getting_started`.

**Time:** ten minutes; the script runs in a few seconds. **Script:** [`t00_first_steps.py`](code/t00_first_steps.py).

## Install

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install "conatus[tutorials]"
python -m conatus doctor
```

The last line says what this machine can run; {doc}`../install` has the other ways to install.

## The equation

A wave $u(x, t)$ on a periodic line of length $2\pi$ steepens under its own advection and is held
back by viscosity $\nu$:

$$
\partial_t u + u\,\partial_x u = \nu\,\partial_x^2 u, \qquad u(x, 0) = \sin x .
$$

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [model]"
:end-before: "# [/model]"
```

The domain, the unknown, the parameter and the equation are all you write. `m.explain()` prints
what Conatus decided from them: the linear term is treated implicitly, the product `u*dx(u)`
explicitly, and the product is dealiased by the 3/2 rule.

```{literalinclude} outputs/t00_first_steps.txt
:language: text
:start-at: "Time stepping:"
:end-at: "Dealiasing:"
```

## Simulate and plot

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [simulate]"
:end-before: "# [/simulate]"
```

A simulation needs a resolution, a scheme and a time step: three arguments, never defaults.
`run(until=t)` advances to $t$, which must be a whole number of steps: the step is never changed
for you.

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [plot]"
:end-before: "# [/plot]"
```

```{image} outputs/t00_first_steps_burgers.png
:alt: Five profiles of u against x from t = 0 to 2: the sine wave tilts forward and its front at x = pi becomes steep.
:width: 100%
```

## One physical number

The kinetic energy $E = \tfrac12 \int u^2\,dx$ starts at $\pi/2$ and decreases as viscosity
dissipates the steepening front.

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [energy]"
:end-before: "# [/energy]"
```

The Cole–Hopf transformation turns the Burgers equation into the heat equation, so the exact
solution is known (Cole 1951; Hopf 1950). The script ends by comparing the energy with it:

```{literalinclude} outputs/t00_first_steps.txt
:language: text
:start-at: "kinetic energy at t"
:end-at: "OK:"
```

## Four changes to try

Change one line of the script, run it again and read what happens.

1. **Less viscosity.** `cn.Model(..., nu=0.05)` makes the front twice as steep. The energy still
   matches the exact solution (change the `0.1` of `cole_hopf_energy` too), to $2\times10^{-10}$.
2. **Too few modes.** `m.simulate(N=32, ...)` cannot resolve the front. Conatus does not raise the
   resolution for you: it warns, once, with an `UnderResolvedWarning` that names the unknown, the
   direction and the spectral tail, and the energy is then only right to about $4\times10^{-6}$.
3. **A larger time step.** `dt=0.05` takes 40 steps instead of 2000. The scheme is third order in
   time, so the energy error grows to about $4\times10^{-6}$; `sim.cfl(safety=0.5)` tells you a
   stable step, it never imposes one.
4. **Another equation.** Drop the nonlinear term: `m.equations("dt(u) - nu*lap(u) = 0")` is the heat
   equation, and the energy decays exactly as $E(t) = \tfrac{\pi}{2} e^{-2\nu t}$.

## Next

{doc}`00_getting_started` takes one problem, convection between two plates, from its equations to
its stability, its onset and a simulation; then each tutorial of {doc}`index` ends on a published
value.

## Output

```{literalinclude} outputs/t00_first_steps.txt
:language: text
```
