# First steps: the Burgers equation in ten minutes

Simulate the viscous Burgers equation, plot its evolution and measure its kinetic energy. Then
change one line at a time and see what changes. The longer path through one problem, from
stability to a simulation of convection rolls, is {doc}`00_getting_started`.

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

You write the domain, the unknown, the parameter and the equation. `m.explain()` prints how
Conatus will treat each term. The linear term is implicit, the product `u*dx(u)` is explicit, and
the product is dealiased by the 3/2 rule:

```{literalinclude} outputs/t00_first_steps.txt
:language: text
:start-at: "Time stepping:"
:end-at: "Dealiasing:"
```

## Simulate and plot

Specify the resolution `N`, the time-stepping scheme and the time step `dt`. `run(until=t)`
advances to time $t$, which must be a whole number of steps.

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [simulate]"
:end-before: "# [/simulate]"
```

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [plot]"
:end-before: "# [/plot]"
```

```{image} outputs/t00_first_steps_burgers.png
:alt: Five profiles of u against x from t = 0 to 2: the sine wave tilts forward and its front at x = pi becomes steep.
:width: 100%
```

## The kinetic energy

The kinetic energy $E = \tfrac12 \int u^2\,dx$ starts at $\pi/2$ and decreases as viscosity
dissipates the steepening front.

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [energy]"
:end-before: "# [/energy]"
```

```{literalinclude} outputs/t00_first_steps.txt
:language: text
:start-at: "kinetic energy at t"
:end-at: "relative error"
```

The second line compares $E$ with the exact solution, given by the Cole–Hopf transformation (Cole
1951; Hopf 1950). The code of that comparison is below.

:::{dropdown} Verification against the exact solution

The Cole–Hopf transformation turns the Burgers equation into the heat equation, whose solution
from $u(x, 0) = \sin x$ is a series of Bessel functions. The script computes the exact energy from
that series and checks the simulation against it to $10^{-9}$ relative.

```{literalinclude} code/t00_first_steps.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

:::

## Changes to try

Change one line of the script, run it again and read what happens.

1. **Less viscosity.** `cn.Model(..., nu=0.05)` makes the front twice as steep. Change the `0.1`
   of `cole_hopf_energy` too: the energy still matches the exact solution, to $2\times10^{-10}$.
2. **Too few modes.** With `m.simulate(N=32, ...)` the front is not resolved. Conatus keeps the
   resolution you gave and warns once, with an `UnderResolvedWarning` that names the unknown, the
   direction and the spectral tail. The energy is then right to about $4\times10^{-6}$.
3. **A larger time step.** `dt=0.05` takes 40 steps instead of 2000. The scheme is third order in
   time, so the energy error grows to about $4\times10^{-6}$. `sim.cfl(safety=0.5)` suggests a
   stable step; the run keeps the `dt` you gave.
4. **Another equation.** Drop the nonlinear term: `m.equations("dt(u) - nu*lap(u) = 0")` is the heat
   equation, and the energy decays exactly as $E(t) = \tfrac{\pi}{2} e^{-2\nu t}$.

## Next

{doc}`00_getting_started` takes one problem, convection between two plates, from its equations to
its stability, its onset and a simulation. Each tutorial of {doc}`index` ends on a published value.

:::{dropdown} The whole output of the script

```{literalinclude} outputs/t00_first_steps.txt
:language: text
:lines: 2-
```

:::
