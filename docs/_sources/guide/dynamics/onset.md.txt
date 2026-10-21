# Onset of convection

The critical Rayleigh number of Rayleigh–Bénard convection between no-slip walls, and the shape of
its mode, in a few calls: from the ready linearization, then from the equations of a normal mode
written by hand.

A layer of fluid heated from below is at rest while the Rayleigh number $Ra$ is small. The rest
state loses stability at $Ra_c = 1707.762$, for rolls of wavenumber $k_c = 3.117$ (Reid & Harris
1958). Finding it means solving two problems at once: for each $k$, the $Ra$ at which the growth
rate of the least stable normal mode $e^{ikx + st}$ crosses zero (the neutral curve), and the
minimum of that curve over $k$.

## From the ready model

`cn.problems.native.rayleigh_benard_evp` is the linearization of the native Rayleigh–Bénard model
about the conduction state. `lin.parametric` keeps $Ra$ and the wavenumber $k$ symbolic in the
matrices of the eigenvalue problem, so that `critical` finds the neutral $Ra$ by Newton on exact
derivatives and minimises it over $k$:

```{literalinclude} _check.py
:language: python
:start-after: "[onset-ready]"
:end-before: "[/onset-ready]"
:dedent: 4
```

```text
CriticalPoint(Ra_c=1707.761777, k=3.116323528, growth 2.4e-15, residual 1.5e-19, verified=True; .explain() for the log)
```

`drift=32` builds the same pencil with 32 Chebyshev modes: a mode is a candidate only if it is
found again there (spurious modes of the boundary treatment are not), and `crit.explain()` gives
the count. The growth of `s` is its real part, read off its name. `k` is the name of the wavenumber
of the one periodic axis (`kx`, `ky` with several: `lin.wavenumber_names()`).

## The mode at onset

```{literalinclude} _check.py
:language: python
:start-after: "[onset-mode]"
:end-before: "[/onset-mode]"
:dedent: 4
```

`res.order()` ranks the modes from the least stable by the growth convention of the eigenvalue.
`md.at("T", z=...)` evaluates the spectral expansion of the eigenfunction at any point,
the walls included (the Chebyshev grid points of `md["T"]` do not reach them): here $T = 0$ and
$u = 0$ on both walls to round-off. Along the complex periodic axis the values are those of
$T(z)\,e^{ikx}$ at the grid points of $x$; the row $x = 0$ is $T(z)$ itself. The normalisation is
declared by `md.explain()`.

## From the equations of a normal mode

The same problem written for the amplitudes $u, w, p, T$ of $e^{ikx + st}$, as on paper: $\partial_x$
becomes $ik$, the walls are named, and the pressure is declared as a pressure:

```{literalinclude} _check.py
:language: python
:start-after: "[onset-primitive]"
:end-before: "[/onset-primitive]"
:dedent: 4
```

Conatus recognises the pressure of a normal-mode model (it enters as $ikp$ and $\partial_z p$, it
has no condition) and places the taus itself: the continuity equation gets the divergence tau of
the wall-normal velocity, as in the vector form; `m.explain(24)` prints it. The critical point is
the same as above to round-off. Declared with `m.scalar("p")` instead, the model is refused with
the fix "declare it with `m.pressure('p')`"; a condition written on the pressure is refused too.

## What you choose, what Conatus refuses to choose

- The resolution, the drift resolution and its tolerance, the bracket of $Ra$, the interval of
  $k$ and the precision `xtol` of the critical value.
- A bracket without a sign change of the growth is refused with the growth at both ends (checked on
  every call) and the sub-brackets where a scan finds one.
- The critical mode is checked against a full QZ at the critical point: if another admissible
  mode were less stable there, the result would be refused.

**See also:** {doc}`eigenvalues`; tutorial {doc}`../../tutorials/00_getting_started` (the
$w$–$\theta$ form).
