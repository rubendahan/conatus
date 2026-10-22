# Conatus

Write a PDE model once, then simulate it, compute its stability and find its invariant solutions from the same object.
Conatus is experimental research software for fluid instability and dynamical systems.

**Status.** CPU wheels for Linux x86-64 (`pip install conatus`); CUDA support is experimental. The
interface can still change; changes come with a deprecation warning and a changelog entry.

## Start here

```{include} tutorials/_generated/start.md
```

## A first calculation

When does flow between two plates become unstable? The script below writes the Orr–Sommerfeld
equation for plane Poiseuille flow, computes the least stable mode at $Re = 10\,000$, $\alpha = 1$,
then searches for the critical Reynolds number. It is the example of the website's home page.

```{literalinclude} ../../web/home/critical_reynolds.py
:language: python
```

What it printed:

```{literalinclude} ../../web/home/critical_reynolds.out
:language: text
:lines: 2-
```

Orszag (1971) gives $c = 0.23752649 + 0.00373967i$ and $Re_c = 5772.22$ at $\alpha_c = 1.02056$
(the script's 1.02055 differs in the last digit).
The residual is the backward error of the computed eigenpair; `drift=96` keeps only the modes found
again with 96 Chebyshev polynomials instead of 64. The {doc}`stability tutorial
<tutorials/01_orr_sommerfeld>` continues from this script.

## Documentation

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Installation
:link: install
:link-type: doc
`pip install conatus`, the CUDA wheel (experimental) or from source; check what your machine can run.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed tutorials by topic, each ending on a published value or a closed form.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
Geometries, equations, boundary conditions, simulations, GPU, dynamics, provenance.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
The public objects, from their docstrings.
:::

:::{grid-item-card} Mathematics
:link: math/index
:link-type: doc
The methods behind each part of the library.
:::
::::

## Principles

- **Declared choices.** Basis, resolution, boundary treatment, dealiasing, time stepper,
  tolerances and inner product are either written by you or shown by `explain()`
  ({doc}`principles`).
- **Residuals with results.** An eigenvalue, steady state or orbit is returned with its residual.
  A check against a finer resolution runs when you ask for it (`drift=` for eigenvalues, a second
  resolution for a solution).
- **Published references.** Each method is tested against a published table or a closed form,
  and each tutorial ends on one.

## Who it is for

Researchers and students in fluid dynamics and applied mathematics who study stability,
transition, pattern formation and invariant solutions.

```{toctree}
:hidden:
:caption: Getting started

install
tutorials/index
```

```{toctree}
:hidden:
:caption: Using Conatus

guide/index
gallery/index
api/index
```

```{toctree}
:hidden:
:caption: About

principles
math/index
citing
```
