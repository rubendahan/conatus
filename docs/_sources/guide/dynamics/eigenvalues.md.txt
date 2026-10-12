# Eigenvalues and stability

Normal modes of a linear problem, each with its backward residual, with spurious modes removed only by a declared test.

A linear model with an eigenvalue, or the linearization of a nonlinear model about a base
state, is discretised into the generalized eigenvalue problem

$$ (L + \sigma M)\,x = 0 $$

for each wavenumber. All finite eigenvalues are computed (dense QZ) unless you ask for a few near
a target (shift-invert Arnoldi). Infinite eigenvalues (rows of boundary conditions) are counted
and removed; every finite mode reports its relative backward residual
$\|(L+\sigma M)x\| / ((\|L\| + |\sigma|\,\|M\|)\,\|x\|)$.

## An eigenvalue problem written as on paper

The Orr–Sommerfeld equation for plane Poiseuille flow: the eigenvalue `c` is declared, the base
flow `U(y)` is a given field, the conditions are named by boundary.

```{literalinclude} _check.py
:language: python
:start-after: "[eig-model]"
:end-before: "[/eig-model]"
:dedent: 4
```

## Solving, with a drift test

```{literalinclude} _check.py
:language: python
:start-after: "[eig-solve]"
:end-before: "[/eig-solve]"
:dedent: 4
```

Spectral tau methods can produce spurious finite eigenvalues. Conatus never removes them by a
hidden threshold: `drift=96` recomputes the spectrum at a second resolution and keeps only the
modes found again within `drift_tol`. The filter, and how many modes it removed, appear in
`res.explain()`. Without `drift=`, every finite mode is returned, and `res.explain()` and the
one-line `repr(res)` say so once ("no spurious-mode filter: tau modes may appear").

A cheaper test runs on every result, with or without `drift=`: the **spectral tail** of each
eigenvector. A spurious tau mode, or a mode the resolution does not hold, has its energy in the
last coefficients of the wall-normal axis (for Orr–Sommerfeld at $N = 64$: a relative tail energy
of 0.96–1.0 for the tau modes $c \approx 0.017 + 1470i$, $10^{-17}$ for the least stable physical
mode). A mode whose tail energy in the last 10 % of the coefficients exceeds
`cn.tolerances.SPURIOUS_TAIL_TOL` ($10^{-3}$, declared) is *flagged*: it stays in `res.values` and
`res.mode(i)`, is listed by `res.explain()` with its tail, its indices are `res.flagged`, and
`res.order()` ranks it after every mode that passes, so `res.order()[0]` is the least stable mode
that passes. `critical` and `least_stable` select only modes that pass. The flag removes nothing
(`drift=` remains the filter that removes modes); `spurious="none"` turns it off, and the result
says so.

A mode can pass the flag and still be under-resolved: for Orr–Sommerfeld at $Re = 10^4$ and
$N = 24$ the least stable mode that passes is $c = 0.580 + 0.024i$ (the converged value is
$0.2375 + 0.0037i$), with a tail of $3.7\times10^{-4}$ and a residual of $10^{-16}$. So the mode a
result **selects** — `res.order()[0]`, `P.least_stable(...)`, the mode of `critical(...)` — is
held to a second, stricter declared level, `cn.tolerances.UNRESOLVED_TAIL_TOL` ($10^{-10}$; measured:
the tail is $10^{-8}$ at $N = 32$ with $c$ off by $4\times10^{-5}$, $10^{-12}$ at $N = 48$ with
$9\times10^{-8}$). Without `drift=`, a selected mode above it gets the sentence "selected mode
under-resolved: ... use drift= or a larger N" in `res.unresolved_note`, in `repr(res)` and
`res.explain()`, and one `cn.errors.ResolutionWarning`; `res.unresolved` lists every passing mode
above the level. Nothing is removed or re-ranked. With `drift=` the drift test is the evidence and
no tail warning is given. `critical()` without a drift pencil prints the tail of the critical mode
and warns when it is above the level (the case above gives $Re_c = 5037.6$ instead of $5772.2$).

The drift test itself is checked before any solve: `drift` must be finer than `N` (equal keeps every
mode, coarser removes resolved ones), `drift_tol` must lie in $(0, 10^{-2}]$
(`cn.tolerances.DRIFT_TOL_MAX`: above it most modes find a neighbour and nothing is filtered), and a
drift test that removes every mode raises `ResolutionError` instead of returning an empty result.

`res.order()` ranks the modes from the least stable by the growth convention of the eigenvalue
(here the phase speed `c`, declared `m.eigenvalue("c", growth="imag")`: $\operatorname{Im} c$), and
`res.explain()` prints them in that order with the rule it used; `res.order("real")` and `res.order("imag")` sort by a part.

## Eigenfunctions on the grid

```{literalinclude} _check.py
:language: python
:start-after: "[eig-mode]"
:end-before: "[/eig-mode]"
:dedent: 4
```

`res.mode(i)` unpacks eigenvector `i` into the named unknowns on the physical (non-dealiased) grid,
tau variables (the fields lifted by a `lift(...)` term) left out: `mode["v"]` (complex values; a vector unknown has its components first),
`mode.coeffs("v")`, `mode.coordinate("y")`, and `mode.at("v", y=points)` for the exact value of the
expansion at any point (the walls included). An eigenvector is defined up to a complex factor; the
declared normalisation (`mode.explain()`) makes the first non-zero unknown have $\max|q| = 1$, real
and positive where $|q|$ is largest, the other unknowns scaled by the same factor; an unknown whose
largest coefficient is below $10^{-10}$ of the eigenvector's counts as zero (round-off is never
scaled to 1). The same works on
`lin.eigenvalues(...)` (unknown names, not the perturbation variables) and on the expert
`conatus.expert.eigenvalues(plan, ...)` of a Cartesian plan.

## What a result holds

Everything `res.explain()` prints is also an attribute, for a script or a notebook:

```python
import conatus as cn

m = cn.problems.native.orr_sommerfeld(Re=1e4, alpha=1.0, flow="poiseuille")
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)
print(res.label, "|", res.method)         # the group solved, and how (dense QZ here)
print(len(res), "modes kept;", res.n_infinite, "infinite eigenvalues (M singular on tau rows)")
print(res.growth_rule())                  # how order() ranks the modes, in words
print(res.growth_note)                    # where that rule comes from (the declared growth=)
print(res.filter_note)                    # the spurious-mode filter that ran (drift here)
print(res.drift[:3])                      # drift ratio of each kept mode
print(res.vectors.shape, res.left)        # right eigenvectors; left ones with left=True
print(res.context is not None)            # what res.mode(i) needs to unpack a vector

raw = m.eigenvalues(N=64)                 # no drift test: the spectral-tail flag only
print(raw.spurious_note)                  # the flag that ran, in words
print(raw.flagged, raw.spurious.sum())    # indices flagged (kept, ranked last), boolean flag
print(raw.tails[:3])                      # relative tail energy of each mode
print(raw.unresolved, raw.unresolved_note)   # passing modes with a large tail; a note if selected
```

## Linearizing a nonlinear model

A model with time derivatives is linearized about a base state given by value or as a function of
the coordinates. Here Rayleigh–Bénard convection about the conduction state, for the wavenumber
$k = \pi$:

```{literalinclude} _check.py
:language: python
:start-after: "[eig-linearize]"
:end-before: "[/eig-linearize]"
:dedent: 4
```

`lin.explain()` states how the base state entered (the zero field, or a non-constant coefficient
and along which coordinates) and which Fourier sizes were chosen to hold `k`.

It also states whether the base is a steady solution: the steady equations and boundary
conditions (time derivatives dropped) are evaluated at the base, the unknowns without a base value
(pressure, taus) fitted by least squares, and the largest relative residual per equation is
printed ("base state: steady residual ..."). Above `cn.tolerances.BASE_RESIDUAL_TOL` ($10^{-6}$) the
base is not a solution: about $T = +z$ instead of $-z$ the residual is 2, in `T(z=-0.5) = 0.5`;
plane Poiseuille without its driving force has 1 in the momentum equation. Such a base gets a
`cn.errors.BaseStateWarning` in v0.1 (refused in v0.2) and a WARNING line in `lin.explain()`; a base that
knowingly is not a solution (a frozen or forced profile) is declared with
`m.linearize(about=..., base_residual="accept")`.

Each wavenumber of a periodic direction is its own set of modes, chosen with `k=`. This holds for
`m.eigenvalues(N, k=...)` on a model with a periodic axis too: without `k=`, v0.1 solves the
$k = 0$ group alone and says so in a `cn.errors.GroupWarning` (required in v0.2). `k` is always a
wavenumber; the number of modes of `method="shift-invert"` is `n_modes=` (an integer `k` on a model
without a periodic direction keeps that former meaning with a `DeprecationWarning`; a non-integer
`k` there is refused, it used to be ignored). A curvilinear geometry chooses its group with `m=`
(and `kz=` or `ell=`).

## What a linearization holds

```python
import numpy as np
import conatus as cn

lin = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1000.0, Pr=1.0)   # below onset
print(lin.about)                          # the base state, by unknown
print(lin.period, lin.base_residual_mode) # None: a steady base; 'check': its residual is checked
print(lin.wavenumber_names())             # {'k': 'x'}: the names parametric(over=...) takes
plan = lin.discretize(24, k=3.1)
print(lin.group(plan, k=3.1))             # the group of the plan that holds k = 3.1
```

## Critical parameters

`m.parametric(N, over=...)` keeps the parameters symbolic in the matrices, so that the neutral
curve and its minimum are found by root finding on exact derivatives (a linearization has the same
call, with its wavenumber `k` as a parameter: {doc}`onset`):

```{literalinclude} _check.py
:language: python
:start-after: "[eig-critical]"
:end-before: "[/eig-critical]"
:dedent: 4
```

Which modes may be the least stable one is decided by the drift test of `drift=96` (as for
`eigenvalues`), never by a hand-written bound on $|c|$; `select=` remains available for a physical
restriction, logged; modes flagged by the spectral-tail test never compete. The growth of a mode
is declared with the eigenvalue: `m.eigenvalue("c", growth="imag")` for a phase speed (mode
$e^{i\alpha(x - ct)}$, growth $\operatorname{Im} c$), `growth="real"` for a rate (mode $e^{st}$),
or `critical(growth=...)` for one call; a linearization's eigenvalue is the $s$ of $e^{st}$
(growth $\operatorname{Re} s$, declared by construction). In v0.1 an undeclared convention is still
guessed from the name (`c` → Im, `s`, `sigma`, `lam` → Re) with a `cn.errors.DefaultChoiceWarning`
naming the rule guessed; it is required from v0.2. The guess fails for a temporal eigenvalue named `c`
(`c*T = kappa*dz(dz(T))`): `res.explain()` then says that the growth part is the same for every mode. A parameter of the pencil that a call leaves out
takes the value declared in the model (logged in `crit.explain()`). The growth of the least stable
admissible mode is computed at both ends of the bracket on every call (with `values=`, `over=` or
neither; logged): a bracket in which it does not change sign is refused with the growth at both
ends and the sub-brackets where a scan of the bracket finds a sign change. A neutral curve can
cross a bracket twice (Orr–Sommerfeld at $\alpha = 1$ between $Re = 5000$ and $40000$: the
lower branch near 5815 and the upper one near 31956): the refusal names both, and each branch is
found from its own bracket.

## Left eigenvectors and sensitivity

```{literalinclude} _check.py
:language: python
:start-after: "[eig-left]"
:end-before: "[/eig-left]"
:dedent: 4
```

With `left=True` the result also carries the left eigenvectors $y$, $y^H(L + \sigma M) = 0$. They
give the derivative of an eigenvalue with respect to any parameter,
$\mathrm d\sigma/\mathrm dp = -\,y^H (\partial_p L + \sigma\,\partial_p M)\,x \,/\, (y^H M x)$.

## What you choose, what Conatus refuses to choose

- The resolution `N`, the second resolution of the drift test and its tolerance.
- The growth convention of the eigenvalue (`m.eigenvalue("c", growth="imag")`).
- Whether flagged modes are flagged (`spurious="tail"`, the default, or `"none"`).
- Dense or sparse solution (`method=`, `target=`, `n_modes=`): never switched for you.
- The wavenumber `k=` of a periodic direction (or `m=`, `kz=`, `ell=` of a curvilinear one).
- A singular pencil (a missing or extra boundary condition) is refused with a rank report.

## Validation

Read the residual of the mode you use and its drift between two resolutions. The spectral tail
of the eigenvector is checked by `verify` (see {doc}`../resolution`).

**See also:** tutorial {doc}`../../tutorials/01_orr_sommerfeld` (Orszag 1971);
{doc}`eigenvalues <../../api/_generated/_stubs/conatus.expert.eigenvalues>`, {doc}`EVPResult <../../api/_generated/_stubs/conatus.expert.EVPResult>`,
{doc}`ParametricEVP <../../api/_generated/_stubs/conatus.expert.ParametricEVP>`, {doc}`critical <../../api/_generated/_stubs/conatus.expert.critical>`.
