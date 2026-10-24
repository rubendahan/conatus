# Nonlinear boundary-value problems and continuation: Bratu

**Goal.** Solve the Bratu problem $u'' + \lambda e^u = 0$ by Newton, see quadratic convergence
reported, then follow the solution branch by pseudo-arclength continuation through its fold.
**Expected.** The closed form $u = 2\ln(\cosh a / \cosh a x)$, $\lambda = 2a^2/\cosh^2 a$ (Bratu 1914;
Gelfand 1963), with the fold at $a \tanh a = 1$: $\lambda_c = 0.87845767978129\ldots$ on $[-1, 1]$
($= 3.51383071912516/4$, the classical value on $[0, 1]$). **Time:** a few seconds on one CPU core.
**Script:** [`t04_bratu.py`](code/t04_bratu.py) · **Tests behind it:** `tests/nlbvp/test_nlbvp.py`,
`tests/continuation/test_continuation.py::test_bratu_branch_through_the_fold`.

## The problem

$$
u'' + \lambda\, e^{u} = 0 \quad \text{on } [-1, 1], \qquad u(\pm 1) = 0 .
$$

For $0 < \lambda < \lambda_c$ the problem has two solutions (lower and upper branch). They merge
at the fold $\lambda_c$, and none exists beyond. A Newton solver alone cannot pass the fold,
because the Jacobian is singular there. Pseudo-arclength continuation parametrises the branch by its arclength instead.

## Write the problem

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

- The model has no `dt()`, and the equation is nonlinear in `u`: `m.explain()` reports a
  nonlinear boundary-value problem and keeps every term as written (there is no implicit/explicit
  split; Newton linearizes the whole residual). The two conditions `u = 0` at the walls are placed
  by the same fixed rule as in a simulation, and listed.
- `exp(u)` has an exact Fréchet derivative ($e^u\,\delta u$), which is what Newton uses. It is
  **not** alias-free for any dealiasing factor, so the factor is declared (`dealias=1`) rather than
  left to the native default; resolution is then checked by convergence in $N$, not assumed.
- `m.compile(64, ...)` returns the compiled problem: Newton and continuation act on
  `pb.problem`, and `pb.fields["u"]` names the unknown in the initial guess.

## Newton

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [newton]"
:end-before: "# [/newton]"
```

The tolerance `tol` is required: it decides the digits of the solution. The damping (default 1,
full Newton steps) and the iteration budget (default 50) have declared defaults; the script gives
both explicitly (1 and 20). All three are printed with the result. `sol.explain()` lists the residual of every iteration and the **observed** order of
convergence $q_k = \log(r_{k+1}/r_k)/\log(r_k/r_{k-1})$, about 2 here, consistent with local
quadratic convergence of Newton's method. Non-convergence raises `NewtonNotConverged` with the history of the iterations.

```{include} media/t04_newton.md
```

## Continuation through the fold

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [continuation]"
:end-before: "# [/continuation]"
```

`ContinuationSettings` requires what sets the accuracy and the scale of the branch (step size and
its bounds, corrector tolerance, number of points); the algorithm controls (adaptation factor,
corrector budget, finite-difference step for $\partial G/\partial\lambda$) have declared defaults,
marked "(default)" by `explain()`. The branch starts on the
lower branch at $\lambda = 0.3$, passes the fold and comes back down the upper branch for a few
points (`max_points`). Up there $u$ grows and $e^u$ needs more modes. At 64 modes its Chebyshev
tail stays below $10^{-12}$ along the whole branch. At too few modes Conatus refuses the step: a
non-constant coefficient whose Chebyshev tail exceeds `ncc_tail_tol` ($10^{-8}$ by default) raises
an error that gives the resolution to use, or the `ncc_cutoff=` to declare. The continuation routine
refines the fold by solving for a zero $\lambda$-component of the tangent (to
$|t_\lambda| \approx 10^{-16}$). `branch.explain()` reports the residual at the located point and
each step-size change.

```{figure} media/t04_branch.png
:alt: The solution branch of the Bratu problem: u(0) rises with lambda, turns back at a red fold near lambda = 0.878 and climbs on the upper branch, the dots lying on a thin closed-form curve.
:width: 100%

The branch followed by pseudo-arclength continuation (the page's settings, run for more points), on the closed form u(0) = 2 ln cosh a. The fold, located by bisection on the tangent, is at λ_c = 0.8784576798.
```

## Check against the closed form

Newton converges quadratically to the closed-form lower-branch solution, and the located fold
agrees with the root of $a\tanh a = 1$ to rounding.

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t04_bratu.txt
:language: text
```

## Going further

- Continuation of an eigenvalue problem's solutions in a domain length (KS equilibria in $L$,
  `KSDomainContinuation`) and bifurcation points located against Kevrekidis, Nicolaenko & Scovel
  (1990): `tests/continuation/test_ks_L.py`, {doc}`05_ks_orbits`.
- Periodic NLBVPs (a Fourier direction coupled by the nonlinearity, declared phase condition):
  `PeriodicNewtonSolver`, see the Swift–Hohenberg example in the gallery.
- Newton, continuation and bifurcations in the user guide: {doc}`../guide/index`.
