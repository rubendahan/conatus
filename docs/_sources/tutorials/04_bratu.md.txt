# 4. Nonlinear boundary-value problems and continuation: Bratu

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

For $\lambda < \lambda_c$ there are two solutions (lower and upper branch); they merge at the fold
$\lambda_c$ and none exists beyond. A Newton solver alone cannot pass the fold: the Jacobian is singular
there. Pseudo-arclength continuation parametrises the branch by its arclength instead.

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

Tolerance, damping and iteration budget have no defaults: they are choices, and they are printed with
the result. `sol.explain()` lists the residual of every iteration and the **observed** order of
convergence $q_k = \log(r_{k+1}/r_k)/\log(r_k/r_{k-1})$ — about 2 here, the signature of an exact
Jacobian. Non-convergence raises `NewtonNotConverged` with the history; it never returns the last
iterate silently.

## Continuation through the fold

```{literalinclude} code/t04_bratu.py
:language: python
:start-after: "# [continuation]"
:end-before: "# [/continuation]"
```

`ContinuationSettings` has no defaults either (step size and its bounds, adaptation factor, corrector
tolerance and budget, finite-difference step for $\partial G/\partial\lambda$). The branch starts on the
lower branch at $\lambda = 0.3$, passes the fold and comes back down the upper branch for a few
points (`max_points`). Up there $u$ grows and $e^u$ needs more modes: at 64 modes its Chebyshev tail
stays below $10^{-12}$ along the whole branch; at too few modes Conatus refuses the step (a
non-constant coefficient whose Chebyshev tail exceeds `ncc_tail_tol`, $10^{-8}$ by default, is never
used or truncated silently: the error gives the resolution to use, or the `ncc_cutoff=` to declare). Folds are *located* (the $\lambda$-component of the tangent changes sign,
refined to $|t_\lambda| \approx 10^{-16}$), not read off the nearest point; `branch.explain()` reports
the residual at the located point and every step-size change.

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
