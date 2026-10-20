# 10. Bifurcations: branch switching, cycles, codimension 2, normal forms

**Goal.** Build the bifurcation diagram of the Lorenz system (pitchfork with a declared symmetry,
Hopf point, its first Lyapunov coefficient, the cycles born there), follow fold and Hopf curves of
the Bogdanov–Takens normal form in two parameters, and derive the Landau and Ginzburg–Landau
coefficients of Swift–Hohenberg rolls.
**Expected.** Lorenz (1963): pitchfork at $r = 1$ with $x^2 = b(r-1)$ on $C^\pm$; Hopf at
$r_H = \sigma(\sigma+b+3)/(\sigma-b-1) = 470/19$ with $\omega^2 = b(\sigma + r_H)$, **subcritical**
(Marsden & McCracken 1976). Bogdanov–Takens (Kuznetsov 2004, §8.4): fold curve $\beta_1 = \beta_2^2/4$,
Hopf curve $\beta_1 = 0$, both ending at $(0,0)$. Swift–Hohenberg (Cross & Hohenberg 1993):
$A^2 = 4r/3$, $\tau_0 = 1$, $\xi_0^2 = 4$, $g_0 = 3$. **Time:** under a minute on one CPU core.
**Script:** [`t10_bifurcations.py`](code/t10_bifurcations.py) · **Tests behind it:**
`tests/bifurcation/test_bifurcation_core.py`, `tests/bifurcation/test_amplitude.py`.

## Write the system

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [system]"
:end-before: "# [/system]"
```

- `ode_problem` writes an ODE from its right-hand side, given as a Python function of the state
  and the parameters. A PDE written as a native model works the same way (Swift–Hohenberg below).
- `ir_system` discretizes once and exposes $M\dot x = G(x;p)$. The Jacobian and the multilinear forms
  $B = F''$, $C = F'''$ are exact symbolic Fréchet derivatives of the equations: **no finite differences in $x$**.
  $\partial G/\partial r$ is exact too: the equations say that $r$ enters linearly, and `explain()` says so.

## Equilibria and branch switching with a declared symmetry

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [equilibria]"
:end-before: "# [/equilibria]"
```

`continue_equilibria` evaluates three test functions at every point (fold: the parameter component of
the tangent; branch point: the determinant of the bordered Jacobian; Hopf: a product over eigenvalue
pairs) and **locates** each sign change by a secant, reporting the residual there. `branch_switch`
solves the algebraic branching equation with the exact second derivatives; the declared
$S = \mathrm{diag}(-1,-1,1)$ classifies the point as a *symmetry-breaking pitchfork*: the kernel is
$S$-odd, and the two half-branches $C^\pm$ are images of each other. Nothing is guessed from a plot.

```{figure} media/t10_lorenz.png
:alt: Bifurcation diagram of the Lorenz system, x against r: a black line along x = 0 that turns dashed at r = 1, where two red branches split off; they turn dashed at the Hopf points near r = 24.7, from which brass lines of cycles bend back.
:width: 100%

The equilibria of the Lorenz system and their stability, as continued by the page: the origin, the pitchfork at r = 1, the branches C± and their subcritical Hopf points, and the cycles born there.
```

## Normal forms

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [normal_forms]"
:end-before: "# [/normal_forms]"
```

The centre-manifold projection (Kuznetsov) gives the pitchfork cubic coefficient and, at the Hopf
point, $c_1$ and the **first Lyapunov coefficient** $\ell_1 = \mathrm{Re}\,c_1/\omega$: positive
here, so the Hopf bifurcation of Lorenz is subcritical. Every quantity is reported with its checks
(residuals of the eigenvectors and of the bordered solves); the normalization $\|q\| = 1$ is declared.

## Cycles from the Hopf point

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [cycles]"
:end-before: "# [/cycles]"
```

The first cycle is *predicted by the normal form* (amplitude declared, $\Delta r = -\mathrm{Re}\,c_1|z|^2/\alpha'$),
corrected by orthogonal collocation (AUTO's method) and continued. Subcritical: the cycles exist for
$r < r_H$. Followed further, their period diverges at
the homoclinic explosion $r \approx 13.926$.

```{figure} media/t10_cycles.png
:alt: Nested closed loops in the (x, z) plane around the point C+, small ones near it and larger ones further out.
:width: 100%

The periodic orbits of the cycle branch (orthogonal collocation, 40 intervals of degree 4) for r from 24.735 down to 22.335: born at the Hopf point around the equilibrium of the continued branch (cross), they grow as r decreases.
```

## Two parameters: fold and Hopf curves

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [codim2]"
:end-before: "# [/codim2]"
```

Minimally augmented systems (Govaerts): one scalar condition $g = 0$ from a bordered matrix (fold), two
for Hopf (bordered $J^2 + \kappa I$, $\kappa = \omega^2$). Along the curves, test functions detect cusp,
Bogdanov–Takens, Bautin, fold–Hopf and Hopf–Hopf points; here both curves end at the BT point, and the
vanishing determinant there is *not* reported as a fold–Hopf point (rejected with its reason).

```{figure} media/t10_bogdanov_takens.png
:alt: Two curves in the parameter plane: a red parabola of folds and a green line of Hopf points at beta1 = 0, meeting at the Bogdanov-Takens point at the origin.
:width: 100%

Codimension two: the fold and Hopf curves of the Bogdanov-Takens normal form, continued in (β₁, β₂). The fold curve follows β₁ = β₂²/4, the Hopf curve β₁ = 0; both end at the BT point.
```

## Landau and Ginzburg–Landau equations of a PDE

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [landau]"
:end-before: "# [/landau]"
```

Swift–Hohenberg is a native model: the operator $1 + \partial_{xx}$ is named once with
`m.define`, the equation is written as on paper, and `dealias=2` is written out although it is
the native default for a cubic nonlinearity ((p+1)/2 with p = 3): the published coefficients do not
depend on a default. The even subspace $u(-x) = u(x)$ removes the translation mode; it is declared
on the compiled problem (expert layer), because the native layer has no `symmetry=` option yet.
`amplitude_equation` rewrites the pitchfork normal form for an **observable** amplitude (here the
$\cos x$ coefficient), and `ginzburg_landau` adds the curvature of the growth rate $\sigma(k)$
(systems rebuilt at $k_c \pm h$, $\pm 2h$, fourth-order differences, step declared). The same call
on stress-free Rayleigh–Bénard rolls gives $\mathrm{Nu} - 1 = 2\varepsilon$ (Malkus & Veronis 1958)
with spectral convergence in $N_z$: `examples/bifurcation/weakly_nonlinear.py`.

## Check against the closed forms

The pitchfork at $r = 1$, the branch $x^2 = b(r-1)$, the Hopf point $r_H = 470/19$ with its
frequency and a positive first Lyapunov coefficient, the fold and Hopf curves of the
Bogdanov–Takens normal form, and the Swift–Hohenberg coefficients $A^2/r = 4/3$, $\xi_0^2 = 4$,
$g_0 = 3$ are all recovered to rounding.

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t10_bifurcations.txt
:language: text
```

## Going further

- Period doubling with branch switching onto the doubled orbit, Neimark–Sacker detection, branch
  points of cycles: `tests/bifurcation/test_bifurcation_core.py` (constructed systems with exact
  multipliers).
- Continuation and bifurcations in the user guide: {doc}`../guide/index`.
