# Bifurcations: the Lorenz equilibria

**Goal.** Follow the equilibria of the Lorenz system as $r$ grows, through the pitchfork at
$r = 1$ (with a declared symmetry) and the Hopf point, and compute the first Lyapunov coefficient
there.
**Expected.** Lorenz (1963): pitchfork at $r = 1$ with $x^2 = b(r-1)$ on $C^\pm$; Hopf at
$r_H = \sigma(\sigma+b+3)/(\sigma-b-1) = 470/19$ with $\omega^2 = b(\sigma + r_H)$, **subcritical**
(Marsden & McCracken 1976).
**Extensions.** Three later sections build on this result; each can be read alone. The cycles born
at the Hopf point. The Bogdanov–Takens normal form in two parameters (Kuznetsov 2004, §8.4): fold
curve $\beta_1 = \beta_2^2/4$, Hopf curve $\beta_1 = 0$, both ending at $(0,0)$. The Landau and
Ginzburg–Landau coefficients of Swift–Hohenberg rolls (Cross & Hohenberg 1993): $A^2 = 4r/3$,
$\tau_0 = 1$, $\xi_0^2 = 4$, $g_0 = 3$.
**Time:** under a minute on one CPU core.
**Script:** [`t10_bifurcations.py`](code/t10_bifurcations.py) · **Tests behind it:**
`tests/bifurcation/test_bifurcation_core.py`, `tests/bifurcation/test_amplitude.py`.

The core result ends with "Normal forms": the Hopf point and the sign of its Lyapunov coefficient.

## Write the system

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [system]"
:end-before: "# [/system]"
```

- `ode_problem` writes an ODE from its right-hand side, given as a Python function of the state
  and the parameters. A PDE written as a native model works the same way (Swift–Hohenberg below).
- `ir_system` discretizes once and exposes $M\dot x = G(x;p)$. The Jacobian and the multilinear forms
  $B = F''$, $C = F'''$ are symbolic Fréchet derivatives of the equations, with no finite
  differences in $x$. $\partial G/\partial r$ is symbolic too: $r$ enters linearly, and `explain()`
  says so.

## Equilibria and branch switching with a declared symmetry

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [equilibria]"
:end-before: "# [/equilibria]"
```

`continue_equilibria` evaluates three test functions at each point (fold: the parameter component of
the tangent; branch point: the determinant of the bordered Jacobian; Hopf: a product over eigenvalue
pairs) and **locates** each sign change by a secant, reporting the residual there. `branch_switch`
solves the algebraic branching equation with the second derivatives. The declared
$S = \mathrm{diag}(-1,-1,1)$ classifies the point as a *symmetry-breaking pitchfork*: the kernel is
$S$-odd, and the two half-branches $C^\pm$ are images of each other. The symmetry declaration and
the branching equation determine the two branch directions.

```{figure} media/t10_lorenz.png
:alt: Bifurcation diagram of the Lorenz system, x against r: a black line along x = 0 that turns dashed at r = 1, where two red branches split off; they turn dashed at the Hopf points near r = 24.7, from which brass lines of cycles bend back.
:width: 100%

The equilibria of the Lorenz system and their stability, as continued by the page (solid: stable, dashed: unstable): the origin, the pitchfork at r = 1, the branches C± and their subcritical Hopf points at r = 24.736842, and the unstable cycles born there.
```

## Normal forms

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [normal_forms]"
:end-before: "# [/normal_forms]"
```

The centre-manifold projection (Kuznetsov) gives the pitchfork cubic coefficient and, at the Hopf
point, $c_1$ and the **first Lyapunov coefficient** $\ell_1 = \mathrm{Re}\,c_1/\omega$: positive
here, so the Hopf bifurcation of Lorenz is subcritical. Each quantity is reported with its checks
(residuals of the eigenvectors and of the bordered solves); the normalization $\|q\| = 1$ is declared.

## Extension: cycles from the Hopf point

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [cycles]"
:end-before: "# [/cycles]"
```

The first cycle is *predicted by the normal form* (amplitude declared, $\Delta r = -\mathrm{Re}\,c_1|z|^2/\alpha'$),
corrected by orthogonal collocation and continued (Doedel, Keller & Kernévez 1991; {doc}`../math/continuation`). Subcritical: the cycles exist for
$r < r_H$. Followed further, their period diverges at
the homoclinic explosion $r \approx 13.926$.

```{figure} media/t10_cycles.png
:alt: Nested closed loops in the (x, z) plane around the point C−, small ones near it and larger ones further out.
:width: 100%

The periodic orbits of the cycle branch (orthogonal collocation, 40 intervals of degree 4) for r from 24.735 down to 22.335 (colour bar): born at the Hopf point around the equilibrium of the continued branch (cross), they grow as r decreases.
```

## Extension: fold and Hopf curves in two parameters

```{literalinclude} code/t10_bifurcations.py
:language: python
:start-after: "# [codim2]"
:end-before: "# [/codim2]"
```

Minimally augmented systems (Govaerts): one scalar condition $g = 0$ from a bordered matrix (fold), two
for Hopf (bordered $J^2 + \kappa I$, $\kappa = \omega^2$). Along the curves, test functions detect cusp,
Bogdanov–Takens, Bautin, fold–Hopf and Hopf–Hopf points. The fold curve passes through the BT point.
The Hopf curve ends there: $\kappa = \omega^2$ reaches 0, and for $\beta_2 > 0$ the equilibrium has a
real pair of eigenvalues $\pm\sqrt{\beta_2}$, which the augmented equations also satisfy with
$\kappa < 0$. The continuation stops at $\kappa = 0$, and every point it keeps is checked against
the eigenvalues of the Jacobian, computed on their own. The vanishing determinant at the BT point is
*not* reported as a fold–Hopf point (rejected with its reason).

```{figure} media/t10_bogdanov_takens.png
:alt: Two curves in the parameter plane: a red parabola of folds through the origin, and a green line of Hopf points at beta1 = 0 for beta2 < 0 that ends at the Bogdanov-Takens point at the origin.
:width: 100%

Codimension two: the fold and Hopf curves of the Bogdanov-Takens normal form, continued in (β₁, β₂). The fold curve follows β₁ = β₂²/4 through the BT point; the Hopf curve β₁ = 0 exists for β₂ < 0 and ends at the BT point.
```

## Extension: Landau and Ginzburg–Landau equations of a PDE

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
Bogdanov–Takens normal form (each Hopf point with the pair $\pm i\sqrt{-\beta_2}$ in the eigenvalues
of its Jacobian), and the Swift–Hohenberg coefficients $A^2/r = 4/3$, $\xi_0^2 = 4$,
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
