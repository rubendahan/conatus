# 1. A first eigenvalue problem: Orr–Sommerfeld

**Goal.** Compute the least stable mode of plane Poiseuille flow at $Re = 10^4$, $\alpha = 1$, and
the critical Reynolds number, with a residual for every number.
**Expected.** $c = 0.23752649 + 0.00373967\,i$ and $Re_c = 5772.22$, $\alpha_c = 1.02056$
(Orszag 1971, *J. Fluid Mech.* 50, 689). **Time:** under a minute on one CPU core.
**Script:** [`t01_orr_sommerfeld.py`](code/t01_orr_sommerfeld.py).

## The equation

For a normal mode $v(y)\,e^{i\alpha(x - ct)}$ of the wall-normal velocity, with $U = 1 - y^2$ and
$\mathcal{L}_2 = D^2 - \alpha^2$,

$$
\frac{1}{i\alpha Re}\,\mathcal{L}_2^2 v - U\,\mathcal{L}_2 v + U'' v + c\,\mathcal{L}_2 v = 0,
\qquad v = v' = 0 \ \text{at}\ y = \pm 1 .
$$

The eigenvalue $c$ appears linearly; Conatus reads the mass matrix off its coefficient.

## Write the problem

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

Things to notice:

- No basis, no size, no tau: `cn.Box(y=cn.walls(-1, 1))` names the geometry, `m.given(U=lambda y:
  1 - y**2)` the base flow (its argument names say it depends on $y$ only), `m.eigenvalue("c", growth="imag")`
  the eigenvalue and how a mode grows (Im $c$ for a phase speed; `growth="real"` for $e^{st}$), `m.define` the operator $\mathcal{L}_2$. The resolution is chosen when solving
  (`N=64`).
- `m.at("walls", "v = 0", "dy(v) = 0")` gives four conditions; each tau is placed by a fixed rule
  (in the equation where $v$ has its highest order, modes $N-1, \dots, N-4$) and
  `m.explain(full=True)` prints them (`m.explain()` is the short summary). The matrices are **bit-identical** to the hand-written IR version
  ([`expert/t01_orr_sommerfeld.py`](code/expert/t01_orr_sommerfeld.py), `add_bc`), which the test suite checks.

## Solve and reject spurious modes

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [solve]"
:end-before: "# [/solve]"
```

The tau method produces spurious eigenvalues with huge $|c|$. Conatus never hides them behind a
threshold: `drift=96` (i.e. `drift_plan=`) solves the same problem at a finer resolution and keeps only modes that
reappear within `drift_tol`. `res.explain()` says how many modes were removed and why, and every
kept mode carries its backward residual $\|(L + cM)x\|$ and its drift.

```{figure} media/t01_spectrum.png
:alt: A few eigenvalues in the plane of phase speed and growth, all below the zero line except one red point just above it near Re c = 0.24.
:width: 100%

The modes kept by the drift test of the page (found again at N = 96), in the plane of phase speed and growth. Only one lies above the dashed line: the Tollmien-Schlichting wave, unstable at Re = 10⁴.
```

## Plot the eigenfunction

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [mode]"
:end-before: "# [/mode]"
```

`res.mode(i)` unpacks the eigenvector into the named unknowns on the physical grid: the tau
variables are left out, `mode["v"]` holds the complex values $v(y_j)$ at the Chebyshev points
`mode.coordinate("y")`, and `mode.coeffs("v")` the Chebyshev coefficients. An eigenvector is only
defined up to a complex factor, so the normalisation is declared and printed by `mode.explain()`:
the first unknown has $\max|v| = 1$, real and positive where $|v|$ is largest (here the centre of
the channel). The physical perturbation is $\operatorname{Re}[v(y)\,e^{i\alpha(x - ct)}]$.

```{figure} media/t01_coefficients.png
:alt: The magnitudes of the 64 Chebyshev coefficients of the mode on a log scale: one sequence falls steadily from 1 to about 1e-12, the other stays near 1e-12.
:width: 100%

The Chebyshev coefficients of the mode (mode.coeffs("v")). The even ones fall to about 10⁻¹² by the last index, the level of the odd ones, which v, even in y, does not need: 64 modes resolve it.
```

## Critical Reynolds number, sensitivity, verification

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [critical]"
:end-before: "# [/critical]"
```

`ParametricEVP` builds $L(Re, \alpha)$ and $M(Re, \alpha)$ once, as exact Laurent polynomials in the
parameters. `drift=96` builds the same pencil at $N = 96$: as for `eigenvalues(drift=)`, only the
modes found again there within `drift_tol` are candidates, so the spurious tau modes never compete
with the physical ones (no hand-written bound on $|c|$). The growth of a mode is
$\operatorname{Im} c$ because $c$ is a phase speed: `critical` and `least_stable` read this off the
declared eigenvalue (`m.eigenvalue("c", growth="imag")`) and print it in `crit.explain()`. `critical` follows the least
stable mode by bordered Newton to $\operatorname{Im} c = 0$ and minimises over $\alpha$; a call that
leaves out a parameter of the pencil uses the value declared in the model (logged).
`P.sensitivity` gives $\partial c / \partial Re$ from the left and right eigenvectors (no finite
differences). `verify` then *reports* residuals, the spectral tail of the eigenvector (is $N = 64$
enough?) and Taylor tests of the exact derivatives. A bracket in which the growth does not change
sign is refused with the growth at both ends and a suggested bracket.

```{figure} media/t01_neutral_curve.png
:alt: The neutral curve of plane Poiseuille flow: a red tongue opening to the right in the plane of Reynolds number (log scale) and wavenumber, its nose marked at Re about 5772 and alpha about 1.02.
:width: 100%

Im c of the least stable mode on a grid of (Re, α), from the parametric pencil of the page. Inside the red line the flow is unstable; its nose is the critical point the script finds.
```

## Check against the reference

**Check.** $c = 0.2375264888 + 0.0037396706\,i$ with backward residual $4\times10^{-16}$, equal to Orszag's
$0.23752649 + 0.00373967\,i$ to all eight published digits; $Re_c = 5772.2218$ (Orszag: 5772.22),
$\alpha_c = 1.020547$ (Orszag: 1.02056).

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t01_orr_sommerfeld.txt
:language: text
```

## Going further

- Same problem with shift-invert (`eigenvalues(plan, method="shift-invert", target=0.24, n_modes=4)`).
- Left eigenvectors and the physical adjoint: {doc}`../guide/operators`.
- Symmetric subspaces of the same problem (half the cost): {doc}`03_symmetries`.
