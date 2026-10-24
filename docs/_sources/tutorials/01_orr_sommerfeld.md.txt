# A first eigenvalue problem: Orr–Sommerfeld

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

- The model declares the geometry and the equations. The native compiler chooses the bases and
  the tau terms when you give `N` at the solve (`N=64`).
- `cn.Box(y=cn.walls(-1, 1))` names the geometry, and `m.given(U=lambda y: 1 - y**2)` the base
  flow (its argument names say it depends on $y$ only). `m.eigenvalue("c", growth="imag")` declares
  the eigenvalue and how a mode grows. For a phase speed the mode is $e^{i\alpha(x - ct)}$, so its
  temporal growth rate is $\alpha \operatorname{Im} c$; with $\alpha > 0$, as here, the sign of
  growth is the sign of $\operatorname{Im} c$ (`growth="real"` is for $e^{st}$).
  `m.define` names the operator $\mathcal{L}_2$.
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

The tau method produces spurious eigenvalues with huge $|c|$. The drift filter compares
eigenvalues at two resolutions: `drift=96` (i.e. `drift_plan=`) solves the same problem at a finer
resolution and keeps only the modes that reappear within `drift_tol`. A mode set aside is not
converged by this test: it may be a tau mode or a physical mode that $N = 64$ does not resolve. The
dense solver also
classifies eigenvalues as finite or infinite, with a separate threshold. `res.explain()` says how
many modes were removed and why. Each kept mode carries its backward residual $\|(L + cM)x\|$ and
its drift.

```{figure} media/t01_spectrum.png
:alt: A few eigenvalues in the plane of phase speed and growth, all below the zero line except one red point just above it near Re c = 0.24.
:width: 90%

Plane Poiseuille flow at Re = 10⁴, α = 1: the 7 modes kept by the drift test of the page (found again at N = 96). A mode varies as exp(iα(x − ct)), so Re c is its phase speed and α Im c its growth rate. Only the Tollmien-Schlichting wave (red), c = 0.23753 + 0.00374i, lies above the dashed line Im c = 0.
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
:alt: The magnitudes of the 64 Chebyshev coefficients of the mode on a log scale, even indices as red dots and odd indices as open blue squares: the even ones fall from 1 to about 4×10⁻¹⁰; the odd ones stay near 3×10⁻¹⁶.
:width: 100%

The Chebyshev coefficients of the mode (mode.coeffs("v")), even and odd indices apart. v is even in y, so only the even coefficients carry it: they fall to 4×10⁻¹⁰ at the last even index. The odd ones stay at the rounding level, about 3×10⁻¹⁶. The largest of the last 10 % of the coefficients is 6×10⁻⁹ of the largest one, the spectral-tail measure that the eigenvalue result reports.
```

## Critical Reynolds number, sensitivity, verification

```{literalinclude} code/t01_orr_sommerfeld.py
:language: python
:start-after: "# [critical]"
:end-before: "# [/critical]"
```

`ParametricEVP` builds $L(Re, \alpha)$ and $M(Re, \alpha)$ once, as Laurent polynomials in the
parameters reconstructed from plans at a few parameter values. `P.explain()` reports the
reconstruction error and the coefficient entries discarded at rounding level. `drift=96`
builds the same pencil at $N = 96$. As for `eigenvalues(drift=)`, only the modes found again there
within `drift_tol` are candidates, so the spurious tau modes are not selected. A mode grows when
$\alpha \operatorname{Im} c > 0$, which for $\alpha > 0$ means $\operatorname{Im} c > 0$: `critical` and `least_stable` read this off the
declared eigenvalue (`m.eigenvalue("c", growth="imag")`) and print it in `crit.explain()`. `critical` follows the least
stable mode by bordered Newton to $\operatorname{Im} c = 0$ and minimises over $\alpha$; a call that
leaves out a parameter of the pencil uses the value declared in the model (logged).
`P.sensitivity` gives $\partial c / \partial Re$ from the left and right eigenvectors (no finite
differences). `verify` then *reports* residuals, the spectral tail of the eigenvector (is $N = 64$
enough?) and Taylor tests of the parameter derivatives. A bracket in which the growth does not change
sign is refused with the growth at both ends and a suggested bracket.

```{figure} media/t01_neutral_curve.png
:alt: The neutral curve of plane Poiseuille flow: a red tongue opening to the right in the plane of Reynolds number (log scale) and wavenumber, its nose marked at Re about 5772 and alpha about 1.02.
:width: 100%

Im c of the least stable mode on a grid of (Re, α), from the parametric pencil of the page. Inside the red line the flow is unstable. Its nose is the critical point the script finds, Re_c = 5772.2 at α_c = 1.0205.
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
