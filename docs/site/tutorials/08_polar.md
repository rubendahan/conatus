# 8. Polar geometry: a drum, a pipe, two rotating cylinders

The same workflow runs on the disk, the annulus and the cylinder, on regular bases, with nothing
divided by $r$.

**Goal.** Solve three classical eigenvalue problems in polar and cylindrical geometry: the modes of
a vibrating drum, written as a native model on `cn.Disk`; the linear stability of Hagen–Poiseuille
flow in a pipe; and the onset of Taylor vortices between rotating cylinders. The last two come from
the library of ready problems (`cn.problems`).
**Expected.** Drum: eigenvalues $-j_{m,n}^2$, the squared zeros of the Bessel functions (closed
form). Pipe, $R = 1000$: rightmost eigenvalues of $(m, k) = (1, 1)$ and $(1, 0)$ to 10 decimals
(Meseguer & Trefethen 2003, *J. Comput. Phys.* 186, Table 1). Taylor–Couette, radius ratio
$\eta = 0.5$, outer cylinder at rest: $Re_c = 68.19$ at $k_c = 3.16$ (Recktenwald, Lücke & Müller
1993, *Phys. Rev. E* 48, 4444, Table I).
**Time:** under a minute on one CPU core.
**Script:** [`t08_polar.py`](code/t08_polar.py) · **Tests behind it:** `tests/polar/`,
`tests/problems/test_problems_cylinders.py` (also against Dedalus to $10^{-13}$).

## Regular bases, nothing divided by r

On the disk each component is expanded in a regular (Zernike-type) basis
$\rho^{|m|}\,P_n^{(a,|m|)}(2\rho^2 - 1)$, vectors are stored in spin components
$u_\pm = u_r \mp i u_\varphi$, and every operator is a sparse map between these bases (Vasil et al.
2016). There is no axis condition to impose and no $1/(r + \varepsilon)$: regularity at $r = 0$ is
built in. You name the geometry and its wall; `explain()` prints the bases it chose.

```{literalinclude} code/t08_polar.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

## A vibrating drum

```{literalinclude} code/t08_polar.py
:language: python
:start-after: "# [disk]"
:end-before: "# [/disk]"
```

- `cn.Disk(radius=1.0)` has one boundary, `wall`. The model declares an eigenvalue `s` and the
  equation $sT = \nabla^2 T$; the condition $T = 0$ is given by name. The tau term is placed by the
  rule of the disk (lifted in the derivative basis), and `explain()` says so.
- The plan has one block per azimuthal number $m$. The expert function `eigenvalues(plan, group)`
  solves one block; the native `drum.eigenvalues(N, m=0)` solves the block $m = 0$ (`m=` names
  the block; without it v0.1 solves the first block and says so in a `DeprecationWarning`).
- The exact answer is $s = -j_{m,n}^2$: separation of variables gives $J_m(\sqrt{-s}\,r)$, and the
  wall condition picks the zeros $j_{m,n}$ of $J_m$.

## Hagen–Poiseuille

```{literalinclude} code/t08_polar.py
:language: python
:start-after: "# [pipe]"
:end-before: "# [/pipe]"
```

- `cn.problems.pipe_evp` returns an ordinary problem: the linearised Navier–Stokes equations about
  $U_0 = (1 - r^2)\,\mathbf{e}_z$, no slip at $r = 1$, with its validation reference.
  `hp.explain()` prints the equations, bases and conditions; open the function to read or copy it.
- The plan has one block per $(m, k_z)$: `plan.group(1, 1.0)`; $k_z = 0$ is a block of the same
  plan. Each block gets its own rank report before any solve.
- Sign convention: Meseguer & Trefethen's Table 1 lists $\operatorname{Im}\lambda$ with the opposite
  sign to the one implied by their own eq. (15) and $V_t = \mathcal{L}V$ (the centre mode advected by
  $W \approx 1$ must have $\operatorname{Im}\lambda \approx -k$); Dedalus agrees with Conatus. The
  comparison is made with the conjugate, and that is written in the check.

## Taylor–Couette

```{literalinclude} code/t08_polar.py
:language: python
:start-after: "# [couette]"
:end-before: "# [/couette]"
```

Units: gap $d = r_o - r_i = 1$, inner wall speed $\Omega_i r_i = 1$, $Re = \Omega_i r_i d/\nu$. The base
flow is circular Couette flow $V(r) = Ar + B/r$. For each $k$ the neutral Reynolds number (largest
growth rate $= 0$) is found by Brent's method on the dense spectrum of the axisymmetric block
$(m, k_z) = (0, k)$; the minimum over three wavenumbers brackets $k_c$.

## Check against the references

The drum eigenvalues equal $-j_{m,n}^2$ to $10^{-12}$ relative. The pipe eigenvalues match
Meseguer & Trefethen's to the 10 published decimals. The Taylor–Couette threshold
$Re_c = 68.1863$, converged in $N$, agrees with the published $68.19$ to its 4 digits.

```{literalinclude} code/t08_polar.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t08_polar.txt
:language: text
```

## Going further

- Time integration on the disk, annulus, sphere, ball and shell uses the same native calls
  (`m.simulate(...)`); see {doc}`../guide/geometries`.
- The ready problems `cn.problems.pipe`, `taylor_couette` and `rotating_shell_convection` give the
  nonlinear versions of these flows.
