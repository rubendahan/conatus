# 3. Symmetries and symmetry subspaces

**Goal.** Compute the Rayleigh–Bénard onset mode in the subspace of solutions symmetric about
mid-depth, with half the unknowns, and see what the subspace hides.
**Expected.** At $(Ra_c, k_c) = (1707.762, 3.1163)$ (Reid & Harris 1958; Chandrasekhar 1961) the
leading growth rate is zero to the published digits; the subspace gives the same eigenvalue to
rounding with half the unknowns. The complementary class decays.
**Time:** a few seconds of CPU.
**Script:** [`t03_symmetries.py`](code/t03_symmetries.py) · **Tests behind it:**
`tests/evp/test_rayleigh_benard.py`.

## The idea

A symmetry is declared, never guessed. You give a group (here the reflection $z \to -z$) and the
parity of each field; Conatus then

1. checks that every equation is **equivariant** and refuses otherwise, naming the offending term;
2. builds bases of the invariant subspace (even or odd Chebyshev polynomials, cosines or sines) and
   drops the boundary conditions that became mirror images of each other;
3. solves only there, on CPU or GPU.

For a vector field you declare only the "scalar" parity: under a reflection the component along the
reflected direction changes sign, the others do not (`du=ODD` means $u_x$ odd, $u_z$ even).

## The problem

The Boussinesq equations of {doc}`02_rayleigh_benard`, written as on paper, on one wavelength
$2\pi/k_c$, and their linearization about the conduction state $T_0 = -z$:

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

## The full space first

`discretize(32, k=K_C)` takes 32 modes across the layer and the smallest Fourier size that holds
$k_c$; the eigenvalues of that wavenumber come with their residuals:

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [full]"
:end-before: "# [/full]"
```

## Two symmetry classes

This step uses the expert layer, because the native layer has no `symmetry=` option yet: the
subspace is declared on the compiled perturbation problem, and the boundary unknowns that the native
layer created (`tau_u1`, `tau_T1`, ..., listed by `m.explain()`) need their parities written out
like the physical fields.

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [subspace]"
:end-before: "# [/subspace]"
```

## Solve in each class

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [solve]"
:end-before: "# [/solve]"
```

`linearize` rebuilds the perturbation problem with `symmetry=`; everything downstream (plan,
eigenvalues, residuals) is unchanged. `symmetry_report()` lists the parity of every component, the equivariance of every
equation and the boundary conditions dropped as mirror images, and always warns that the subspace
**hides symmetry-breaking instabilities**. Here the other class is the check: its leading mode decays
strongly, so the onset mode found in the symmetric class is the leading mode of the
full problem. A dense QZ costs $O(n^3)$; halving $n$ saves 2–8× depending on the size.

## A wrong declaration is refused

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [guard]"
:end-before: "# [/guard]"
```

## Check

At Reid & Harris' $(Ra_c, k_c)$ the leading growth rate vanishes to the published digits, the
onset class gives the same eigenvalue to rounding with half the unknowns, and the other class
decays.

```{literalinclude} code/t03_symmetries.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t03_symmetries.txt
:language: text
```

## Going further

- Time-stepping in a subspace: `rayleigh_benard_2d(..., symmetric=True)` in `tests/ivp/ivp_cases.py`
  (reflection $x \to -x$, same trajectory as the full space to $10^{-12}$ with half the unknowns);
  initial data outside the subspace is refused.
- Shift-reflect, m-fold rotations, equatorial parity: `ShiftReflect`, `AzimuthalMFold`,
  `EquatorialParity` (see {doc}`../guide/bases`).
- Symmetric orbits of Kuramoto–Sivashinsky: {doc}`05_ks_orbits`.
