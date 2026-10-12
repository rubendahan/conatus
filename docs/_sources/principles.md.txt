# Principles

Conatus is written for people who publish the numbers it computes, and its code enforces the rules
that follow rather than recommending them.

## 1. No silent substitution

A numerical method is a list of choices: basis and resolution, boundary treatment (tau and lift
positions), dealiasing factor, time stepper and step, tolerances, inner product, definition of the
adjoint. **Conatus never changes one of these for you.** It may *suggest*; it never *modifies*.

Concretely:

- **Dealiasing is declared.** Unless you give `dealias=`, a native model uses the factor
  $(p+1)/2$ for the highest polynomial degree $p$ of its equations in the unknowns, never below 3/2
  (3/2 for quadratic terms, 2 for cubic ones), and says so in `explain()` and in the *Methods*
  paragraph; a value you give always wins; an expert basis requires the factor. There
  is no opaque `dealias=True`, and `explain()` warns when a nonlinear term (`exp`, `tanh`…) is not
  alias-free at the declared factor.
- **Automatic means a fixed rule, printed.** Each boundary condition, the pressure gauge and the
  implicit or explicit treatment of each term are placed by deterministic rules and listed by
  `explain()`; when a rule cannot decide, the model is refused with the way to say it. A
  structural rank report refuses an under- or over-determined system *before* any factorisation.
- **No hidden cut-offs.** Truncations of non-constant coefficients and removed matrix entries are
  counted and reported; filters on eigenvalues are written by the user and logged with the number
  of modes they remove.
- **No regularisation of singularities.** Curvilinear geometries use regular bases (Zernike-type
  polynomials on the disk, spin-weighted harmonics on the sphere); there is no `1/r → 1/(r + ε)`.
- **Transpose is not adjoint.** The physical adjoint is $L^\dagger = W^{-1} L^* W$ for the inner
  product $W$ you declare; a plain matrix transpose is never presented as an adjoint.
- **CFL is not error control.** A stability-limited step (`CFL`) and an accuracy-controlled step
  (`EmbeddedError`) are different objects; neither is chosen for you.
- **Name clashes are reported.** `conatus.expert.explain()` lists every name that two modules export
  with different meanings, instead of letting the last import win.

Every silent substitution found in the code is treated as a bug and fixed with a test that failed
before the fix: for example, an eigenvalue problem whose eigenvalue appeared in no equation once
returned an empty spectrum instead of an error, and a Newton solver dropped an imaginary part
without checking it.

## 2. Validation by residual, never by a bare number

An eigenvalue, a steady state or a periodic orbit is not "correct" because it looks right. Each
solver returns what is needed to judge it:

| Object | What is reported |
|---|---|
| Eigenpair | backward residual $\|(L + \sigma M)x\|$, left residual, resolution drift against a finer plan, spectral tail of the eigenvector |
| Sensitivity $\partial\sigma/\partial p$ | Taylor test of the exact derivative against a central difference |
| Newton solution (NLBVP, equilibria, orbits) | residual history (quadratic convergence visible), spectral tail |
| Time integration | observed order of convergence, manufactured solutions, exact decay rates of known modes |
| Floquet exponent | convergence in resolution and in number of harmonics, determinant winding |
| Adjoint | inner-product identity $\langle Jv, w\rangle = \langle v, J^T w\rangle$ and Taylor order 2 |

`verify(...)` gathers these into a report whose lines say *PASS* with the measured value and
threshold, or ask for an inspection by hand. There is no `CORRECT = True`.

For example, the first tutorial returns the least stable Orr–Sommerfeld eigenvalue with its
backward residual, its drift against a finer resolution, and the number of spurious modes that the
drift test removed ({doc}`tutorials/01_orr_sommerfeld`).

## 3. Every capability is tied to a published value

Each method enters the library with a test against a published table or a closed form (Orszag
1971 for Orr–Sommerfeld, Reid & Harris (1958) for Rayleigh–Bénard onset, Cvitanović et al. A test that only compares two backends of Conatus can hide an error common to
both: each compiler and problem class also checks a *non-zero* dynamical quantity against an exact
value. The tutorials follow the same rule: each ends on an `assert` against its reference.

## 4. Everything explains itself

Models, simulations, plans, solvers and results have `explain()`. A simulation also writes the
*Methods* paragraph of a paper (`sim.methods()`), built only from what the objects hold, with the
BibTeX of the cited methods. See {doc}`guide/explain`.

## 5. Provenance

A run records its commit (and whether the tree was dirty), environment, parameters, tolerances and
inner product. See {doc}`guide/runs` and {doc}`guide/reproducibility`.
