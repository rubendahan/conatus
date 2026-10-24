# Principles

Conatus is written for people who publish the numbers it computes. This page lists the library's
design goals. Each section says which checks the code implements for them.

## 1. Numerical choices are declared

A numerical method is a list of choices: basis and resolution, boundary treatment (tau and lift
positions), dealiasing factor, time stepper and step, tolerances, inner product, definition of the
adjoint. The compiler records the choices it makes automatically. You can override the ones the
API exposes. Where Conatus has a better option to offer, it suggests it in a message and leaves
the problem as you wrote it.

In practice:

- **Dealiasing is declared.** Unless you give `dealias=`, a native model uses the factor
  $(p+1)/2$ for the highest polynomial degree $p$ of its equations in the unknowns and the given
  fields (each given coefficient counts as one factor), with a floor of 3/2. This gives 3/2 for
  quadratic terms and 2 for cubic ones. `explain()` and the *Methods* paragraph state the factor.
  A value you give takes precedence, and an expert basis requires one. `explain()` warns when a
  nonlinear term (`exp`, `tanh`…) is not alias-free at the declared factor.
- **Automatic choices follow fixed rules.** Each boundary condition, the pressure gauge and the
  implicit or explicit treatment of each term are placed by deterministic rules. `explain()` lists
  them. When a rule cannot decide, the model is refused with a message that says what to declare.
  A structural rank report refuses an under- or over-determined system before any factorisation.
- **Cut-offs are counted.** Truncations of non-constant coefficients and removed matrix entries
  are counted and reported. The drift filter on eigenvalues is requested by the user and logs the
  number of modes it removes. The dense eigensolver also classifies eigenvalues as finite or
  infinite with its own threshold ({doc}`math/eigenproblems`).
- **Singular coordinates use regular bases.** Curvilinear geometries use bases that are regular at
  the axis and the poles (Zernike-type polynomials on the disk, spin-weighted harmonics on the
  sphere). There is no `1/r → 1/(r + ε)`.
- **The adjoint uses your inner product.** For the inner product $\langle a, b\rangle_W = a^H W b$
  you declare, with $W$ Hermitian positive definite, the physical adjoint is
  $L^\dagger = W^{-1} L^H W$, where $L^H$ is the conjugate transpose. For real states and
  $W$ equal to the identity it reduces to the plain transpose.
- **Stability and accuracy are separate controls.** The CFL controller limits the step for
  stability. `EmbeddedError` estimates the temporal error. You choose which one to use.
- **Name clashes are listed.** `conatus.expert.explain()` lists every name that two modules export
  with different meanings, and how each clash is handled.

A silent substitution found in the code is treated as a bug and fixed with a test that failed
before the fix. One example: an eigenvalue problem whose eigenvalue appeared in no equation once
returned an empty spectrum instead of an error. Another: a Newton solver dropped an imaginary part
without checking it.

## 2. Results come with their checks

Each solver returns the quantities needed to judge its result. Validate with residuals,
refinement studies and independent reference values together:

| Object | What is reported |
|---|---|
| Eigenpair | backward residual $\|(L + \sigma M)x\|$; on request, the left residual (`left=`) and the drift against a finer plan (`drift=`); spectral tail of the eigenvector |
| Sensitivity $\partial\sigma/\partial p$ | Taylor test of the derivative against a central difference |
| Newton solution (NLBVP, equilibria, orbits) | residual history (quadratic convergence visible), spectral tail |
| Time integration | observed order of convergence, manufactured solutions, decay rates of known modes |
| Floquet exponent | convergence in resolution and in number of harmonics, determinant winding |
| Adjoint | inner-product identity $\langle Jv, w\rangle = \langle v, J^T w\rangle$ and Taylor order 2 |

`verify(...)` gathers these checks into a report. Each line gives the measurement, the threshold
and the scope of the check, or asks for an inspection by hand.

For example, the first tutorial returns the least stable Orr–Sommerfeld eigenvalue with its
backward residual, its drift against a finer resolution, and the number of modes that the drift
test set aside as not converged ({doc}`tutorials/01_orr_sommerfeld`).

## 3. Methods are tested against published values

Each method enters the library with a test against a published table or a closed form: Orszag
(1971) for Orr–Sommerfeld, Reid & Harris (1958) for Rayleigh–Bénard onset, and Cvitanović,
Davidchack & Siminos (2010) for the Kuramoto–Sivashinsky orbits, among others.

A test that only compares two backends of Conatus can hide an error common to both. Each compiler
and problem class therefore also checks a non-zero dynamical quantity against an exact value. Each
tutorial ends on an `assert` against its reference.

## 4. Inspecting objects

Models, simulations, plans, solvers and results have `explain()`. The API reference lists the
fields each one reports. A simulation can also draft the *Methods* paragraph of a paper
(`sim.methods()`) from the settings it holds, with the BibTeX of the cited methods. See
{doc}`guide/explain`.

## 5. Provenance

When you ask for it, a run records its commit (and whether the tree was dirty), environment,
parameters, tolerances and inner product. See {doc}`guide/runs` and {doc}`guide/reproducibility`.
