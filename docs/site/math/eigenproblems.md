# Eigenproblems, residuals and the physical adjoint

## Convention

A linear eigenproblem is stored as a generalised pencil, one per group,

$$ (L + \sigma M)\, x = 0 . $$

Linearising an initial-value problem about a base state replaces $\partial_t$ by $\sigma$, so
$\sigma$ is the temporal growth rate of the normal mode $x\,e^{\sigma t}$. A model may instead
declare its own eigenvalue (for example the phase speed $c$ of Orr–Sommerfeld), which may appear
anywhere in the equations as long as they are affine in it.

## Solvers

- **Dense QZ** on the full pencil. Because the boundary and tau rows have no $M$ part, the pencil
  has infinite eigenvalues. A pair $(\alpha, \beta)$ is classified infinite when $\beta = 0$, or
  when $|\beta| \le 10^3\,\varepsilon\,|\alpha|$ (rows equilibrated, scaled by
  $\|L\|/\|M\|$) and $M$ is singular. The rank of $M$ counts its singular values above
  $n\,\varepsilon\,\sigma_{\max}(M)$. With $M$ nonsingular every eigenvalue is finite and is
  returned. The infinite ones are counted in `n_infinite`; the log states the rank of $M$ and
  names any pair the rule cannot classify. Left eigenvectors are available.
- **Shift-invert Arnoldi** for a few eigenvalues near a target, valid for singular,
  non-symmetric $M$.

Before solving, the pencil is checked for regularity at a generic $\sigma$; a singular pencil
(a missing condition) is refused.

## Residuals

For each mode the relative backward residual

$$ \eta = \frac{\lVert (L + \sigma M) x\rVert}{(\lVert L\rVert + |\sigma|\,\lVert M\rVert)\,\lVert x\rVert} $$

is reported. A small $\eta$ says the pair is an exact eigenpair of a nearby pencil; whether the
eigenvalue is accurate also depends on its conditioning, which is large for non-normal operators
such as shear flows at high Reynolds number.

## Spurious modes and the drift test

Tau discretisations of fourth-order problems can produce a few spurious eigenvalues of large
modulus, sometimes with positive growth rate, that move by orders of magnitude when $N$ changes.
They are removed by the drift test: the spectrum is computed at two resolutions and only the
eigenvalues found again within a tolerance are kept. The number of rejected modes is reported.
The drift test runs only when you ask for it (`drift=`).

## Critical parameters and sensitivities

The sensitivity of a simple eigenvalue to a parameter $p$ follows from the left eigenvector $y$:

$$ \frac{d\sigma}{dp} = -\,\frac{y^H\,(\partial_p L + \sigma\,\partial_p M)\,x}{y^H M x}. $$

Neutral curves and critical points are found by Newton on the growth rate with this exact
derivative, safeguarded by a bracket. The matrices' dependence on the parameters (including a
wavenumber treated as continuous) is reconstructed exactly once, so a parameter sweep does not
re-discretise. The selected mode is checked against a full QZ at the end: if another admissible
mode is less stable there, the result is refused.

## Inner products and adjoints

The adjoint depends on the inner product. With a weighted product
$\langle a, b\rangle_W = a^H W b$ (for example the kinetic energy, with the exact Gram matrix of the
Chebyshev polynomials), the physical adjoint of a matrix operator is

$$ A^\dagger = W^{-1} A^H W, $$

not the transpose. When $W$ is singular (tau columns carry no energy), the adjoint is refused;
left eigenvectors, which do not need $W$, are used for sensitivities instead.

## References

- S. A. Orszag, *Accurate solution of the Orr–Sommerfeld stability equation*, J. Fluid Mech. 50
  (1971) 689–703.
- D. R. Gardner, S. A. Trogdon and R. W. Douglass, *A modified tau spectral method that eliminates
  spurious eigenvalues*, J. Comput. Phys. 80 (1989).
- J. P. Boyd, *Chebyshev and Fourier Spectral Methods* (2001), ch. 7 (the drift test).
- P. J. Schmid and D. S. Henningson, *Stability and Transition in Shear Flows*, Springer (2001).
- P. Luchini and A. Bottaro, *Adjoint equations in stability analysis*, Annu. Rev. Fluid Mech. 46
  (2014) 493–517.
