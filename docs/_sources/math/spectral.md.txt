# Spectral discretisation and the tau method

## Bases

Each coordinate carries one basis, fixed by the geometry:

- a **periodic** coordinate on $[0, L)$ uses a Fourier series, real (cosines and sines,
  `RealFourier`) or complex (`ComplexFourier`, with `complex=True`);
- a **bounded** coordinate $z \in [a, b]$ uses Chebyshev polynomials $T_n(t)$ of the mapped
  variable $t = 2(z-a)/(b-a) - 1$;
- curvilinear coordinates use the regular bases of {doc}`curvilinear`.

The problem is linear in each Fourier wavenumber with constant coefficients along periodic
directions, so it separates into independent systems, one per **group** (one wavenumber per
periodic direction). Only the bounded direction is coupled within a group. At most one direction
may be bounded in a Cartesian box: a product of two Chebyshev directions does not separate into
banded blocks.

## Sparse operators (Olver–Townsend)

Differentiating a Chebyshev series is dense in the Chebyshev basis but sparse between bases: with
the Gegenbauer (ultraspherical) polynomials $C^{(\lambda)}_n$,

$$
\frac{d T_n}{dt} = n\, C^{(1)}_{n-1}, \qquad
\frac{d C^{(\lambda)}_n}{dt} = 2\lambda\, C^{(\lambda+1)}_{n-1},
$$

so each derivative is a single super-diagonal mapping $C^{(\lambda)} \to C^{(\lambda+1)}$. The
conversions $S_\lambda: C^{(\lambda)} \to C^{(\lambda+1)}$ have two diagonals, and multiplication
by a smooth coefficient $a(t)$ is banded, with half-bandwidth the number of Chebyshev coefficients
kept for $a$. An equation of order $q$ in the bounded direction is written in $C^{(q)}$: every term
is converted up to that space. The resulting matrices are banded plus a few dense rows; no dense
differentiation matrix is ever formed, and the condition number grows only mildly with $N$.

Variable coefficients (a base flow $U(y)$, a metric factor) enter as non-constant coefficients.
All their Chebyshev coefficients are used unless you ask for a truncation; the size of the
expansion and its tail are reported, so an under-resolved coefficient is visible.

## Boundary conditions: the generalised tau method

With $N$ coefficients per unknown and an equation of order $q$, the equation leaves $q$ degrees of
freedom free. The tau method adds $q$ unknown scalars $\tau_1, \dots, \tau_q$ (functions of the
other coordinates) to the equation, multiplied by fixed polynomials, the **lifts**, and imposes the
$q$ boundary conditions as extra rows. In the ultraspherical form the lift of $\tau_k$ is the last
modes of $C^{(q)}$, so the taus only perturb the highest modes of the equation and the system stays
square and banded with a border.

You do not write taus. For a native model the compiler applies one fixed rule: each boundary
condition constrains one unknown $f$, and its tau goes into the equation in which $f$ has its
highest derivative along the bounded coordinate; the taus of one equation occupy the modes
$N-1, N-2, \dots$ in the order the conditions were declared. When the rule cannot decide (the
highest order appears in two equations) the model is refused with the place where to say
`into=`. For incompressible flow the pressure is recognised from the structure of the equations
(it appears only under a gradient and pairs with a divergence constraint without time derivative):
a constant gauge tau is added to the constraint with $\int p = 0$, the standard closure that removes
the pressure's free constant.

## Well-posedness before solving

Before any factorisation each group is checked: number of unknowns (coefficients, taus, constants)
against number of equations (interior rows, boundary rows, gauges), structural rank by bipartite
matching, and, for small blocks, numerical rank by singular values. A missing or redundant
condition, a pure Neumann problem without a gauge, or a Helmholtz problem exactly on an eigenvalue
is refused with the group concerned, instead of producing a meaningless solution.

## Nonlinear terms and dealiasing

Products are evaluated on a grid (pseudospectrally) and transformed back. A quadratic product of
two series truncated at $N$ modes has $2N$ modes; evaluating it on $3N/2$ points removes the aliased
part exactly (the "3/2 rule"); a product of $p$ fields needs $(p+1)N/2$ points. The native layer
reads the highest polynomial degree $p$ of its equations, counting each spatially varying given
field as one factor (its coefficients are truncated at the same $N$ modes), and uses
$\max(3/2, (p+1)/2)$ in every
direction by default (3/2 for Navier–Stokes, 2 for the cubic term of Swift–Hohenberg), and prints
the factor with its reason; `dealias=` changes it. Terms that are not polynomial in the unknowns
(`exp(u)`, `1/T`, a pointwise function) are listed by `explain()`: no factor removes their
aliasing. The expert layer requires the factor explicitly.

## Infinite and semi-infinite intervals

A bounded coordinate can be mapped to the real line or a half-line by a declared map $y = m(\xi)$,
for example $y = L\,\mathrm{artanh}\,\xi$ or the rational map $y = y_0 + L(1+\xi)/(1-\xi)$. The
derivative $d/dy = \mu(\xi)\,d/d\xi$ with $\mu = 1/m'$ is banded when $\mu$ is a polynomial, and
otherwise $\mu$ is expanded to round-off like any variable coefficient.

## References

- S. Olver and A. Townsend, *A fast and well-conditioned spectral method*, SIAM Review 55 (2013).
- K. J. Burns, G. M. Vasil, J. S. Oishi, D. Lecoanet and B. P. Brown, *Dedalus: a flexible
  framework for numerical simulations with spectral methods*, Phys. Rev. Research 2 (2020) 023068.
- J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd ed., Dover (2001).
- C. Lanczos, *Trigonometric interpolation of empirical and analytical functions*, J. Math. Phys.
  17 (1938) (the tau method).
