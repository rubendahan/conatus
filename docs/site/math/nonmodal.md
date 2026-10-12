# Non-modal analysis: transient growth

Shear flows are governed by non-normal operators: perturbations can grow by orders of magnitude
before decaying, although every eigenmode decays. This growth is measured in a declared norm.

## The norm

All quantities are measured in the kinetic energy of the perturbation,

$$ E = \tfrac12 \sum_i w_i \int |s_i|^2\, dy , $$

with declared weights $w_i$ and the exact Gram matrix of the Chebyshev polynomials. With $F$ a
Cholesky factor, $\lVert F x\rVert_2^2 = E(x)$, and every operator norm below is a 2-norm after this
change of variables.

## From the tau pencil to a generator

The discretised linearised Navier–Stokes equations form a pencil $M x' + L x = 0$ with infinite
eigenvalues (continuity, boundary rows, taus). An ordered generalised Schur decomposition separates
the finite part,

$$ (-L) Z = Q S, \quad M Z = Q T, \qquad x = Z_1 q, \quad T_{11} q' = S_{11} q , $$

and a thin QR of $F C Z_1$ (with $C$ the map to the physical state) gives the generator
$\tilde A$ in coordinates orthonormal for the energy. Then

$$ \lVert e^{tA}\rVert_E = \lVert e^{t\tilde A}\rVert_2 , $$

without any truncated eigenvector expansion.

## Quantities

- **Transient growth** $G(t) = \max_{x_0} E(t)/E(0) = \lVert e^{t\tilde A}\rVert_2^2$ and its
  optimal initial condition (leading right singular vector).
- **Numerical abscissa**: the initial slope $\max \lambda(\tilde A + \tilde A^H)/2$.

## Time-periodic base flows

For a base flow periodic in time the propagator $\Phi(t_0, T)$ is built by the
same implicit–explicit schemes as the time stepper, applied to all unit initial states at once: the
base-flow terms are explicit and evaluated at the exact stage times, the time step is $T/n$ exactly,
and the gain is the largest singular value of $F\Phi F^{-1}$ restricted to admissible initial
states (those satisfying continuity and the boundary conditions). Whether inadmissible inputs are
allowed is an explicit choice.

## References

- L. N. Trefethen, A. E. Trefethen, S. C. Reddy and T. A. Driscoll, *Hydrodynamic stability without
  eigenvalues*, Science 261 (1993) 578–584.
- K. M. Butler and B. F. Farrell, *Three-dimensional optimal perturbations in viscous shear flow*,
  Phys. Fluids A 4 (1992) 1637–1650.
- S. C. Reddy and D. S. Henningson, *Energy growth in viscous channel flows*, J. Fluid Mech. 252
  (1993) 209–238.
- P. J. Schmid, *Nonmodal stability theory*, Annu. Rev. Fluid Mech. 39 (2007) 129–162.
