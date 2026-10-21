# Invariant solutions: Newton–Krylov–hookstep

Equilibria, travelling waves, periodic orbits and relative periodic orbits are fixed points of a
map built from the time stepper. Conatus finds them by a matrix-free Newton method with a trust
region. The solver reports the residual of the discretized equations; check spatial and temporal
convergence separately.

## Formulations

Write the discrete flow map over a time $T$ as $\Phi_T$ (the map of the time stepper itself, with
$dt = T/n$), and the velocity field of the semi-discrete system as
$v(u) = M^{-1}(F(u) - L u)$.

- **Equilibrium**: $v(u) = 0$.
- **Travelling wave** $u(x - ct)$: $v(u) + c\,\partial_x u = 0$, unknowns $(u, c)$, with the phase
  condition $\langle \partial_x u^*, \delta u\rangle = 0$ that removes the free translation.
- **Periodic orbit**: $\Phi_T(u) = u$, unknowns $(u, T)$, phase condition
  $\langle v(u^*), \delta u\rangle = 0$ against time shifts.
- **Relative or pre-periodic orbit**: $S\,\tau_\ell\,\Phi_T(u) = u$ with a shift $\tau_\ell$ and
  optionally a reflection $S$, unknowns $(u, T, \ell)$, with both phase conditions.

Because $\Phi_T$ is the discrete map, the reported residual is that of the discrete time-stepper
problem. It does not estimate the time-discretisation error.

## Newton–Krylov

The Newton correction solves $J\,\delta z = -R(z)$ by GMRES. The Jacobian is not formed. Its
products with a vector use the tangent of the discrete time stepper ({doc}`chaos` and the adjoint
pages describe the same tangent), and the derivative with respect to $T$ is the velocity at the end
point.

## Hookstep

GMRES gives an Arnoldi basis $V_{m+1}$ and a Hessenberg matrix $\bar H$. The Newton step minimises
$\lVert \beta e_1 - \bar H y\rVert$. If it is longer than the trust radius $\Delta$, the hookstep
(Viswanath 2007) replaces it by

$$ y(\mu) = W\,\mathrm{diag}\!\left(\frac{s_i}{s_i^2 + \mu}\right) U^T \beta e_1,
\qquad \lVert y(\mu)\rVert = \Delta, $$

from the singular value decomposition $\bar H = U\,\mathrm{diag}(s)\,W^T$, with $\mu$ found by
bisection. The radius follows the ratio of actual to predicted reduction: halved and retried in the
same Krylov space when the step is poor, doubled when it is good at the boundary. Every tolerance,
budget and radius is an argument; a search that does not converge raises an error with its history.

## Stability of a solution

Once found, an equilibrium's stability follows from the eigenvalues of its Jacobian, and a periodic
orbit's from its Floquet multipliers (eigenvalues of $D\Phi_T$, by Arnoldi on the tangent map). A
solution computed in a symmetry subspace is declared stable only after the symmetry-breaking
directions have been checked ({doc}`symmetry`).

## Edge states and minimal seeds

Between laminar and turbulent behaviour lies the edge of chaos. Bisection on the amplitude of an
initial condition, repeated along the trajectory, tracks the edge state; the converged edge
trajectory is then a seed for Newton. The same bisection on the energy of an optimised initial
condition brackets the transition threshold of a minimal seed.

## References

- D. Viswanath, *Recurrent motions within plane Couette turbulence*, J. Fluid Mech. 580 (2007)
  339–358.
- P. Cvitanović, R. L. Davidchack and E. Siminos, *On the state space geometry of the
  Kuramoto–Sivashinsky flow in a periodic domain*, SIAM J. Appl. Dyn. Syst. 9 (2010) 1–33.
- G. Kawahara, M. Uhlmann and L. van Veen, *The significance of simple invariant solutions in
  turbulent flows*, Annu. Rev. Fluid Mech. 44 (2012) 203–225.
- J. F. Gibson, J. Halcrow and P. Cvitanović, *Visualizing the geometry of state space in plane
  Couette flow*, J. Fluid Mech. 611 (2008) 107–130.
- J. D. Skufca, J. A. Yorke and B. Eckhardt, *Edge of chaos in a parallel shear flow*, Phys. Rev.
  Lett. 96 (2006) 174101.
- C. C. T. Pringle and R. R. Kerswell, *Using nonlinear transient growth to construct the minimal
  seed for shear flow turbulence*, Phys. Rev. Lett. 105 (2010) 154502.
