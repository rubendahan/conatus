# Symmetry subspaces, slices and phase-space geometry

## Computing in an invariant subspace

If the equations are equivariant under a reflection $S$ (for example $x \to -x$ with $u$ odd and $T$
even), the solutions with $S u = u$ form an invariant subspace $V_S$. Conatus can compute entirely
in $V_S$: the user declares the symmetry and the parity of each scalar field (the parities of vector
components follow: the component normal to the mirror changes sign), and the library

1. checks that the equations and the boundary conditions are equivariant, and refuses otherwise,
   naming the offending term;
2. keeps only the independent coefficients: along a reflected Chebyshev direction the polynomials
   of the right parity, along a Fourier direction cosines or sines, and only the taus and boundary
   conditions that are not mirror images of each other;
3. runs the transforms, the nonlinear terms on the folded grid, and the banded solves in $V_S$,
   which halves each group;
4. provides the embedding $E: V_S \to V$ and the projection $P: V \to V_S$ to move a symmetric
   solution into the full space and back. $PE$ is the identity on $V_S$; $EP$ is the projector of
   $V$ onto $V_S$, not the identity of $V$.

In curvilinear geometries each symmetry (azimuthal rotation by $2\pi/n$, equatorial reflection,
axial reflection) acts on coefficients as a signed permutation; the subspace keeps one
representative per orbit. Regularity at the axis is not a symmetry choice: the basis imposes it.

## What a subspace hides

A computation in $V_S$ cannot see instabilities that break the symmetry, and `explain()` says so.
For a reflection $R$ with $R^2 = I$, a Jacobian that commutes with $R$ splits into a symmetric
block ($Rv = v$) and an antisymmetric block ($Rv = -v$). `symmetry_breaking_check` covers two
cases. For an equilibrium of a one-dimensional periodic problem that is symmetric under a
reflection (`conatus.ecs`), it computes both blocks densely, measures the coupling between them,
and counts the eigenvalues with positive real part in each. For a curvilinear eigenvalue problem
with a declared subspace (`conatus.discretize.curvilinear_symmetry`), it computes the leading
eigenvalue in each of the other isotypic components, reusing the same discretisation with the
opposite parities. Floquet multipliers of orbits are not covered: for a periodic orbit, compute
them in the full space. A symmetric solution is declared stable only after one of these
full-space checks.

## Slices

A continuous symmetry (translation along a periodic direction) is reduced by a slice. In the
first-Fourier-mode slice, each state is shifted so that the phase of its first Fourier mode is
zero; the shift is reconstructed by integrating the phase velocity. Relative periodic orbits become
periodic orbits of the reduced flow. States where the first mode vanishes (the slice border) are
detected and reported.

## Trajectories, sections and manifolds

- **Poincaré sections**: crossings of a hypersurface in a declared direction are located on the
  dense output of the time stepper (or exactly, by a partial step of the discrete map); successive
  crossings give the first-return map.
- **Unstable manifolds of equilibria**: by the parametrisation method, a polynomial expansion in
  the unstable directions, then globalised by time integration.
- **Connections**: heteroclinic orbits between invariant solutions by shooting with projection
  boundary conditions onto the stable and unstable subspaces.

## References

- M. Golubitsky and I. Stewart, *The Symmetry Perspective*, Birkhäuser (2002).
- N. B. Budanur, P. Cvitanović, R. L. Davidchack and E. Siminos, *Reduction of SO(2) symmetry for
  spatially extended dynamical systems*, Phys. Rev. Lett. 114 (2015) 084102.
- X. Cabré, E. Fontich and R. de la Llave, *The parameterization method for invariant manifolds I:
  manifolds associated to non-resonant subspaces*, Indiana Univ. Math. J. 52 (2003) 283–328.
- W.-J. Beyn, *The numerical computation of connecting orbits in dynamical systems*, IMA J. Numer.
  Anal. 10 (1990) 379–405.
- P. Cvitanović, R. Artuso, R. Mainieri, G. Tanner and G. Vattay, *Chaos: Classical and Quantum*,
  ChaosBook.org.
