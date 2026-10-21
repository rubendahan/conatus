# Continuation and bifurcations

## Pseudo-arclength continuation

A branch of solutions of $G(x, p) = 0$ is followed in the combined unknown $z = (x, p)$ with no
privileged parameter (Keller 1977). From a converged point $z_k$ with unit tangent $t_k$:

- **predictor** $z_{\text{pred}} = z_k + \Delta s\, t_k$;
- **corrector**: Newton on the bordered system
  $$
  \begin{pmatrix} G_x & G_p \\ t_x^T W & t_p \end{pmatrix}
  \begin{pmatrix} \delta x \\ \delta p \end{pmatrix}
  = - \begin{pmatrix} G \\ t_k \cdot (z - z_{\text{pred}}) \end{pmatrix};
  $$
- **next tangent**: the same bordered matrix with right-hand side $(0, 1)$, normalised and
  oriented like the previous one.

At a fold $G_x$ is singular but the bordered matrix is not, so the branch passes through the fold
and $t_p$ changes sign exactly there. The weights $W$ of the norm, the step bounds, the adaptation
factor, the tolerances and the budgets are arguments. You supply the initial step and its bounds;
the algorithm adapts the step within them using the documented controls.

## Events

Test functions are evaluated along the branch; a sign change between two points is located by a
secant in arclength, each iterate being a corrected point:

- **fold** (limit point): $t_p$ changes sign;
- **branch point**: the determinant of the bordered Jacobian changes sign;
- **Hopf point**: a complex pair of eigenvalues of the Jacobian crosses the imaginary axis
  (with the mass matrix: the generalised problem $J v = \sigma M v$).

## Branch switching

At a branch point the new tangent is the null vector of the Jacobian that is not along the current
branch; with a declared symmetry the pitchfork is classified and one half-branch is chosen. At a
Hopf point a family of periodic orbits starts with a declared first amplitude. At a period-doubling
point of an orbit the doubled orbit is started from the Floquet vector.

## Periodic orbits by collocation

Periodic orbits of low-dimensional systems (and of Galerkin truncations) are computed by orthogonal
collocation. The orbit is a piecewise polynomial on a mesh in time, and the differential equation
holds at the Gauss points of each mesh interval. The period is an unknown, and an integral phase
condition fixes the time origin (Doedel, Keller and Kernévez 1991). The Floquet multipliers of the
orbit detect limit points of cycles, branch points, period doubling and Neimark–Sacker
bifurcations.

## Codimension two

Curves of folds and of Hopf points are continued in two parameters with minimally augmented
systems (a bordered matrix whose last component vanishes on the curve). Along them, the cusp,
Bogdanov–Takens, Bautin, fold–Hopf and Hopf–Hopf points are detected.

## Normal forms and amplitude equations

The derivatives of the right-hand side with respect to the state are exact: the second and third
Fréchet derivatives $B(u, v)$ and $C(u, v, w)$ are derived symbolically from the equations. They give
the cubic coefficient of a pitchfork and the first Lyapunov coefficient of a Hopf point
(super- or subcritical), and the coefficients of the Landau and Ginzburg–Landau amplitude equations
by a weakly nonlinear expansion.

## References

- H. B. Keller, *Numerical solution of bifurcation and nonlinear eigenvalue problems*, in
  Applications of Bifurcation Theory, Academic Press (1977).
- E. L. Allgower and K. Georg, *Numerical Continuation Methods*, Springer (1990).
- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed., Springer (2004).
- W. Govaerts, *Numerical Methods for Bifurcations of Dynamical Equilibria*, SIAM (2000).
- E. Doedel, H. B. Keller and J.-P. Kernévez, *Numerical analysis and control of bifurcation
  problems*, Int. J. Bifurcation Chaos 1 (1991) (AUTO).
- A. Dhooge, W. Govaerts and Yu. A. Kuznetsov, *MATCONT*, ACM Trans. Math. Softw. 29 (2003).
- M. C. Cross and P. C. Hohenberg, *Pattern formation outside of equilibrium*, Rev. Mod. Phys. 65
  (1993) 851.
