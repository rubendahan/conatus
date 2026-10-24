# Continuation and bifurcations

## Pseudo-arclength continuation

A branch of solutions of $G(x, p) = 0$ is followed in the combined unknown $z = (x, p)$ with no
privileged parameter (Keller 1977). Lengths are measured with declared positive weights
$W = \mathrm{diag}(w)$ on $x$ and weight one on $p$:
$$ \langle (a_x, a_p), (b_x, b_p)\rangle_W = a_x^T W b_x + a_p b_p, \qquad
\lVert z\rVert_W^2 = \langle z, z\rangle_W . $$
From a converged point $z_k$ with tangent $t_k = (t_x, t_p)$ of unit $W$-norm:

- **predictor** $z_{\text{pred}} = z_k + \Delta s\, t_k$;
- **corrector**: Newton on $G(x, p) = 0$ together with the phase condition
  $$ h(z) = \langle t_k, z - z_{\text{pred}}\rangle_W
         = t_x^T W (x - x_{\text{pred}}) + t_p\,(p - p_{\text{pred}}) = 0 , $$
  which keeps the corrected point on the hyperplane through $z_{\text{pred}}$ that is
  $W$-orthogonal to $t_k$. The derivative of $h$ is the row $(t_x^T W, t_p)$, so each Newton step
  solves the bordered system
  $$
  \begin{pmatrix} G_x & G_p \\ t_x^T W & t_p \end{pmatrix}
  \begin{pmatrix} \delta x \\ \delta p \end{pmatrix}
  = - \begin{pmatrix} G \\ h(z) \end{pmatrix};
  $$
- **next tangent**: the same bordered matrix with right-hand side $(0, 1)$, normalised in the
  $W$-norm and oriented so that $\langle t_{k+1}, t_k\rangle_W \ge 0$.

At a simple fold, $G_x$ has a one-dimensional null space and $G_p$ is not in its range. The
bordered matrix is then nonsingular when $t_k$ is close enough to the branch tangent, so the
corrector converges through the fold, and $t_p$ changes sign there. The weights $W$ of the norm, the step bounds, the adaptation
factor, the tolerances and the budgets are arguments. You supply the initial step and its bounds;
the algorithm adapts the step within them using the documented controls.

## Events

Test functions are evaluated along the branch; a sign change between two points is located by a
secant in arclength, each iterate being a corrected point. Detection relies on a sign change, so
it assumes a regular crossing: a test function that touches zero without changing sign, or two
crossings within one step, are not detected.

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
  problems (II): bifurcation in infinite dimensions*, Int. J. Bifurcation Chaos 1 (1991) 745–772 (AUTO).
- A. Dhooge, W. Govaerts and Yu. A. Kuznetsov, *MATCONT: a MATLAB package for numerical bifurcation analysis of ODEs*, ACM Trans.
  Math. Softw. 29 (2003) 141–164.
- M. C. Cross and P. C. Hohenberg, *Pattern formation outside of equilibrium*, Rev. Mod. Phys. 65
  (1993) 851.
