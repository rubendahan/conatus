# Curvilinear geometries: regular bases

Disks, annuli, pipes, spheres, balls and spherical shells are discretised with bases that are
regular at the axis and at the poles. No condition is imposed at $r = 0$, and no term is
regularised as $1/(r + \varepsilon)$.

## Disk, annulus and pipe

In polar coordinates $(\phi, r)$ each azimuthal mode $e^{im\phi}$ separates. A smooth scalar on the
disk of radius $R$ has the form

$$ f_m(r) = \rho^{|m|}\, w(\rho^2), \qquad \rho = r/R, $$

with $w$ smooth. The discretisation approximates $w$ by a finite polynomial in $\rho^2$. A vector is written in spin components $u_\pm = u_r \pm i u_\phi$, which are
smooth when $u_\pm = \rho^{|m\pm1|} \times$ (a smooth function of $\rho^2$). Conatus expands these
factors in normalised Jacobi polynomials of $z = 2\rho^2 - 1$ (a Zernike basis), which
stays well conditioned at large $|m|$ where a monomial-weighted Chebyshev basis does not.
Gradient, divergence, curl and Laplacian map these spaces into each other with banded matrices.
The disk has one boundary, $r = R$, so a second-order equation takes one condition there.

On the annulus $r_i \le r \le r_o$ there is no axis: the radial direction is Chebyshev, the
operators are written multiplied by powers of $r$ (polynomial coefficients, hence banded), and a
second-order equation takes one condition on each wall. The pipe is the disk times a periodic axial
coordinate.

## Sphere, ball and shell

On the sphere, vectors and tensors are expanded in spin-weighted spherical harmonics: the tangent
components are taken in the spin basis $e_\pm = (e_\theta \mp i e_\phi)/\sqrt2$, and a component of
spin $s$ is a series of ${}_sY_\ell^m$, regular at both poles. Surface gradient and divergence raise
or lower the spin and act diagonally in $\ell$.

In the ball the radial dependence of each $(\ell, m)$ is again a power of $r$ times a Jacobi
polynomial in $2r^2 - 1$, the three-dimensional analogue of the disk, so fields are regular at the
centre. In the shell $r_i \le r \le r_o$ the radial direction is Chebyshev; second-order operators
are written in first-order form (the Laplacian as the divergence of a gradient plus a lifted tau
term), which keeps the $\ell = 0$ mode well posed.

## Conventions

The resolution of a sphere direction is the number of degrees $L_{\max} + 1$. Coordinates are
ordered $(\phi, r)$ on the disk and annulus, $(\phi, r, z)$ in the pipe, $(\phi, \theta)$ on the
sphere and $(\phi, \theta, r)$ in the ball and shell; the native layer prints the bases it chose.

## References

- G. M. Vasil, K. J. Burns, D. Lecoanet, S. Olver, B. P. Brown and J. S. Oishi, *Tensor calculus in
  polar coordinates using Jacobi polynomials*, J. Comput. Phys. 325 (2016) 53–73.
- G. M. Vasil, D. Lecoanet, K. J. Burns, J. S. Oishi and B. P. Brown, *Tensor calculus in spherical
  coordinates using Jacobi polynomials. Part-I: mathematical analysis and derivations*,
  J. Comput. Phys. X 3 (2019) 100013; D. Lecoanet et al., *Part-II: implementation and examples*,
  ibid. 100012.
- J. P. Boyd and F. Yu, *Comparing seven spectral methods for interpolation and for solving the
  Poisson equation in a disk*, J. Comput. Phys. 230 (2011) 1408–1438.
