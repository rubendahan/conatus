# Floquet–Hill analysis of time-periodic flows

## Floquet exponents

The stability of a base flow that is periodic in time, with period $T = 2\pi/\omega$, is governed
by a linear system $x' = A(t)\,x$ with $A(t + T) = A(t)$ continuous. Floquet's theorem states that
a fundamental matrix can be written $X(t) = P(t)\,e^{tB}$, with $P$ invertible and periodic and $B$
a constant matrix ($P$ and $B$ may be complex). The eigenvalues $\mu$ of $B$ are the Floquet
exponents, and $e^{\mu T}$ are the eigenvalues of the monodromy matrix $X(T)\,X(0)^{-1}$. An
eigenvector $v$ of $B$ gives a **Floquet mode**
$$ x(t) = e^{\mu t}\,\hat x(t), \qquad \hat x(t) = P(t)\,v \ \text{periodic} . $$
A general solution is a combination of Floquet modes. When $B$ is not diagonalisable, polynomial
factors in $t$ also appear. The exponents are defined modulo $i\omega$: $e^{\mu t}\hat x(t) =
e^{(\mu + ik\omega)t}\,\big(e^{-ik\omega t}\hat x(t)\big)$ only relabels the harmonics. A real
operator has $\mu$ and $\bar\mu$ as exponents. Conatus always reports the folded value and says how
it folded it. Hill's method below looks for Floquet modes.

## Hill's method

Instead of integrating over one period (a monodromy matrix), the periodic part is expanded in
Fourier harmonics in time,

$$ x(y, t) = e^{\mu t} \sum_{n=-N_h}^{N_h} x_n(y)\, e^{i n \omega t}, $$

and the base flow $U(y, t) = \sum_m U_m(y)\,e^{i m\omega t}$ couples harmonic $n$ to $n \pm m$. Each
harmonic is discretised in space with the same sparse spectral operators as a steady problem. The
result is a block-banded matrix $M(\mu)$, block tridiagonal when the base flow has a single
harmonic, and the exponents are the values where it is singular: either the roots of
$\det M(\mu)$, found by a secant method from a declared seed, or the eigenvalues of the
corresponding generalised problem near a shift. Block elimination along the harmonics keeps the
cost linear in the number of harmonics.

A small residual alone does not validate a root, because the residual of a nearly singular
matrix is small everywhere. The checks below are recorded evidence about a computed root; they are
not a proof that it is an exponent of the continuous problem. `root.certify(residual_tol=..., convergence_tol=...)` marks a root
certified only if the secant converged, the residual is below the declared tolerance, and the root moves by less than
`convergence_tol` between two resolutions. Two further checks are run on request and recorded
when passed to `certify`: the contrast $\sigma_{\min}/\sigma_2$ of $M(\mu)$ (`contrast(mu)`, small
problems) and the argument principle (`winding(center, radius, n_points)`, the winding of
$\det M$ around the root); `root.explain()` lists which checks were run.

## References

- G. Floquet, *Sur les équations différentielles linéaires à coefficients périodiques*, Ann. Sci.
  ENS 12 (1883) 47–88.
- G. W. Hill, *On the part of the motion of the lunar perigee which is a function of the mean
  motions of the sun and moon*, Acta Math. 8 (1886) 1–36.
- S. H. Davis, *The stability of time-periodic flows*, Annu. Rev. Fluid Mech. 8 (1976) 57–74.
