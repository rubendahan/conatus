# Floquet–Hill analysis of time-periodic flows

## Floquet exponents

The stability of a base flow that is periodic in time, with period $2\pi/\omega$, is governed by a
linear system with periodic coefficients. By Floquet's theorem its solutions are
$x(t) = e^{\mu t}\,\hat x(t)$ with $\hat x$ periodic; the Floquet exponents $\mu$ are defined modulo
$i\omega$ (adding $i k\omega$ only relabels the harmonics), and a real operator has $\mu$ and
$\bar\mu$ as exponents. Conatus always reports the folded value and says how it folded it.

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

A root is never accepted on a residual alone (the residual of a nearly singular matrix is small
everywhere); it is certified by the contrast between the two smallest singular values and by the
argument principle (the winding of $\det M$ around the root).

## References

- G. Floquet, *Sur les équations différentielles linéaires à coefficients périodiques*, Ann. Sci.
  ENS 12 (1883).
- G. W. Hill, *On the part of the motion of the lunar perigee which is a function of the mean
  motions of the sun and moon*, Acta Math. 8 (1886).
- C. Davis, *The stability of time-periodic flows*, Annu. Rev. Fluid Mech. 8 (1976) 57–74.
