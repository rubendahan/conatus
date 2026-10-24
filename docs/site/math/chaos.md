# Lyapunov exponents, covariant vectors and ensembles

## What is computed

The Lyapunov spectrum is that of the discrete map $\Phi_{dt}$ of the time stepper (an implicit–
explicit Runge–Kutta scheme at fixed step), not of a separately discretised variational equation.
The tangent vectors are advanced by the derivative of $\Phi_{dt}$ itself. Differentiating the stage
equation of {doc}`timestepping` gives, for stage $i$,

$$ (M + dt\,H_{ii} L)\,\delta X_i = M\,\delta X_0 + dt \sum_{k<i} A_{ik}\, F'(X_k)\,\delta X_k
 - dt \sum_{k<i} H_{ik}\, L\,\delta X_k , $$

with the stage states $X_k$ of the trajectory and $F'$ the symbolic Fréchet derivative of the
nonlinear terms, evaluated on the same dealiased grid. The diagonal implicit term $H_{ii} L$ appears
only on the left; both sums run over the earlier stages $k < i$. This is the derivative of the
discrete step, up to rounding. It is not the derivative of the exact flow of the continuous
equations. The exponents of $\Phi_{dt}$ are expected to approach those of the flow as $dt \to 0$.
The difference is a time-discretisation error; measure it by repeating the computation at a
smaller $dt$. Multistep schemes are refused: their tangent lives in an augmented state and carries
the parasitic roots of the scheme.

## Algorithm

- **Benettin / QR**: $k$ orthonormal tangent vectors are advanced with the trajectory and
  re-orthonormalised by QR every few steps (positive diagonal of $R$);
  $\lambda_i = \frac1T \sum \log R_{ii}$, with $T$ the physical time covered by the sum. Transients of the trajectory and of the tangent vectors are
  discarded before averaging, both with declared lengths.
- **Statistical error**: the average is split into blocks, and the standard error of the block
  means is reported. This error bar assumes that the block means are nearly independent. The
  lag-one correlation of consecutive blocks is reported to check that assumption.
- **Covariant Lyapunov vectors** (Ginelli et al. 2007): the $R$ factors are stored, a backward
  iteration with upper-triangular matrices gives the coefficients of each covariant vector in the
  Gram–Schmidt basis; principal angles between the unstable and stable subspaces measure
  hyperbolicity.
- **Kaplan–Yorke dimension** $D_{KY} = j + \sum_{i\le j}\lambda_i/|\lambda_{j+1}|$, with an error
  propagated from the exponents.
- An invariant subspace can be imposed (for example zero mean, or the Galilean direction removed)
  by a declared projection before each QR.

## Ensembles

An ensemble is $B$ states of one discretised problem advanced together: per stage, one evaluation
of the nonlinear terms for all members, one sparse product by $M$ and $L$, one stacked banded solve.
Parameters that enter the nonlinear terms become per-member coefficients; a parameter inside $L$
gives per-member factorisations. The same machinery advances $B$ trajectories times $k$ tangent
vectors at once, on the CPU or the GPU.

The fate of each member (relaminarised, transitioned, in which basin) is decided by declared
criteria, and transition probabilities come with a Wilson confidence interval.

## References

- G. Benettin, L. Galgani, A. Giorgilli and J.-M. Strelcyn, *Lyapunov characteristic exponents for
  smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 2:
  numerical application*, Meccanica 15 (1980) 21–30.
- I. Shimada and T. Nagashima, *A numerical approach to ergodic problem of dissipative dynamical
  systems*, Prog. Theor. Phys. 61 (1979) 1605–1616.
- F. Ginelli, P. Poggi, A. Turchi, H. Chaté, R. Livi and A. Politi, *Characterizing dynamics with
  covariant Lyapunov vectors*, Phys. Rev. Lett. 99 (2007) 130601.
- J. L. Kaplan and J. A. Yorke, *Chaotic behavior of multidimensional difference equations*,
  Lecture Notes in Math. 730, Springer (1979).
- E. B. Wilson, *Probable inference, the law of succession, and statistical inference*, J. Am.
  Stat. Assoc. 22 (1927) 209–212.
