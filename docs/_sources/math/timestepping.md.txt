# Implicit–explicit time stepping

## The split

After spatial discretisation an initial-value problem becomes, in each group,

$$ M \frac{dX}{dt} + L X = F(X, t), $$

where $X$ holds the spectral coefficients and the taus, $M$ and $L$ are the sparse matrices of the
linear terms (the boundary and gauge rows have no $M$ part), and $F$ collects the nonlinear and
forcing terms, evaluated pseudospectrally. Stiff linear terms (viscosity, diffusion, pressure) are
treated implicitly and $F$ explicitly.

In the native layer the split follows one rule: a term that is linear in the unknowns, with
constant, parameter or given-field coefficients, goes to $L$; a nonlinear term or a source goes to
$F$; each term moves across the equals sign with its sign. `explicit(...)` and `implicit(...)` in
the equation text override the rule for one term (an `implicit` nonlinear term is refused).
`explain()` lists where every term went.

## Schemes

Two families are available:

- **IMEX Runge–Kutta** `RK111`, `RK222`, `RK443` (Ascher, Ruuth and Spiteri 1997). A stage reads
  $$ (M + k H_{ii} L)\, X_i = M X_0 + k \sum_{j<i} A_{ij} F(X_j, t_j) - k \sum_{j<i} H_{ij} L X_j , $$
  with $(A, H)$ the explicit and implicit tableaux. In these tableaux all $H_{ii}$ are equal, so one
  factorisation of $M + k H_{ii} L$ per group serves every stage of the run.
- **Semi-implicit BDF** `SBDF1` to `SBDF4` (Wang and Ruuth 2008): multistep, one solve per step;
  the first steps are taken with a declared start-up procedure, printed by `explain()`.

The time step is fixed and chosen by you. The CFL estimate (`sim.cfl("u", safety=0.5)`, with your
safety factor) limits the step for stability. It does not estimate the time error. The stop time
must be a whole number of fixed time steps from the current time.

## Factorisations

Each group's matrix $M + k H L$ is banded with a border (the tau columns and the boundary rows).
It is factorised once by a banded LU with the border eliminated through a small Schur complement
(capacitance) system; all groups are solved together in one batched call, on the CPU or on the GPU.

## Watching the taus

The tau variables absorb the residual of the truncated equations. Their size along a run is a
cheap diagnostic of resolution: a growing tau means the top modes are not resolved. Conatus can
monitor them (`TauWatch`) and warn above a declared threshold.

## Exact restart

The state of a run (coefficients, time, step count and, for multistep schemes, the history) is
written in full, so that a restarted run continues bit for bit.

## References

- U. M. Ascher, S. J. Ruuth and R. J. Spiteri, *Implicit-explicit Runge–Kutta methods for
  time-dependent partial differential equations*, Appl. Numer. Math. 25 (1997) 151–167.
- D. Wang and S. J. Ruuth, *Variable step-size implicit-explicit linear multistep methods for
  time-dependent partial differential equations*, J. Comput. Math. 26 (2008) 838–855.
- K. J. Burns et al., Phys. Rev. Research 2 (2020) 023068.
