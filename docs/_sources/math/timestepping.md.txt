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

Three families are available:

- **IMEX Runge–Kutta** `RK111`, `RK222`, `RK443` (Ascher, Ruuth and Spiteri 1997), `RKSMR` and
  `RKGFY` (their sources are listed in the time-stepper guide). With step $k$, stage $i$ solves
  $$ (M + k H_{ii} L)\, X_i = M X_0 + k \sum_{j<i} A_{ij} F(X_j, t_j) - k \sum_{j<i} H_{ij} L X_j , $$
  with $(A, H)$ the explicit and implicit tableaux and $t_j = t_0 + c_j k$. The diagonal term
  $H_{ii} L X_i$ has been moved to the left, so both sums run over the earlier stages $j < i$.
  In `RK111`, `RK222`, `RK443` and `RKGFY` all $H_{ii}$ of the implicit stages are equal, so one
  factorisation of $M + k H_{ii} L$ per group serves every stage. `RKSMR` has three different
  diagonal entries and needs one factorisation per stage.
- **Semi-implicit BDF** `SBDF1` to `SBDF4` (Wang and Ruuth 2008): multistep, one solve per step.
- **Crank–Nicolson multistep** `CNAB1`, `CNAB2`, `MCNAB2` and `CNLF2` (Wang and Ruuth 2008):
  Crank–Nicolson-type implicit part, Adams–Bashforth or leapfrog explicit part.

The multistep schemes take their first steps with a declared start-up procedure, printed by
`explain()`.

The time step is fixed and chosen by you. The CFL estimate (`sim.cfl("u", safety=0.5)`, with your
safety factor) limits the step for stability. It does not estimate the time error. The stop time
must be a whole number of fixed time steps from the current time.

## Factorisations

Each group's matrix $M + k H L$ is banded with a border (the tau columns and the boundary rows).
It is factorised once by a banded LU with the border eliminated through a small Schur complement
(capacitance) system; all groups are solved together in one batched call, on the CPU or on the GPU.

## Watching the taus

The tau variables absorb the residual of the truncated equations. Their size along a run is a
cheap diagnostic. Growth of the taus relative to the size of the fields and of the forcing can
indicate that the highest modes are not resolved. A tau also grows when the solution or the forcing
grows, and in fourth-order equations the taus can amplify rounding errors while the fields are
unaffected. A growing tau is therefore a reason to check the resolution, not a proof of
under-resolution. `TauWatch` records the largest tau coefficient of each tau variable, without
normalisation, and warns when it exceeds its first recorded value by a declared factor. To confirm
under-resolution, repeat the run at a larger $N$ and inspect the decay of the spectral
coefficients.

## Exact restart

The state of a run (coefficients, time, step count and, for multistep schemes, the history) is
written in full. A Runge–Kutta step starts from a single state, so no stage data need to be kept
between steps. Restarted with the same problem, scheme, step and build, the run continues bit for
bit.

## References

- U. M. Ascher, S. J. Ruuth and R. J. Spiteri, *Implicit-explicit Runge–Kutta methods for
  time-dependent partial differential equations*, Appl. Numer. Math. 25 (1997) 151–167.
- D. Wang and S. J. Ruuth, *Variable step-size implicit-explicit linear multistep methods for
  time-dependent partial differential equations*, J. Comput. Math. 26 (2008) 838–855.
- K. J. Burns et al., Phys. Rev. Research 2 (2020) 023068.
