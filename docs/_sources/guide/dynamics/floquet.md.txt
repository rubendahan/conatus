# Floquet analysis

Stability of time-periodic flows by the Hill method: the Floquet exponents are eigenvalues of one large, structured problem.

For a linear system with a periodic coefficient, $\dot q + L(t)\,q = 0$ with period $2\pi/\omega$,
solutions are $q = e^{\mu t} p(t)$ with $p$ periodic. Expanding $p = \sum_n p_n e^{i n \omega t}$
turns the problem into the Hill eigenvalue problem

$$ \big(L_0 + (\mu + i n\omega) M\big)\,p_n + \sum_{m \neq 0} L_m\, p_{n-m} = 0 , $$

block-banded, with time discretised spectrally like space. The exponents $\mu$ are defined modulo
$i\omega$.

## A first example: Mathieu's equation

```{literalinclude} _check.py
:language: python
:start-after: "[fl-mathieu]"
:end-before: "[/fl-mathieu]"
:dedent: 4
```

`HillBlocks` takes the diagonal block, the couplings by harmonic and the number of harmonics `Nh`.
`exponents_dense` returns all exponents; `secant_root` refines one of them on the exact
determinant and returns a `SecantRoot` (`mu`, `converged`, `residual` = the scaled determinant at
`mu`, `iterations`, `reasons`). A search that ends on `max_iterations` raises `ConvergenceError`
carrying the last iterate (`err.result`) unless `allow_unconverged=True`, which returns it with
`converged=False`.

## A discretised time-periodic problem

For a time-periodic flow given as a discretised problem, `hill_from_plan` builds the Hill operator
from the harmonics of the base flow and checks that the operator is linear in it. The compiler
extracts the harmonic matrices $L_m$ of the formula above from the plan. The spatial resolution,
the number of harmonics `Nh` and the number of base-flow harmonics kept are arguments of the call.

## Any model about a periodic base flow

In the native layer, `m.linearize(about={u: lambda y, t: ...}, period=P).floquet(N, k=...,
method="hill", Nh=..., targets=[...], tol=1e-12)` gives the exponents of one wavenumber, from the same
operator as the transient gain ({doc}`nonmodal`). The expert form
`conatus.growth.floquet_exponents(build, base_harmonics, kx, kz, method=..., fields=...)` takes a
linear problem per wave; `method="monodromy"` compares with the eigenvalues of the time-stepped
monodromy matrix.

In a spherical shell, where the Hill operator would be very large, `conatus.floquet.shell_floquet`
uses the one-period map of the time stepper and Arnoldi iteration instead: {doc}`../shells`.

## Validation

An exponent is trusted when it is found again: refine the spatial resolution and the number of
harmonics `Nh` until its digits stop moving, and check that its residual (`residual` of a
`SecantRoot`, `residuals` of a native `floquet` result) stays at round-off. Exponents are defined
modulo $i\omega$: compare them after folding the imaginary part into one band.

**See also:** {doc}`HillBlocks <../../api/_generated/_stubs/conatus.expert.HillBlocks>`,
{doc}`hill_from_plan <../../api/_generated/_stubs/conatus.expert.hill_from_plan>`,
{doc}`secant_root <../../api/_generated/_stubs/conatus.expert.secant_root>`, and the mathematics in
{doc}`../../math/floquet`.
