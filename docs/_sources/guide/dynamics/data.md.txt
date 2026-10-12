# Data-driven analysis

Dynamic mode decomposition and spectral proper orthogonal decomposition of snapshots, in the inner product of the physics rather than that of the array.

Snapshots are vectors of values on a grid. Every decomposition takes a `DataInnerProduct`, built
from the quadrature weights of the grid and per-component weights, so that modes are orthogonal in
the energy norm of the flow. The example uses snapshots of the linear Ginzburg–Landau equation.

```{literalinclude} _check.py
:language: python
:start-after: "[dd-setup]"
:end-before: "[/dd-setup]"
:dedent: 4
```

## Dynamic mode decomposition

```{literalinclude} _check.py
:language: python
:start-after: "[dd-dmd]"
:end-before: "[/dd-dmd]"
:dedent: 4
```

`rank` and `amplitudes` are required. On snapshots of a linear system in the span of a few modes,
DMD returns their eigenvalues to rounding. `edmd` and its dictionaries (`monomials`, `rbf`) extend it
to observables of the state.

## Spectral POD

The Welch segmentation (`nfft`, `noverlap`, `window`), the treatment of the mean and one- or
two-sided spectra are all arguments.

**See also:** tutorial {doc}`../../tutorials/13_sensitivity_data` (Towne et al. 2018);
{doc}`dmd <../../api/_generated/_stubs/conatus.expert.dmd>`, {doc}`spod <../../api/_generated/_stubs/conatus.expert.spod>`, {doc}`pod <../../api/_generated/_stubs/conatus.expert.pod>`,
{doc}`DataInnerProduct <../../api/_generated/_stubs/conatus.expert.DataInnerProduct>`.
