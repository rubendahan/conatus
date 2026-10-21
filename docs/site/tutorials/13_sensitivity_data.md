# Where a global mode is born, and what the data say about it

Compute the sensitivity of an eigenvalue to changes in μ(x), then recover the model's eigenvalues from snapshots.

**Goal.** Compute the direct and adjoint global modes of the linear Ginzburg–Landau equation, its
wavemaker (Giannetti & Luchini 2007) and the gradient of the eigenvalue with respect to the
"base flow" μ(x) (Marquet, Sipp & Jacquin 2008); then decompose data produced by the same model:
DMD of a transient.
**Expected.** Wavemaker maximum S₀(0) = |χ|/√π and dλ₀/dμ₂ = 1/(4χ²) (closed forms, Chomaz 2005)
to 1e-9; DMD returns the four least stable eigenvalues to 1e-9.
**Time:** about 20 s of CPU.
**Script:** [`t13_sensitivity_data.py`](code/t13_sensitivity_data.py) · **Tests behind it:**
`tests/sensitivity/`, `tests/datadriven/`.
The Ginzburg–Landau model comes ready (`linear_ginzburg_landau`, expert layer): the sensitivity
tools need it as a function of the "base flow" μ(x), which they differentiate.

## The model

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/model]"
```

## Structural and base-flow sensitivity

The adjoint is declared (`adjoint="continuous"`: the formal adjoint with its checked boundary
conditions). The inner product is the energy of the system. `explain()` states each normalization
that is applied.

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [sensitivity]"
:end-before: "# [/sensitivity]"
```

```{figure} media/t13_modes.png
:alt: Top: two bell-shaped curves, the direct global mode peaking downstream (right) and the adjoint mode upstream (left), overlapping little. Bottom: their product, the wavemaker, a single bump centred at x = 0 with its maximum marked in red.
:width: 100%

The direct and adjoint global modes of the Ginzburg-Landau model of the page (top) and the wavemaker S(x) = |q†(x)| |q(x)| (bottom). Advection pushes the direct mode downstream and the adjoint upstream; S is largest where they overlap, the region where a local change of the model moves the eigenvalue most. Its maximum, at |x| < 3×10⁻¹⁰, equals Chomaz's closed form |χ|/√π to a relative 7×10⁻¹².
```

## Data: DMD in the exact norm

The data are grid values on Gauss nodes with Fejér weights. This quadrature integrates the resolved
field without error, so the data norm is the energy norm of the model (EᴴWE = 2I).

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [data]"
:end-before: "# [/data]"
```

```{figure} media/t13_dmd.png
:alt: Eigenvalues of the model in grey; four of them circled in red and numbered 1 to 4, each with a black cross from the decomposition of the data at its centre.
:width: 100%

Dynamic mode decomposition of a transient made of four global modes (numbered by growth rate): the DMD eigenvalues (crosses), computed from the snapshots alone, against the model's (red circles). The distances, too small to see here, are 2×10⁻¹⁴, 4×10⁻¹³, 2×10⁻¹², 5×10⁻¹³ for modes 1 to 4.
```

## Checks

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output

```{literalinclude} outputs/t13_sensitivity_data.txt
:language: text
```
