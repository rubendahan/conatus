# 13. Where a global mode is born, and what the data say about it

The adjoint tells where to act; the data, read in the right norm, tell what the operator predicts.

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
conditions); the product is the energy of the system; nothing is normalized for you without being
said in `explain()`.

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [sensitivity]"
:end-before: "# [/sensitivity]"
```

```{figure} media/t13_modes.png
:alt: Two bell-shaped curves: the direct global mode peaks downstream (right) and the adjoint mode upstream (left), barely overlapping.
:width: 100%

The direct and adjoint global modes of the Ginzburg-Landau model of the page. Advection pushes the direct mode downstream and the adjoint upstream: the operator is far from normal.
```

```{figure} media/t13_wavemaker.png
:alt: A single bump centred at x = 0, the structural sensitivity, with its maximum marked in red.
:width: 100%

The wavemaker S(x) = |q†(x)| |q(x)|, where the overlap of the two modes is largest: the region where a local change of the model moves the eigenvalue most. The maximum matches Chomaz's closed form.
```

## Data: DMD in the exact norm

Grid values on Gauss nodes with Fejér weights: the quadrature is exact for the resolved field, so
the data norm is the energy norm of the model (EᴴWE = 2I).

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [data]"
:end-before: "# [/data]"
```

```{figure} media/t13_dmd.png
:alt: Eigenvalues of the model in grey; four of them circled in red, each with a black cross from the decomposition of the data exactly at its centre.
:width: 100%

Dynamic mode decomposition of a transient made of four global modes: the DMD eigenvalues (crosses) land on the model's (red circles) to 10⁻⁹, from the snapshots alone.
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
