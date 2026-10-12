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

## Data: DMD in the exact norm

Grid values on Gauss nodes with Fejér weights: the quadrature is exact for the resolved field, so
the data norm is the energy norm of the model (EᴴWE = 2I).

```{literalinclude} code/t13_sensitivity_data.py
:language: python
:start-after: "# [data]"
:end-before: "# [/data]"
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
