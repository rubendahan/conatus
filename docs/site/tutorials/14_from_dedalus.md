# Importing a Dedalus script

Write a convection cell as a Conatus model, import the same cell from a Dedalus 3 script, and
check that the two agree.

**Goal.** The onset of two-dimensional Boussinesq convection between a hot and a cold wall:

- write it as a native Conatus model: equations as on paper, conditions by wall name, no tau
  terms;
- give a Dedalus 3 script of the same cell to `cn.from_dedalus` and read its line-by-line report;
- run the translation and compare it with the native model as the wall-normal resolution grows;
- optionally, run the original script and compare its result with the translation.

**Expected.** The report has no unsupported line. The native model and the translation are two
tau formulations of the same equations, so they differ by a truncation error that decays
spectrally with $N_z$, down to rounding at $N_z = 32$. When the original script can be run, the
check script finds the translation equal to its result to rounding ($< 10^{-10}$ relative after
40 steps of SBDF2).
**Time:** about 5 s of CPU; the optional check takes under a minute, most of it at import.
**Scripts:** [`t14_from_dedalus.py`](code/t14_from_dedalus.py) (no Dedalus needed),
[`rb_cell_dedalus.py`](code/rb_cell_dedalus.py) (the Dedalus script, written for this tutorial),
[`t14b_dedalus_check.py`](code/t14b_dedalus_check.py) (runs the original script and compares).
**Tests behind it:** `tests/translate/`, `tests/parse/`.
**Guide:** {doc}`../guide/dedalus`, {doc}`../guide/equations`.

## The cell as a native model

The driver is [`t14_from_dedalus.py`](code/t14_from_dedalus.py). To run the blocks of this page
yourself, save [`rb_cell_dedalus.py`](code/rb_cell_dedalus.py) in your working directory first:
the translation reads it. The driver starts with its imports:

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [native]"
:end-before: "# [/native]"
```

You name the unknowns by their role and write the equations as on paper. `lap`, `div` and `grad`
are the operators of the box's coordinates. The unknown `th` is the temperature minus the
conduction profile $1 - z$, so it vanishes on both walls, and `u·ez` is the advection of that
profile. Lengths are in units of the depth and times in units of the thermal diffusion time. The
walls are `bottom` and `top`. The native layer adds one tau term per wall condition by a fixed
rule, and `explain()` prints them.

## The same cell as a Dedalus script

[`rb_cell_dedalus.py`](code/rb_cell_dedalus.py) is a Dedalus 3 script written for this tutorial.
It is a separate file: the driver reads it but does not run it, so do not paste it into the
driver. It reduces the equations to first order and declares its own tau fields and lifts:

```{literalinclude} code/rb_cell_dedalus.py
:language: python
:caption: rb_cell_dedalus.py (separate file, not run by the driver)
:start-after: "# Coordinates, bases"
```

It uses a fixed time step and a deterministic start. The translator marks a `CFL` or a
`fill_random` line **unsupported** and says what to write instead: Conatus runs with a fixed
`dt`, and the random layout of `fill_random` cannot be reproduced.

## Translate and read the report

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [translate]"
:end-before: "# [/translate]"
```

The translator parses the script with `ast` and reads it statement by statement with a restricted
interpreter. It does not execute the script or import Dedalus. In the output below, look for:

- `e_z*tau_z(u_bot)`: a product of two vectors, which the source language reads as a tensor
  product. The report says `outer`.
- `trace(Gu)`: it becomes `div(u)` plus the tau term, and `div(outer(ez, ...))` becomes `dz(...)`.
  Both are exact identities, and both are listed.
- `dealias=1.5`, written in the script. When a script leaves `dealias` out, the translator writes
  the source default of 1 explicitly and reports it.
- The `while` loop: it becomes `solver.run(n)` with the script's fixed `dt`.

## Run the translation

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [run]"
:end-before: "# [/run]"
```

`tr.ivp_solver` takes the script's scheme and `dt`, and replays its `th['g'] = ...` line on the
Conatus grid. SBDF2 is a two-step scheme. `startup="lower_order"` starts it with one first-order
step, as the source script does.

## Native model and translation, side by side

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [compare]"
:end-before: "# [/compare]"
```

The translation keeps the script's tau terms, and the native model places its own. These are two
discretisations of the same equations, so they agree up to the truncation error. The difference
falls spectrally as $N_z$ grows and reaches rounding at $N_z = 32$. Both converge to the same
solution, and the difference measures how far the coarser grids are from it.
`overrides={"N_z": nz}` changes one top-level assignment of the script, and the report lists the
change.

```{figure} media/t14_convergence.png
:alt: The relative difference between the native model and the translated script on a log scale, falling steeply with the number of wall-normal modes down to rounding.
:width: 100%

Relative difference between the native model and the translated script after 40 SBDF2 steps, against nz. The two tau formulations of the same equations differ by an amount that falls spectrally down to rounding.
```

## Check

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t14_from_dedalus.txt
:language: text
```

## Optional: run the original script

[`t14b_dedalus_check.py`](code/t14b_dedalus_check.py) runs the original script, if Dedalus is
installed (`micromamba install -c conda-forge dedalus`), and compares its result with the
translation:

```{literalinclude} code/t14b_dedalus_check.py
:language: python
:caption: t14b_dedalus_check.py (separate script, needs Dedalus)
:start-after: "# [dedalus]"
:end-before: "# [/dedalus]"
```

The translation keeps the script's bases, tau terms, lifts, scheme, start-up and dealiasing, so
the difference is rounding.

Its output:

```{literalinclude} outputs/t14b_dedalus_check.txt
:language: text
```

## Going further

- `print(tr.to_script())` writes a runnable Conatus program (it replays the translation and runs it), with the report as comments.
- For a problem built inside a function, use `cn.from_dedalus(script, entry=("f", {...}))`.
- {doc}`../guide/dedalus` lists the translated subset and the translator's tests on the official
  examples. The relative differences between translated and original results are 1.3e-15 for
  KdV–Burgers, 7e-16 for Poisson, 1.3e-15 for the shear flow, 4e-16 for the Rayleigh–Bénard
  eigenvalue problem and 5e-14 for the pipe-flow eigenvalues on the disk.
