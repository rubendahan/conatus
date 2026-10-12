# 14. Coming from Dedalus

Write a convection cell as a Conatus model, read the same cell from a Dedalus script, and check
that the two agree.

**Goal.** The onset of two-dimensional Boussinesq convection between a hot and a cold wall:

- write it as a native Conatus model: equations as on paper, conditions by wall name, no tau
  terms;
- give a Dedalus 3 script of the same cell to `cn.from_dedalus` and read its line-by-line report;
- run the translation and compare it with the native model as the wall-normal resolution grows;
- optionally, run the script with Dedalus itself and compare it with the translation.

**Expected.** The report has no unsupported line. The native model and the translation are two
tau formulations of the same equations, so they differ by a truncation error that decays
spectrally with $N_z$, down to rounding at $N_z = 32$. With Dedalus installed, the check script
finds the translation equal to Dedalus to rounding ($< 10^{-10}$ relative after 40 steps of
SBDF2).
**Time:** about 5 s of CPU; the Dedalus check takes under a minute, most of it importing Dedalus.
**Scripts:** [`t14_from_dedalus.py`](code/t14_from_dedalus.py) (no Dedalus needed),
[`rb_cell_dedalus.py`](code/rb_cell_dedalus.py) (the Dedalus script, written for this tutorial),
[`t14b_dedalus_check.py`](code/t14b_dedalus_check.py) (the comparison with Dedalus).
**Tests behind it:** `tests/translate/`, `tests/parse/`, and the official Dedalus examples in
`tests/translate/run_dedalus_examples.py` (fetched at test time, never stored here).
**Guide:** {doc}`../guide/dedalus`, {doc}`../guide/equations`.

## The cell as a native model

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [native]"
:end-before: "# [/native]"
```

You name the unknowns by their role and write the equations as you would on paper; `lap`, `div`
and `grad` are the operators of the box's coordinates. The unknown `th` is the temperature minus
the conduction profile $1 - z$, so it vanishes on both walls, and `u·ez` is the advection of that
profile; lengths are in units of the depth and times of the thermal diffusion time. The walls are `bottom` and `top`. There is
no tau field and no lift: the native layer adds one tau term per wall condition by a fixed rule,
which `explain()` prints.

## The same cell in Dedalus

[`rb_cell_dedalus.py`](code/rb_cell_dedalus.py) is the script a Dedalus user would write for this
cell. It reduces the equations to first order, so it declares the tau fields and the lifts
itself:

```{literalinclude} code/rb_cell_dedalus.py
:language: python
:start-after: "# Coordinates, bases"
```

It uses a fixed time step and a deterministic start. Conatus never adapts `dt` by itself, and
`fill_random` cannot be repeated outside Dedalus. Given a script with a `CFL` or a `fill_random`
line, the translator marks that line **unsupported** and says what to write instead.

## Translate and read the report

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [translate]"
:end-before: "# [/translate]"
```

The script is parsed with `ast` and read statement by statement by a restricted interpreter.
Nothing in it is executed, and Dedalus is not imported. In the output below, look for:

- `e_z*tau_z(u_bot)`: a product of two vectors, which Dedalus reads as a tensor product. The
  report says `outer`.
- `trace(Gu)`: it becomes `div(u)` plus the tau term, and `div(outer(ez, ...))` becomes `dz(...)`.
  Both are exact identities, and both are listed.
- `dealias=1.5`, written in the script. When a script leaves `dealias` out, the translator writes
  the Dedalus default of 1 explicitly and reports it.
- The `while` loop: it becomes `solver.run(n)` with the script's fixed `dt`.

## Run the translation

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [run]"
:end-before: "# [/run]"
```

`tr.ivp_solver` takes the script's scheme and `dt`, and replays its `th['g'] = ...` line on the
Conatus grid. SBDF2 is a two-step scheme: `startup="lower_order"` is the start-up Dedalus uses
(one first-order step), written here explicitly.

## Native model and translation, side by side

```{literalinclude} code/t14_from_dedalus.py
:language: python
:start-after: "# [compare]"
:end-before: "# [/compare]"
```

The translation keeps the script's tau terms; the native model places its own. These are two
discretisations of the same equations, so they agree up to the truncation error, not to
rounding. The difference falls spectrally as $N_z$ grows and reaches rounding at $N_z = 32$:
both converge to the same solution, and the difference measures how far the coarser grids are
from it. `overrides={"N_z": nz}` changes one top-level assignment of the script, and the report
lists the change.

## Compare with Dedalus

[`t14b_dedalus_check.py`](code/t14b_dedalus_check.py) runs the same file with Dedalus, if
Dedalus is installed (`micromamba install -c conda-forge dedalus`), and compares the result with
the translation:

```{literalinclude} code/t14b_dedalus_check.py
:language: python
:start-after: "# [dedalus]"
:end-before: "# [/dedalus]"
```

Both codes now use the same method: the same bases, tau terms, lifts, scheme, start-up and
dealiasing. So the difference is rounding.

## Output (as executed)

```{literalinclude} outputs/t14_from_dedalus.txt
:language: text
```

The Dedalus check:

```{literalinclude} outputs/t14b_dedalus_check.txt
:language: text
```

## Going further

- `print(tr.to_script())` writes a readable Conatus script, with the report as comments.
- For a problem built inside a function, use `cn.from_dedalus(script, entry=("f", {...}))`.
- {doc}`../guide/dedalus` lists the translated subset and the comparisons on the official Dedalus
  examples. These include KdV–Burgers (1.6e-15), Poisson (7e-16), shear flow (1.3e-15), the
  Rayleigh–Bénard eigenvalue problem (4e-16) and the pipe-flow eigenvalues on the disk (5e-14).
