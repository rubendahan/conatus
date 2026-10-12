# Errors and what they mean

Every error says what is wrong, where, why it is refused, and what to write instead.

Conatus refuses rather than guesses. When a problem is ill-posed, a choice is missing or a solver
does not converge, it raises an exception that says **what** is wrong, **where** (equation, term,
variable, group), **why** it is refused and a suggested **fix**, which it never applies for you
(see {doc}`../principles`). The last line of every message links to the section of this page
named after the exception class.

All errors derive from `ConatusError`, and each also derives from the built-in exception its code
raised before 0.1 (`ValueError`, `TypeError`, `KeyError`, `RuntimeError`,
`NotImplementedError`, `ImportError`, `FileNotFoundError`), so `except ValueError:` keeps working.

```python
import conatus.errors
print(conatus.errors.explain())   # the tree of exception classes
```

## Warnings

A warning never replaces a result: the result is returned. A warning is shown once per line of
your script, and `explain()` does not repeat every one (it keeps the base residual of a
linearization, the group in a result's header and a growth rule "inferred from the name"), so read
the warnings. A statement that the answer may not be the one the call meant is a
`conatus.errors.CorrectnessWarning` (a `UserWarning`, shown by Python's default filters wherever it
is raised):

| Warning | When |
|---|---|
| `BaseStateWarning` | `m.linearize(about=...)` about a base whose steady residual exceeds `BASE_RESIDUAL_TOL` (not declared with `base_residual="accept"`); refused in v0.2 |
| `GroupWarning` | `eigenvalues()` without `k=` on a periodic model (only k = 0 is solved), or without `m=` on a curvilinear one (only the first group); required in v0.2 |

Resolution detections are `ResolutionWarning`, `UnderResolvedWarning`, `InitialDataWarning` and
`IllPosedWarning`. A former spelling with the same meaning (a renamed argument) is a
`DeprecationWarning`, which Python shows only for code run as `__main__`.

## Catalogue

The list below is generated from `conatus.errors` at each build.

```{include} _generated/errors_list.md
```
