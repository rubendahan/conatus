# Errors and what they mean

A `ConatusError` states what is wrong, where (equation, term, variable, group), why the call is
refused, and a suggested fix. It is raised when a problem is ill-posed, a required choice is
missing or a solver does not converge. The fix is a suggestion: Conatus does not apply it for you
({doc}`../principles`). The last line of each message links to the section of this page named
after the exception class. An unexpected Python exception (an `IndexError` from inside Conatus, for
example) is a bug: please report it.

All errors derive from `ConatusError`, and each also derives from the built-in exception its code
raised before 0.1 (`ValueError`, `TypeError`, `KeyError`, `RuntimeError`,
`NotImplementedError`, `ImportError`, `FileNotFoundError`), so `except ValueError:` keeps working.

```python
import conatus.errors
print(conatus.errors.explain())   # the tree of exception classes
```

## Warnings

A warning comes with a returned result. Read the warning's category and message before you use
that result. Python shows a warning once per line of your script. `explain()` repeats only some of
them: the base residual of a linearization, the group in a result's header, and a growth rule
"inferred from the name". When the answer may not be the one the call meant, the warning is a
`conatus.errors.CorrectnessWarning`. It is a `UserWarning`, so Python's default filters show it:

| Warning | When |
|---|---|
| `BaseStateWarning` | `m.linearize(about=...)` about a base whose steady residual exceeds `BASE_RESIDUAL_TOL` (not declared with `base_residual="accept"`); refused from 0.3 |
| `GroupWarning` | `eigenvalues()` without `k=` on a periodic model (only k = 0 is solved), or without `m=` on a curvilinear one (only the first group); required from 0.3 |

Resolution detections are `ResolutionWarning`, `UnderResolvedWarning`, `InitialDataWarning` and
`IllPosedWarning`. A former spelling with the same meaning (a renamed argument) is a
`DeprecationWarning`, which Python shows only for code run as `__main__`.

## Catalogue

The list below is generated from `conatus.errors` at each build.

```{include} _generated/errors_list.md
```
