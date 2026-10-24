# Scientific examples

One script per canonical case, written as in the paper that introduced it: the equation, physical
comments, `explain()`, and a result checked against a published value or a closed form. Each
script ends with an `assert` against that value, with its tolerance, so it stops with an error if
the value is not reproduced. The pages below are generated from
`examples/` at each build; the values are those recorded in `examples/README.md`.

```{include} _generated/table.md
```

Where a script needs a capability that the library does not have, it says so and says what it does
instead.

```{toctree}
:hidden:
:glob:

_generated/*
```
