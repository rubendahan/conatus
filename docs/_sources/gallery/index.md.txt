# Gallery: fluids everywhere

One script per canonical case, written as in the paper that introduced it: the equation, physical
comments, `explain()`, and a result checked against a published value or a closed form. Every
script checks its result and stops with an error if it disagrees with the reference. The pages below are generated from
`examples/` at each build; the values are those recorded in `examples/README.md`.

```{include} _generated/table.md
```

No missing capability met while writing the gallery was worked around silently: each script says
what it does instead.

```{toctree}
:hidden:
:glob:

_generated/*
```
