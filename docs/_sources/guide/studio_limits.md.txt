# Studio limits and timings

Every number the Studio, the hub, the relay and the telemetry use on their own is declared in one
table, `conatus.studio.constants`. Print it with
`python -c "import conatus.studio.constants as c; print(c.explain())"`. The table below is
generated from it; a test fails if the code and the table differ. The last column says how each
value is changed: some are command-line options, others are constants of the source code.

```{include} _generated/studio_constants.md
```
