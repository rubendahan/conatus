# API reference

Conatus has two layers. Most scripts only need the first.

- **The user layer**, `import conatus as cn`: named geometries (`cn.Box`, `cn.Channel`, `cn.Disk`,
  …), the model in which equations are written as on paper (`cn.Model`), the objects it returns
  (simulation, linearisation, compiled problem), and the utilities (`cn.problems`, `cn.runs`,
  `cn.doctor()`, `cn.from_dedalus`, `cn.parse_equation`). See {doc}`user`.
- **The expert layer**, `conatus.expert`: the mathematical intermediate representation (fields,
  bases, operators, problems written term by term) and every toolkit that works on it (solvers,
  time stepping, adjoints, invariant solutions, continuation, bifurcations, Floquet–Hill,
  non-modal analysis, chaos, data-driven methods, telemetry, provenance). See {doc}`expert`.

The two layers meet in one place: `m.compile(N).problem` is an ordinary expert-layer problem, and
`sim.solver` is an ordinary expert-layer time stepper, so every toolkit applies to a native model.

```{toctree}
:maxdepth: 2

user
expert
```
