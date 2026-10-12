# Expert layer: `conatus.expert`

```python
import conatus.expert as ex          # or: from conatus.expert import *
```

The expert layer is the mathematical intermediate representation that native models compile to,
together with every toolkit that operates on it. Use it to write a problem term by term (bases,
tau variables and lifts chosen by hand), to reach options the native layer does not expose, or to
apply a toolkit to `m.compile(N).problem` or `sim.solver`.

A name reachable in both layers means the same thing in both: the same object, or, for
`discretize`, `linearize`, `methods` and `verify`, the package `cn.<name>` whose call is the
function `conatus.expert.<name>` (`cn.discretize(problem)` works; `cn.discretize.Plan` too).
`conatus.api` is the former name of the expert layer: it still works, with a `FutureWarning`, and
will be removed. Two names keep their former meaning only there: `sensitivity` (now the package
`conatus.sensitivity`; the dense-eigenproblem helper is `conatus.solvers.sensitivity`) and
`Channel` (now `ChannelBaseFlow`; `cn.Channel` is the native geometry).

`conatus.expert` re-exports the public objects of every user-facing module under their own names
and adds nothing. Name clashes are never resolved silently; `conatus.expert.explain()` lists them:

- two modules export different objects under the same name: the first one is kept, the other stays
  reachable as `conatus.<module>.<name>`;
- a name with two unrelated meanings is not exported at all: `record` is refused
  (`AmbiguousNameError`) and exported as `telemetry_record` and `provenance_record`;
- an object named like a module (`linearize`, `discretize`, `methods`): the module stays
  `conatus.<module>`, callable as the function;
- generic helpers stay in their module; the run registry is a module, `conatus.expert.runs`
  (= `cn.runs`). The compiled runtime and the MPI backend are left out of the single import.

```{eval-rst}
.. autofunction:: conatus.expert.explain
```

```{toctree}
:maxdepth: 1

_generated/reference
```
