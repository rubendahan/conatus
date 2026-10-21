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

`conatus.expert` re-exports, under their own names, the public objects of the user-facing modules
that this documentation shows. Since 0.1.1 the list is reviewed name by name. A helper, a constant
or a result type that no page shows stays in its module (`conatus.contour.box`,
`conatus.timestep.rk_tableau`). Written as `conatus.expert.<name>`, such a name still works in 0.1.x
with a `DeprecationWarning` that names the module, and stops working in 0.3. Every exception and
warning class of `conatus.errors` is exported (each has its entry in {doc}`../guide/errors`).

The Studio's telemetry recorder and wire protocol are not part of the expert layer. Import them as
`conatus.telemetry`. The modules themselves stay reachable as attributes (`conatus.expert.ecs`,
`conatus.expert.provenance`), but their canonical name is `conatus.<module>`.
`conatus.expert.explain()` lists the name clashes and how each one is handled:

- two modules export different objects under the same name: the first one is kept, the other stays
  reachable as `conatus.<module>.<name>`;
- a name with two unrelated meanings is not exported at all: `record` is refused
  (`AmbiguousNameError`); the run provenance record is exported as `provenance_record`, and the
  telemetry frame is `conatus.telemetry.record`;
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
